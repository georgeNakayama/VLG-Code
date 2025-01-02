import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Optional, Union, Dict, Any, Literal
import tiktoken
import hydra
import loralib as lora
import logging 
log = logging.getLogger(__name__)
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd: int, n_head: int, lora_attn_dim: int, lora_attn_alpha: int, lora_attn_dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = lora.MergedLinear(
            n_embd, n_embd * 3, 
            r=lora_attn_dim, 
            lora_alpha=lora_attn_alpha, 
            lora_dropout=lora_attn_dropout, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=True,
            merge_weights=False
        )
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, n_embd: int, n_head: int, lora_attn_dim: int, lora_attn_alpha: int, lora_attn_dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, lora_attn_dim, lora_attn_alpha, lora_attn_dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    _target_: str = "models.decoders.lora_gpt2.GPT"
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    encoder: Literal["gpt2"] = "gpt2"
    lora_attn_dim: int = 0
    lora_embd_dim: int = 0
    lora_attn_alpha: int = 2
    lora_attn_dropout: float = 0.1
    new_vocab_size: int = 0
    ignore_index: Optional[int] = None

class GPT(nn.Module):

    def __init__(
        self, 
        block_size: int, 
        vocab_size: int, 
        n_layer: int, 
        n_head: int, 
        n_embd: int, 
        encoder: Literal["gpt2"], 
        ignore_index: Optional[int] = None,
        lora_embd_dim: int = 128,
        lora_attn_dim: int = 128,
        lora_attn_alpha: int = 2,
        new_vocab_size: int = 0,
        lora_attn_dropout: float = 0.1,):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.ignore_index = ignore_index if ignore_index is not None else -100
        self.encoder = tiktoken.get_encoding(encoder)
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_attn_dropout = lora_attn_dropout
        self.lora_embd_dim = lora_embd_dim
        self.new_vocab_size = new_vocab_size

        self.transformer = dict(
            wte = lora.Embedding(self.vocab_size, self.n_embd, self.lora_embd_dim),
            wpe = lora.Embedding(self.block_size, self.n_embd, self.lora_embd_dim),
            h = nn.ModuleList([Block(n_embd, n_head, lora_attn_dim, lora_attn_alpha, lora_attn_dropout) for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.n_embd),
        )
        if new_vocab_size > 0:
            self.transformer["new_wte"] = nn.Embedding(new_vocab_size, self.n_embd)
            self.new_lm_head = nn.Linear(self.n_embd, new_vocab_size, bias=False)
            self.new_lm_head.ZEROINIT = True

        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        self.transformer = nn.ModuleDict(self.transformer)
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if hasattr(module, 'ZEROINIT'):
                torch.nn.init.zeros_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def encode_text(self, text: str)-> List[int]:
        return self.encoder.encode(text)

    def decode_text(self, tokens: List[int])-> str:
        return self.encoder.decode(tokens)
    
    def forward(self, idx: torch.Tensor, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        
        if self.new_vocab_size > 0:
            new_idx_mask = idx >= self.vocab_size
            new_idx = idx - self.vocab_size
            new_tok_emb = self.transformer.new_wte(new_idx[new_idx_mask])
            idx = idx[~new_idx_mask]
        else:
            new_idx_mask = torch.zeros_like(idx, dtype=torch.bool)
            
        original_tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        tok_emb = torch.zeros(B, T, self.n_embd, device=idx.device)
        if self.new_vocab_size > 0:
            tok_emb[new_idx_mask] = new_tok_emb
        tok_emb[~new_idx_mask] = original_tok_emb
        
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if self.new_vocab_size > 0:
            new_logits = self.new_lm_head(x) # (B, T, new_vocab_size)
            logits = torch.cat([logits, new_logits], dim=-1) # (B, T, vocab_size+new_vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.ignore_index, reduction='mean')
        return logits, loss

    @classmethod
    def from_pretrained(
        cls, 
        model_type: Literal['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], 
        original_model_config: GPTConfig) -> Tuple[nn.Module, GPTConfig]:
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        log.info("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        model_config = {
            'gpt2':         GPTConfig(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  GPTConfig(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   GPTConfig(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      GPTConfig(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        model_config.vocab_size = 50257 # always 50257 for GPT model checkpoints
        model_config.block_size = 1024 # always 1024 for GPT model checkpoints
        model_config.lora_attn_dim = original_model_config.lora_attn_dim
        model_config.lora_attn_alpha = original_model_config.lora_attn_alpha
        model_config.lora_attn_dropout = original_model_config.lora_attn_dropout
        model_config.lora_embd_dim = original_model_config.lora_embd_dim
        model_config.new_vocab_size = original_model_config.new_vocab_size
        model_config.ignore_index = original_model_config.ignore_index
        # create a from-scratch initialized minGPT model
        model: GPT = hydra.utils.instantiate(model_config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        extra_keys = [k for k in sd_keys if k not in sd_keys_hf]
        sd_keys = [k for k in sd_keys if k not in extra_keys]
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model, model_config

    