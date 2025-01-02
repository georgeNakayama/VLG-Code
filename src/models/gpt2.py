import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
from collections import defaultdict
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Optional, Union, Dict, Any, Literal
from data.datasets.panel_configs import StandardizeConfig
from data.garment_tokenizers.special_tokens import PanelEdgeTypeIndices, PanelEdgeTypeV3
import hydra
import math
# -----------------------------------------------------------------------------


@dataclass
class GPTConfig:
    _target_: str = "models.decoders.gpt2.GPT"
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    encoder: Literal["gpt2"] = "gpt2"

class GPTToken(nn.Module):

    def __init__(
        self, 
        block_size: int, 
        n_layer: int, 
        n_head: int, 
        n_embd: int, 
        vocab_size: int,
        ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.n_embd),
            wpe = nn.Embedding(self.block_size, self.n_embd),
            h = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(n_embd, n_head, dim_feedforward=n_embd * 4, dropout=0, activation=F.gelu, batch_first=True, norm_first=True),
                n_layer, norm=nn.LayerNorm(n_embd)
            ),
        ))
        self.proj_in = nn.Linear(n_embd, n_embd, bias=False)
        
        self.proj_feature_txt = nn.Linear(1024, n_embd, bias=False)

        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

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
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(
        self,
        caption_features: torch.FloatTensor,
        start_idx: int = 0,
        end_idx: int = 1024,
        temperature: int = 1.0,
        top_k: Optional[int] = None,
    ):
        assert caption_features.shape[0] == 1, "batch size must be 1"
        input_ids = torch.full((1, 1), start_idx, dtype=torch.long, device=caption_features.device)
        while True:
            output_dict = self.forward(
                caption_features, 
                input_ids,
                labels=None)
            logits = output_dict['logits']
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if idx_next == end_idx or input_ids.shape[1] == self.block_size:
                break
        return {"output_ids":input_ids}
    
    def forward(
        self,
        caption_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        **kwargs,
    ):
        # idx is of shape (B, T)
        B, T = input_ids.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        # forward the token and posisition embeddings
        image_feature = self.proj_feature_txt(caption_features)
        
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (B, T, n_embd)
        
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        x = x * math.sqrt(self.n_embd)
        x = self.proj_in(x)
        x = self.transformer.h(x, image_feature, tgt_mask=nn.Transformer.generate_square_subsequent_mask(x.shape[1], x.device))
        # forward the final layernorm and the classifier
        logits = self.lm_head(x) # (B, T, vocab_size)
        ce_loss = None
        total_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), reduction='mean')
            total_loss = ce_loss
        
        return_dict = {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "logits": logits
        }
        return return_dict

    @classmethod
    def from_pretrained(cls, model_type: Literal['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']) -> Tuple[nn.Module, GPTConfig]:
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        model_config = {
            'gpt2':         GPTConfig(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  GPTConfig(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   GPTConfig(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      GPTConfig(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        model_config.vocab_size = 50257 # always 50257 for GPT model checkpoints
        model_config.block_size = 1024 # always 1024 for GPT model checkpoints
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
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
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

    