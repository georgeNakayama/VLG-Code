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

def make_mlp(input_dim, hidden_dim, output_dim, num_layers, dropout=0):
    """ Very simple multi-layer perceptron (also called FFN)"""
    h = [input_dim] + [hidden_dim] * (num_layers - 1)
    layers = []
    for i in range(num_layers - 1):
        layers.append(nn.Linear(h[i], h[i +1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(h[-1], output_dim))
    layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

def _discretize(x, bin, bounds, dim):
    min_bounds = torch.tensor(bounds[:dim]).cuda()
    max_bounds = torch.tensor(bounds[dim:]).cuda()
    x = torch.minimum(max_bounds.cuda(), x)
    x = torch.maximum(min_bounds.cuda(), x)
    x = (x - min_bounds) / (max_bounds - min_bounds)
    x = torch.round(x * bin) / bin
    x = x * (max_bounds - min_bounds) + min_bounds
    return x


@dataclass
class GPTConfig:
    _target_: str = "models.decoders.gpt2.GPT"
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    encoder: Literal["gpt2"] = "gpt2"

class GPTTokenRegression(nn.Module):

    def __init__(
        self, 
        block_size: int, 
        n_layer: int, 
        n_head: int, 
        n_embd: int, 
        vocab_size: int,
        panel_edge_indices: PanelEdgeTypeIndices,
        gt_stats: StandardizeConfig,
        edge_loss_weight: float = 0.1,
        bin_num: int = 256,
        verts_bounds: List[float] = [-4, -4, 4, 4],
        transf_bounds: List[float] = [-4, -4, -4, -1, -1, -1, -1, 4, 4, 4, 1, 1, 1, 1],
        ):
        super().__init__()
        self.gt_stats = gt_stats
        self.edge_loss_weight = edge_loss_weight
        self.bin_num = bin_num
        self.verts_bounds = verts_bounds
        self.transf_bounds = transf_bounds
        self.block_size = block_size
        self.panel_edge_indices = panel_edge_indices
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.zero_vert = - torch.tensor(self.gt_stats.vertices.shift) / torch.tensor(self.gt_stats.vertices.scale)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.n_embd),
            wpe = nn.Embedding(self.block_size, self.n_embd),
            h = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(n_embd, n_head, dim_feedforward=n_embd * 4, dropout=0, activation=F.gelu, batch_first=True, norm_first=True),
                n_layer, norm=nn.LayerNorm(n_embd)
            ),
        ))
        self.proj_in = nn.Linear(n_embd, n_embd, bias=False)
        self.transf_proj = nn.Sequential(
            nn.Linear(self.panel_edge_indices.get_index_param_num(self.panel_edge_indices.move_idx), n_embd, bias=True),
            nn.GELU(),
            nn.Linear(n_embd, n_embd)
        )
        self.vertex_proj = nn.Sequential(
            nn.Linear(2, n_embd, bias=True),
            nn.GELU(),
            nn.Linear(n_embd, n_embd)
        )
        
        self.transformation_fc = nn.Sequential(*[
            nn.Linear(n_embd, n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(n_embd, self.panel_edge_indices.get_index_param_num(self.panel_edge_indices.move_idx)),
            nn.Dropout(0.0),
        ])
        self.transformation_fc.train()
        for p in self.transformation_fc.parameters():
            p.requires_grad = True
            
        line_curve_out_dim = 4 if self.panel_edge_indices.get_token_indices(PanelEdgeTypeV3.CUBIC) == -1 else 6
        if self.panel_edge_indices.get_token_indices(PanelEdgeTypeV3.ARC) != -1:
            line_curve_out_dim += 2
        
        self.line_curve_fc = nn.Sequential(*[
            nn.Linear(n_embd, n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(n_embd, 4),
            nn.Dropout(0.0),
        ])
        self.line_curve_fc.train()
        
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
        pred_params = defaultdict(list) 
        pred_endpoints = None
        pred_endpoints_mask = None
        pred_transformations = None
        pred_transformations_mask = None
        while True:
            output_dict = self.forward(
                caption_features, 
                input_ids, 
                labels=None,
                param_target_endpoints=pred_endpoints, 
                param_target_endpoints_mask=pred_endpoints_mask,
                param_target_transformations=pred_transformations,
                param_target_transformations_mask=pred_transformations_mask,
                return_hidden=True)
            logits = output_dict['logits']
            hidden_states = output_dict['hidden'][:, -1].unsqueeze(1)
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            if torch.isin(idx_next, torch.tensor(self.panel_edge_indices.get_all_edge_indices()).to(input_ids)).flatten():
                edge_type = self.panel_edge_indices.get_index_token(idx_next)
                panel_params = self.line_curve_fc(hidden_states)
                if edge_type == PanelEdgeTypeV3.LINE:
                    panel_params = panel_params[..., :2]
                elif edge_type == PanelEdgeTypeV3.CURVE:
                    panel_params = panel_params[..., :4]
                elif edge_type == PanelEdgeTypeV3.CLOSURE_CURVE:
                    panel_params = panel_params[..., 2:4]
                elif edge_type == PanelEdgeTypeV3.CLOSURE_LINE:
                    panel_params = self.zero_vert.to(panel_params).reshape(1, 1, 2)
                pred_params[idx_next.item()].append(panel_params)
                if edge_type.is_closure():
                    pred_endpoints = torch.cat([pred_endpoints, self.zero_vert.to(panel_params).reshape(1, 1, 2)], dim=1) if pred_endpoints is not None else self.zero_vert.to(panel_params).reshape(1, 1, 2)
                else:
                    pred_endpoints = torch.cat([pred_endpoints, panel_params[..., :2]], dim=1) if pred_endpoints is not None else panel_params[..., :2]
                pred_endpoints_mask = torch.ones(1, pred_endpoints.shape[1]).to(pred_endpoints).bool()
            elif idx_next == self.panel_edge_indices.get_token_indices(PanelEdgeTypeV3.MOVE):
                transf_params = self.transformation_fc(hidden_states)
                pred_params[idx_next.item()].append(transf_params)
                pred_transformations = torch.cat([pred_transformations, transf_params], dim=1) if pred_transformations is not None else transf_params
                pred_transformations_mask = torch.ones(1, pred_transformations.shape[1]).to(pred_transformations).bool()
            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if idx_next == end_idx or input_ids.shape[1] == self.block_size:
                break
        pred_params = {k: torch.cat(v, dim=1)[0] for k, v in pred_params.items()}
        return {"output_ids":input_ids, "params": pred_params}
    
    def forward(
        self,
        caption_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        param_targets: Optional[Dict[int, torch.FloatTensor]] = None,
        param_target_endpoints: Optional[torch.FloatTensor] = None,
        param_target_endpoints_mask: Optional[torch.BoolTensor] = None,
        param_target_transformations: Optional[torch.FloatTensor] = None,
        param_target_transformations_mask: Optional[torch.BoolTensor] = None,
        param_target_masks: Optional[Dict[int, torch.BoolTensor]] = None,
        return_hidden: bool = False,
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
        
        edge_mask = torch.isin(input_ids, torch.tensor(self.panel_edge_indices.get_all_edge_indices()).to(input_ids))
        transf_mask = input_ids == self.panel_edge_indices.get_token_indices(PanelEdgeTypeV3.MOVE)
        
            
        x = tok_emb + pos_emb
        if edge_mask.any():
            _endpoints = param_target_endpoints[param_target_endpoints_mask]
            _endpoints = _discretize(_endpoints, self.bin_num, self.verts_bounds, 2)
            edge_embeds = self.vertex_proj(_endpoints)
            x[edge_mask] = x[edge_mask] + edge_embeds
        
        if transf_mask.any():
            _transformations = param_target_transformations[param_target_transformations_mask]
            _transformations = _discretize(_transformations, self.bin_num, self.transf_bounds, 7)
            transf_embeds = self.transf_proj(_transformations)
            x[transf_mask] = x[transf_mask] + transf_embeds
        
        # forward the blocks of the transformer
        x = x * math.sqrt(self.n_embd)
        x = self.proj_in(x)
        x = self.transformer.h(x, image_feature, tgt_mask=nn.Transformer.generate_square_subsequent_mask(x.shape[1], x.device))
        # forward the final layernorm and the classifier
        logits = self.lm_head(x) # (B, T, vocab_size)
        param_preds = {}
        edge_type_losses = {}
        total_loss = None
        ce_loss = None
        total_edge_loss = None
        # edge losses
        if param_targets is not None:
            total_edge_loss = 0
            edge_panel_mask_dict = {}
            for index in self.panel_edge_indices.get_all_indices():
                token_mask = labels[..., 1:] == index
                if token_mask.any():
                    token_mask = torch.cat(
                        [
                            token_mask,
                            torch.zeros((token_mask.shape[0], 1)).bool().cuda(),
                        ],
                        dim=1,
                    )
                    edge_panel_mask_dict[index] = token_mask
            for ind in param_target_masks.keys():
                mask = edge_panel_mask_dict[ind]
                if not mask.any():
                    edge_type_losses[f"{self.panel_edge_indices.get_index_token(ind).value}_loss"] = torch.zeros(1).to(x)
                    continue
                panel_embeds = x[mask]
                edge_type = self.panel_edge_indices.get_index_token(ind)
                if edge_type == PanelEdgeTypeV3.MOVE:
                    panel_params = self.transformation_fc(panel_embeds)
                else:
                    panel_params = self.line_curve_fc(panel_embeds)
                    if edge_type == PanelEdgeTypeV3.CUBIC:
                        panel_params = panel_params[:, :-2]
                    elif edge_type == PanelEdgeTypeV3.ARC:
                        panel_params = torch.cat([panel_params[:, :2], panel_params[:, 6:]], dim=-1)
                    elif edge_type == PanelEdgeTypeV3.LINE:
                        panel_params = panel_params[:, :2]
                    elif edge_type == PanelEdgeTypeV3.CURVE:
                        panel_params = panel_params[:, :4]
                    elif edge_type == PanelEdgeTypeV3.CLOSURE_CURVE:
                        panel_params = panel_params[:, 2:4]
                    elif edge_type == PanelEdgeTypeV3.CLOSURE_ARC:
                        panel_params = panel_params[:, 6:]
                    elif edge_type == PanelEdgeTypeV3.CLOSURE_CUBIC:
                        panel_params = panel_params[:, 2:6]
                
                param_preds[ind] = torch.zeros(*param_targets[ind].shape).to(panel_params)
                param_preds[ind][param_target_masks[ind]] = panel_params 
                loss = torch.sum((param_preds[ind] - param_targets[ind]) ** 2, -1).sum(1) / (torch.sum(param_target_masks[ind], 1) + 1e-5)
                loss = loss.mean()
                total_edge_loss += loss
                edge_type_losses[f"{self.panel_edge_indices.get_index_token(ind).value}_loss"] = loss
            total_loss = self.edge_loss_weight * total_edge_loss

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), reduction='mean')
            total_loss = ce_loss + total_loss if total_loss is not None else ce_loss
        
        return_dict = {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "edge_loss": total_edge_loss,
            "params": param_preds,
            "logits": logits,
        }
        return_dict.update(edge_type_losses)
        if return_hidden:
            return_dict["hidden"] = x
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

    