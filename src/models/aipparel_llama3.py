# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Llava-NeXT model."""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from enum import Enum

from transformers.modeling_outputs import ModelOutput

from transformers import MllamaForConditionalGeneration, MllamaConfig
from .encodings import SinusoidalEncoding

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

def _prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape so it can be used by attn module
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(
        inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
    )

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = (
        (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
    )
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask, full_text_row_masked_out_mask


class AIpparelMllamaNextConfig(MllamaConfig):
    model_type = "aipparel_llama3"
    
    def __init__(
        self,
        transf_token="<transformation>",
        transf_token_index=32001,
        line_token="<line>",
        line_token_index=32002,
        quadratic_token="<quadratic>",
        quadratic_token_index=32003,
        cubic_token="cubic",
        cubic_token_index=32004,
        arc_token="<arc>",
        arc_token_index=32005,
        cline_token="<closure_line>",
        cline_token_index=32006,
        cquadratic_token="<closure_quadratic>",
        cquadratic_token_index=32007,
        ccubic_token="<closure_cubic>",
        ccubic_token_index=32008,
        carc_token="<closure_arc>",
        carc_token_index=32009,
        pattern_start_token_index=32010,
        pattern_end_token_index=32011,
        panel_start_token_index=32012,
        panel_end_token_index=32013,
        edge_loss_weight: float = 1.0,
        num_freq: int = 9,
        num_regression_layers: int = 2,
        zero_tensor: torch.Tensor = None,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        
        self.edge_loss_weight = edge_loss_weight
        self.num_freq = num_freq
        self.num_regression_layers = num_regression_layers
        
        self.pattern_start_token_index = pattern_start_token_index
        self.pattern_end_token_index = pattern_end_token_index
        self.panel_start_token_index = panel_start_token_index
        self.panel_end_token_index = panel_end_token_index
        
        self.transf_token = transf_token
        self.transf_token_index = transf_token_index
        self.line_token = line_token
        self.line_token_index = line_token_index
        self.quadratic_token = quadratic_token
        self.quadratic_token_index = quadratic_token_index
        self.cubic_token = cubic_token
        self.cubic_token_index = cubic_token_index
        self.arc_token = arc_token
        self.arc_token_index = arc_token_index
        self.cline_token = cline_token
        self.cline_token_index = cline_token_index
        self.cquadratic_token = cquadratic_token
        self.cquadratic_token_index = cquadratic_token_index
        self.ccubic_token = ccubic_token
        self.ccubic_token_index = ccubic_token_index
        self.carc_token = carc_token
        self.carc_token_index = carc_token_index
        
        self.zero_tensor = zero_tensor
        
        
    def get_all_edge_indices(self, ret_dict=True):
        if ret_dict:
            return {
                self.transf_token: self.transf_token_index,
                self.line_token: self.line_token_index,
                self.quadratic_token: self.quadratic_token_index,
                self.cubic_token: self.cubic_token_index,
                self.arc_token: self.arc_token_index,
                self.cline_token: self.cline_token_index,
                self.cquadratic_token: self.cquadratic_token_index,
                self.ccubic_token: self.ccubic_token_index,
                self.carc_token: self.carc_token_index,
            }
        return [
            self.transf_token_index,
            self.line_token_index,
            self.quadratic_token_index,
            self.cubic_token_index,
            self.arc_token_index,
            self.cline_token_index,
            self.cquadratic_token_index,
            self.ccubic_token_index,
            self.carc_token_index,
        ]
        

@dataclass
class AIpparelMllamaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    edge_type_losses: Optional[Dict[int, torch.FloatTensor]] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    params: Optional[Dict[int, torch.FloatTensor]] = None
    params_mask_dict: Optional[Dict[int, torch.BoolTensor]] = None



class AIpparelMllavaNextForConditionalGeneration(MllamaForConditionalGeneration):
    config_class = AIpparelMllamaNextConfig
    def __init__(self, config: AIpparelMllamaNextConfig):
        super().__init__(config)
        self.initialize_panel_edge_modules()
    
    def initialize_panel_edge_modules(self):
        self.regression_head = make_mlp(
            self.config.text_config.hidden_size, 
            self.config.text_config.hidden_size,
            7+8,
            self.config.num_regression_layers
        )

        self.vertex_encoding = SinusoidalEncoding(
            in_dim=2,
            num_frequencies=self.config.num_freq,
            min_freq_exp=0.0, max_freq_exp=self.config.num_freq - 1, include_input=True
        )
        self.trasl_encoding = SinusoidalEncoding(
            in_dim=3,
            num_frequencies=self.config.num_freq,
            min_freq_exp=0.0, max_freq_exp=self.config.num_freq - 1, include_input=True
        )
        self.transf_proj = nn.Sequential(
            nn.Linear(
                self.trasl_encoding.get_out_dim()+4, 
                self.config.text_config.hidden_size, 
                bias=True
                ),
            nn.GELU(),
            nn.Linear(
                self.config.text_config.hidden_size, 
                self.config.text_config.hidden_size
                )
        )
        self.vertex_proj = nn.Sequential(
            nn.Linear(
                self.vertex_encoding.get_out_dim(), 
                self.config.text_config.hidden_size, 
                bias=True
                ),
            nn.GELU(),
            nn.Linear(
                self.config.text_config.hidden_size, 
                self.config.text_config.hidden_size
                )
        )
        
        self.regression_head[-2].weight.data.zero_()
        self.vertex_proj[-1].weight.data.zero_()
        self.transf_proj[-1].weight.data.zero_()

    def resize_token_embeddings(self, new_num_tokens: int):
        # mysterious 8
        new_embeddings = nn.Embedding(self.config.text_config.vocab_size + new_num_tokens + 8, self.config.text_config.hidden_size)
        old_embeddings = self.get_input_embeddings()
        new_embeddings.weight.data[: self.config.text_config.vocab_size, :] = old_embeddings.weight.data[:self.config.text_config.vocab_size, :].clone()
        mean, std = old_embeddings.weight.data.mean(0), old_embeddings.weight.data.std(0) * 1e-5
        new_embeddings.weight.data[self.config.text_config.vocab_size:self.config.text_config.vocab_size + new_num_tokens] = mean[None] + std[None] * torch.randn(new_num_tokens, self.config.text_config.hidden_size)
        new_embeddings.weight.data[-8:] = old_embeddings.weight.data[-8:, :].clone()
        self.set_input_embeddings(new_embeddings)
        old_lm_head = self.get_output_embeddings()
        new_lm_head = nn.Linear(self.config.text_config.hidden_size, self.config.text_config.vocab_size + new_num_tokens, bias=False)
        new_lm_head.weight.data[: self.config.text_config.vocab_size, :] = old_lm_head.weight.data[:self.config.text_config.vocab_size, :].clone()
        new_lm_head.weight.data[self.config.text_config.vocab_size:] = mean[None, :] + std[None, :] * torch.randn(new_num_tokens, self.config.text_config.hidden_size)
        self.config.text_config.vocab_size = self.config.text_config.vocab_size + new_num_tokens
        self.vocab_size = self.config.text_config.vocab_size
        self.set_output_embeddings(new_lm_head)

    def is_closure(self, token_id):
        return token_id in [
            self.config.cline_token_index,
            self.config.cquadratic_token_index,
            self.config.ccubic_token_index,
            self.config.carc_token_index,
        ]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        
        pattern_params: Optional[Dict[int, torch.FloatTensor]] = None,
        pattern_params_mask: Optional[Dict[int, torch.BoolTensor]] = None,
        pattern_endpoints: Optional[torch.FloatTensor] = None,
        pattern_endpoint_masks: Optional[torch.BoolTensor] = None,
        pattern_transfs: Optional[torch.FloatTensor] = None,
        pattern_transf_masks: Optional[torch.BoolTensor] = None
    ) -> Union[Tuple, AIpparelMllamaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> prompt = "<|image|>If I had to write a haiku for this one"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> output = model.generate(**inputs, max_new_tokens=15)

        >>> prompt_len = inputs.input_ids.shape[-1]
        >>> generated_ids = output[:, prompt_len:]
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        >>> print(generated_text)
        [', it would be:.\\nA stop sign in Chinatown.\\n']
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]
        
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if input_ids is not None and labels is not None:
            transf_mask = input_ids == self.config.get_all_edge_indices(ret_dict=False)[0]
            edge_mask = torch.isin(input_ids, torch.tensor(self.config.get_all_edge_indices(ret_dict=False)[1:]).to(input_ids))
            if (edge_mask is not None and pattern_endpoints is not None and pattern_endpoint_masks is not None):
                assert edge_mask.sum() == pattern_endpoint_masks.sum(), "edge mask has shape {} but endpoints mask has shape {}" \
                    .format(edge_mask.sum(), pattern_endpoint_masks.sum())
                _endpoints = pattern_endpoints[pattern_endpoint_masks]
                edge_embeds = self.vertex_proj(self.vertex_encoding(_endpoints))
                inputs_embeds[edge_mask] = inputs_embeds[edge_mask] + edge_embeds
                    
            if (transf_mask is not None and pattern_transfs is not None and pattern_transf_masks is not None):
                assert transf_mask.sum() == pattern_transf_masks.sum()
                _transformations = pattern_transfs[pattern_transf_masks]
                transf_embeds = self.transf_proj(torch.cat([self.trasl_encoding(_transformations[:, :3]), _transformations[:, 3:]], dim=1))
                inputs_embeds[transf_mask] = inputs_embeds[transf_mask] + transf_embeds

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )
        ce_loss = None
        logits = outputs.logits
        total_loss = None
        edge_type_losses = {}
        edge_loss = None
        if labels is not None:
            # ce loss
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction='none')
            shift_labels = shift_labels.view(logits.shape[0], -1)
            ce_loss = ce_loss.view(logits.shape[0], -1).sum(dim=1) / (shift_labels != -100).sum(1)
            
            # Regression loss
            last_hidden_state = outputs.hidden_states[-1]
            param_preds = {k:torch.zeros_like(v) for k,v in pattern_params.items()}
            if pattern_params_mask is not None:
                edge_loss = 0
                for edge_type, ind in self.config.get_all_edge_indices(ret_dict=True).items():
                    mask = labels[..., 1:] == ind
                    mask = torch.cat([mask, torch.zeros_like(mask)[..., :1]], dim=1)
                    if not mask.any():
                        edge_type_losses[f"{edge_type}_loss"] = torch.zeros(1).to(last_hidden_state.device)
                        continue
                    panel_embeds = last_hidden_state[mask]
                    panel_params = self.regression_head(panel_embeds)
                    if ind == self.config.cline_token_index:
                        continue
                    if ind == self.config.transf_token_index:
                        # transf
                        panel_params = panel_params[:, :7]
                    elif ind == self.config.line_token_index:
                        # line
                        panel_params = panel_params[:, 7:9]
                    elif ind == self.config.quadratic_token_index:
                        # quadratic
                        panel_params = panel_params[:, 7:11]
                    elif ind == self.config.cubic_token_index:
                        # cubic
                        panel_params = panel_params[:, 7:13]
                    elif ind == self.config.arc_token_index:
                        # arc
                        panel_params = torch.cat([panel_params[:, 7:9], panel_params[:, 13:15]], dim=-1)
                    elif ind == self.config.cquadratic_token_index:
                        # c_quadratic
                        panel_params = panel_params[:, 9:11]
                    elif ind == self.config.ccubic_token_index:
                        # c_cubic
                        panel_params = panel_params[:, 9:13]
                    elif ind == self.config.carc_token_index:
                        # c_arc
                        panel_params = panel_params[:, 13:15]
                        
                    param_preds[ind][pattern_params_mask[ind]] = panel_params 
                    loss = torch.sum((param_preds[ind] - pattern_params[ind]) ** 2, -1).sum(1) / (torch.sum(pattern_params_mask[ind], 1) + 1e-5)
                    edge_loss += loss
                    edge_type_losses[f"{edge_type}_loss"] = loss.mean()
                
                total_loss = self.config.edge_loss_weight * edge_loss
            
            total_loss += ce_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (total_loss,) + output if loss is not None else output

        return AIpparelMllamaCausalLMOutputWithPast(
            loss=total_loss,
            ce_loss=ce_loss,
            edge_loss=edge_loss,
            edge_type_losses=edge_type_losses,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        num_logits_to_keep=None,
        last_hidden_state=None,
        param_dict=defaultdict(torch.Tensor),
        **kwargs,
    ):

        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if len(past_key_values) != 0:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
                last_hidden_state = last_hidden_state[:, -cache_position.shape[0]:, :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
                last_hidden_state = last_hidden_state[:, [-1], :]

        pattern_transf_masks = None
        pattern_endpoint_masks = None
        pattern_endpoints = None
        pattern_transfs = None
        if last_hidden_state is not None:
            pattern_transf_masks = input_ids == self.config.get_all_edge_indices(ret_dict=False)[0]
            pattern_endpoint_masks = torch.isin(input_ids, torch.tensor(self.config.get_all_edge_indices(ret_dict=False)[1:]).to(input_ids))
            if pattern_endpoint_masks.any():
                assert pattern_endpoint_masks.shape[1] == last_hidden_state.shape[1]
                pattern_endpoints = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1], 2).to(last_hidden_state)
                for ind in self.config.get_all_edge_indices(ret_dict=False):

                    mask = input_ids == ind
                    if not mask.any():
                        continue
                    if ind != self.config.get_all_edge_indices(ret_dict=False)[0]:
                        pattern_endpoint_masks |= mask
                        if self.is_closure(ind):
                            pattern_endpoints[mask] = self.config.zero_tensor.to(last_hidden_state)
                        else:
                            edge_embeds = last_hidden_state[mask]
                            pattern_endpoints[mask] = self.regression_head(edge_embeds)[:, 7:9]
                        
                    panel_params = self.regression_head(last_hidden_state[mask])

                    if ind == self.config.cline_token_index:
                        panel_params = torch.empty(0)
                    if ind == self.config.transf_token_index:
                        # transf
                        panel_params = panel_params[:, :7]
                    elif ind == self.config.line_token_index:
                        # line
                        panel_params = panel_params[:, 7:9]
                    elif ind == self.config.quadratic_token_index:
                        # quadratic
                        panel_params = panel_params[:, 7:11]
                    elif ind == self.config.cubic_token_index:
                        # cubic
                        panel_params = panel_params[:, 7:13]
                    elif ind == self.config.arc_token_index:
                        # arc
                        panel_params = torch.cat([panel_params[:, 7:9], panel_params[:, 13:15]], dim=-1)
                    elif ind == self.config.cquadratic_token_index:
                        # c_quadratic
                        panel_params = panel_params[:, 9:11]
                    elif ind == self.config.ccubic_token_index:
                        # c_cubic
                        panel_params = panel_params[:, 9:13]
                    elif ind == self.config.carc_token_index:
                        # c_arc
                        panel_params = panel_params[:, 13:15]
                
                    if not ind in param_dict:
                        param_dict[ind] = panel_params
                    else:
                        param_dict[ind] = torch.cat([param_dict[ind], panel_params])

            if pattern_transf_masks.any():
                assert pattern_transf_masks.shape[1] == last_hidden_state.shape[1]
                pattern_transfs = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1], 7).to(last_hidden_state)
                transf_embeds = last_hidden_state[pattern_transf_masks]
                pattern_transfs[pattern_transf_masks] = self.regression_head(transf_embeds)[:, :7]

        # TODO: we have no attention_mask so this won't work, check if we really won't need attention mask and find another way
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if len(past_key_values) != 0:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_attention_mask,
                "pattern_endpoints": pattern_endpoints,
                "pattern_endpoint_masks": pattern_endpoint_masks,
                "pattern_transfs": pattern_transfs,
                "pattern_transf_masks": pattern_transf_masks,
            }
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["aspect_ratio_ids"] = aspect_ratio_ids
            model_inputs["aspect_ratio_mask"] = aspect_ratio_mask

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
            )
        
        # add last hidden state for param computation
        model_kwargs["last_hidden_state"] = outputs.hidden_states[-1]
        return model_kwargs
