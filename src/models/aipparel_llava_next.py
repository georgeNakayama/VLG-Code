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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from enum import Enum

from transformers.modeling_outputs import ModelOutput

from transformers import LlavaNextConfig
from transformers import LlavaNextForConditionalGeneration
from .aipparel_llava import AIpparelLlavaCausalLMOutputWithPast
from .encodings import SinusoidalEncoding
class PanelEdgeType(Enum):
    LINE = 0 
    CLOSURE_LINE = 1
    CURVE = 2
    CLOSURE_CURVE = 3
    CUBIC_CURVE = 4
    CLOSURE_CUBIC_CURVE = 5
    ARC = 6
    CLOSURE_ARC = 7
    TRANSFORMATION = 8
    
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

class AIpparelLlavaNextConfig(LlavaNextConfig):
    model_type = "aipparel_llava_next"
    
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
class AIpparelLlavaNextCausalLMOutputWithPast(ModelOutput):
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
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    params: Optional[Dict[int, torch.FloatTensor]] = None
    params_mask_dict: Optional[Dict[int, torch.BoolTensor]] = None



class AIpparelLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    config_class = AIpparelLlavaNextConfig
    def __init__(self, config: AIpparelLlavaNextConfig):
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


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
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
    ) -> Union[Tuple, AIpparelLlavaCausalLMOutputWithPast]:
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
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        legacy_processing = False
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # if the number of image tokens is more than image embeddings seq length, then prob we expanded it in processing
            # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
            # In case we're in decoding stage, legacy behavior is checked by presence of pixel values even if use_cache=True
            legacy_processing = (
                (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
            ) or (input_ids.shape[-1] == 1 and pixel_values is not None)

        image_features = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_newline=self.image_newline,
            )

        if legacy_processing:
            logger.warning_once(
                "Expanding inputs for image tokens in LLaVa-NeXT should be done in processing. "
                "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.50."
            )
            if input_ids.shape[1] != 1:
                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, position_ids, labels, _ = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
            else:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)[-target_length:]

        # TODO: @raushan retain only the new behavior after v4.47
        elif image_features is not None:
            n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
            n_image_features = image_features.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        if input_ids is not None and labels is not None:
            transf_mask = input_ids == self.config.get_all_edge_indices(ret_dict=False)[:1]
            edge_mask = torch.isin(input_ids, self.config.get_all_edge_indices(ret_dict=False)[1:])
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
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        total_loss = None
        ce_loss = None
        edge_type_losses = {}
        edge_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

            # Regression loss
            last_hidden_state = output.hidden_states[-1]
            param_preds = {k:torch.zeros_like(v) for k,v in param_targets.items()}
            if param_target_masks is not None:
                edge_loss = 0
                for i, (edge_type, ind) in self.config.get_all_edge_indices(ret_dict=True).items():
                    mask = labels[..., 1:] == ind
                    mask = torch.cat([mask, torch.zeros_like(mask)[..., :1]], dim=1)
                    if not mask.any():
                        edge_type_losses[f"{edge_type}_loss"] = torch.zeros(1).to(last_hidden_state.device)
                        continue
                    panel_embeds = last_hidden_state[mask]
                    panel_params = self.regression_head(panel_embeds)
                    if i == 0:
                        # transf
                        panel_params = panel_params[:, :7]
                    elif i == 1:
                        # line
                        panel_params = panel_params[:, 7:9]
                    elif i == 2:
                        # quadratic
                        panel_params = panel_params[:, 7:11]
                    elif i == 3:
                        # cubic
                        panel_params = panel_params[:, 7:13]
                    elif i == 4:
                        # arc
                        panel_params = torch.cat([panel_params[:, 7:9], panel_params[:, 13:15]], dim=-1)
                    elif i == 6:
                        # c_quadratic
                        panel_params = panel_params[:, 9:11]
                    elif i == 7:
                        # c_cubic
                        panel_params = panel_params[:, 11:13]
                    elif i == 8:
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

        return AIpparelLlavaCausalLMOutputWithPast(
            loss=total_loss,
            ce_loss=ce_loss,
            edge_loss=edge_loss,
            edge_type_losses=edge_type_losses,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes

        return model_inputs