# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Llava model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import AutoModel, AutoConfig, LlavaConfig, LlavaForConditionalGeneration
from transformers.modeling_outputs import ModelOutput
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

class AIpparelLlavaConfig(LlavaConfig):
    model_type = "aipparel_llava"
    
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_seq_length=576,
        transf_token_index=32001,
        line_token_index=32002,
        quadratic_token_index=32003,
        cubic_token_index=32004,
        arc_token_index=32005,
        cline_token_index=32006,
        cquadratic_token_index=32007,
        ccubic_token_index=32008,
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
        
        super().__init__(
            vision_config,
            text_config,
            ignore_index,
            image_token_index,
            projector_hidden_act,
            vision_feature_select_strategy,
            vision_feature_layer,
            image_seq_length,
            **kwargs)
        
        self.edge_loss_weight = edge_loss_weight
        self.num_freq = num_freq
        self.num_regression_layers = num_regression_layers
        
        self.transf_token_index = transf_token_index
        self.line_token_index = line_token_index
        self.quadratic_token_index = quadratic_token_index
        self.cubic_token_index = cubic_token_index
        self.arc_token_index = arc_token_index
        self.cline_token_index = cline_token_index
        self.cquadratic_token_index = cquadratic_token_index
        self.ccubic_token_index = ccubic_token_index
        self.carc_token_index = carc_token_index
        self.pattern_start_token_index = pattern_start_token_index
        self.pattern_end_token_index = pattern_end_token_index
        self.panel_start_token_index = panel_start_token_index
        self.panel_end_token_index = panel_end_token_index
        
        

@dataclass
class AIpparelLlavaCausalLMOutputWithPast(ModelOutput):
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
    edge_type_loss: Dict[str, torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    params: Optional[Dict[int, torch.FloatTensor]] = None
    params_mask_dict: Optional[Dict[int, torch.BoolTensor]] = None

class AIpparelLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    config_class = AIpparelLlavaConfig
    def __init__(self, config: AIpparelLlavaConfig):
        super().__init__(config)
        self.initialize_panel_edge_modules()

    def initialize_panel_edge_modules(self):
        self.transformation_fc = make_mlp(
            self.config.text_config.hidden_size, 
            self.config.text_config.hidden_size,
            7,
            self.config.num_regression_layers
            )
            
        self.line_curve_fc = make_mlp(
            self.config.text_config.hidden_size, 
            self.config.text_config.hidden_size,
            8,
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
        
        self.line_curve_fc[-2].weight.data.zero_()
        self.transformation_fc[-2].weight.data.zero_()
        self.vertex_proj[-1].weight.data.zero_()
        self.transf_proj[-1].weight.data.zero_()

        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
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
        endpoints: Optional[torch.FloatTensor] = None,
        endpoints_mask: Optional[torch.BoolTensor] = None,
        transformations: Optional[torch.FloatTensor] = None,
        transformations_mask: Optional[torch.BoolTensor] = None
    ) -> Union[Tuple, AIpparelLlavaCausalLMOutputWithPast]:

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
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

        if legacy_processing:
            logger.warning_once(
                "Expanding inputs for image tokens in LLaVa should be done in processing. "
                "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.50."
            )
            # prefill stage vs decoding stage (legacy behavior copied)
            if input_ids.shape[1] != 1:
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
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
            n_image_features = image_features.shape[0] * image_features.shape[1]

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
            edge_panel_mask_dict = self.prepare_edge_panel_mask_dict_for_labels(input_ids, labels)
            edge_mask, transf_mask = self.prepare_edge_transf_masks_for_inputs(input_ids)
            if (edge_mask is not None and endpoints is not None and endpoints_mask is not None):
                assert edge_mask.sum() == endpoints_mask.sum(), "edge mask has shape {} but endpoints mask has shape {}" \
                    .format(edge_mask.sum(), endpoints_mask.sum())
                for i in range(inputs_embeds.shape[0]):
                    _endpoints = endpoints[i][endpoints_mask[i]]
                    edge_embeds = self.vertex_proj(self.vertex_encoding(_endpoints))
                    inputs_embeds[i, edge_mask[i]] = inputs_embeds[i, edge_mask[i]] + edge_embeds
                    
            if (transf_mask is not None and transformations is not None and transformations_mask is not None):
                assert transf_mask.sum() == transformations_mask.sum()
                for i in range(inputs_embeds.shape[0]):
                    _transformations = transformations[i][transformations_mask[i]]
                    transf_embeds = self.transf_proj(torch.cat([self.trasl_encoding(_transformations[:, :3]), _transformations[:, 3:]], dim=1))
                    inputs_embeds[i, transf_mask[i]] = inputs_embeds[i, transf_mask[i]] + transf_embeds

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

        ce_loss = None
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
        edge_type_losses = {}
        total_loss = None
        if param_target_masks is not None:
            total_edge_loss = 0
            for ind in param_target_masks.keys():
                mask = edge_panel_mask_dict[ind]
                if not mask.any():
                    edge_type_losses[f"{self.config.panel_edge_indices.get_index_token(ind).value}_loss"] = torch.zeros(1).to(last_hidden_state.device)
                    continue
                panel_embeds = last_hidden_state[mask]
                edge_type = self.config.panel_edge_indices.get_index_token(ind)
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
                    
                param_preds[ind][param_target_masks[ind]] = panel_params 
                loss = torch.sum((param_preds[ind] - param_targets[ind]) ** 2, -1).sum(1) / (torch.sum(param_target_masks[ind], 1) + 1e-5)
                total_edge_loss += loss
                edge_type_losses[f"{self.config.panel_edge_indices.get_index_token(ind).value}_loss"] = loss.mean()
            
            total_loss = self.config.edge_loss_weight * total_edge_loss
        
        if ce_loss is not None:
            total_loss += ce_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            last_hidden_state=last_hidden_state,
            param_preds=param_preds,
            params_mask_dict=param_target_masks
        )
        
AutoConfig.register("aipparel_llava", AIpparelLlavaConfig)
AutoModel.register(AIpparelLlavaConfig, AIpparelLlavaForConditionalGeneration)
