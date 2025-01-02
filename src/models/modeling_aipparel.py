from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import BitsAndBytesConfig, CLIPVisionModel, LlavaConfig, LlavaForConditionalGeneration, LlavaLlamaModel
from transformers.configuration_utils import CONFIG_MAPPING, PretrainedConfig
from transformers.models.auto import AutoConfig
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from data.garment_tokenizers.special_tokens import PanelEdgeTypeIndices, PanelEdgeTypeV3
from .encodings import SinusoidalEncoding, DiscreteEncoding
from data.datasets.panel_configs import StandardizeConfig

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

class AIpparelConfig(PretrainedConfig):
    model_type = "aipparel"
    sub_configs = {"llm_config": AutoConfig}
    
    def __init__(
        self,
        llm_config=None,
        panel_edge_indices=None, 
        edge_loss_weight: float = 1.0,
        num_freq: int = 9,
        num_regression_layers: int = 2,
    ):
        self.edge_loss_weight = edge_loss_weight
        self.num_freq = num_freq
        self.num_regression_layers = num_regression_layers
        self.panel_edge_indices = panel_edge_indices
        
        if isinstance(llm_config, dict):
            llm_config["model_type"] = (
                llm_config["model_type"] if "model_type" in llm_config else "clip_vision_model"
            )
            llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)
        elif llm_config is None:
            # default to llava
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )
            text_config = CONFIG_MAPPING["llama"]()
            
            llm_config = CONFIG_MAPPING["llava"](
                vision_config=vision_config,
                text_config=text_config,
                ignore_index=-100,
                image_token_index=32000,
                projector_hidden_act="gelu",
                vision_feature_select_strategy="default",
                vision_feature_layer=-2,
                image_seq_length=576,
            )

        self.llm_config = llm_config
        self.transformer_hidden_size = self.llm_config.text_config.hidden_size
        self.image_token_index = self.llm_config.image_token_index
        self.image_seq_length = self.llm_config.image_seq_length
        super().__init__(**kwargs)
        
@dataclass
class AIpparelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
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

class AIpparelPretrainedModel(PreTrainedModel):
    config_class = AIpparelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

class AIpparelForConditionalGeneration(GenerationMixin):
    def __init__(
        self,
        config: AIpparelConfig
    ):
        super(GarmentTokenRegressionModel, self).__init__(config.llm_config)
        self.config = config
        self.initialize_panel_edge_modules()
        
    def _init_weights(self, module):
        super().init_weights()
        # zero init the last layers of the regression and pe heads. 
        self.line_curve_fc[-2].weight.data.zero_()
        self.transformation_fc[-2].weight.data.zero_()
        self.vertex_proj[-1].weight.data.zero_()
        self.transf_proj[-1].weight.data.zero_()

    def initialize_panel_edge_modules(self, config):
        in_dim = self.config.transformer_hidden_size
        self.transformation_fc = make_mlp(
            in_dim, 
            in_dim,
            self.config.panel_edge_indices.get_index_param_num(self.config.panel_edge_indices.move_idx),
            self.config.num_regression_layers
            )
            
        line_curve_out_dim = 4 if self.config.panel_edge_indices.get_token_indices(PanelEdgeTypeV3.CUBIC) == -1 else 6
        if self.config.panel_edge_indices.get_token_indices(PanelEdgeTypeV3.ARC) != -1:
            line_curve_out_dim += 2
        
        self.line_curve_fc = make_mlp(
            in_dim, 
            in_dim,
            line_curve_out_dim,
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
            nn.Linear(self.trasl_encoding.get_out_dim()+4, in_dim, bias=True),
            nn.GELU(),
            nn.Linear(in_dim, in_dim)
        )
        self.vertex_proj = nn.Sequential(
            nn.Linear(self.vertex_encoding.get_out_dim(), in_dim, bias=True),
            nn.GELU(),
            nn.Linear(in_dim, in_dim)
        )
        
    
                
    
    def prepare_edge_transf_masks_for_inputs(
        self, 
        input_ids, 
        return_separate=False,
        pad_for_image=True,
    ):
        edge_mask = torch.isin(input_ids, torch.tensor(self.config.panel_edge_indices.get_all_edge_indices()).to(input_ids))
        transf_mask = input_ids == self.config.panel_edge_indices.get_token_indices(PanelEdgeTypeV3.MOVE)
        
        if return_separate:
            edge_mask_dict = {}
            for ind in self.config.panel_edge_indices.get_all_edge_indices():
                mask = input_ids == ind
                if not mask.any():
                    continue
                edge_mask_dict[ind] = mask
            return edge_mask, transf_mask, edge_mask_dict
        return edge_mask, transf_mask
    
    def prepare_edge_panel_mask_dict_for_labels(self, input_ids, labels):
        edge_panel_mask_dict = {}
        for index in self.config.panel_edge_indices.get_all_indices():
            token_mask = labels[:, 1:] == index
            token_mask = torch.cat(
                [
                    token_mask,
                    torch.zeros((token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            edge_panel_mask_dict[index] = token_mask
        return edge_panel_mask_dict
    
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
        transformations_mask: Optional[torch.BoolTensor] = None,
    ):
        
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
        assert not legacy_processing
        
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

        if image_features is not None:
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

        # CE loss
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
            return (total_loss,) + output if loss is not None else output

        return AIpparelOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            last_hidden_state=last_hidden_state,
            param_preds=param_preds,
            params_mask_dict=param_target_masks
        )
        
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None, 
        inputs_embeds=None, 
        images=None, 
        last_hidden_state=None,
        edge_mask=None, 
        transf_mask=None, 
        endpoints=None, 
        endpoints_mask=None, 
        transformations=None, 
        transformations_mask=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
            last_hidden_state = last_hidden_state[:, -1:, :]
        if self.config.pos_embed and \
            last_hidden_state is not None:
            endpoints, endpoints_mask = None, None 
            transformations, transformations_mask = None, None
            edge_mask, transf_mask, edge_mask_dict = self.prepare_edge_transf_masks_for_inputs(input_ids, return_separate=True, pad_for_image=past_key_values is None)
            if edge_mask.any():
                assert edge_mask.shape[1] == last_hidden_state.shape[1]
                endpoints = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1], 2).to(last_hidden_state)
                endpoints_mask = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1]).bool().cuda()
                for ind, mask in edge_mask_dict.items():
                    if not mask.any():
                        continue
                    endpoints_mask |= mask
                    edge_type = self.panel_edge_indices.get_index_token(ind)
                    edge_embeds = last_hidden_state[mask]
                    if edge_type.is_closure():
                        endpoints[mask] = self.zero_tensor.to(edge_embeds)
                    if edge_type == PanelEdgeTypeV3.CUBIC:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                    elif edge_type == PanelEdgeTypeV3.ARC:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                    elif edge_type == PanelEdgeTypeV3.LINE:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                    elif edge_type == PanelEdgeTypeV3.CURVE:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                        
        
            if transf_mask.any():
                assert transf_mask.shape[1] == last_hidden_state.shape[1]
                transformations = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1], 7).to(last_hidden_state)
                transformations_mask = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1]).bool().cuda()
                transf_embeds = last_hidden_state[transf_mask]
                transformations[transf_mask] = self.model.transformation_fc(transf_embeds)
                transformations_mask[transf_mask] = True
            
            

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
                "edge_mask": edge_mask,
                "transf_mask": transf_mask,
                "endpoints": endpoints,
                "endpoints_mask": endpoints_mask,
                "transformations": transformations,
                "transformations_mask": transformations_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(self,
        outputs: AIpparelOutputWithPast,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            standardize_cache_format=standardize_cache_format,
        )
        model_kwargs["last_hidden_state"] = outputs.hidden_states[-1]
        return model_kwargs