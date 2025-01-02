from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import BitsAndBytesConfig, CLIPVisionModel, LlavaConfig, LlavaForConditionalGeneration, LlavaLlamaModel
from transformers.configuration_utils import CONFIG_MAPPING, PretrainedConfig
from transformers.models.auto import AutoConfig



class GarmentTokenMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GarmentTokenMetaModel, self).__init__(config)

        self.config = config


class AIpparelConfig(PretrainedConfig):
    model_type = "aipparel"
    sub_configs = {"llm_config": AutoConfig}
    
    def __init__(
        self,
        llm_config=None,
        edge_loss_weight: float = 1.0,
        **kwargs,
    ):
        self.edge_loss_weight = edge_loss_weight
        
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
        super().__init__(**kwargs)

class GarmentTokenModel(LlavaLlamaModel):
    def __init__(
        self,
        config,
    ):
        super(GarmentTokenModel, self).__init__(config)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class AIpparelForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(
        self,
        config: AIpparelConfig, 
        **kwargs,
    ):
        super().__init__(config)


    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        question_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        inference: bool = False,
        **kwargs,
    ):
        
        batch_size = images_clip.shape[0]
        assert batch_size == len(offset) - 1



        images_clip_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_clip_i = (
                images_clip[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_clip_list.append(images_clip_i)
        images_clip = torch.cat(images_clip_list, dim=0)

        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=False,
        )

        logits = output.logits

        loss = output.loss
        
        return {"total_loss":loss, "ce_loss": loss, "logits":logits}

    def evaluate(
        self,
        images_clip,
        input_ids,
        attention_mask,
        max_new_tokens=32
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_ids = outputs.sequences


        return {"output_ids": output_ids}
    
