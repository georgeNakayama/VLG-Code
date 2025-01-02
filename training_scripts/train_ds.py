import argparse
import os
import shutil
import sys
import time
from functools import partial
from dataclasses import dataclass, field, asdict
import logging 
log = logging.getLogger(__name__)
import hydra
from typing import Literal, List
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from typing import Optional
from peft import LoraConfig, get_peft_model

from experiment_hydra import ExperimentWrappper, MyExperimentConfig
from models.garment_token import GarmentTokenConfig, GarmentTokenForCausalLM
from models.garment_token_regression import GarmentTokenRegressionForCausalLM
from models.llava import conversation as conversation_lib
from data.datasets.dataset_garment_as_token_qva import QVAGarmentTokenDataset, QVAGarmentTokenDatasetConfig
from data.data_wrappers.garment_image_data_wrapper import GarmentImageDataWrapper, GarmentImageDataWrapperConfig
from data.datasets.utils import PanelEdgeTypeV2, SpecialTokensV2
from trainers.llava_trainer import FinetuneLlavaTrainer, FinetuneLlavaTrainerConfig
from data.datasets.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN)

@dataclass 
class SystemConfig:
    wandb_username: Optional[str] = None
    output: str = "./"


@dataclass 
class LoraArguments():
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]

@dataclass
class MainConfig:
    version: str
    model_type: Literal["garment_token", "garment_token_regression"]
    model_max_length: int
    system: SystemConfig = field(default_factory=SystemConfig)
    experiment: MyExperimentConfig = field(default_factory=MyExperimentConfig)
    trainer: FinetuneLlavaTrainerConfig = field(default_factory=FinetuneLlavaTrainerConfig)
    data_wrapper: GarmentImageDataWrapperConfig = field(default_factory=GarmentImageDataWrapperConfig)
    model: GarmentTokenConfig = field(default_factory=GarmentTokenConfig)
    precision: Literal["bf16", "fp16"] = "bf16"
    lora_args: LoraArguments = field(default_factory=LoraArguments)
    eval_only: bool = False
    gen_only: bool = False
    conv_type: Literal["default", "v0", "v1", "vicuna_v1", "llama_2", "plain", "v0_plain", "llava_v0", "v0_mmtag", "llava_v1", "v1_mmtag", "llava_llama_2", "mpt"] = "llava_v1"
    pre_trained: Optional[str] = None
    from_start: bool = False
    storage_dir: Optional[str] = None

@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(cfg: MainConfig):
    log.info(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Output directory : {output_dir}")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = (ddp_rank == 0)
    system_info = cfg.system
    
    experiment: ExperimentWrappper = ExperimentWrappper(
        cfg, 
        output_dir, 
        system_info.wandb_username, 
        master_process=master_process,
        no_sync=False)
    log.info("Experiment Wrapper created.")
    
    data_wrapper: GarmentImageDataWrapper = hydra.utils.instantiate(
        cfg.data_wrapper,
        output_dir=output_dir
    )
    log.info("Dataset created.")

    trainer: FinetuneLlavaTrainer = hydra.utils.instantiate(
        cfg.trainer,
        experiment_tracker=experiment,
        data_wrapper=data_wrapper,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        ddp_local_rank=ddp_local_rank,
        precision=cfg.precision,
        model_type=cfg.model_type,
    )
    
    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.version,
        cache_dir=None,
        model_max_length=cfg.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    tokenizer.pad_token = tokenizer.unk_token
    all_new_tokens = data_wrapper.dataset.garment_tokenizer.get_all_token_names()
    num_added_tokens = tokenizer.add_tokens(all_new_tokens)
    if master_process:
        log.info(f"Added {num_added_tokens} tokens to the tokenizer.")
    token_name2_idx_dict = {}
    for token in all_new_tokens:
        token_idx = tokenizer(token, add_special_tokens=False).input_ids[0]
        token_name2_idx_dict[token] = token_idx
        
    if master_process:
        log.info(f"Token name to index dictionary: {token_name2_idx_dict}")
    data_wrapper.dataset.garment_tokenizer.set_token_indices(token_name2_idx_dict)

    if cfg.model.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    torch_dtype = torch.float32
    if cfg.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        torch_dtype = torch.half
        
    if cfg.model_type == "garment_token":
        model = GarmentTokenForCausalLM.from_pretrained(
            cfg.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **cfg.model, vision_tower=data_wrapper.dataset.vision_tower
        )
    elif cfg.model_type == "garment_token_regression":
        model = GarmentTokenRegressionForCausalLM.from_pretrained(
            cfg.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **cfg.model, vision_tower=data_wrapper.dataset.vision_tower, 
            panel_edge_indices=data_wrapper.dataset.garment_tokenizer.panel_edge_type_indices, gt_stats=data_wrapper.dataset.garment_tokenizer.gt_stats
        )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=ddp_local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[cfg.conv_type]

    lora_r = cfg.lora_args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = cfg.lora_args.lora_alpha
        lora_dropout = cfg.lora_args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, cfg.lora_args.lora_target_modules
        )
        if master_process:
            print(lora_target_modules)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    trainable_layers = ["lm_head", "embed_tokens"] 
    if cfg.model_type == "garment_token_regression":
        if getattr(model.model, 'model', None) is not None:
            trainable_layers += model.model.model.get_trainable_param_names()
            model.model.model.init_trainable_params()
        else:
            trainable_layers += model.model.get_trainable_param_names()
            model.model.init_trainable_params()
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in trainable_layers
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    for name, module in model.named_parameters():
        print(name, module.shape, module.requires_grad)
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    trainer.training_setup(model, tokenizer, cfg.conv_type, cfg.from_start, cfg.pre_trained)
    
    if cfg.eval_only:
        trainer.eval_step(trainer.start_epoch * trainer.steps_per_epoch)
        trainer.generation_step(trainer.start_epoch * trainer.steps_per_epoch)
    elif cfg.gen_only:
        trainer.generation_step(trainer.start_epoch)
    else:
        trainer.fit()


if __name__ == "__main__":
    main()
