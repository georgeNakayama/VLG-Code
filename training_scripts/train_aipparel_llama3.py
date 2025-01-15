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
from typing import Literal
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from typing import Optional
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from models.aipparel_llama3 import AIpparelMllavaNextForConditionalGeneration
from data.garment_tokenizers.special_tokens import PanelEdgeTypeV3
from data.collate_fns import collate_fn
from trainers.train import train
from trainers.generate import generate

@dataclass
class MainConfig:
    version: str
    precision: Literal["bf16", "fp16"] = "bf16"
    eval_only: bool = False
    gen_only: bool = False
    gen_split: Literal["train", "val"] = "train"
    resume: Optional[str] = None
    from_start: bool = False
    grad_accumulation_steps: int = 1
    num_steps: int = 10000
    warmup_steps: int = 100
    save_freq: int = 500
    batch_size: int = 16
    optimizer: dict = field(default_factory=lambda: {"lr": 1e-4, "beta1": 0.9, "beta2": 0.999})

@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(cfg: MainConfig):
    log.info(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Output directory : {output_dir}")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = (ddp_rank == 0)
    
    garment_tokenizer = hydra.utils.instantiate(
        cfg.garment_tokenizer,
    )
    
    dataset_train = hydra.utils.instantiate(
        cfg.dataset,
        split="train"
    )
    
    if ddp_rank == 0:
        log.info(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = hydra.utils.instantiate(
        cfg.dataset,
        split="val"
    )
    if ddp_rank == 0:
        log.info(f"--> Validation Set Length = {len(dataset_val)}")
    
    # Create model
    processor = transformers.AutoProcessor.from_pretrained(
        cfg.version,
    )
    
    # processor.tokenizer.add_tokens("<pad>", special_tokens=True)
    processor.tokenizer.pad_token = "<|finetune_right_pad_id|>"
    all_new_tokens = garment_tokenizer.get_all_token_names()
    num_added_tokens = processor.tokenizer.add_tokens(all_new_tokens, special_tokens=True)
    if master_process:
        log.info(f"Added {num_added_tokens} tokens to the tokenizer.")
    token_name2_idx_dict = {}
    for token in all_new_tokens:
        token_idx = processor.tokenizer(token, add_special_tokens=False).input_ids[0]
        token_name2_idx_dict[token] = token_idx
        
    if master_process:
        log.info(f"Token name to index dictionary: {token_name2_idx_dict}")
    garment_tokenizer.set_token_indices(token_name2_idx_dict)

    torch_dtype = torch.float32
    if cfg.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        torch_dtype = torch.half

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model = AIpparelMllavaNextForConditionalGeneration.from_pretrained(
        cfg.version,
        **model_dict
    )
    
    model.config.transf_token = PanelEdgeTypeV3.MOVE.value
    model.config.transf_token_index = garment_tokenizer.panel_edge_type_indices.move_idx

    model.config.line_token = PanelEdgeTypeV3.LINE.value
    model.config.line_token_index = garment_tokenizer.panel_edge_type_indices.line_idx

    model.config.quadratic_token = PanelEdgeTypeV3.CURVE.value
    model.config.quadratic_token_index = garment_tokenizer.panel_edge_type_indices.curve_idx

    model.config.cubic_token = PanelEdgeTypeV3.CUBIC.value
    model.config.cubic_token_index = garment_tokenizer.panel_edge_type_indices.cubic_idx

    model.config.arc_token = PanelEdgeTypeV3.ARC.value
    model.config.arc_token_index = garment_tokenizer.panel_edge_type_indices.arc_idx

    model.config.cline_token = PanelEdgeTypeV3.CLOSURE_LINE.value
    model.config.cline_token_index = garment_tokenizer.panel_edge_type_indices.closure_line_idx

    model.config.cquadratic_token = PanelEdgeTypeV3.CLOSURE_CURVE.value
    model.config.cquadratic_token_index = garment_tokenizer.panel_edge_type_indices.closure_curve_idx

    model.config.ccubic_token = PanelEdgeTypeV3.CLOSURE_CUBIC.value
    model.config.ccubic_token_index = garment_tokenizer.panel_edge_type_indices.closure_cubic_idx

    model.config.carc_token = PanelEdgeTypeV3.CLOSURE_ARC.value
    model.config.carc_token_index = garment_tokenizer.panel_edge_type_indices.closure_arc_idx

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    vision_tower = model.vision_model
    vision_tower.to(dtype=torch_dtype, device=ddp_local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.multi_modal_projector.parameters():
        p.requires_grad = False

    model.resize_token_embeddings(len(processor.tokenizer) - model.vocab_size)
    # for name, module in model.named_parameters():
    #     print(name, module.shape, module.requires_grad)
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params} ({trainable_params/total_params:.4f})")
    if cfg.eval_only:
        pass
    elif cfg.gen_only:
        if cfg.gen_split == "train":
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_train, shuffle=False, drop_last=False
            )
            loader = DataLoader(
                dataset_train,
                batch_size=1,
                shuffle=False,
                num_workers=12,
                pin_memory=False,
                sampler=sampler,
                collate_fn=partial(
                    collate_fn,
                    processor=processor,
                    garment_tokenizer=garment_tokenizer,
                    model_version=cfg.version
                ),
            )
            gen_num = 100
        elif cfg.gen_split == "val":
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_val, shuffle=False, drop_last=False
            )
            loader = DataLoader(
                dataset_val,
                batch_size=1,
                shuffle=False,
                num_workers=12,
                pin_memory=False,
                sampler=sampler,
                collate_fn=partial(
                    collate_fn,
                    processor=processor,
                    garment_tokenizer=garment_tokenizer,
                    model_version=cfg.version,
                    inference=True
                ),
            )
            gen_num = -1
        os.makedirs(os.path.join(output_dir, f"generation_outputs_{cfg.gen_split}"), exist_ok=True)
        generate(
            cfg,
            model, 
            loader, 
            garment_tokenizer,
            processor,
            ddp_rank, 
            ddp_world_size,
            os.path.join(output_dir, f"generation_outputs_{cfg.gen_split}"),
            gen_num
        )
    else:
        optimizer_config = {
            "type": "AdamW",
            "params": {
                "lr": cfg.optimizer.lr,
                "weight_decay": 0.0,
                "betas": (cfg.optimizer.beta1, cfg.optimizer.beta2)
            }
        }
        scheduler_config = {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": cfg.num_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": cfg.optimizer.lr,
                "warmup_num_steps": cfg.warmup_steps,
                "warmup_type": "linear"
            }
        }
        ds_config = {
            "train_micro_batch_size_per_gpu": cfg.batch_size,
            "gradient_accumulation_steps": cfg.grad_accumulation_steps,
            "fp16": {
                "enabled": cfg.precision == "fp16",
            },
            "bf16": {
                "enabled": cfg.precision == "bf16",
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },

        }
        ds_config["optimizer"] = optimizer_config
        ds_config["scheduler"] = scheduler_config
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=dataset_train,
            collate_fn=partial(
                collate_fn,
                processor=processor,
                garment_tokenizer=garment_tokenizer,
                model_version=cfg.version,
                inference=False
            ),
            config=ds_config,
        )
    train(
        cfg,
        model_engine, 
        optimizer, 
        train_loader, 
        scheduler,
        garment_tokenizer,
        ddp_rank, 
        ddp_world_size,
        output_dir
    )


if __name__ == "__main__":
    main()
