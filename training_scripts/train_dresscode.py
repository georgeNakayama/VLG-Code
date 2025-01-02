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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from experiment_hydra import ExperimentWrappper, MyExperimentConfig
from models.gpt2_token_regression import GPTTokenRegression, GPTConfig
from models.llava import conversation as conversation_lib
from data.data_wrappers.garment_image_data_wrapper import GarmentImageDataWrapper, GarmentImageDataWrapperConfig
from trainers.dresscode_trainer import DressCodeTrainer, FinetuneLlavaTrainerConfig
# My modules
import sys, os
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pkg_path = "{}/sewformer/SewFactory/packages".format(root_path)
sys.path.insert(0, pkg_path) 
print(pkg_path)
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
    system: SystemConfig = field(default_factory=SystemConfig)
    experiment: MyExperimentConfig = field(default_factory=MyExperimentConfig)
    trainer: FinetuneLlavaTrainerConfig = field(default_factory=FinetuneLlavaTrainerConfig)
    data_wrapper: GarmentImageDataWrapperConfig = field(default_factory=GarmentImageDataWrapperConfig)
    model: GPTConfig = field(default_factory=GPTConfig)
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    model_type: Literal["token", "regression"] = "regression"
    eval_only: bool = False
    gen_only: bool = False
    pre_trained: Optional[str] = None
    storage_dir: Optional[str] = None

@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(cfg: MainConfig):
    log.info(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Output directory : {output_dir}")
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    system_info = cfg.system
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
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

    trainer: DressCodeTrainer = hydra.utils.instantiate(
        cfg.trainer,
        experiment_tracker=experiment,
        data_wrapper=data_wrapper,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        ddp_local_rank=ddp_local_rank,
        precision=cfg.precision,
        model_type=cfg.model_type,
    )
    
    all_new_tokens = data_wrapper.dataset.garment_tokenizer.get_all_token_names()
    all_new_tokens = ["<pad>"] + all_new_tokens
    if master_process:
        log.info(f"Added {len(all_new_tokens)} tokens to the tokenizer.")
    token_name2_idx_dict = {token:i for i, token in enumerate(all_new_tokens)}
    if master_process:
        log.info(f"Token name to index dictionary: {token_name2_idx_dict}")
    data_wrapper.dataset.garment_tokenizer.set_token_indices(token_name2_idx_dict)
    if cfg.model_type == "token":
        model_without_ddp: GPTTokenRegression = hydra.utils.instantiate(
            cfg.model, vocab_size=len(all_new_tokens))
    elif cfg.model_type == "regression":
        model_without_ddp: GPTTokenRegression = hydra.utils.instantiate(
            cfg.model, 
            panel_edge_indices=data_wrapper.dataset.garment_tokenizer.panel_edge_type_indices, 
            vocab_size=len(all_new_tokens),
            gt_stats=data_wrapper.dataset.garment_tokenizer.gt_stats)
    model_without_ddp = model_without_ddp.to(trainer.device)
    model = DDP(model_without_ddp, device_ids=[ddp_local_rank], output_device=ddp_local_rank)

    for name, module in model.named_parameters():
        print(name, module.shape, module.requires_grad)
        
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    trainer.training_setup(model, model_without_ddp, cfg.pre_trained)
    
    if cfg.eval_only:
        trainer.eval_step(trainer.start_epoch * trainer.steps_per_epoch)
        trainer.generation_step(trainer.start_epoch * trainer.steps_per_epoch)
    elif cfg.gen_only:
        trainer.generation_step(trainer.start_epoch)
    else:
        trainer.fit()


if __name__ == "__main__":
    main()
