from distutils import dir_util
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import os 
from pprint import pprint 
from typing import Optional
import logging
from typing import Literal
import loralib as lora
log = logging.getLogger(__name__)
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)
from dataclasses import dataclass, field

import hydra
from trainers.garment_token_trainer import GarmentTokenTrainer, GarmentTokenTrainerConfig
from experiment_hydra import ExperimentWrappper, MyExperimentConfig
from data.data_wrappers.garment_token_data_wrapper_v2 import GarmentTokenDataWrapperV2, GarmentTokenDataWrapperV2Config
from models.decoders.gpt2 import GPT, GPTConfig
@dataclass 
class SystemConfig:
    wandb_username: Optional[str] = None
    output: str = "./"
    

@dataclass
class MainConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    experiment: MyExperimentConfig = field(default_factory=MyExperimentConfig)
    data_wrapper: GarmentTokenDataWrapperV2Config = field(default_factory=GarmentTokenDataWrapperV2Config)
    trainer: GarmentTokenTrainerConfig = field(default_factory=GarmentTokenTrainerConfig)
    model: GPTConfig = field(default_factory=GPTConfig)
    random_seed: Optional[int] = None
    test_only: bool = False
    from_pretrained: Optional[Literal['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']] = None
    pre_trained: Optional[str] = None
    storage_dir: Optional[str] = None

@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(cfg: MainConfig):
    np.set_printoptions(precision=4, suppress=True)
    # import pdb; pdb.set_trace()
    system_info = cfg.system
    log.info(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg.storage_dir = output_dir
    # DDP
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = (ddp_rank == 0)
    log.info(f"{__file__}::Start running basic DDP example on rank {ddp_rank}.")

    experiment: ExperimentWrappper = ExperimentWrappper(
        cfg, 
        output_dir, 
        system_info.wandb_username, 
        master_process=master_process,
        no_sync=False)
    log.info("Experiment Wrapper created.")
    
    
    # Dataset Wrapper
    data_wrapper: GarmentTokenDataWrapperV2 = hydra.utils.instantiate(
        cfg.data_wrapper, 
        process_rank=ddp_rank, 
        num_processes=ddp_world_size,
        output_dir=output_dir)
    log.info("Data wrapper created.")
    
    if cfg.from_pretrained is not None:
        model, model_config =  hydra.utils.get_class(cfg.model._target_).from_pretrained(cfg.from_pretrained, cfg.model)
    else:
        model: GPT = hydra.utils.instantiate(cfg.model)
        
    log.info("Model created.")

    trainer: GarmentTokenTrainer = hydra.utils.instantiate(
        cfg.trainer, 
        experiment_tracker=experiment, 
        data_wrapper=data_wrapper,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        random_seed=cfg.random_seed) 
    
    # --- Model ---
    model_without_ddp = model
    torch.cuda.set_device(ddp_local_rank)
    model.cuda(ddp_local_rank)

    # Wrap model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfg.pre_trained is not None and os.path.exists(cfg.pre_trained):
        ckpt = torch.load(cfg.pre_trained, map_location="cuda:{}".format(ddp_local_rank))
        model.load_state_dict(ckpt["model_state_dict"])
        for module in model.modules():
            if isinstance(module, lora.MergedLinear) or \
                isinstance(module, lora.Linear) or \
                isinstance(module, lora.Embedding) or \
                isinstance(module, lora.ConvLoRA):
                if module.merge_weights:
                    module.merged = True 
            
        step = ckpt['epoch']
        if master_process:
            log.info("Load Pre-step-trained model: {}".format(cfg.pre_trained))
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Number of params: {n_parameters}')

    if not cfg.test_only:    
        trainer.fit(model, model_without_ddp)
    else:
        trainer.eval_step(step, model, model_without_ddp, data_wrapper.loaders.validation)

        
if __name__ == '__main__':
    main()