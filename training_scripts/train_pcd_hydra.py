from distutils import dir_util
from pathlib import Path
import argparse
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import yaml
from pprint import pprint 
from typing import Optional
import logging

log = logging.getLogger(__name__)

# My modules
import sys, os
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pkg_path = "{}/sewformer/SewFactory/packages".format(root_path)
sys.path.insert(0, pkg_path) 
print(pkg_path)
from dataclasses import dataclass, field

import hydra
import data
import models
from metrics.eval_pcd_metrics import eval_pcd_metrics
from trainers.pcd_trainer_hydra import TrainerPCD, TrainerPCDConfig
from experiment_hydra import ExperimentWrappper, MyExperimentConfig
from data.data_wrappers.wrapper_pcd_hydra import PCDDataWrapperConfig, PCDDatasetWrapper
from models.pcd2garment.garment_pcd_hydra import PCDModelConfig, GarmentPCDLossConfig, GarmentPCD
@dataclass 
class SystemConfig:
    wandb_username: Optional[str] = None
    output: str = "./"
    

@dataclass
class MainConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    experiment: MyExperimentConfig = field(default_factory=MyExperimentConfig)
    data_wrapper: PCDDataWrapperConfig = field(default_factory=PCDDataWrapperConfig)
    trainer: TrainerPCDConfig = field(default_factory=TrainerPCDConfig)
    model: PCDModelConfig = field(default_factory=PCDModelConfig)
    random_seed: Optional[int] = None
    test_only: bool = False
    step_trained: Optional[str] = None
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
    rank = dist.get_rank()
    log.info(f"INFO::{__file__}::Start running basic DDP example on rank {rank}.")
    multiprocess = True

    experiment: ExperimentWrappper = ExperimentWrappper(
        cfg, 
        output_dir, 
        system_info.wandb_username, 
        no_sync=False)
    log.info("Experiment Wrapper created.")
    
    
    model: GarmentPCD = hydra.utils.instantiate(cfg.model, num_panel_queries=cfg.data_wrapper.dataset.panel_info.max_pattern_len)
    log.info("Model created.")
    # Dataset Wrapper
    data_wrapper: PCDDatasetWrapper = hydra.utils.instantiate(
        cfg.data_wrapper, 
        multiprocess=multiprocess, 
        random_seed=cfg.random_seed,
        output_dir=output_dir)
    log.info("Data wrapper created.")

    trainer: TrainerPCD = hydra.utils.instantiate(
        cfg.trainer, 
        experiment_tracker=experiment, 
        data_wrapper=data_wrapper,
        multiprocess=multiprocess,
        random_seed=cfg.random_seed) 
    
    trainer.init_randomizer()
    # --- Model ---
    model_without_ddp = model
    # DDP
    torch.cuda.set_device(rank)
    model.cuda(rank)

    # Wrap model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfg.step_trained is not None and os.path.exists(cfg.step_trained):
        model.load_state_dict(torch.load(cfg.step_trained, map_location="cuda:{}".format(rank))["model_state_dict"])
        log.info("Train::Info::Load Pre-step-trained model: {}".format(cfg.step_trained))
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Train::Info::Number of params: {n_parameters}')

    if not cfg.test_only:    
        trainer.fit(model, model_without_ddp, rank)
    else:
        cfg.model.criteria.lepoch = -1
        if cfg.pre_trained is None or not os.path.exists(cfg.pre_trained):
            log.info("Train::Error:Pre-trained model should be set for test only mode")
            raise ValueError("Pre-trained model should be set for test")

    # --- Final evaluation ----
    if rank == 0:
        save_to = os.path.join(output_dir, "final_eval")
        os.makedirs(save_to, exist_ok=True)
        model.load_state_dict(experiment.get_best_model()['model_state_dict'])
        datawrapper = trainer.datawraper
        final_metrics = eval_pcd_metrics(model, model_without_ddp.criteria, datawrapper, save_to, rank, 'validation')
        experiment.add_statistic('valid_on_best', final_metrics, log='Validation metrics')
        json.dump(final_metrics, open(os.path.join(save_to, "validation", 'final_metrics.json'), 'w'), indent=4)
        pprint(final_metrics)
        final_metrics = eval_pcd_metrics(model, model_without_ddp.criteria, datawrapper, save_to, rank, 'test')
        experiment.add_statistic('test_on_best', final_metrics, log='Test metrics')
        json.dump(final_metrics, open(os.path.join(save_to, "test", 'final_metrics.json'), 'w'), indent=4)
        pprint(final_metrics)
        
if __name__ == '__main__':
    main()