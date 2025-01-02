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

import yaml
from pprint import pprint

# My modules
import sys, os
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pkg_path = "{}/sewformer/SewFactory/packages".format(root_path)
sys.path.insert(0, pkg_path) 
print(pkg_path)


import customconfig
import data
import models
from metrics.eval_ldm_metrics import eval_ldm_metrics
from trainers import TrainerLDM
from experiment import ExperimentWrappper
from models.diffusion.ldm import Diffuser

def get_values_from_args():
    """command line arguments to control the run for running wandb Sweeps!"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', help='YAML configuration file', type=str, default='./models/att/att.yaml')
    parser.add_argument('--test-only', '-t',  action='store_true', default=False)
    parser.add_argument('--local_rank', default=0)
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config, args

if __name__ == '__main__':
    from pprint import pprint 
    np.set_printoptions(precision=4, suppress=True)
    # import pdb; pdb.set_trace()
    config, args = get_values_from_args()
    system_info = customconfig.Properties('./system.json')
    # DDP
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    print(f"INFO::{__file__}::Start running basic DDP example on rank {rank}.")
    config['trainer']['multiprocess'] = True

    experiment = ExperimentWrappper(
        config,  # set run id in cofig to resume unfinished run!
        system_info['wandb_username'],
        no_sync=False) 
    
    # Dataset Class
    data_class = getattr(data, config['dataset']['class'])
    dataset = data_class(system_info['datasets_path'], config['dataset'], gt_caching=True, feature_caching=True)
    trainer = TrainerLDM(
            config['trainer'], experiment, dataset, config['data_split'], 
            with_norm=True, with_visualization=config['trainer']['with_visualization'])  # only turn on visuals on custom garment data
    trainer.init_randomizer()

    # --- Model ---
    model: Diffuser = models.build_model(config)
    model_without_ddp = model
    # DDP
    torch.cuda.set_device(rank)
    model.cuda(rank)

    # Wrap model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if config["NN"]["step-trained"] is not None and os.path.exists(config["NN"]["step-trained"]):
        model.load_state_dict(torch.load(config["NN"]["step-trained"], map_location="cuda:{}".format(rank))["model_state_dict"])
        print("Train::Info::Load Pre-step-trained model: {}".format(config["NN"]["step-trained"]))
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Train::Info::Number of params: {n_parameters}')

    if not args.test_only:    
        trainer.fit(model, model_without_ddp, rank, config)
    else:
        config["NN"]["loss"]["lepoch"] = -1
        if config["NN"]["pre-trained"] is None or not os.path.exists(config["NN"]["pre-trained"]):
            print("Train::Error:Pre-trained model should be set for test only mode")
            raise ValueError("Pre-trained model should be set for test")

    # --- Final evaluation ----
    if rank == 0:
        print(system_info["output"])
        save_to = os.path.join(system_info["output"], experiment.run_id, "final_eval")
        os.makedirs(save_to, exist_ok=True)
        model.load_state_dict(experiment.get_best_model()['model_state_dict'])
        datawrapper = trainer.datawraper
        final_metrics, recon_final_metrics = eval_ldm_metrics(model, model.module.first_stage_criterions, datawrapper, save_to, rank, 'validation')
        experiment.add_statistic('valid_on_best', final_metrics, log='Validation metrics')
        experiment.add_statistic('valid_on_best_recon', recon_final_metrics, log='Validation metrics reconstruction')
        pprint(final_metrics)
        pprint(recon_final_metrics)
        final_metrics, recon_final_metrics = eval_ldm_metrics(model, model.module.first_stage_criterions, datawrapper, save_to, rank, 'test')
        experiment.add_statistic('test_on_best', final_metrics, log='Test metrics')
        experiment.add_statistic('test_on_best_recon', recon_final_metrics, log='Test metrics reconstruction')
        pprint(final_metrics)
        pprint(recon_final_metrics)
        
