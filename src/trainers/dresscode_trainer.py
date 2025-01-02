# Training loop func
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.data_loaders.infinite_loader import InfiniteDataLoader
import traceback
from typing import Literal, Optional, List, Dict
import logging
log = logging.getLogger(__name__)
import torch
import wandb as wb
from PIL import Image
from dataclasses import dataclass, field
import hydra, os
import deepspeed
import numpy as np
import transformers
from functools import partial
import shutil
# My modules

from data.data_wrappers.garment_image_data_wrapper import GarmentImageDataWrapper
from data.datasets.dataset_garment_as_token_qva import QVAGarmentTokenDataset
from data.pattern_converter import NNSewingPattern
from data.datasets.utils import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, PanelEdgeTypeV2, SpecialTokensV2, DecodeErrorTypes
from data.panel_classes import PanelClasses
from .trainer_hydra import Trainer, OptimizerConfig, SchedulerConfig, TrainerConfig
from .llava_trainer import FinetuneLlavaTrainer, FinetuneLlavaTrainerConfig
from models.gpt2_token_regression import GPTTokenRegression
from experiment_hydra import ExperimentWrappper
from models.decoders.lora_gpt2 import GPT
from trainers.utils import dict_to_cuda, AverageMeter, ProgressMeter, master_log, dict_to_cpu, dict_to_dtype
from data.garment_tokenizers.special_tokens import PanelEdgeTypeV3
from convert_zero_to_torch import _get_fp32_state_dict_from_zero_checkpoint
import torch.distributed as dist
import time
import inspect
import sys
import loralib as lora
import random


class DressCodeTrainer:
    def __init__(
        self,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        experiment_tracker: ExperimentWrappper, 
        data_wrapper: GarmentImageDataWrapper, 
        ddp_rank: int, 
        ddp_local_rank: int,
        ddp_world_size: int,
        random_seed: int,
        max_norm: float,
        precision: Literal["bf16", "fp16", "fp32"],
        lr: float,
        grad_accumulation_steps: int, 
        batch_size: int,
        epoches: int,
        steps_per_epoch: int,
        eval_freq: int,
        log_freq: int,
        model_type: Literal["token", "regression"],
        ):
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler
        self.experiment = experiment_tracker
        self.datawrapper = data_wrapper
        self.random_seed = random_seed
        self.lr = lr
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.device = f"cuda:{ddp_local_rank}"
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        self.master_process = (self.ddp_rank == 0)
        self.max_norm = max_norm
        self.eval_freq=eval_freq
        self.log_freq=log_freq
        self.grad_accumulation_steps=grad_accumulation_steps
        self.batch_size=batch_size
        self.epoches=epoches
        self.steps_per_epoch=steps_per_epoch
        self.precision = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        self.scaler = torch.cuda.amp.GradScaler(enabled=(precision == 'float16'))
        self.start_epoch = 0
        self.model_type = model_type
        self.set_seeds()
        
    def set_seeds(self):
        torch.manual_seed(self.random_seed + self.ddp_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed + self.ddp_rank)
    
    def _add_optimizer(self, model_without_ddp: GPTTokenRegression):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in model_without_ddp.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        weight_decay = self.optimizer_config.weight_decay
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.master_process:
            log.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            log.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(hydra.utils.get_class(self.optimizer_config._target_)).parameters
        use_fused = fused_available and self.device_type == "cuda"
        if self.master_process:
            log.info(f"using fused AdamW: {use_fused}")
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(self.optimizer_config, _convert_="partial", params=optim_groups, lr=float(self.lr), fused=use_fused)
        if self.master_process:
            log.info('Using {} optimizer'.format(self.optimizer_config._target_))
    
    def _add_scheduler(self):
        if self.scheduler_config._target_ is not None:
            self.scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(self.scheduler_config, optimizer=self.optimizer, steps_per_epoch=self.steps_per_epoch, epochs=self.epoches)
        else:
            self.scheduler = None  # no scheduler
            if self.master_process:
                log.warn('no learning scheduling set')
                
    def training_setup(
        self, 
        model: DDP,
        model_without_ddp: GPTTokenRegression, 
        resume: Optional[str]=None
        ):
        self._add_optimizer(model_without_ddp)
        self._add_scheduler()
        self.model = model
        self.model_without_ddp = model_without_ddp
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.datawrapper.training, shuffle=True, drop_last=False
            )
        self.train_loader = DataLoader(
                self.datawrapper.training,
                batch_size=self.batch_size,
                num_workers=12,
                pin_memory=True,
                sampler=train_sampler,
                collate_fn=partial(
                    self.datawrapper.collate_fn,
                    local_rank=self.ddp_local_rank,
                ),
            )
        train_val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.datawrapper.training, shuffle=False, drop_last=False
        )
        self.train_val_loader = DataLoader(
            self.datawrapper.training,
            batch_size=1,
            shuffle=False,
            num_workers=12,
            pin_memory=False,
            sampler=train_val_sampler,
            collate_fn=partial(
                self.datawrapper.collate_fn,
                local_rank=self.ddp_local_rank,
            ),
        )
        if self.datawrapper.validation is not None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.datawrapper.validation, shuffle=False, drop_last=False
            )
            self.val_loader = DataLoader(
                self.datawrapper.validation,
                batch_size=1,
                shuffle=False,
                num_workers=12,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    self.datawrapper.collate_fn,
                    local_rank=self.ddp_local_rank,
                ),
            )
        else:
            self.val_loader = None
            
        self._start_experiment(resume)
        master_log(self.ddp_local_rank, log, 'NN training Using device: {}'.format(self.device))
        
        self.log_dir = self.experiment.local_output_dir
    def eval_setup(
        self, 
        model: DDP,
        model_without_ddp: GPTTokenRegression, 
        ):
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.model_without_ddp.eval()
        self.model.eval()
        self.experiment.init_run()
        
        assert self.datawrapper.validation is not None
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.datawrapper.validation, shuffle=False, drop_last=False
        )
        self.val_loader = DataLoader(
            self.datawrapper.validation,
            batch_size=1,
            shuffle=False,
            num_workers=12,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                self.datawrapper.collate_fn,
                local_rank=self.ddp_local_rank,
            ),
        )
        train_val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.datawrapper.training, shuffle=False, drop_last=False
        )
        self.train_val_loader = DataLoader(
            self.datawrapper.training,
            batch_size=1,
            shuffle=False,
            num_workers=12,
            pin_memory=False,
            sampler=train_val_sampler,
            collate_fn=partial(
                self.datawrapper.collate_fn,
                local_rank=self.ddp_local_rank,
            ),
        )
        self.log_dir = self.experiment.local_output_dir
        if self.master_process:
            wb.watch(self.model, log='all')
        
    
    @torch.no_grad()  
    def generation_step(self, step: int, subset: Literal["validation", "train"]="validation", num_gen=-1):
        if subset == "validation":
            loader = self.val_loader
            gen_len = len(self.val_loader)
        elif subset == "train":
            loader = self.train_val_loader
            print(len(self.datawrapper.training))
            print("Training set length is {}".format(len(self.train_val_loader)))
            gen_len = min(len(self.train_val_loader), 1000)
        else:
            raise ValueError(f"Subset {subset} not supported")
        if num_gen > 0:
            gen_len = min(gen_len, num_gen)
        eval_sample_time = AverageMeter("Time", ":6.3f")
        self.model.eval()
        if self.val_loader is None:
            log.info("No validation data provided, skipping validation")
            return None, None
        all_patterns = []
        all_gt_patterns = []
        all_names = []
        all_captions = []
        all_output_texts = []
        all_errored_texts = []
        all_sample_types = []
        save_path = f'{self.experiment.local_output_dir}/step_{step}/{subset}'
        os.makedirs(save_path, exist_ok=True)
        dist.barrier()
        for i, input_dict in enumerate(loader):
            if i == gen_len:
                break
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)
            end = time.time()
            output_dict = self.model.module.generate(
                input_dict["caption_features"], 
                start_idx=self.datawrapper.dataset.garment_tokenizer.special_token_indices.pattern_start_idx,
                end_idx=self.datawrapper.dataset.garment_tokenizer.special_token_indices.pattern_end_idx,
                top_k=1
            )
            output_dict = dict_to_cpu(output_dict)
            output_dict = dict_to_dtype(output_dict, torch.float32)
            output_text, patterns, error_type = self.datawrapper.dataset.decode(output_dict)
            gt_output_text, gt_patterns, _ = self.datawrapper.dataset.decode({
                'output_ids': input_dict['input_ids'],
                'params': {k:v[0].cpu().numpy() for k, v in input_dict['param_targets'].items()} if self.model_type == "regression" else None,
            })
            try:
                os.makedirs(os.path.join(save_path, input_dict['data_names'][0]), exist_ok=True)
                patterns.serialize(os.path.join(save_path, input_dict['data_names'][0]), to_subfolder=False, spec_only=False, tag=f"pred")
                gt_patterns.serialize(os.path.join(save_path, input_dict['data_names'][0]), to_subfolder=False, spec_only=False, tag=f"gt")
            except Exception as e:
                print(e) 
                pass
            if error_type != DecodeErrorTypes.NO_ERROR:
                all_errored_texts.append((error_type, output_text))
            eval_sample_time.update(time.time() - end)
            end = time.time()
            all_names.append(input_dict["data_names"][0])
            all_captions.append(input_dict["captions"][0])  
            all_gt_patterns.append(gt_patterns)
            all_patterns.append(patterns)
            all_output_texts.append(output_text)
            log.info("Rank: {}, Progress: [{}/{}]".format(self.ddp_rank, i, gen_len))
            
        
        # save all the errored texts as single txt file
        with open(os.path.join(save_path, f"errored_texts{self.ddp_local_rank}.txt"), "w") as f:
            for text in all_errored_texts:
                f.write("Error Type: " + str(text[0]) + "\n")
                f.write("Output Text: " + str(text[1]) + "\n")
        log.info(f"Saved {len(all_errored_texts)} errored texts to {save_path}")
            
        (total_num_panel_correct, 
         num_edge_accs, 
         num_edge_correct_accs, 
         vertex_L2s, 
         transl_l2s, 
         rots_l2s, 
         stitch_accs, 
         sorted_inds) = self.datawrapper.dataset.evaluate_patterns(all_patterns, all_gt_patterns)
        
        modewise_panel_accs = {}
        total_panel_accs = total_num_panel_correct.mean()
        modewise_edge_accs = {}
        total_edge_acc = num_edge_accs[num_edge_accs != -1].sum() \
            / max((num_edge_accs != -1).sum(), 1)
        modewise_edge_correct_accs = {}
        total_edge_correct_acc = num_edge_correct_accs[num_edge_correct_accs != -1].sum() \
            / max((num_edge_correct_accs != -1).sum(), 1)
        modewise_vertex_L2s = {}
        total_vertex_L2 = vertex_L2s[vertex_L2s != -1].sum() \
            / max((vertex_L2s != -1).sum(), 1)
        modewise_transl_l2s = {}
        total_transl_l2 = transl_l2s[transl_l2s != -1].sum() \
            / max((transl_l2s != -1).sum(), 1)
        modewise_rots_l2s = {}
        total_rots_l2 = rots_l2s[rots_l2s != -1].sum() \
            / max((rots_l2s != -1).sum(), 1)
        modewise_stitch_accs = {}
        total_stitch_acc = stitch_accs[stitch_accs != -1].sum() \
            / max((stitch_accs != -1).sum(), 1)
        
        all_sample_types = np.array(all_sample_types)
        for i, mode in enumerate(self.datawrapper.dataset.get_mode_names()):
            mode_mask = all_sample_types == i
            mode_mask = torch.from_numpy(mode_mask).to(num_edge_accs).bool()
            if not mode_mask.any():
                continue
            modewise_panel_accs[mode] = total_num_panel_correct[mode_mask].mean()
            modewise_edge_accs[mode] = num_edge_accs[torch.logical_and(mode_mask, num_edge_accs != -1)].sum() \
                / max((num_edge_accs[mode_mask] != -1).sum(), 1)
            modewise_edge_correct_accs[mode] = num_edge_correct_accs[torch.logical_and(mode_mask, num_edge_correct_accs != -1)].sum() \
                / max((num_edge_correct_accs[mode_mask] != -1).sum(), 1)
            modewise_vertex_L2s[mode] = vertex_L2s[torch.logical_and(mode_mask, vertex_L2s != -1)].sum() \
                / max((vertex_L2s[mode_mask] != -1).sum(), 1)
            modewise_transl_l2s[mode] = transl_l2s[torch.logical_and(mode_mask, transl_l2s != -1)].sum() \
                / max((transl_l2s[mode_mask] != -1).sum(), 1)
            modewise_rots_l2s[mode] = rots_l2s[torch.logical_and(mode_mask, rots_l2s != -1)].sum() \
                / max((rots_l2s[mode_mask] != -1).sum(), 1)
            modewise_stitch_accs[mode] = stitch_accs[torch.logical_and(mode_mask, stitch_accs != -1)].sum() \
                / max((stitch_accs[mode_mask] != -1).sum(), 1)

        # save top error
        if len(sorted_inds) != 0:
            for k in range(min(10, len(sorted_inds))):
                _save_path = os.path.join(self.experiment.local_output_dir, f"epoch_{step}", subset, f"top_error_sample_{self.ddp_local_rank*10 + k}")
                os.makedirs(_save_path, exist_ok=True)
                choice = sorted_inds[k]
                top_num_panel_accuracy, top_num_edge_acc, top_num_edge_correct_acc, top_vertex_L2, top_transl_l2, top_rots_l2, top_stitch_acc, _ = self.datawrapper.dataset.evaluate_patterns([all_patterns[choice]], [all_gt_patterns[choice]])
                try:
                    final_dir = all_patterns[choice].serialize(_save_path, to_subfolder=False, tag=f'_pred')
                except:
                    final_dir = None
                gt_final_dir = all_gt_patterns[choice].serialize(_save_path, to_subfolder=False, tag=f'_gt')
                output_text = all_output_texts[choice]
                f = open(os.path.join(_save_path, "output.txt"), "w")
                f.write(f"Output: {output_text}\n")
                f.write(f"Input Caption: {all_captions[choice]}\n")
                f.write(f"Data Name: {all_names[choice]}\n")
                f.write(f"Num panel_accuracy: {top_num_panel_accuracy}\n")
                f.write(f"Num edge_accuracy: {top_num_edge_acc}\n")
                f.write(f"Num correct edge_accuracy: {top_num_edge_correct_acc}\n")
                f.write(f"vertex_L2: {top_vertex_L2}\n")
                f.write(f"translation_L2: {top_transl_l2}\n")
                f.write(f"rotation_L2: {top_rots_l2}\n")
                f.write(f"stitch_acc: {top_stitch_acc}\n")
                f.close()
                log.info(f"Saved {final_dir} and {gt_final_dir}")
            # save random
            for k in range(10):
                _save_path = os.path.join(self.experiment.local_output_dir, f"epoch_{step}", subset, f"random_sample_{self.ddp_local_rank*10 + k}")
                os.makedirs(_save_path, exist_ok=True)
                choice = random.randint(0, len(all_patterns)-1)
                top_num_panel_accuracy, top_num_edge_acc, top_num_edge_correct_acc, top_vertex_L2, top_transl_l2, top_rots_l2, top_stitch_acc, _ = self.datawrapper.dataset.evaluate_patterns([all_patterns[choice]], [all_gt_patterns[choice]])
                try:
                    final_dir = all_patterns[choice].serialize(_save_path, to_subfolder=False, tag=f'_pred')
                except:
                    final_dir = None
                gt_final_dir = all_gt_patterns[choice].serialize(_save_path, to_subfolder=False, tag=f'_gt')
                output_text = all_output_texts[choice]
                f = open(os.path.join(_save_path, "output.txt"), "w")
                f.write(f"Output: {output_text}\n")
                f.write(f"Input Caption: {all_captions[choice]}\n")
                f.write(f"Data Name: {all_names[choice]}\n")
                f.write(f"Num panel_accuracy: {top_num_panel_accuracy}\n")
                f.write(f"Num edge_accuracy: {top_num_edge_acc}\n")
                f.write(f"Num correct edge_accuracy: {top_num_edge_correct_acc}\n")
                f.write(f"vertex_L2: {top_vertex_L2}\n")
                f.write(f"translation_L2: {top_transl_l2}\n")
                f.write(f"rotation_L2: {top_rots_l2}\n")
                f.write(f"stitch_acc: {top_stitch_acc}\n")
                f.close()
                log.info(f"Saved {final_dir} and {gt_final_dir}")
                
        eval_sample_time.all_reduce()
        for mode in modewise_panel_accs.keys():
            dist.all_reduce(modewise_panel_accs[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_edge_accs[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_edge_correct_accs[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_vertex_L2s[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_transl_l2s[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_rots_l2s[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_stitch_accs[mode], dist.ReduceOp.AVG)
        dist.all_reduce(total_panel_accs, dist.ReduceOp.AVG)
        dist.all_reduce(total_edge_acc, dist.ReduceOp.AVG)
        dist.all_reduce(total_edge_correct_acc, dist.ReduceOp.AVG)
        dist.all_reduce(total_vertex_L2, dist.ReduceOp.AVG)
        dist.all_reduce(total_transl_l2, dist.ReduceOp.AVG)
        dist.all_reduce(total_rots_l2, dist.ReduceOp.AVG)
        if self.master_process:
            log.info(
                "{}: Step: [{}]\t"
                "Sample: {:.3f} ({:.3f})\t"
                "Num Panel Accuracy: {:.4f}\t"
                "Num Edge Accuracy: {:.4f}\t"
                "Num Correct Edge Accuracy: {:.4f}\t"
                "Vertex L2: {:.4f}\t"
                "translation L2: {:.4f}\t"
                "rotation L2: {:.4f}\t"
                "stitch acc: {:.4f}\t".format(
                    subset,
                    step,
                    eval_sample_time.val,
                    eval_sample_time.avg,
                    total_panel_accs.cpu().item(),
                    total_edge_acc.cpu().item(),
                    total_edge_correct_acc.cpu().item(),
                    total_vertex_L2.cpu().item(),
                    total_transl_l2.cpu().item(),
                    total_rots_l2.cpu().item(),
                    total_stitch_acc.cpu().item(),
                )
            )
            log_dict = {
                f"val/{subset}_num_panel_accuracy": total_panel_accs.cpu().item(),
                f"val/{subset}_num_edge_accuracy": total_edge_acc.cpu().item(),
                f"val/{subset}_num_correct_edge_accuracy": total_edge_correct_acc.cpu().item(),
                f"val/{subset}_vertex_L2": total_vertex_L2.cpu().item(),
                f"val/{subset}_translation_L2": total_transl_l2.cpu().item(),
                f"val/{subset}_rotation_L2": total_rots_l2.cpu().item(),
                f"val/{subset}_stitch_acc": total_stitch_acc.cpu().item(),
                f"val/{subset}_sample_time": eval_sample_time.avg,
            }
            for mode in modewise_panel_accs.keys():
                log_dict.update({
                    f"val/{subset}_{mode}_num_panel_accuracy": modewise_panel_accs[mode].cpu().item(),
                    f"val/{subset}_{mode}_num_edge_accuracy": modewise_edge_accs[mode].cpu().item(),
                    f"val/{subset}_{mode}_num_correct_edge_accuracy": modewise_edge_correct_accs[mode].cpu().item(),
                    f"val/{subset}_{mode}_vertex_L2": modewise_vertex_L2s[mode].cpu().item(),
                    f"val/{subset}_{mode}_translation_L2": modewise_transl_l2s[mode].cpu().item(),
                    f"val/{subset}_{mode}_rotation_L2": modewise_rots_l2s[mode].cpu().item(),
                    f"val/{subset}_{mode}_stitch_acc": modewise_stitch_accs[mode].cpu().item(),
                })
            wb.log(log_dict, step=step)
            
    @torch.no_grad()   
    def eval_step(self, step: int):
        # validation loss
        eval_forward_time = AverageMeter("Time", ":6.3f")
        eval_data_time = AverageMeter("Data", ":6.3f")
        eval_losses = AverageMeter("Loss", ":.4f")
        ce_losses = AverageMeter("CE Loss", ":.4f")
        if self.model_type == "regression":
            edge_losses = AverageMeter("Edge Loss", ":.4f")
            edge_type_losses = {self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value: AverageMeter(f"{self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value} Loss", ":.4f") for ind in self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_all_indices()}
        self.model.eval()
        if self.val_loader is None:
            log.info("No validation data provided, skipping validation")
            return None, None
        end = time.time()
        for kk, input_dict in enumerate(self.val_loader):
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)
            
            with torch.cuda.amp.autocast(dtype=self.precision):
                output_dict = self.model(**input_dict)
                loss = output_dict["total_loss"] / self.grad_accumulation_steps
                
            ce_loss = output_dict["ce_loss"]
            if self.model_type == "regression":
                edge_loss = output_dict.get("edge_loss", 0)
                edge_losses.update(edge_loss.mean(), input_dict["caption_features"].size(0))
                for k, meter in edge_type_losses.items():
                    if f"{k}_loss" in output_dict:
                        meter.update(output_dict[f"{k}_loss"], input_dict["caption_features"].size(0))
            
                    
            eval_losses.update(loss.mean(), input_dict["caption_features"].size(0))
            ce_losses.update(ce_loss.mean(), input_dict["caption_features"].size(0))
            
            eval_forward_time.update(time.time() - end)
            end = time.time()
            log.info("Rank: {}, Progress: [{}/{}]".format(self.ddp_rank, kk, len(self.val_loader)))
        
        eval_data_time.all_reduce()
        eval_forward_time.all_reduce()
        eval_losses.all_reduce()
        ce_losses.all_reduce()
        if self.model_type == "regression":
            edge_losses.all_reduce()
            for k, meter in edge_type_losses.items():
                meter.all_reduce()
        if self.master_process:
            save_path = f'{self.experiment.local_output_dir}/step_{step}'
            os.makedirs(save_path, exist_ok=True)
            log_str = "Validation: Epoch: [{}]\tData: {:.3f} ({:.3f})\tForward: {:.3f} ({:.3f})\tLoss: {:.4f} ({:.4f})\tCE Loss: {:.4f} ({:.4f})\t".format(
                step,
                eval_data_time.val,
                eval_data_time.avg,
                eval_forward_time.val,
                eval_forward_time.avg,
                eval_losses.val,
                eval_losses.avg,
                ce_losses.val,
                ce_losses.avg,
            )
            if self.model_type == "regression":
                log_str = log_str + "Edge Loss: {:.4f} ({:.4f})\t".format(
                    edge_losses.val,
                    edge_losses.avg,
                )
            log.info(log_str)
            wandb_log_dict = {
                "val/data_time": eval_data_time.avg,
                "val/forward_time": eval_forward_time.avg,
                "val/loss": eval_losses.avg,
                "val/ce_loss": ce_losses.avg
            }
            if self.model_type == "regression":
                wandb_log_dict["val/edge_loss"] = edge_losses.avg
                for k, meter in edge_type_losses.items():
                    wandb_log_dict[f"val/{k}_loss"] = meter.avg
            wb.log(wandb_log_dict, step=step)
        return eval_losses.avg

    def _save_images(self, outlines, panel_names, template_name, path):
        pattern = NNSewingPattern(view_ids=False, panel_classifier=None, template_name=template_name)
        pattern.pattern_from_tensors(outlines, 
                        panel_rotations=None,
                        panel_translations=None, 
                        stitches=None,
                        padded=True, 
                        panel_names=panel_names)
        pattern.name = template_name
        pattern.serialize(path, to_subfolder=True)


    def _start_experiment(self, resume: Optional[str]=None):
        # resume deepspeed checkpoint
        self.experiment.init_run()
        if resume:
            state_dict = torch.load(resume, map_location="cpu")
            self.model.load_state_dict(state_dict["model_state_dict"])
            self.start_epoch = state_dict["epoch"]
            log.info("resume training from {}, start from epoch {}".format(
                    resume, self.start_epoch
                ))
                
            dist.barrier()
        if self.master_process:
            wb.watch(self.model, log='all')
    def fit(self):
        """Fit provided model to reviosly configured dataset"""
        
        if not self.datawrapper:
            raise RuntimeError('{}::Error::fit before dataset was provided. run use_dataset() first'.format(self.__class__.__name__))
        
        self._fit_loop()
        master_log(self.ddp_local_rank, log, "Finished training")

    def _save_checkpoint(self, epoch):
        save_dir = os.path.join(self.log_dir, f"ckpt_{epoch}.pth")
        if self.master_process:
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                 },save_dir)
            log.info(f"Saved checkpoint to {save_dir}")
        dist.barrier()
    
    def _fit_loop(self):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""
        train_loader_iter = iter(self.train_loader)
        for epoch in range(self.start_epoch, self.epoches):
            last_epoch = (epoch == self.epoches - 1)
            train_batch_time = AverageMeter("Time", ":6.3f")
            train_data_time = AverageMeter("Data", ":6.3f")
            train_losses = AverageMeter("Total Loss", ":.4f")
            ce_losses = AverageMeter("CE Loss", ":.4f")
            all_meters = [train_batch_time, train_data_time, train_losses, ce_losses]
            if self.model_type == "regression":
                edge_losses = AverageMeter("Edge Loss", ":.4f")
                edge_type_losses = {self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value: AverageMeter(f"{self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value} Loss", ":.4f") for ind in self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_all_indices()}
                all_meters += [edge_losses] + list(edge_type_losses.values())
            progress = ProgressMeter(
                log, 
                self.ddp_local_rank, 
                self.steps_per_epoch,
                all_meters,
                prefix="Epoch: [{}]".format(epoch),
            )
            self.model.train()
            for step in range(self.steps_per_epoch):
                # training step
                for i in range(self.grad_accumulation_steps):
                    self.model.require_backward_grad_sync = (i == self.grad_accumulation_steps - 1)
                    end = time.time()
                    try:
                        input_dict = next(train_loader_iter)
                    except:
                        train_loader_iter = iter(self.train_loader)
                        input_dict = next(train_loader_iter)
                    
                    train_data_time.update(time.time() - end)
                    input_dict = dict_to_cuda(input_dict)
                    
                    with torch.cuda.amp.autocast(dtype=self.precision):
                        output_dict = self.model(**input_dict)
                        loss = output_dict["total_loss"] / self.grad_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                    # measure elapsed time
                    train_batch_time.update(time.time() - end)
                    train_losses.update(output_dict["total_loss"], input_dict["caption_features"].size(0))
                    ce_losses.update(output_dict["ce_loss"], input_dict["caption_features"].size(0))
                    if self.model_type == "regression":
                        edge_loss = output_dict.get("edge_loss", 0)
                        edge_losses.update(edge_loss, input_dict["caption_features"].size(0))
                        for k, meter in edge_type_losses.items():
                            if f"{k}_loss" in output_dict:
                                meter.update(output_dict[f"{k}_loss"], input_dict["caption_features"].size(0))
                if self.max_norm != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                # flush the gradients as soon as we can, no need for this memory anymore
                self.optimizer.zero_grad(set_to_none=True)
                
                if step % self.log_freq == 0 or True:
                    if self.ddp_world_size > 1:
                        train_batch_time.all_reduce()
                        train_data_time.all_reduce()
                        train_losses.all_reduce()
                        ce_losses.all_reduce()
                        if self.model_type == "regression":
                            edge_losses.all_reduce()
                            for k, meter in edge_type_losses.items():
                                meter.all_reduce()

                    if self.master_process:
                        progress.display(step + 1)
                        log_dict = {
                            "train/loss": train_losses.avg,
                            "train/ce_loss": ce_losses.avg,
                            "train/batch_time": train_batch_time.avg,
                            "train/data_time": train_data_time.avg,
                        }
                        if self.model_type == "regression":
                            log_dict["train/edge_loss"] = edge_losses.avg
                            for k, meter in edge_type_losses.items():
                                log_dict[f"train/{k}_loss"] = meter.avg
                        wb.log(log_dict, step=step + epoch * self.steps_per_epoch)

                    train_batch_time.reset()
                    train_data_time.reset()
                    train_losses.reset()
                    ce_losses.reset()
                    if self.model_type == "regression":
                        edge_losses.reset()
                        for k, meter in edge_type_losses.items():
                            meter.reset()
            
                if step != 0:
                    curr_lr = self.scheduler.get_last_lr()
                    if self.master_process:
                        wb.log({"train/lr": curr_lr[0]}, step + epoch * self.steps_per_epoch)
                
                
            if (epoch % self.eval_freq == 0 ) or last_epoch:
                self._save_checkpoint(epoch)
                self.model.eval()
                try:
                    self.eval_step(step + epoch * self.steps_per_epoch)
                    self.generation_step(step + epoch * self.steps_per_epoch, 'train', num_gen = 10)
                    self.generation_step(step + epoch * self.steps_per_epoch, 'validation', num_gen = 10)
                except Exception as e:
                    log.error(f"Error in evaluation step: {e}")
                    
            