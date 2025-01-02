# Training loop func
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.data_loaders.infinite_loader import InfiniteDataLoader
import traceback
from typing import Literal, Optional, List
import logging
log = logging.getLogger(__name__)
import torch
import wandb as wb
from dataclasses import dataclass, field
import hydra, os
import numpy as np
# My modules

from data.data_wrappers.garment_token_data_wrapper_v2 import GarmentTokenDataWrapperV2
from data.pattern_converter import NNSewingPattern
from data.panel_classes import PanelClasses
from .trainer_hydra import Trainer, OptimizerConfig, SchedulerConfig, TrainerConfig
from .text_trainer import TextTrainer, TextTrainerConfig
from experiment_hydra import ExperimentWrappper
from models.decoders.lora_gpt2 import GPT
import torch.distributed as dist
import time
import inspect
import sys
import loralib as lora
import random

@dataclass
class FinetuneGarmentTokenTrainerV2Config(TextTrainerConfig):
    _target_: str = "trainers.text_trainer.FinetuneGarmentTokenTrainerV2"
    
    
    

class FinetuneGarmentTokenTrainerV2(TextTrainer):
    def __init__(
        self,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        experiment_tracker: ExperimentWrappper, 
        data_wrapper: GarmentTokenDataWrapperV2, 
        ddp_rank: int, 
        ddp_local_rank: int,
        ddp_world_size: int,
        random_seed: int,
        max_steps: int,
        lr: float,
        max_norm: float,
        steps_per_eval: int,
        steps_per_save: int,
        grad_accum_steps: int,
        ):
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler
        self.experiment = experiment_tracker
        self.datawrapper = data_wrapper
        self.random_seed = random_seed
        self.max_steps = max_steps
        self.lr = lr
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.device = f"cuda:{ddp_local_rank}"
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        self.grad_accum_steps = grad_accum_steps // self.ddp_world_size
        self.master_process = (self.ddp_rank == 0)
        self.max_norm = max_norm
        self.steps_per_eval=steps_per_eval
        self.steps_per_save=steps_per_save
        self.set_seeds()
        

    
    def eval_step(self, step: int, model: nn.parallel.DistributedDataParallel, model_without_ddp: GPT, valid_loader: DataLoader):
            # validation loss
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = len(valid_loader)
                for i, (x, y) in enumerate(valid_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if self.master_process:
                log.info(f"validation loss: {val_loss_accum.item():.4f}")
            # sampling 
            if self.master_process:
                max_length = model_without_ddp.block_size // 2
                input_texts = [
                    "Sewing pattern for jumpsuit sleeveless:",
                    "Sewing pattern for wb dress sleeveless:",
                    "Sewing pattern for dress sleeveless:",
                    "Sewing pattern for tee sleeveless:",
                    "Sewing pattern for skirt 2 panels:",
                    "Sewing pattern for skirt 4 panels:",
                    "Sewing pattern for skirt 8 panels:",
                    "Sewing pattern for tee:",
                    "Sewing pattern for wb pants straight:",
                    "Sewing pattern for pants straight sides:",
                    ]
                input_texts = random.choices(input_texts, k=1)
                num_return_sequences = 4
                tokens = [torch.tensor(model_without_ddp.encode_text(input_text), dtype=torch.long).reshape(1, -1).repeat_interleave(num_return_sequences, 0) for input_text in input_texts]
                sample_rng = torch.Generator(device=self.device)
                sample_rng.manual_seed(42 + self.ddp_rank)
                all_generated_tokens = []
                for token, input_text in zip(tokens, input_texts):
                    xgen = token.to(self.device)
                    terminated_mask = torch.zeros(num_return_sequences, dtype=torch.bool, device=self.device)
                    while xgen.size(1) < max_length:
                        # forward the model to get the logits
                        if terminated_mask.all():
                            break
                        with torch.no_grad():
                            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                                logits, loss = model(xgen) # (B, T, vocab_size)
                            # take the logits at the last position
                            logits = logits[:, -1, :] # (B, vocab_size)
                            # get the probabilities
                            probs = F.softmax(logits, dim=-1)
                            # do top-k sampling of 50 (huggingface pipeline default)
                            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                            # select a token from the top-k probabilities
                            # note: multinomial does not demand the input to sum to 1
                            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                            # gather the corresponding indices
                            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                            terminated_mask = terminated_mask | (xcol == self.datawrapper.dataset.stop_token)
                            # append to the sequence
                            xgen = torch.cat((xgen, xcol), dim=1)
                    xgen[:, -1] = self.datawrapper.dataset.stop_token # force the model to stop
                    all_generated_tokens.append((input_text, " ".join(input_text.split(" ")[3:])[:-1], xgen[:, token.shape[1]:]))
                # print the generated text
                table = wb.Table(columns=["Sample", "Condition Class", "Predicted Indices"])
                for text_cond, cond_class, xgen in all_generated_tokens:
                    for i in range(num_return_sequences):
                        tokens = xgen[i, :max_length].tolist()
                        tokens = np.array(tokens, dtype=np.uint16)
                        panels, panel_names, panel_descriptions = self.datawrapper.dataset.decode_pattern(tokens)
                        save_path = f'{self.experiment.local_output_dir}/step_{step}'
                        os.makedirs(save_path, exist_ok=True)
                        try:
                            self._save_images(outlines=panels, panel_names=panel_names, template_name="_".join(cond_class.split(" ") + [str(i)]), path=save_path)
                        except:
                            continue
                        decoded = " ".join(map(str, tokens))
                        log.info(f"rank {self.ddp_rank} sample {i}: {decoded}")
                        table.add_data(str(i), text_cond, decoded)
            else:
                table = None
            return val_loss_accum.item(), table
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




    def fit(self, model: nn.parallel.DistributedDataParallel, model_without_ddp: GPT):
        """Fit provided model to reviosly configured dataset"""
        if model_without_ddp.lora_attn_dim > 0:
            log.info(f"Using LORA Fintuning with dimension {model_without_ddp.lora_attn_dim}")
            lora.mark_only_lora_as_trainable(model_without_ddp)
            if model_without_ddp.new_vocab_size > 0:
                model_without_ddp.lm_head.requires_grad_(True)
                model_without_ddp.new_lm_head.requires_grad_(True)
                
        if not self.datawrapper:
            raise RuntimeError('{}::Error::fit before dataset was provided. run use_dataset() first'.format(self.__class__.__name__))
        self._add_optimizer(model_without_ddp)
        self._add_scheduler(len(self.datawrapper.loaders.train))
        self.es_tracking = []  # early stopping init
        # TODO
        start_epoch = self._start_experiment(model)
        log.info('NN training Using device: {}'.format(self.device))
        
        self.folder_for_preds = self.experiment.local_output_dir
        
        self._fit_loop(model, model_without_ddp, self.datawrapper.loaders.train, self.datawrapper.loaders.validation, start_epoch=start_epoch)
        if self.master_process:
            log.info("Finished training")

    def _fit_loop(self, model: nn.parallel.DistributedDataParallel, model_without_ddp: GPT, train_loader: DataLoader, valid_loader: DataLoader, start_epoch: int):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""

        global best_valid_loss

        if self.master_process:
            best_valid_loss = self.experiment.last_best_validation_loss()
            best_valid_loss = torch.tensor(best_valid_loss) if best_valid_loss is not None else None
            best_epoch = best_valid_loss is None
        model.to(self.device)
        step = 0
        train_loader_iter = iter(train_loader)
        while step < self.max_steps:
            if self.master_process:
                t0 = time.time()
            last_step = step == self.max_steps - 1
            
            if (step % self.steps_per_eval == 0 ) or last_step:
                valid_loss, sample_table = self.eval_step(step, model, model_without_ddp, valid_loader)
                if self.master_process:
                    wb.log({'valid_loss': valid_loss}, step=step)
                    wb.log({'samples': sample_table}, step=step)
                    if best_valid_loss is None or valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_epoch = True
                    if step > 0 and (step % self.steps_per_save == 0 or last_step):
                        self._save_checkpoint(model, step, valid_loss, best_epoch)
                    
            # training step
            model.train()
            self.optimizer.zero_grad()
            loss_accum = 0.0        
            tokens_processed = torch.zeros(1, device=self.device)
            for i in range(self.grad_accum_steps):
                x, y = next(train_loader_iter)
                x, y = x.to(self.device), y.to(self.device)
                tokens_processed += x.shape[0] * x.shape[1]
                model.require_backward_grad_sync = i == self.grad_accum_steps - 1
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN. Scale the loss here so it comes out right
                loss = loss / self.grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(tokens_processed, op=dist.ReduceOp.SUM)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.step()
            if self.device_type == "cuda":
                torch.cuda.synchronize() # wait for the GPU to finish work
            
            if self.master_process:
                t1 = time.time()
                dt = t1 - t0
                tokens_per_sec = tokens_processed.cpu().item() / dt
                loss_dict = {
                    'loss': loss_accum, 
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'grad_norm': norm,
                    'dt': dt,
                    'tok_per_sec': tokens_per_sec
                }
                wb.log(loss_dict, step=step)
                log.info(f"step: {step:02d} | loss: {loss_accum:.6f} | lr: {self.optimizer.param_groups[0]['lr']:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            step += 1
            