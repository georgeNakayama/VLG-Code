# Training loop func
from pathlib import Path
from torch import nn
import torch.nn.functional as F
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

from data.data_wrappers.finetune_garment_token_data_wrapper import GarmentTokenDataWrapper, DataLoaderLite
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

@dataclass
class FinetuneGarmentTokenTrainerConfig(TextTrainerConfig):
    _target_: str = "trainers.text_trainer.FinetuneGarmentTokenTrainer"
    
    
    

class FinetuneGarmentTokenTrainer(TextTrainer):
    def __init__(
        self,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        experiment_tracker: ExperimentWrappper, 
        data_wrapper: GarmentTokenDataWrapper, 
        ddp_rank: int, 
        ddp_local_rank: int,
        ddp_world_size: int,
        random_seed: int,
        max_steps: int,
        lr: float,
        total_batch_size: int,
        max_norm: float,
        steps_per_eval: int,
        steps_per_save: int,
        ):
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler
        self.experiment = experiment_tracker
        self.datawrapper = data_wrapper
        self.random_seed = random_seed
        self.max_steps = max_steps
        self.lr = lr
        self.total_batch_size = total_batch_size
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.device = f"cuda:{ddp_local_rank}"
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        self.grad_accum_steps = self.total_batch_size // (self.datawrapper.batch_size * self.datawrapper.sequence_length * self.ddp_world_size)
        self.master_process = (self.ddp_rank == 0)
        self.max_norm = max_norm
        self.steps_per_eval=steps_per_eval
        self.steps_per_save=steps_per_save
        self.set_seeds()
        

    

    
    
    def eval_step(self, step: int, model: nn.parallel.DistributedDataParallel, model_without_ddp: GPT, valid_loader: DataLoaderLite):
            # validation loss
            model.eval()
            valid_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = valid_loader.next_batch()
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
                max_length = model_without_ddp.block_size
                input_texts = [
                    "Sewing pattern for sleeveless jumpsuit:",
                    "Sewing pattern for sleeveless dress:",
                    "Sewing pattern for sleeveless tee:",
                    "Sewing pattern for skirt with 2 panels:",
                    "Sewing pattern for skirt with 4 panels:",
                    "Sewing pattern for skirt with 8 panels:",
                    "Sewing pattern for tee:",
                    "Sewing pattern for straight pants:",
                    ]
                num_return_sequences = 2
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
                            terminated_mask = terminated_mask | (xcol == self.datawrapper.stop_token)
                            # append to the sequence
                            xgen = torch.cat((xgen, xcol), dim=1)
                    xgen[:, -1] = self.datawrapper.stop_token # force the model to stop
                    all_generated_tokens.append((input_text, " ".join(input_text.split(" ")[3:])[:-1], xgen[:, token.shape[1]:]))
                # print the generated text
                table = wb.Table(columns=["Sample", "Condition Class", "Predicted Indices"])
                for text_cond, cond_class, xgen in all_generated_tokens:
                    for i in range(num_return_sequences):
                        tokens = xgen[i, :max_length].tolist()
                        tokens = np.array(tokens, dtype=np.uint16)
                        panels, invalid_panel_mask, invalid_edge_mask, panel_names = self._decode_tokens(tokens, model_without_ddp)
                        save_path = f'{self.experiment.local_output_dir}/step_{step}'
                        os.makedirs(save_path, exist_ok=True)
                        self._save_images(outlines=panels, panel_names=panel_names, template_name="_".join(cond_class.split(" ") + [str(i)]), path=save_path)
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


    def _decode_tokens( 
        self, 
        token_sequence, 
        model_without_ddp: GPT, 
        pad_panels_to_len=14, 
        pad_panel_num=23,
        min_range = np.array([-1.69588947, -2.79317152, -0.59022361, -2.00742926]),
        max_range = np.array([ 3.93797993,  2.79317152,  2.83853553,  3.74875851]),
        edge_scale = np.array([26.63403268, 29.54941741,  0.27706816,  0.2372687 ]),
        edge_shift = np.array([1.14394751e-17, 5.51454519e-19, 1.63532172e-01, 7.63001275e-02]),
                    ): 
        # find the panel starts and ends
        garment_end = np.where(token_sequence == self.datawrapper.stop_token)[0]
        if len(garment_end) > 0:
            token_sequence = token_sequence[:garment_end[0]]
        else:
            token_sequence = token_sequence
        panel_starts = np.where(token_sequence == self.datawrapper.panel_start)[0]
        panel_ends = np.where(token_sequence == self.datawrapper.panel_end)[0]

        panels = np.zeros((pad_panel_num, pad_panels_to_len, 4))
        panel_names = ["NONE" for _ in range(pad_panel_num)]
        invalid_panel_mask = np.ones(pad_panel_num, dtype=bool)
        invalid_edge_mask = np.ones((pad_panel_num, pad_panels_to_len), dtype=bool)

        if len(panel_starts) < len(panel_ends):
            panel_ends = panel_ends[:len(panel_starts)]
        if len(panel_starts) > len(panel_ends):
            panel_starts = panel_starts[:len(panel_ends)]
        if np.any(panel_starts > panel_ends):
            return panels, invalid_panel_mask, invalid_edge_mask, panel_names
        n_panels = len(panel_starts)
        
        current_mark = 0
        for i in range(n_panels):
            panel_start = panel_starts[i]
            panel_end = panel_ends[i]
            panel_description = model_without_ddp.decode_text(token_sequence[current_mark:panel_start])
            panel_names[i] = panel_description if panel_description != '' else "NONE"
            panel = token_sequence[panel_start+1:panel_end]
            current_mark = panel_end + 1
            if len(panel) % 4 != 0:
                continue

            edges = panel.reshape(-1, 4)
            edges = edges.astype(float)
            edges = ((edges - model_without_ddp.vocab_size) + 0.5) / self.datawrapper.bin_size
            edges = edges * (max_range - min_range) + min_range
            edges = edges * edge_scale + edge_shift
            panels[i, :len(edges)] = edges
            invalid_edge_mask[i, :len(edges)] = False
            invalid_panel_mask[i] = False
            
        return panels, invalid_panel_mask, invalid_edge_mask, panel_names

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

    def _fit_loop(self, model: nn.parallel.DistributedDataParallel, model_without_ddp: GPT, train_loader: DataLoaderLite, valid_loader: DataLoaderLite, start_epoch: int):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""

        global best_valid_loss

        if self.master_process:
            best_valid_loss = self.experiment.last_best_validation_loss()
            best_valid_loss = torch.tensor(best_valid_loss) if best_valid_loss is not None else None
            best_epoch = best_valid_loss is None
        model.to(self.device)
        for step in range(self.max_steps):
            if self.master_process:
                t0 = time.time()
            last_step = step == self.max_steps - 1
            
            if step % self.steps_per_eval == 0 or last_step:
                valid_loss, sample_table = self.eval_step(step, model, model_without_ddp, valid_loader)
                if self.master_process:
                    wb.log({'valid_loss': valid_loss}, step=step)
                    wb.log({'samples': sample_table}, step=step)
                    if best_valid_loss is None or valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_epoch = True
                if step > 0 and (step % self.steps_per_save == 0 or last_step) and self.master_process:
                    self._save_checkpoint(model, step, valid_loss, best_epoch)
                    
                    
            model.train()
            self.optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(self.grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                # added after video, this field is also used by the forward pass.
                model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)
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
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.step()
            if self.device_type == "cuda":
                torch.cuda.synchronize() # wait for the GPU to finish work
            if self.master_process:
                t1 = time.time()
                dt = t1 - t0
                tokens_processed = train_loader.B * train_loader.T * self.grad_accum_steps * self.ddp_world_size
                tokens_per_sec = tokens_processed / dt
                loss_dict = {
                    'loss': loss_accum, 
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'grad_norm': norm,
                    'dt': dt,
                    'tok_per_sec': tokens_per_sec
                }
                wb.log(loss_dict, step=step)
                log.info(f"step: {step:02d} | loss: {loss_accum:.6f} | lr: {self.optimizer.param_groups[0]['lr']:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")