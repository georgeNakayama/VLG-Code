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
import hydra
# My modules

from data.data_wrappers.text_data_wrapper import TextDataWrapper, DataLoaderLite
from .trainer_hydra import Trainer, OptimizerConfig, SchedulerConfig, TrainerConfig
from experiment_hydra import ExperimentWrappper
from models.decoders.gpt2 import GPT
import torch.distributed as dist
import time
import inspect
import sys

@dataclass
class TextTrainerConfig():
    _target_: str = "trainers.text_trainer.TextTrainer"
    lr: float = 0.001
    total_batch_size: int = 524288 # 2**19, ~0.5M, in number of tokens
    max_steps: int = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
    max_norm: float = 1.0
    steps_per_eval: int = 250
    steps_per_save: int = 5000
    
    
    

class TextTrainer():
    def __init__(
        self,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        experiment_tracker: ExperimentWrappper, 
        data_wrapper: TextDataWrapper, 
        ddp_rank: int, 
        ddp_local_rank: int,
        ddp_world_size: int,
        random_seed: int,
        max_steps: int,
        lr: float,
        total_batch_size: int,
        max_norm: float,
        steps_per_eval: int,
        steps_per_save: int
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
    def set_seeds(self):
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
        
        
    
    def _add_optimizer(self, model_without_ddp: GPT):
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
    
    def _add_scheduler(self, steps_per_epoch):
        if self.scheduler_config._target_ is not None:
            self.scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(self.scheduler_config, optimizer=self.optimizer, steps_per_epoch=1, epochs=self.max_steps)
        else:
            self.scheduler = None  # no scheduler
            if self.master_process:
                log.warn('no learning scheduling set')
    
    def _restore_run(self, model: nn.parallel.DistributedDataParallel):
        """Restore the training process from the point it stopped at. 
            Assuming 
                * Current wb.config state is the same as it was when run was initially created
                * All the necessary training objects are already created and only need update
                * All related object types are the same as in the resuming run (model, optimizer, etc.)
                * Self.run_id is properly set
            Returns id of the next epoch to resume from. """
        
        # get latest checkoint info
        checkpoint = self.experiment.get_checkpoint_file(device=self.device)  # latest

        # checkpoint loaded correctly
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208

        # new epoch id
        return checkpoint['epoch'] + 1
    
    def _start_experiment(self, model: nn.parallel.DistributedDataParallel):
        self.experiment.init_run()

        if self.experiment.get_pre_trained() is not None:
            checkpoint = torch.load(self.experiment.get_pre_trained(), map_location=self.device)
            # checkpoint loaded correctly
            model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208

            # new epoch id
            start_epoch = checkpoint['epoch'] + 1
            if self.master_process:
                log.info('Resumed run {} from epoch {}'.format(self.experiment.get_pre_trained(), start_epoch))
        else:
            start_epoch = 0
            # record configurations of data and model
        if self.master_process:
            wb.watch(model, log='all')
        return start_epoch
    
    def _save_checkpoint(self, model: nn.parallel.DistributedDataParallel, step: int, valid_loss: float, best: bool):
        checkpoint_dict = {
            'epoch': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': valid_loss
        }
        if self.scheduler is not None:
            checkpoint_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        self.experiment.save_checkpoint(
            checkpoint_dict,
            aliases=['best'] if best else [], 
            step=step
        )
        
    def fit(self, model: nn.parallel.DistributedDataParallel, model_without_ddp: GPT):
        """Fit provided model to reviosly configured dataset"""

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
                wb.log({'validation_loss': val_loss_accum.item()}, step=step)
            
            # sampling 
            if self.master_process:
                num_return_sequences = 4
                max_length = 32
                input_text = "Hello, I'm a language model,"
                tokens = model_without_ddp.encode_text(input_text)
                tokens = torch.tensor(tokens, dtype=torch.long)
                tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
                xgen = tokens.to(self.device)
                sample_rng = torch.Generator(device=self.device)
                sample_rng.manual_seed(42 + self.ddp_rank)
                while xgen.size(1) < max_length:
                    # forward the model to get the logits
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
                        # append to the sequence
                        xgen = torch.cat((xgen, xcol), dim=1)
                # print the generated text
                table = wb.Table(columns=["Sample", "Condition Text", "Predicted Text"])
                for i in range(num_return_sequences):
                    tokens = xgen[i, :max_length].tolist()
                    decoded = model_without_ddp.decode_text(tokens)
                    log.info(f"rank {self.ddp_rank} sample {i}: {decoded}")
                    table.add_data(str(i), input_text, decoded)
                wb.log({'samples': table}, step=step)
            return val_loss_accum.item()
    def _fit_loop(self, model: nn.parallel.DistributedDataParallel, model_without_ddp: GPT, train_loader: DataLoaderLite, valid_loader: DataLoaderLite, start_epoch: int):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""

        global best_valid_loss
        best_epoch = False
        if self.master_process:
            best_valid_loss = self.experiment.last_best_validation_loss()
            best_valid_loss = torch.tensor(best_valid_loss) if best_valid_loss is not None else None
        model.to(self.device)
        for step in range(self.max_steps):
            if self.master_process:
                t0 = time.time()
            last_step = step == self.max_steps - 1
            
            if step % self.steps_per_eval == 0 or last_step:
                valid_loss = self.eval_step(step, model, model_without_ddp, valid_loader)
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