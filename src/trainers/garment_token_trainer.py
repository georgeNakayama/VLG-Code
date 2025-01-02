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
from shapely import Polygon
from trimesh.path.polygons import sample
# My modules

from data.data_wrappers.garment_token_data_wrapper_v2 import GarmentTokenDataWrapperV2
from data.pattern_converter import NNSewingPattern
from data.panel_classes import PanelClasses
from .trainer_hydra import Trainer, OptimizerConfig, SchedulerConfig, TrainerConfig
from .text_trainer import TextTrainer, TextTrainerConfig
from experiment_hydra import ExperimentWrappper
from models.decoders.gpt2 import GPT
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import time
import inspect
import sys
# from metrics.composed_loss_hydra import ComposedLoss, ComposedPatternLoss

@dataclass
class GarmentTokenTrainerConfig(TextTrainerConfig):
    _target_: str = "trainers.text_trainer.GarmentTokenTrainer"
    
    
class GarmentTokenTrainer(TextTrainer):
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
        self.grad_accum_steps = len(self.datawrapper.loaders.train)
        self.master_process = (self.ddp_rank == 0)
        self.max_norm = max_norm
        self.steps_per_eval=steps_per_eval
        self.steps_per_save=steps_per_save
        # self.composed_loss=ComposedPatternLoss()
        self.set_seeds()
        
    
    
    def eval_step(self, step: int, model: nn.parallel.DistributedDataParallel, model_without_ddp: GPT, valid_loader: DataLoader):
            # validation loss
            model.eval()
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
                cat_to_id = {
                    'jumpsuit_sleeveless': 0, 
                    'wb_dress_sleeveless': 1, 
                    'dress_sleeveless': 2, 
                    'tee_sleeveless': 3, 
                    'skirt_2_panels': 4, 
                    'pants_straight_sides': 5, 
                    'tee': 6, 
                    'skirt_8_panels': 7, 
                    'wb_pants_straight': 8, 
                    'skirt_4_panels': 9
                }
                id_to_cat = {v: k for k, v in cat_to_id.items()}
                
                num_return_sequences = len(cat_to_id.keys())
                max_length = model_without_ddp.block_size
                input_cond = valid_loader.class_tokens
                tokens = torch.tensor(input_cond, dtype=torch.long).reshape(num_return_sequences, 1)
                xgen = tokens.to(self.device)
                sample_rng = torch.Generator(device=self.device)
                sample_rng.manual_seed(42 + self.ddp_rank)
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
                # print the generated text
                table = wb.Table(columns=["Sample", "Condition Class", "Predicted Indices"])
                chamfer_loss = []
                for i in range(num_return_sequences):
                    tokens = xgen[i, :max_length].cpu().numpy()
                    cond, tokens = tokens[0].item(), tokens[1:]
                    tokens = tokens.astype(np.uint16)
                    panels, invalid_panel_mask, invalid_edge_mask = self.datawrapper.dataset.decode_pattern(tokens)
                    save_dir = os.path.join(self.experiment.local_output_dir, f'step_{step}')
                    os.makedirs(save_dir, exist_ok=True)
                    self._save_images(outlines=panels, garment_class=id_to_cat[cond - self.datawrapper.dataset.bin_size], path=save_dir)
                    decoded = " ".join(map(str, tokens))
                    log.info(f"rank {self.ddp_rank} sample {i}: {decoded}")
                    table.add_data(str(i), str(input_cond[i]), decoded)
            else:
                table = None
            return val_loss_accum.item(), table
    def _save_images(self, outlines, garment_class, path):
        pattern = NNSewingPattern(view_ids=False, panel_classifier=PanelClasses(self.datawrapper.panel_classification))
        pattern.name = garment_class
        pattern.pattern_from_tensors(outlines, 
                        panel_rotations=None,
                        panel_translations=None, 
                        stitches=None,
                        padded=True)
        pattern.serialize(path, to_subfolder=False)


    def _decode_tokens( 
        self, 
        token_sequence, 
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
        invalid_panel_mask = np.ones(pad_panel_num, dtype=bool)
        invalid_edge_mask = np.ones((pad_panel_num, pad_panels_to_len), dtype=bool)

        if len(panel_starts) < len(panel_ends):
            panel_ends = panel_ends[:len(panel_starts)]
        if len(panel_starts) > len(panel_ends):
            panel_starts = panel_starts[:len(panel_ends)]
        if np.any(panel_starts > panel_ends):
            return panels, invalid_panel_mask, invalid_edge_mask
        n_panels = len(panel_starts)
        
        for i in range(n_panels):
            panel_start = panel_starts[i]
            panel_end = panel_ends[i]
            panel = token_sequence[panel_start+1:panel_end]
            if len(panel) % 4 != 0:
                continue

            edges = panel.reshape(-1, 4)
            edges = edges.astype(float)
            edges = edges  / self.datawrapper.bin_size
            edges = edges * (max_range - min_range) + min_range
            edges = edges * edge_scale + edge_shift
            panels[i, :len(edges)] = edges
            invalid_edge_mask[i, :len(edges)] = False
            invalid_panel_mask[i] = False
            
        return panels, invalid_panel_mask, invalid_edge_mask

        gt = {
            'outlines': padded_panels, 'num_edges': num_edges,
            'num_panels': num_panels, 'empty_panels_mask': empty_panels_mask
        }

        return gt, incorect_panels


    def _retrieve_closest_garment(self, 
                                  gt_token_sequences, 
                                  pred_token_sequences, 
                                  bin_size=128, 
                                  categories=10,
                                  base_vocab_size=0,):
        offset = base_vocab_size + bin_size + categories

        # 1 = garment start, 2 = panel start, 3 = panel end, 4 = garment end, 5 = zero padding
        s_tokens = np.arange(5) + offset

        # find the garment start
        start = np.where(pred_token_sequences == s_tokens[0])[0]
        end = np.where(pred_token_sequences == s_tokens[3])[0]

        # check if panels starting and endings are correct or not
        if len(start) == 0 or len(end) == 0:
            return None
        if start[0] > end[0]:
            return None
        
        pred_token_sequences= pred_token_sequences[start[0]+1:end[0]]
        
        gt_token_sequences_modified = gt_token_sequences[:, 2:].detach().numpy()
        padd = len(pred_token_sequences)
        
        result = np.sum(gt_token_sequences_modified[:, :padd] == pred_token_sequences, axis=1)
        closest = np.argmax(result)
        
        return gt_token_sequences[closest].detach().numpy(), np.max(result) / padd
        

    def _fit_loop(self, model: nn.parallel.DistributedDataParallel, model_without_ddp: GPT, train_loader: DataLoader, valid_loader: DataLoader, start_epoch: int):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""

        global best_valid_loss

        if self.master_process:
            best_valid_loss = self.experiment.last_best_validation_loss()
            best_valid_loss = torch.tensor(best_valid_loss) if best_valid_loss is not None else None
            best_epoch = best_valid_loss is None
        model.to(self.device)
        for step in range(self.max_steps):
            
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
                    
            if self.master_process:
                t0 = time.time()   
            model.train()
            self.optimizer.zero_grad()
            loss_accum = 0.0
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
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


    def chamfer(self, 
                pred_panels: torch.FloatTensor, 
                gt_panels: torch.FloatTensor):
        """
        Args:
            pred_panels [batch x max_n_panels x points x 2]: predicted panel outlines, the valid pred panels are at the beginning of the tensor (number of valid panels is stored in pred_panel_nums)
            gt_panels [batch x max_n_panels x points x 2]: predicted panel outlines, the valid GT panels are at the beginning of the tensor (number of valid panels is stored in gt_panel_nums)
            pred_panel_nums [batch]: number of pred panels in each pattern
            gt_panels [batch]: number of GT panels in each pattern
            """
        
        
        device = pred_panels.device
        pred_panels = pred_panels.to(device)
        gt_panels = gt_panels.to(device)

        concat_pred_panels = pred_panels.reshape(pred_panels.shape[0], -1, 2)

        overall_chamfer = torch.zeros(gt_panels.shape[0])
        for i in range(gt_panels.shape[1]):
            dist = torch.cdist(concat_pred_panels, gt_panels[:,i, :,:])
            dist = dist.reshape(pred_panels.shape[0], pred_panels.shape[1], pred_panels.shape[2], -1)
            min_dist0, _ = torch.min(dist, dim=3)
            min_dist1, _ = torch.min(dist, dim=2)

            minimum = min_dist0.mean(dim=2) + min_dist1.mean(dim=2)
            increment, _ = torch.min(minimum, dim=1)
            overall_chamfer += increment

        return overall_chamfer 
    
    
    
    def pred_to_polygon(
            self, 
            out_patterns:torch.FloatTensor,                     # ground_truth['outlines'], 
            valid_panel_num: torch.IntTensor,                   # ground_truth['num_panels'].to(torch.int32), 
            valid_edge_num: torch.IntTensor,                    # ground_truth['num_edges'].to(torch.int32),
            valid_panel_mask: Optional[torch.BoolTensor]=None,  # ~ground_truth["empty_panels_mask"].to(torch.bool),
            num_points_sample:int=100):
            device = out_patterns.device
            def batch_discretize_curves(starts, ends, control_scale, num_lines=10):
                # starts, ends: [n_pts, 2]
                # out: [n_pts, num_lines, 2]
                n_pts = starts.shape[0]
                edge = ends - starts
                valid_curve_mask = ~torch.all(torch.isclose(control_scale, torch.zeros_like(control_scale), atol=1e-2), -1)
                n_valid = valid_curve_mask.sum()
                edge_perp = torch.cat([-edge[:, 1:], edge[:, :1]], dim=-1)
                control_start = starts + control_scale[:, :1] * edge
                control_point = control_start + control_scale[:, 1:] * edge_perp
                
                out_pts = torch.zeros(n_pts + valid_curve_mask.sum()*num_lines + 1, 2, device=device, dtype=starts.dtype)
                index = torch.arange(n_pts, device=device) + torch.cumsum(valid_curve_mask, dim=0) * num_lines
                index = index.unsqueeze(-1).repeat_interleave(2, -1)
         
                out_pts.scatter_(0, index, starts)
                index = torch.arange(valid_curve_mask.sum()*num_lines, device=device).reshape(-1, num_lines) + torch.cumsum(~valid_curve_mask, dim=0)[valid_curve_mask].unsqueeze(-1) + torch.arange(n_valid, device=device).unsqueeze(-1)
                index = index.unsqueeze(-1).repeat_interleave(2, -1).reshape(-1, 2)
                control_point = control_point[valid_curve_mask].unsqueeze(-2) #[n_pts, 1, 2]
                ts = torch.linspace(0, 1, num_lines, dtype=starts.dtype, device=starts.device).reshape( 1, num_lines, 1)
                center_points = (1 - ts) ** 2 * starts[valid_curve_mask].unsqueeze(-2) + 2 * ts * (1 - ts) * control_point + ts ** 2 * ends[valid_curve_mask].unsqueeze(-2)
                center_points = center_points.reshape(-1, 2)
                out_pts.scatter_(0, index, center_points)
                out_pts[-1] = ends[-1]
                
                return out_pts
            
            # (ground truth is already scaled back again)
            # scale = torch.tensor(self.gt_outline_stats["scale"]).to(device)
            # shift = torch.tensor(self.gt_outline_stats["shift"]).to(device)
            batch_size, max_n_panel, _, _ = out_patterns.shape
            all_pts_samples = torch.zeros((batch_size, max_n_panel, num_points_sample, 2), device=device)
            for pattern_idx in range(batch_size):
                num_panel = valid_panel_num[pattern_idx]
                _out_patterns = out_patterns[pattern_idx]
                _valid_edge_num = valid_edge_num[pattern_idx]
                if valid_panel_mask is not None:
                    assert valid_panel_mask[pattern_idx].sum() == num_panel
                    _out_patterns = _out_patterns[valid_panel_mask[pattern_idx]]
                    _valid_edge_num = _valid_edge_num[valid_panel_mask[pattern_idx]]
                for panel in range(num_panel):
                    num_edge = _valid_edge_num[panel]
                    de_panel = _out_patterns[panel, :num_edge, :] # * scale + shift (ground truth is already scaled back again)
                    aug_points = torch.zeros((1, 2), dtype=de_panel.dtype).to(de_panel.device)
                    aug_points = torch.cat((aug_points, de_panel[..., :2]), dim=0) # [b, n_panel, n_edge+1, 2]
                    aug_points = torch.cumsum(aug_points, dim=0)
                    num_points = aug_points.shape[0]

                    starts = aug_points[:num_points-1, :]
                    ends = aug_points[1:num_points, :]
                    control_scale = de_panel[:, 2:]
                    all_points = batch_discretize_curves(starts, ends, control_scale)
                    sampled_pts = self.sample_points(all_points, num_points_sample)
                    all_pts_samples[pattern_idx, panel] = sampled_pts

            return all_pts_samples # batch x panels x points x 2

        
    def sample_points(self, triangle_points: torch.FloatTensor, num_points=100):
        # triangle_points: [n_points, 2]
        device = triangle_points.device
        dtype=triangle_points.dtype
        # converts it into a shapely polygon
        polygon = triangle_points.cpu().numpy()
        polygon = tuple(polygon)
        polygon = Polygon(polygon)
        
        # sample points from shapely polygon, make sure it is at least as many as num_points
        points_sampled = sample(polygon, num_points)
        while len(points_sampled) < num_points:
            points_sampled = np.concatenate([points_sampled, sample(polygon, num_points)], axis=0)
        points_sampled = points_sampled[:num_points]

        return torch.from_numpy(points_sampled).to(device=device, dtype=dtype)