# Training loop func
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from .text_trainer import TextTrainer, TextTrainerConfig
from models.garment_token import GarmentTokenForCausalLM, GarmentTokenConfig
from experiment_hydra import ExperimentWrappper
from models.decoders.lora_gpt2 import GPT
from trainers.utils import dict_to_cuda, AverageMeter, ProgressMeter, master_log, dict_to_cpu, dict_to_dtype
from data.garment_tokenizers.special_tokens import PanelEdgeTypeV3
from convert_zero_to_torch import _get_fp32_state_dict_from_zero_checkpoint
import torch.distributed as dist
from .llava_trainer import FinetuneLlavaTrainer, FinetuneLlavaTrainerConfig
import time
import inspect
import sys
import loralib as lora
import random


    
    

class FinetuneLlavaSewFactoryTrainer(FinetuneLlavaTrainer):
    def __init__(
        self,
        experiment_tracker: ExperimentWrappper, 
        data_wrapper: GarmentImageDataWrapper, 
        ddp_rank: int, 
        ddp_local_rank: int,
        ddp_world_size: int,
        precision: str, 
        lr: float,
        beta1: float, 
        beta2: float, 
        grad_accumulation_steps: int, 
        batch_size: int,
        epoches: int,
        steps_per_epoch: int,
        eval_freq: int,
        log_freq: int,
        model_type: Literal["garment_token", "garment_token_regression"]
        ):
        super().__init__(
            experiment_tracker,
            data_wrapper,
            ddp_rank,
            ddp_local_rank,
            ddp_world_size,
            precision,
            lr,
            beta1,
            beta2,
            grad_accumulation_steps,
            batch_size,
            epoches,
            steps_per_epoch,
            eval_freq,
            log_freq,
            model_type,
        )
        


    def generation_step(self, step: int, subset: Literal["validation", "train"]="validation"):
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
        
        eval_sample_time = AverageMeter("Time", ":6.3f")
        self.model_engine.eval()
        if self.val_loader is None:
            log.info("No validation data provided, skipping validation")
            return None, None
        all_patterns = []
        all_gt_patterns = []
        all_questions = []
        all_image_paths = []
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
            input_dict["images_clip"] = input_dict["images_clip"].to(self.cast_dtype)
            
            if "questions_pattern_endpoints" in input_dict and input_dict["questions_pattern_endpoints"].shape[1] > 0:
                input_dict["questions_pattern_endpoints"] = input_dict["questions_pattern_endpoints"].to(self.cast_dtype)
            else:
                input_dict["questions_pattern_endpoints"] = None
            
            if "questions_pattern_transformations" in input_dict and input_dict["questions_pattern_transformations"].shape[1] > 0:
                input_dict["questions_pattern_transformations"] = input_dict["questions_pattern_transformations"].to(self.cast_dtype)
            else:
                input_dict["questions_pattern_transformations"] = None
            
            
            end = time.time()
            output_dict = self.model_engine.module.evaluate(
                input_dict["images_clip"],
                input_dict["question_ids"],
                input_dict["question_attention_masks"],
                endpoints=input_dict["questions_pattern_endpoints"],
                endpoints_mask=input_dict["questions_pattern_endpoints_mask"],
                transformations=input_dict["questions_pattern_transformations"],
                transformations_mask=input_dict["questions_pattern_transformations_mask"],
                max_new_tokens=2100
            )
            output_dict = dict_to_cpu(output_dict)
            output_dict = dict_to_dtype(output_dict, torch.float32)
            output_dict["input_mask"] = torch.arange(output_dict["output_ids"].shape[1]).reshape(1, -1) >= input_dict["question_ids"].shape[1]
            output_text, patterns, error_type = self.datawrapper.dataset.decode(output_dict, self.tokenizer)
            try:
                data_name = input_dict["gt_patterns"][0][-1].name
                os.makedirs(os.path.join(save_path, data_name), exist_ok=True)
                patterns.serialize(os.path.join(save_path, data_name), spec_only=False, to_subfolder=False, tag=f'_pred')
                for gt_pattern in input_dict["gt_patterns"][0]:
                    gt_pattern.serialize(os.path.join(save_path, data_name), spec_only=False, to_subfolder=False, tag=f'_gt')
                f = open(os.path.join(save_path, data_name, "output.txt"), "w")
                question = input_dict["questions_list"][0]
                input_string = " ".join(list(map(str, input_dict["input_ids"][0].cpu().numpy().tolist())))
                output_string = " ".join(list(map(str, output_dict["output_ids"][0].cpu().numpy().tolist())))
                question_string = " ".join(list(map(str, input_dict["question_ids"][0].cpu().numpy().tolist())))
                input_text = self.tokenizer.decode(input_dict["input_ids"][0][input_dict["input_ids"][0] != -200], skip_special_tokens=True)
                f.write(f"Input Id: {input_string}\n")
                f.write(f"Input text: {input_text}\n")
                f.write(f"Question: {question}\n")
                f.write(f"Output Text: {output_text}\n")
                f.write(f"Question Ids: {question_string}\n")
                f.write(f"Output Ids: {output_string}\n")
                f.close()
                cond_img = Image.open(input_dict["image_paths"][0])
                cond_img.save(os.path.join(save_path, data_name, 'input.png'))
            except Exception as e:
                log.error(e)
                pass
            if error_type != DecodeErrorTypes.NO_ERROR:
                all_errored_texts.append((error_type, output_text))
            eval_sample_time.update(time.time() - end)
            end = time.time()
            all_gt_patterns.append(input_dict["gt_patterns"][0])
            all_sample_types.append(input_dict["sample_type"][0].cpu().numpy())  
            all_patterns.append(patterns)
            all_questions.append(input_dict["questions_list"][0])
            all_image_paths.append(input_dict["image_paths"][0])
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
         sorted_inds) = self.datawrapper.dataset.evaluate_patterns(all_patterns, [p[-1] for p in all_gt_patterns])
        
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
        
        # save top error
        if len(sorted_inds) != 0:
            for k in range(min(10, len(sorted_inds))):
                _save_path = os.path.join(self.experiment.local_output_dir, f"epoch_{step}", subset, f"top_error_sample_{self.ddp_local_rank*10 + k}")
                os.makedirs(_save_path, exist_ok=True)
                choice = sorted_inds[k]
                top_num_panel_accuracy, top_num_edge_acc, top_num_edge_correct_acc, top_vertex_L2, top_transl_l2, top_rots_l2, top_stitch_acc, _ = self.datawrapper.dataset.evaluate_patterns([all_patterns[choice]], [all_gt_patterns[choice][-1]])
                try:
                    final_dir = all_patterns[choice].serialize(_save_path, to_subfolder=False, tag=f'_pred')
                except:
                    final_dir = None
                gt_final_dir = all_gt_patterns[choice][-1].serialize(_save_path, to_subfolder=False, tag=f'_gt')
                if all_sample_types[choice] in [0, 3]:
                    cond_img = Image.open(all_image_paths[choice])
                    cond_img.save(Path(gt_final_dir) / 'input.png')
                question = all_questions[choice]
                output_text = all_output_texts[choice]
                f = open(os.path.join(_save_path, "output.txt"), "w")
                f.write(f"Question: {question}\n")
                f.write(f"Output: {output_text}\n")
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
                top_num_panel_accuracy, top_num_edge_acc, top_num_edge_correct_acc, top_vertex_L2, top_transl_l2, top_rots_l2, top_stitch_acc, _ = self.datawrapper.dataset.evaluate_patterns([all_patterns[choice]], [all_gt_patterns[choice][-1]])
                try:
                    final_dir = all_patterns[choice].serialize(_save_path, to_subfolder=False, tag=f'_pred')
                except:
                    final_dir = None
                gt_final_dir = all_gt_patterns[choice][-1].serialize(_save_path, to_subfolder=False, tag=f'_gt')
                cond_img = Image.open(all_image_paths[choice])
                cond_img.save(Path(gt_final_dir) / 'input.png')
                question = all_questions[choice]
                output_text = all_output_texts[choice]
                f = open(os.path.join(_save_path, "output.txt"), "w")
                f.write(f"Question: {question}\n")
                f.write(f"Output: {output_text}\n")
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
            wb.log(log_dict, step=step)
            
    @torch.no_grad()   
    def eval_step(self, step: int):
        # validation loss
        eval_forward_time = AverageMeter("Time", ":6.3f")
        eval_data_time = AverageMeter("Data", ":6.3f")
        eval_losses = AverageMeter("Loss", ":.4f")
        ce_losses = AverageMeter("CE Loss", ":.4f")
        edge_losses = AverageMeter("Edge Loss", ":.4f")
        edge_type_losses = {self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value: AverageMeter(f"{self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value} Loss", ":.4f") for ind in self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_all_indices()}
        self.model_engine.eval()
        if self.val_loader is None:
            log.info("No validation data provided, skipping validation")
            return None, None
        
        end = time.time()
        for kk, input_dict in enumerate(self.val_loader):
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)
            input_dict["images_clip"] = input_dict["images_clip"].to(self.cast_dtype)
            if "param_targets" in input_dict:
                for k in input_dict["param_targets"].keys():
                    input_dict["param_targets"][k] = input_dict["param_targets"][k].to(self.cast_dtype)
                    
            if "param_target_endpoints" in input_dict:
                input_dict["param_target_endpoints"] = input_dict["param_target_endpoints"].to(self.cast_dtype)
            
            if "param_target_transformations" in input_dict:
                input_dict["param_target_transformations"] = input_dict["param_target_transformations"].to(self.cast_dtype)
            eval_data_time.update(time.time() - end)
            end = time.time()
            
            output_dict = self.model_engine(**input_dict)
            loss = output_dict["total_loss"]
            ce_loss = output_dict["ce_loss"]
            edge_loss = output_dict.get("edge_loss", 0)
            edge_losses.update(edge_loss.mean(), input_dict["images_clip"].size(0))
            for k, meter in edge_type_losses.items():
                if f"{k}_loss" in output_dict:
                    meter.update(output_dict[f"{k}_loss"], input_dict["images_clip"].size(0))
                    
            eval_losses.update(loss.mean(), input_dict["images_clip"].size(0))
            ce_losses.update(ce_loss.mean(), input_dict["images_clip"].size(0))
            
            eval_forward_time.update(time.time() - end)
            end = time.time()
            log.info("Rank: {}, Progress: [{}/{}]".format(self.ddp_rank, kk, len(self.val_loader)))
        
        eval_data_time.all_reduce()
        eval_forward_time.all_reduce()
        eval_losses.all_reduce()
        ce_losses.all_reduce()
        if self.model_type == "garment_token_regression":
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
            if self.model_type == "garment_token_regression":
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
            if resume.endswith(".bin"):
                self.start_epoch = self.epoches
                self.model_engine.module.load_state_dict(torch.load(resume, map_location=self.device))
            else:
                latest_path = os.path.join(resume, 'latest')
                if os.path.isfile(latest_path):
                    with open(latest_path, 'r') as fd:
                        tag = fd.read().strip()
                else:
                    raise ValueError(f"Unable to find 'latest' file at {latest_path}")

                ds_checkpoint_dir = os.path.join(resume, tag)
                if os.path.isdir(ds_checkpoint_dir) and len(os.listdir(ds_checkpoint_dir)) != self.ddp_world_size + 1:
                    state_dict = _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir, True)
                    self.model_engine.module.load_state_dict(state_dict, strict=False)
                else:
                    load_path, client_state = self.model_engine.load_checkpoint(resume)
                    
                with open(os.path.join(resume, "latest"), "r") as f:
                    ckpt_dir = f.readlines()[0].strip()
                self.start_epoch = (
                    int(ckpt_dir.replace("global_step", "")) // self.steps_per_epoch
                )
            master_log(
                log,
                "resume training from {}, start from epoch {}".format(
                    resume, self.start_epoch
                )
            )
        if self.master_process:
            wb.watch(self.model_engine, log='all')
    def fit(self):
        """Fit provided model to reviosly configured dataset"""
        
        if not self.datawrapper:
            raise RuntimeError('{}::Error::fit before dataset was provided. run use_dataset() first'.format(self.__class__.__name__))
        
        
        
        if self.master_process:
            encoded_pattern = self.datawrapper.training[0][0]
            print(encoded_pattern)
        
        self._fit_loop()
        master_log(self.ddp_local_rank, log, "Finished training")

    def _save_checkpoint(self, epoch, valid_loss):
        save_dir = os.path.join(self.log_dir, f"ckpt_{epoch}")
        if self.master_process:
            torch.save(
                {"epoch": epoch},
                os.path.join(self.log_dir,"loss_{:.4f}.pth".format(valid_loss)),
            )
        dist.barrier()
        self.model_engine.save_checkpoint(save_dir)
    
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
            edge_losses = AverageMeter("Edge Loss", ":.4f")
            edge_type_losses = {self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value: AverageMeter(f"{self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value} Loss", ":.4f") for ind in self.datawrapper.dataset.garment_tokenizer.panel_edge_type_indices.get_all_indices()}
            all_meters += [edge_losses]
            progress = ProgressMeter(
                log, 
                self.ddp_local_rank, 
                self.steps_per_epoch,
                all_meters,
                prefix="Epoch: [{}]".format(epoch),
            )
            self.model_engine.train()
            for step in range(self.steps_per_epoch):
                # training step
                for i in range(self.grad_accumulation_steps):
                    end = time.time()
                    try:
                        input_dict = next(train_loader_iter)
                    except:
                        train_loader_iter = iter(self.train_loader)
                        input_dict = next(train_loader_iter)
                    
                    train_data_time.update(time.time() - end)
                    input_dict = dict_to_cuda(input_dict)
                    input_dict["images_clip"] = input_dict["images_clip"].to(self.cast_dtype)
                    if "param_targets" in input_dict:
                        for k in input_dict["param_targets"].keys():
                            input_dict["param_targets"][k] = input_dict["param_targets"][k].to(self.cast_dtype)
                            
                    if "param_target_endpoints" in input_dict:
                        input_dict["param_target_endpoints"] = input_dict["param_target_endpoints"].to(self.cast_dtype)
                    
                    if "param_target_transformations" in input_dict:
                        input_dict["param_target_transformations"] = input_dict["param_target_transformations"].to(self.cast_dtype)
                    output_dict = self.model_engine(**input_dict)
                    loss = output_dict["total_loss"]
                    self.model_engine.backward(loss.mean())
                    self.model_engine.step()
                    # measure elapsed time
                    train_batch_time.update(time.time() - end)
                    ce_loss = output_dict["ce_loss"]
                    train_losses.update(loss.mean().item(), input_dict["images_clip"].size(0))
                    ce_losses.update(ce_loss.mean().item(), input_dict["images_clip"].size(0))
                    edge_loss = output_dict.get("edge_loss", 0)
                    edge_losses.update(edge_loss.mean().item(), input_dict["images_clip"].size(0))
                    for k, meter in edge_type_losses.items():
                        if f"{k}_loss" in output_dict:
                            meter.update(output_dict[f"{k}_loss"].item(), input_dict["images_clip"].size(0))
                if step % self.log_freq == 0 or True:
                    if self.ddp_world_size > 1:
                        train_batch_time.all_reduce()
                        train_data_time.all_reduce()
                        train_losses.all_reduce()
                        ce_losses.all_reduce()
                        if self.model_type == "garment_token_regression":
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
                        if self.model_type == "garment_token_regression":
                            log_dict["train/edge_loss"] = edge_losses.avg
                            for k, meter in edge_type_losses.items():
                                log_dict[f"train/{k}_loss"] = meter.avg
                        wb.log(log_dict, step=step + epoch * self.steps_per_epoch)

                    train_batch_time.reset()
                    train_data_time.reset()
                    train_losses.reset()
                    ce_losses.reset()
                    if self.model_type == "garment_token_regression":
                        edge_losses.reset()
                        for k, meter in edge_type_losses.items():
                            meter.reset()
            
                if step != 0:
                    curr_lr = self.scheduler.get_last_lr()
                    if self.master_process:
                        wb.log({"train/lr": curr_lr[0]}, step + epoch * self.steps_per_epoch)
                
                
            if (epoch % self.eval_freq == 0 ) or last_epoch:
                self._save_checkpoint(epoch, 0)
                