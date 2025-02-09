import os 
import wandb as wb
from omegaconf import OmegaConf
import torch
from trainers.convert_zero_to_torch import _get_fp32_state_dict_from_zero_checkpoint
from trainers.utils import dict_to_cuda, AverageMeter, ProgressMeter
import time
import logging
log = logging.getLogger(__name__)

def _start_experiment(
    cfg,
    model,
    ddp_rank,
    resume,
):
    assert resume, f"Must resume forom checkpoint to evaluate!"
    # resume deepspeed checkpoint
    os.environ['WANDB_DIR'] = cfg.wandb_info.wandb_dir
    os.environ['WANDB_CACHE_DIR'] = cfg.wandb_info.wandb_cache_dir
    if ddp_rank == 0:
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wb.init(
            name=cfg.run_name, project=cfg.project, config=config, 
            resume='allow', id=cfg.run_id,    # Resuming functionality
            dir=cfg.run_local_path
        )
    if resume:
        latest_path = os.path.join(resume, 'latest')
        if os.path.isfile(latest_path):
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

        ds_checkpoint_dir = os.path.join(resume, tag)
        state_dict = _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir, True)
        model.load_state_dict(state_dict, strict=False)
            
        if ddp_rank == 0:
            log.info(
                "Loaded checkpoint from {}".format(resume)
            )
    if ddp_rank == 0:
        wb.watch(model, log='all')

def evaluate(
    cfg,
    model, 
    loader, 
    garment_tokenizer,
    ddp_rank,
    ddp_world_size,
):
    """Evaluate model on validation dataset"""
    
    _start_experiment(
        cfg, 
        model, 
        ddp_rank, 
        ddp_world_size,
        cfg.resume)
    
    mode_names = loader.dataset.get_mode_names()
    cast_dtype = torch.half if cfg.precision == "fp16" else (torch.bfloat16 if cfg.precision == "bf16" else torch.float32)
    model = model.to(f"cuda:{ddp_rank}")
    model.eval()
    model = model.to(cast_dtype)
    
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Total Loss", ":.4f")
    modewise_total_losses = [AverageMeter(f"{mode} Total Loss", ":.4f") for mode in mode_names]
    ce_losses = AverageMeter("CE Loss", ":.4f")
    modewise_ce_losses = [AverageMeter(f"{mode} CE Loss", ":.4f") for mode in mode_names]
    all_meters = [batch_time, data_time, losses, ce_losses] + modewise_total_losses + modewise_ce_losses
    edge_losses = AverageMeter("Edge Loss", ":.4f")
    modewise_edge_losses = [AverageMeter(f"{mode} Edge Loss", ":.4f") for mode in mode_names]
    all_meters += modewise_edge_losses
    edge_type_losses = {garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value: AverageMeter(f"{garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value} Loss", ":.4f") for ind in garment_tokenizer.panel_edge_type_indices.get_all_indices()}
    all_meters += [edge_losses]
    progress = ProgressMeter(
        log, 
        ddp_rank, 
        1,
        all_meters,
        prefix="Eval:",
    )

    if ddp_rank == 0:
        log.info("Starting evaluation...")
        
    with torch.no_grad():
        for i, input_dict in enumerate(loader):
            end = time.time()
            data_time.update(time.time() - end)
            
            input_dict = dict_to_cuda(input_dict)
            
            input_dict["pixel_values"] = input_dict["pixel_values"].to(cast_dtype)
            B = input_dict["pixel_values"].size(0)
            for k in input_dict["pattern_params"].keys():
                input_dict["pattern_params"][k] = input_dict["pattern_params"][k].to(cast_dtype)
            input_dict["pattern_endpoints"] = input_dict["pattern_endpoints"].to(cast_dtype)
            input_dict["pattern_transfs"] = input_dict["pattern_transfs"].to(cast_dtype)
            
            output = model(
                input_ids=input_dict["input_ids"],
                pixel_values=input_dict["pixel_values"],
                attention_mask=input_dict["attention_mask"],
                aspect_ratio_ids=input_dict["aspect_ratio_ids"],
                aspect_ratio_mask=input_dict["aspect_ratio_mask"],
                cross_attention_mask=input_dict["cross_attention_mask"],
                labels=input_dict["labels"],
                pattern_params=input_dict["pattern_params"],
                pattern_params_mask=input_dict["pattern_params_mask"],
                pattern_endpoints=input_dict["pattern_endpoints"],
                pattern_endpoint_masks=input_dict["pattern_endpoint_masks"],
                pattern_transfs=input_dict["pattern_transfs"],
                pattern_transf_masks=input_dict["pattern_transf_masks"],
            )
            
            loss = output.loss
            ce_loss = output.ce_loss
            edge_loss = output.edge_loss

            # measure elapsed time
            batch_time.update(time.time() - end)
            
            losses.update(loss.mean(), B)
            ce_losses.update(ce_loss.mean(), B)
            edge_losses.update(edge_loss.mean(), B)
            
            for k, meter in edge_type_losses.items():
                if f"{k}_loss" in output.edge_type_losses.keys():
                    meter.update(output.edge_type_losses[f"{k}_loss"], B)
                    
            for i, mode in enumerate(mode_names):
                mask = input_dict["sample_type"] == i
                if not mask.any():
                    continue
                mode_loss = loss[mask]
                modewise_total_losses[i].update(mode_loss.mean(), mode_loss.size(0))
                mode_ce_loss = ce_loss[mask]
                modewise_ce_losses[i].update(mode_ce_loss.mean(), mode_ce_loss.size(0))
                mode_edge_loss = edge_loss[mask]
                modewise_edge_losses[i].update(mode_edge_loss.mean(), mode_edge_loss.size(0))

            if ddp_world_size > 1:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                for i, mode in enumerate(mode_names):
                    modewise_total_losses[i].all_reduce()
                    modewise_ce_losses[i].all_reduce()
                    modewise_edge_losses[i].all_reduce()
                edge_losses.all_reduce()
                for k, meter in edge_type_losses.items():
                    meter.all_reduce()

            if ddp_rank == 0:
                progress.display(i + 1)
                log_dict = {
                    "val/loss": losses.avg,
                    "val/ce_loss": ce_losses.avg,
                    "val/batch_time": batch_time.avg,
                    "val/data_time": data_time.avg,
                }
                log_dict["val/edge_loss"] = edge_losses.avg
                for k, meter in edge_type_losses.items():
                    log_dict[f"val/{k}_loss"] = meter.avg
                for k, mode in enumerate(mode_names):
                    log_dict[f"val/{mode}_loss"] = modewise_total_losses[i].avg
                    log_dict[f"val/{mode}_ce_loss"] = modewise_ce_losses[i].avg
                    log_dict[f"val/{mode}_edge_loss"] = modewise_edge_losses[i].avg
                wb.log(log_dict)

    if ddp_rank == 0:
        log.info("Finished evaluation")
