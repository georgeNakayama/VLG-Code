import os 
import wandb as wb
import OmegaConf
import _get_fp32_state_dict_from_zero_checkpoint
from trainers.utils import dict_to_cuda, AverageMeter, ProgressMeter, master_log, dict_to_cpu, dict_to_dtype
import logging
log = logging.getLogger(__name__)

def _start_experiment(
    cfg,
    model,
    ddp_rank,
    ddp_world_size
)
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
    start_step = 0
    if resume:
        latest_path = os.path.join(resume, 'latest')
        if os.path.isfile(latest_path):
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

        ds_checkpoint_dir = os.path.join(resume, tag)
        if from_start or (os.path.isdir(ds_checkpoint_dir) and len(os.listdir(ds_checkpoint_dir)) != self.ddp_world_size + 1):
            state_dict = _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir, True)
            model.module.load_state_dict(state_dict, strict=False)
        else:
            load_path, client_state = model.load_checkpoint(resume)
            
        with open(os.path.join(resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        if not from_start:
            start_step = (
                int(ckpt_dir.replace("global_step", ""))
            )
        if ddp_rank == 0:
            log.info(
                "resume training from {}, start from epoch {}".format(
                    resume, start_step
                )
            )
    if ddp_rank == 0:
        wb.watch(model, log='all')
    return start_step

def train(
    cfg,
    model, 
    optimizer, 
    loader, 
    scheduler, 
    garment_tokenizer,
    ddp_rank,
    ddp_world_size):
    """Fit provided model to reviosly configured dataset"""
    
    start_step = _start_experiment(cfg, model, ddp_rank, ddp_world_size)
    _fit_loop(
        model, 
        optimizer, 
        loader, 
        scheduler, 
        garment_tokenizer, 
        cfg.precision, 
        cfg.grad_accumulation_steps, 
        cfg.num_steps, 
        start_step,
        cfg.save_freq)
    if ddp_rank == 0: 
        log.info("Finished training")
        
        
def _fit_loop(
    model, 
    optimizer, 
    loader, 
    scheduler, 
    garment_tokenizer,
    precision,
    grad_accumulation_steps,
    num_steps, 
    start_step,
    save_freq):
    loader_iter = iter(loader)
    mode_names = loader.dataset.get_mode_names()
    cast_dtype = torch.half if precision == "fp16" else (torch.bfloat16 if precision == "bf16" else torch.float32)
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
        ddp_local_rank, 
        1,
        all_meters,
        prefix="Step:",
    )
    model.train()
    for step in range(start_step, num_steps):
        last_step = (step == num_steps - 1)
        
        for i in range(grad_accumulation_steps):
            end = time.time()
            try:
                input_dict = next(loader_iter)
            except:
                loader_iter = iter(loader)
                input_dict = next(loader_iter)
            
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
                labels=input_dict["labels"],
                pixel_values=input_dict["pixel_values"],
                attention_mask=input_dict["attention_mask"],
                image_sizes=input_dict["image_sizes"],
                pattern_params=input_dict["pattern_params"],
                pattern_params_mask=input_dict["pattern_params_mask"],
                pattern_endpoints=input_dict["pattern_endpoints"],
                pattern_endpoint_masks=input_dict["pattern_endpoint_masks"],
                pattern_transfs=input_dict["pattern_transfs"],
                pattern_transf_masks=input_dict["pattern_transf_masks"],
            )
            loss = output.loss
            model.backward(loss.mean())
            model.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            ce_loss = output.ce_loss
            losses.update(loss.mean(), B)
            ce_losses.update(ce_loss.mean(), B)
            edge_loss = output_dict.edge_loss
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
        
        # logging
        if ddp_world_size > 1:
            train_batch_time.all_reduce()
            train_data_time.all_reduce()
            train_losses.all_reduce()
            ce_losses.all_reduce()
            for i, mode in enumerate(mode_names):
                modewise_total_losses[i].all_reduce()
                modewise_ce_losses[i].all_reduce()
                modewise_edge_losses[i].all_reduce()
                edge_losses.all_reduce()
                for k, meter in edge_type_losses.items():
                    meter.all_reduce()

            if ddp_rank == 0:
                progress.display(step + 1)
                log_dict = {
                    "train/loss": train_losses.avg,
                    "train/ce_loss": ce_losses.avg,
                    "train/batch_time": train_batch_time.avg,
                    "train/data_time": train_data_time.avg,
                }
                log_dict["train/edge_loss"] = edge_losses.avg
                for k, meter in edge_type_losses.items():
                    log_dict[f"train/{k}_loss"] = meter.avg
                for k, mode in enumerate(mode_names):
                    log_dict[f"train/{mode}_loss"] = modewise_total_losses[i].avg
                    log_dict[f"train/{mode}_ce_loss"] = modewise_ce_losses[i].avg
                    log_dict[f"train/{mode}_edge_loss"] = modewise_edge_losses[i].avg
                wb.log(log_dict, step=step)

                train_batch_time.reset()
                train_data_time.reset()
                train_losses.reset()
                ce_losses.reset()
                for k, mode in enumerate(mode_names):
                    modewise_total_losses[k].reset()
                    modewise_ce_losses[k].reset()
                    modewise_edge_losses[k].reset()
                edge_losses.reset()
                for k, meter in edge_type_losses.items():
                    meter.reset()
        
            if step != 0:
                curr_lr = scheduler.get_last_lr()
                if ddp_rank == 0:
                    wb.log({"train/lr": curr_lr[0]}, step)
            
            
        if (step % save_freq == 0) or last_step:
            model.save_checkpoint(os.path.join(log_dir, f"ckpt_{epoch}"))