import logging

log = logging.getLogger(__name__)
import os
import time

import torch
import torch.distributed as dist
import wandb as wb

from trainers.utils import dict_to_cuda, start_experiment, AverageMeter, ProgressMeter


def train(
    cfg,
    model: torch.nn.Module,
    loader,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    garment_tokenizer,
    ddp_rank: int,
    ddp_world_size: int,
    log_dir: os.PathLike,
):
    """Fit provided model to reviosly configured dataset"""

    start_step = start_experiment(
        cfg, model, ddp_rank, ddp_world_size, cfg.resume, cfg.from_start
    )
    _fit_loop(
        model,
        loader,
        scheduler,
        garment_tokenizer,
        ddp_rank,
        ddp_world_size,
        log_dir,
        cfg.precision,
        cfg.grad_accumulation_steps,
        cfg.num_steps,
        start_step,
        cfg.save_freq,
    )
    if ddp_rank == 0:
        log.info("Finished training")


def _fit_loop(
    model,
    loader,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    garment_tokenizer,
    ddp_rank: int,
    ddp_world_size: int,
    log_dir: str,
    precision: str,
    grad_accumulation_steps: int,
    num_steps: int,
    start_step: int,
    save_freq: int,
) -> None:

    mode_names = loader.dataset.get_mode_names()
    cast_dtype = (
        torch.half
        if precision == "fp16"
        else (torch.bfloat16 if precision == "bf16" else torch.float32)
    )

    ##################################################
    #                  Meters Setup                  #
    ##################################################
    batch_time_meter = AverageMeter("Time", ":6.3f")
    data_time_meter = AverageMeter("Data", ":6.3f")
    total_loss_meter = AverageMeter("Total Loss", ":.4f")
    ce_loss_meter = AverageMeter("CE Loss", ":.4f")
    edge_loss_meter = AverageMeter("Edge Loss", ":.4f")

    modewise_total_loss_meters = [
        AverageMeter(f"{mode} Total Loss", ":.4f") for mode in mode_names
    ]
    modewise_ce_loss_meters = [
        AverageMeter(f"{mode} CE Loss", ":.4f") for mode in mode_names
    ]
    modewise_edge_loss_meters = [
        AverageMeter(f"{mode} Edge Loss", ":.4f") for mode in mode_names
    ]
    edge_type_loss_meters = {
        garment_tokenizer.panel_edge_type_indices.get_index_token(
            ind
        ).value: AverageMeter(
            f"{garment_tokenizer.panel_edge_type_indices.get_index_token(ind).value} Loss",
            ":.4f",
        )
        for ind in garment_tokenizer.panel_edge_type_indices.get_all_indices()
    }

    all_meters: list[AverageMeter] = (
        [
            batch_time_meter,
            data_time_meter,
            total_loss_meter,
            ce_loss_meter,
            edge_loss_meter,
        ]
        + modewise_total_loss_meters
        + modewise_ce_loss_meters
        + [meter for meter in edge_type_loss_meters.values()]
    )
    num_edge_type_losses = len(edge_type_loss_meters)

    progress = ProgressMeter(
        log,
        ddp_rank,
        1,
        all_meters[:-num_edge_type_losses],
        prefix="Step:",
    )

    ##################################################
    #                 Model Training                 #
    ##################################################
    model.train()
    if ddp_rank == 0:
        log.info("Start Training...")

    train_sampler: torch.utils.data.DistributedSampler = loader.data_sampler
    current_epoch = 0
    train_sampler.set_epoch(current_epoch)
    loader_iter = iter(loader)

    model.module.initialize_weights_for_panel_modules()
    for step in range(start_step, num_steps):
        is_last_step = step == num_steps - 1
        for i in range(grad_accumulation_steps):
            start_time = time.time()
            try:
                input_dict = next(loader_iter)
            except StopIteration:
                current_epoch += 1
                train_sampler.set_epoch(current_epoch)
                loader_iter = iter(loader)
                input_dict = next(loader_iter)
            data_time_meter.update(time.time() - start_time)

            input_dict = dict_to_cuda(input_dict)
            input_dict["pixel_values"] = input_dict["pixel_values"].to(cast_dtype)
            batch_size = input_dict["pixel_values"].size(0)
            for k in input_dict["pattern_params"].keys():
                input_dict["pattern_params"][k] = input_dict["pattern_params"][k].to(
                    cast_dtype
                )
            input_dict["pattern_endpoints"] = input_dict["pattern_endpoints"].to(
                cast_dtype
            )
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
                train_step=step,
                ddp_rank=ddp_rank,
            )
            dist.barrier()

            loss = output.loss
            model.backward(loss.mean())
            model.step()
            ce_loss = output.ce_loss
            edge_loss = output.edge_loss

            batch_time_meter.update(time.time() - start_time)
            total_loss_meter.update(loss.mean(), batch_size)
            ce_loss_meter.update(ce_loss.mean(), batch_size)
            if edge_loss is not None:
                edge_loss_meter.update(edge_loss.mean(), batch_size)

            for k, meter in edge_type_loss_meters.items():
                if f"{k}_loss" in output.edge_type_losses.keys():
                    meter.update(output.edge_type_losses[f"{k}_loss"], batch_size)

            for i, _ in enumerate(mode_names):
                mask = input_dict["sample_type"] == i
                if not mask.any():
                    continue

                mode_batch_size = mask.sum()
                modewise_total_loss_meters[i].update(loss[mask].mean(), mode_batch_size)
                modewise_ce_loss_meters[i].update(ce_loss[mask].mean(), mode_batch_size)

                if edge_loss is not None:
                    modewise_edge_loss_meters[i].update(
                        edge_loss[mask].mean(), mode_batch_size
                    )

        # logging
        if ddp_world_size > 1:
            for meter in all_meters:
                meter.all_reduce()

            if ddp_rank == 0:
                progress.display(step + 1)
                log_dict = {f"train/{meter.name}": meter.avg for meter in all_meters}
                wb.log(log_dict, step=step)

            for meter in all_meters:
                meter.reset()

        if step != 0:
            curr_lr = scheduler.get_last_lr()
            if ddp_rank == 0:
                wb.log({"train/lr": curr_lr[0]}, step)

        if (step % save_freq == 0 and step > 0) or is_last_step:
            model.save_checkpoint(os.path.join(log_dir, f"ckpt_{step}"))
