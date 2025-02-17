import logging

log = logging.getLogger(__name__)
import os
import time

from omegaconf import OmegaConf
import torch
import wandb as wb

from trainers.convert_zero_to_torch import _get_fp32_state_dict_from_zero_checkpoint
from trainers.utils import dict_to_cuda, AverageMeter, ProgressMeter


def _start_experiment(
    cfg,
    model,
    ddp_rank,
    resume,
):
    assert resume, f"Must resume forom checkpoint to evaluate!"
    # resume deepspeed checkpoint
    os.environ["WANDB_DIR"] = cfg.wandb_info.wandb_dir
    os.environ["WANDB_CACHE_DIR"] = cfg.wandb_info.wandb_cache_dir

    if ddp_rank == 0:
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wb.init(
            name=cfg.run_name,
            project=cfg.project,
            config=config,
            resume="allow",
            id=cfg.run_id,  # Resuming functionality
            dir=cfg.run_local_path,
        )
    if resume:
        latest_path = os.path.join(resume, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

        ds_checkpoint_dir = os.path.join(resume, tag)
        state_dict = _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir, True)
        model.load_state_dict(state_dict, strict=False)

        if ddp_rank == 0:
            log.info("Loaded checkpoint from {}".format(resume))
    if ddp_rank == 0:
        wb.watch(model, log="all")


def evaluate(
    cfg,
    model,
    loader,
    garment_tokenizer,
    ddp_rank,
    ddp_world_size,
):
    """Evaluate model on validation dataset"""

    _start_experiment(cfg, model, ddp_rank, cfg.resume)

    mode_names = loader.dataset.get_mode_names()
    model = model.to(f"cuda:{ddp_rank}")
    model.eval()
    cast_dtype = torch.float32
    model = model.to(cast_dtype)

    batch_time_meter = AverageMeter("Time", ":6.3f")
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

    if ddp_rank == 0:
        log.info("Starting evaluation...")

    batch_size = 1

    with torch.no_grad():

        for step, input_dict in enumerate(loader):
            start_time = time.time()

            input_dict = dict_to_cuda(input_dict)

            input_dict["pixel_values"] = input_dict["pixel_values"].to(cast_dtype)
            B = input_dict["pixel_values"].size(0)
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
                train_step=10000,
            )

            loss = output.loss
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

            if ddp_world_size > 1:
                for meter in all_meters:
                    meter.all_reduce()

            if ddp_rank == 0:
                progress.display(step + 1)

        if ddp_rank == 0:
            progress.display(step + 1)
            log_dict = {f"val/{meter.name}": meter.avg for meter in all_meters}
            wb.log(log_dict)

    if ddp_rank == 0:
        log.info("Finished evaluation")
