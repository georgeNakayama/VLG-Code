import logging

log = logging.getLogger(__name__)
import os
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
import torch.distributed as dist
import wandb as wb

from trainers.convert_zero_to_torch import _get_fp32_state_dict_from_zero_checkpoint
from trainers.utils import dict_to_cpu, dict_to_dtype, AverageMeter, dict_to_cuda



def _start_experiment(
    cfg,
    model,
    ddp_rank,
    ddp_world_size,
    resume,
):
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
    start_step = 0
    if resume:
        if os.path.isfile(resume):
            # assume single torch checkpoint
            state_dict = torch.load(resume)
            model.load_state_dict(state_dict, strict=False)
            return 0

        # assume deepspeed checkpoint
        latest_path = os.path.join(resume, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

        ds_checkpoint_dir = os.path.join(resume, tag)
        print(f"The checkpointdir that we will load is {ds_checkpoint_dir}")
        state_dict = _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir, True)
        model.load_state_dict(state_dict, strict=False)

        with open(os.path.join(resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        start_step = int(ckpt_dir.replace("global_step", ""))
        if ddp_rank == 0:
            log.info(
                "resume training from {}, start from epoch {}".format(
                    resume, start_step
                )
            )
    if ddp_rank == 0:
        wb.watch(model, log="all")
    return start_step


@torch.no_grad()
def generate(
    cfg,
    model,
    loader,
    garment_tokenizer,
    processor,
    ddp_rank,
    ddp_world_size,
    log_dir,
    gen_num=-1,
):
    if gen_num == -1:
        gen_num = len(loader)
    assert cfg.resume is not None, "Need a checkpoint path"

    if ddp_rank == 0:
        log.info(f"Logging samples to {log_dir}")

    step = _start_experiment(cfg, model, ddp_rank, ddp_world_size, cfg.resume)
    dist.barrier()

    mode_names = loader.dataset.get_mode_names()
    model = model.to(f"cuda:{ddp_rank}")
    model.eval()
    cast_dtype = torch.float32
    model = model.to(cast_dtype)
    dist.barrier()

    eval_sample_time = AverageMeter("Time", ":6.3f")

    all_patterns = []
    all_gt_patterns = []
    all_questions = []
    all_image_paths = []
    all_output_texts = []
    all_sample_types = []
    for i, input_dict in enumerate(loader):
        if i == gen_num:
            break

        input_dict = dict_to_cuda(input_dict)
        input_dict["pixel_values"] = input_dict["pixel_values"].to(cast_dtype)
        if (
            "pattern_endpoints" in input_dict
            and input_dict["pattern_endpoints"].shape[1] > 0
        ):
            input_dict["pattern_endpoints"] = input_dict["pattern_endpoints"].to(
                cast_dtype
            )
        else:
            input_dict["pattern_endpoints"] = None

        if (
            "pattern_transfs" in input_dict
            and input_dict["pattern_transfs"].shape[1] > 0
        ):
            input_dict["pattern_transfs"] = input_dict["pattern_transfs"].to(cast_dtype)
        else:
            input_dict["pattern_transfs"] = None

        param_dict = {}

        output_tensor = model.generate(
            input_ids=input_dict["input_ids"],
            pixel_values=input_dict["pixel_values"],
            attention_mask=input_dict["attention_mask"],
            aspect_ratio_ids=input_dict["aspect_ratio_ids"],
            aspect_ratio_mask=input_dict["aspect_ratio_mask"],
            cross_attention_mask=input_dict["cross_attention_mask"],
            pattern_endpoints=input_dict["pattern_endpoints"],
            pattern_endpoint_masks=input_dict["pattern_endpoint_masks"],
            pattern_transfs=input_dict["pattern_transfs"],
            pattern_transf_masks=input_dict["pattern_transf_masks"],
            param_dict=param_dict,
            max_new_tokens=2100,
            ddp_rank=ddp_rank,
        )
        param_dict = dict_to_dtype(param_dict)  # to torch.float32
        param_dict = dict_to_cpu(param_dict)

        output_tensor = output_tensor.float()
        output_text, patterns, _ = garment_tokenizer.decode_tensor(
            output_tensor, param_dict, processor
        )

        try:
            # --- Prepare logging directory and serialize patterns ---
            # Determine the name from the last ground truth pattern.
            data_name = input_dict["gt_patterns"][0][-1].name
            data_log_dir = os.path.join(log_dir, data_name)
            os.makedirs(data_log_dir, exist_ok=True)

            # Serialize predicted patterns.
            patterns.serialize(
                data_log_dir,
                spec_only=False,
                with_3d=False,
                with_text=False,
                view_ids=False,
                to_subfolder=False,
                tag="_pred",
            )

            # Serialize each ground truth pattern.
            for gt_pattern in input_dict["gt_patterns"][0]:
                gt_pattern.serialize(
                    data_log_dir,
                    spec_only=False,
                    with_3d=False,
                    with_text=False,
                    view_ids=False,
                    to_subfolder=False,
                    tag="_gt",
                )

            # --- Write output text to file ---
            output_file_path = os.path.join(data_log_dir, "output.txt")
            question = input_dict["questions_list"][0]
            with open(output_file_path, "w") as output_file:
                output_file.write(f"Question: {question}\n")
                output_file.write(f"Output Text: {output_text}\n")

            # --- Save input image if it exists ---
            image_path = input_dict["image_paths"][0]
            if os.path.isfile(image_path):
                cond_img = Image.open(image_path)
                cond_img.save(os.path.join(data_log_dir, "input.png"))

        except Exception as e:
            log.error(e)


        # --- Update timing and accumulate sample information ---
        end = time.time()
        eval_sample_time.update(time.time() - end)

        all_gt_patterns.append(input_dict["gt_patterns"][0])
        all_sample_types.append(input_dict["sample_type"][0].cpu().numpy())
        all_patterns.append(patterns)
        all_questions.append(input_dict["questions_list"][0])
        all_image_paths.append(input_dict["image_paths"][0])
        all_output_texts.append(output_text)

        log.info("Rank: {}, Progress: [{}/{}]".format(ddp_rank, i, gen_num))


    # --- Evaluate patterns ---
    (
        total_num_panel_correct,
        num_edge_accs,
        num_edge_correct_accs,
        vertex_L2s,
        transl_l2s,
        rots_l2s,
        stitch_accs,
        sorted_inds,
    ) = garment_tokenizer.evaluate_patterns(
        all_patterns, [p[-1] for p in all_gt_patterns]
    )

    # Compute overall metrics.
    total_panel_accs = total_num_panel_correct.mean()
    total_edge_acc = num_edge_accs[num_edge_accs != -1].sum() / max((num_edge_accs != -1).sum(), 1)
    total_edge_correct_acc = num_edge_correct_accs[num_edge_correct_accs != -1].sum() / max(
        (num_edge_correct_accs != -1).sum(), 1
    )
    total_vertex_L2 = vertex_L2s[vertex_L2s != -1].sum() / max((vertex_L2s != -1).sum(), 1)
    total_transl_l2 = transl_l2s[transl_l2s != -1].sum() / max((transl_l2s != -1).sum(), 1)
    total_rots_l2 = rots_l2s[rots_l2s != -1].sum() / max((rots_l2s != -1).sum(), 1)
    total_stitch_acc = stitch_accs[stitch_accs != -1].sum() / max((stitch_accs != -1).sum(), 1)


    # --- Compute mode-wise metrics ---
    modewise_panel_accs = {}
    modewise_edge_accs = {}
    modewise_edge_correct_accs = {}
    modewise_vertex_L2s = {}
    modewise_transl_l2s = {}
    modewise_rots_l2s = {}
    modewise_stitch_accs = {}

    for mode_idx, mode in enumerate(mode_names):
        # Create a mask for the current mode.
        mode_mask = np.array(all_sample_types) == mode_idx
        mode_mask = torch.from_numpy(mode_mask).to(num_edge_accs).bool()
        if not mode_mask.any():
            continue

        modewise_panel_accs[mode] = total_num_panel_correct[mode_mask].mean()
        modewise_edge_accs[mode] = num_edge_accs[
            torch.logical_and(mode_mask, num_edge_accs != -1)
        ].sum() / max((num_edge_accs[mode_mask] != -1).sum(), 1)
        modewise_edge_correct_accs[mode] = num_edge_correct_accs[
            torch.logical_and(mode_mask, num_edge_correct_accs != -1)
        ].sum() / max((num_edge_correct_accs[mode_mask] != -1).sum(), 1)
        modewise_vertex_L2s[mode] = vertex_L2s[
            torch.logical_and(mode_mask, vertex_L2s != -1)
        ].sum() / max((vertex_L2s[mode_mask] != -1).sum(), 1)
        modewise_transl_l2s[mode] = transl_l2s[
            torch.logical_and(mode_mask, transl_l2s != -1)
        ].sum() / max((transl_l2s[mode_mask] != -1).sum(), 1)
        modewise_rots_l2s[mode] = rots_l2s[
            torch.logical_and(mode_mask, rots_l2s != -1)
        ].sum() / max((rots_l2s[mode_mask] != -1).sum(), 1)
        modewise_stitch_accs[mode] = stitch_accs[
            torch.logical_and(mode_mask, stitch_accs != -1)
        ].sum() / max((stitch_accs[mode_mask] != -1).sum(), 1)


    # --- Reduce metrics across processes ---
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


    # --- Log final metrics if master process ---
    if ddp_rank == 0:
        log.info(
            "Step: {:03d}\t"
            "Sample: {:.3f} ({:.3f})\t"
            "Num Panel Accuracy: {:.8f}\t"
            "Num Edge Accuracy: {:.8f}\t"
            "Num Correct Edge Accuracy: {:.8f}\t"
            "Vertex L2: {:.8f}\t"
            "Translation L2: {:.8f}\t"
            "Rotation L2: {:.8f}\t"
            "Stitch Acc: {:.8f}\t".format(
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
            "val/num_panel_accuracy": total_panel_accs.cpu().item(),
            "val/num_edge_accuracy": total_edge_acc.cpu().item(),
            "val/num_correct_edge_accuracy": total_edge_correct_acc.cpu().item(),
            "val/vertex_L2": total_vertex_L2.cpu().item(),
            "val/translation_L2": total_transl_l2.cpu().item(),
            "val/rotation_L2": total_rots_l2.cpu().item(),
            "val/stitch_acc": total_stitch_acc.cpu().item(),
            "val/sample_time": eval_sample_time.avg,
        }

        for mode in modewise_panel_accs.keys():
            log_dict.update({
                f"val/{mode}_num_panel_accuracy": modewise_panel_accs[mode].cpu().item(),
                f"val/{mode}_num_edge_accuracy": modewise_edge_accs[mode].cpu().item(),
                f"val/{mode}_num_correct_edge_accuracy": modewise_edge_correct_accs[mode].cpu().item(),
                f"val/{mode}_vertex_L2": modewise_vertex_L2s[mode].cpu().item(),
                f"val/{mode}_translation_L2": modewise_transl_l2s[mode].cpu().item(),
                f"val/{mode}_rotation_L2": modewise_rots_l2s[mode].cpu().item(),
                f"val/{mode}_stitch_acc": modewise_stitch_accs[mode].cpu().item(),
            })

        wb.log(log_dict, step=step)
