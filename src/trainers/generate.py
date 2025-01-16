import os 
import wandb as wb
from omegaconf import OmegaConf
import torch
from trainers.convert_zero_to_torch import _get_fp32_state_dict_from_zero_checkpoint
from trainers.utils import dict_to_cpu, dict_to_dtype, AverageMeter
import time
import logging
import torch.distributed as dist
from PIL import Image
import numpy as np 
log = logging.getLogger(__name__)

def _start_experiment(
    cfg,
    model,
    ddp_rank,
    ddp_world_size,
    resume,
):
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
            
        with open(os.path.join(resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
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
    gen_num=-1):
    if gen_num == -1:
        gen_num = len(loader)
    assert cfg.resume is not None, "Need a checkpoint path"
    step = _start_experiment(
        cfg, 
        model, 
        ddp_rank, 
        ddp_world_size,
        cfg.resume)
    
    mode_names = loader.dataset.get_mode_names()
    model = model.to(f"cuda:{ddp_rank}")
    model.eval()
    cast_dtype = torch.half if cfg.precision == "fp16" else (torch.bfloat16 if cfg.precision == "bf16" else torch.float32)
    eval_sample_time = AverageMeter("Time", ":6.3f")
    all_patterns = []
    all_gt_patterns = []
    all_questions = []
    all_image_paths = []
    all_output_texts = []
    all_errored_texts = []
    all_sample_types = []
    dist.barrier()
    for i, input_dict in enumerate(loader):
        if i == gen_num:
            break
        
        input_dict["pixel_values"] = input_dict["pixel_values"].to(cast_dtype)
        if "pattern_endpoints" in input_dict and input_dict["pattern_endpoints"].shape[1] > 0:
            input_dict["pattern_endpoints"] = input_dict["pattern_endpoints"].to(cast_dtype)
        else:
            input_dict["pattern_endpoints"] = None
        
        if "pattern_transfs" in input_dict and input_dict["pattern_transfs"].shape[1] > 0:
            input_dict["pattern_transfs"] = input_dict["pattern_transfs"].to(cast_dtype)
        else:
            input_dict["pattern_transfs"] = None
            
        output_dict = model.generate(
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
            max_new_tokens=2100
        )
        
        output_dict = dict_to_cpu(output_dict)
        output_dict = dict_to_dtype(output_dict, torch.float32)
        output_dict["input_mask"] = torch.arange(output_dict["output_ids"].shape[1]).reshape(1, -1) >= input_dict["question_ids"].shape[1]
        output_text, patterns, error_type = garment_tokenizer.decode(output_dict, processor)
        try:
            data_name = input_dict["gt_patterns"][0][-1].name
            os.makedirs(os.path.join(log_dir, data_name), exist_ok=True)
            patterns.serialize(os.path.join(log_dir, data_name), spec_only=False, with_3d=False, with_text=False, view_ids=False, to_subfolder=False, tag=f'_pred')
            for gt_pattern in input_dict["gt_patterns"][0]:
                gt_pattern.serialize(os.path.join(log_dir, data_name), spec_only=False, with_3d=False, with_text=False, view_ids=False, to_subfolder=False, tag=f'_gt')
            f = open(os.path.join(log_dir, data_name, "output.txt"), "w")
            question = input_dict["questions_list"][0]
            f.write(f"Question: {question}\n")
            f.write(f"Output Text: {output_text}\n")
            f.close()
            if os.path.isfile(input_dict["image_paths"][0]):
                cond_img = Image.open(input_dict["image_paths"][0])
                cond_img.save(os.path.join(log_dir, data_name, 'input.png'))
        except Exception as e:
            log.error(e)
            pass
        eval_sample_time.update(time.time() - end)
        end = time.time()
        all_gt_patterns.append(input_dict["gt_patterns"][0])
        all_sample_types.append(input_dict["sample_type"][0].cpu().numpy())  
        all_patterns.append(patterns)
        all_questions.append(input_dict["questions_list"][0])
        all_image_paths.append(input_dict["image_paths"][0])
        all_output_texts.append(output_text)
        log.info("Rank: {}, Progress: [{}/{}]".format(ddp_rank, i, gen_num))
        (total_num_panel_correct, 
         num_edge_accs, 
         num_edge_correct_accs, 
         vertex_L2s, 
         transl_l2s, 
         rots_l2s, 
         stitch_accs, 
         sorted_inds) = garment_tokenizer.evaluate_patterns(all_patterns, [p[-1] for p in all_gt_patterns])
        
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
        
        all_sample_types = np.array(all_sample_types)
        for i, mode in enumerate(mode_names):
            mode_mask = all_sample_types == i
            mode_mask = torch.from_numpy(mode_mask).to(num_edge_accs).bool()
            if not mode_mask.any():
                continue
            modewise_panel_accs[mode] = total_num_panel_correct[mode_mask].mean()
            modewise_edge_accs[mode] = num_edge_accs[torch.logical_and(mode_mask, num_edge_accs != -1)].sum() \
                / max((num_edge_accs[mode_mask] != -1).sum(), 1)
            modewise_edge_correct_accs[mode] = num_edge_correct_accs[torch.logical_and(mode_mask, num_edge_correct_accs != -1)].sum() \
                / max((num_edge_correct_accs[mode_mask] != -1).sum(), 1)
            modewise_vertex_L2s[mode] = vertex_L2s[torch.logical_and(mode_mask, vertex_L2s != -1)].sum() \
                / max((vertex_L2s[mode_mask] != -1).sum(), 1)
            modewise_transl_l2s[mode] = transl_l2s[torch.logical_and(mode_mask, transl_l2s != -1)].sum() \
                / max((transl_l2s[mode_mask] != -1).sum(), 1)
            modewise_rots_l2s[mode] = rots_l2s[torch.logical_and(mode_mask, rots_l2s != -1)].sum() \
                / max((rots_l2s[mode_mask] != -1).sum(), 1)
            modewise_stitch_accs[mode] = stitch_accs[torch.logical_and(mode_mask, stitch_accs != -1)].sum() \
                / max((stitch_accs[mode_mask] != -1).sum(), 1)
                
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
        if ddp_rank == 0:
            log.info(
                "Step: {:03d}\t"
                "Sample: {:.3f} ({:.3f})\t"
                "Num Panel Accuracy: {:.8f}\t"
                "Num Edge Accuracy: {:.8f}\t"
                "Num Correct Edge Accuracy: {:.8f}\t"
                "Vertex L2: {:.8f}\t"
                "translation L2: {:.8f}\t"
                "rotation L2: {:.8f}\t"
                "stitch acc: {:.8f}\t".format(
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
                f"val/num_panel_accuracy": total_panel_accs.cpu().item(),
                f"val/num_edge_accuracy": total_edge_acc.cpu().item(),
                f"val/num_correct_edge_accuracy": total_edge_correct_acc.cpu().item(),
                f"val/vertex_L2": total_vertex_L2.cpu().item(),
                f"val/translation_L2": total_transl_l2.cpu().item(),
                f"val/rotation_L2": total_rots_l2.cpu().item(),
                f"val/stitch_acc": total_stitch_acc.cpu().item(),
                f"val/sample_time": eval_sample_time.avg,
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
