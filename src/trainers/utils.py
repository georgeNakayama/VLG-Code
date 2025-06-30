from enum import Enum
import logging

log = logging.getLogger(__name__)
import pathlib
import os

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import wandb as wb

from trainers.convert_zero_to_torch import _get_fp32_state_dict_from_zero_checkpoint


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
        elif isinstance(input_dict[k], dict):
            input_dict[k] = dict_to_cuda(input_dict[k])
    return input_dict


def dict_to_cpu(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cpu()
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cpu() for ele in v]
        elif isinstance(input_dict[k], dict):
            input_dict[k] = dict_to_cpu(input_dict[k])
    return input_dict


def dict_to_dtype(input_dict, dtype=torch.float32, target_keys=None):
    for k, v in input_dict.items():
        if target_keys is not None:
            if k not in target_keys:
                continue

        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.to(dtype=dtype)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.to(dtype=dtype) for ele in v]
        elif isinstance(input_dict[k], dict):
            input_dict[k] = dict_to_dtype(input_dict[k], dtype)
    return input_dict


def master_log(local_rank: int, logger: logging.Logger, *args):
    if local_rank == 0:
        logger.info(*args)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class ProgressMeter(object):
    def __init__(self, logger, local_rank, num_batches, meters, prefix=""):
        self.logger = logger
        self.local_rank = local_rank
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        master_log(self.local_rank, self.logger, "\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        master_log(self.local_rank, self.logger, " ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        val = val.item() if isinstance(val, torch.Tensor) and val.dim() == 0 else val
        n = n.item() if isinstance(n, torch.Tensor) and n.dim() == 0 else n

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def start_experiment(
    cfg,
    model,
    ddp_rank: int,
    ddp_world_size: int,
    resume: pathlib.Path | str | None,
    from_start: bool = True,
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

    if resume is not None:
        if os.path.isfile(resume):
            # assume single torch checkpoint
            state_dict = torch.load(resume)
            model.load_state_dict(state_dict, strict=False)
            return 0

        # assume deepspeed checkpoint
        resume = pathlib.Path(resume)
        latest_path = resume / "latest"
        if not latest_path.exists():
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")
        with open(latest_path, "r") as fd:
            tag = fd.read().strip()

        ds_checkpoint_dir = resume / tag
        ds_checkpoint_mismatch = ds_checkpoint_dir.exists() and len(os.listdir(ds_checkpoint_dir)) != ddp_world_size + 1

        if from_start or ds_checkpoint_mismatch:
            if ddp_rank == 0 and not from_start:
                log.warn("Resuming from checkpoint but number of GPUs is mismatched!")

            state_dict = _get_fp32_state_dict_from_zero_checkpoint(
                ds_checkpoint_dir, True
            )
            model.module.load_state_dict(state_dict, strict=False)
        else:
            _, _ = model.load_checkpoint(resume)

        with open(latest_path, "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        if not from_start:
            start_step = int(ckpt_dir.replace("global_step", ""))

        if ddp_rank == 0:
            log.info(
                "Resume training from {}, start from epoch {}".format(
                    resume, start_step
                )
            )
    if ddp_rank == 0:
        wb.watch(model, log="all")
    return start_step
