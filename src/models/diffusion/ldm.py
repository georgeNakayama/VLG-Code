# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from einops import rearrange

from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler
)
from transformers import CLIPModel

# from michelangelo.models.tsal.tsal_base import ShapeAsLatentPLModule
from ..pcd2garment.garment_pcd import GarmentPCD
from ..pcd2garment.garment_pcd import SetCriterionWithOutMatcher
from ..pcd2garment.garment_pcd import build as build_first_stage
from ..diffusion.denoiser import ConditionalDenoiser
from ..diffusion.denoiser import build as build_diffusion_denoiser
from .inference_utils import ddim_sample
from .clip_encoder_factory import FrozenCLIPImageGridEmbedder
from .schedulers import build_denoise_scheduler, build_noise_scheduler
SchedulerType = Union[DDIMScheduler, KarrasVeScheduler, DPMSolverMultistepScheduler]


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Diffuser(nn.Module):
    def __init__(self, 
                 loss_cfg: Dict, 
                 scheduler_cfg: Dict, 
                 first_stage_model: GarmentPCD, 
                 first_stage_criterions: SetCriterionWithOutMatcher,
                 cond_stage_model: FrozenCLIPImageGridEmbedder,
                 denoiser: ConditionalDenoiser,
                 noise_scheduler: DDPMScheduler,
                 denoise_scheduler: SchedulerType, 
                 scale_by_std: bool = False,
                 z_scale_factor: float = 1.0,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: Union[Tuple[str], List[str]] = ()):

        super().__init__()

        self.loss_cfg=loss_cfg
        self.scheduler_cfg=scheduler_cfg

        # 1. initialize first stage. 
        # Note: the condition model contained in the first stage model.
        self.first_stage_model: GarmentPCD = first_stage_model
        self.first_stage_criterions: SetCriterionWithOutMatcher = first_stage_criterions
        self.instantiate_first_stage()

        # 2. initialize conditional stage
        # self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_model: FrozenCLIPImageGridEmbedder = cond_stage_model

        # 3. diffusion model
        self.model: ConditionalDenoiser = denoiser


        self.noise_scheduler: DDPMScheduler = noise_scheduler
        self.denoise_scheduler: SchedulerType = denoise_scheduler

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    @torch.no_grad()
    def empty_img_cond(self, cond):

        return torch.zeros_like(cond, device=cond.device)

    def instantiate_first_stage(self):
        self.first_stage_model = self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        
        self.first_stage_criterions.train = disabled_train
        for param in self.first_stage_criterions.parameters():
            param.requires_grad = False

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def encode_text(self, text):

        b = text.shape[0]
        text_tokens = rearrange(text, "b t l -> (b t) l")
        text_embed = self.first_stage_model.model.encode_text_embed(text_tokens)
        text_embed = rearrange(text_embed, "(b t) d -> b t d", b=b)
        text_embed = text_embed.mean(dim=1)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        return text_embed

    @torch.no_grad()
    def encode_first_stage(self, surface: torch.FloatTensor, sample_posterior=True):

        _, z_q, _ = self.first_stage_model.encode_pcd(surface, sample_posterior)
        z_q = self.z_scale_factor * z_q

        return z_q

    @torch.no_grad()
    def decode_first_stage(self, z_q: torch.FloatTensor, **kwargs):

        z_q = 1. / self.z_scale_factor * z_q
        latents = self.first_stage_model.decode_garment(z_q, **kwargs)
        return latents
    
    @torch.no_grad()
    def reconstruct_first_stage(self, surface: torch.FloatTensor, **kwargs):
        z_q = self.encode_first_stage(surface)
        latents = self.decode_first_stage(z_q, **kwargs)
        return latents

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 \
                and batch_idx == 0 and self.ckpt_path is None:
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")

            z_q = self.encode_first_stage(batch[self.first_stage_key])
            z = z_q.detach()

            del self.z_scale_factor
            self.register_buffer("z_scale_factor", 1. / z.flatten().std())
            print(f"setting self.z_scale_factor to {self.z_scale_factor}")

            print("### USING STD-RESCALING ###")

    def compute_loss(self, model_outputs, split):
        """

        Args:
            model_outputs (dict):
                - x_0:
                - noise:
                - noise_prior:
                - noise_pred:
                - noise_pred_prior:

            split (str):

        Returns:

        """

        pred = model_outputs["pred"]

        if self.noise_scheduler.prediction_type == "epsilon":
            target = model_outputs["noise"]
        elif self.noise_scheduler.prediction_type == "sample":
            target = model_outputs["x_0"]
        else:
            raise NotImplementedError(f"Prediction Type: {self.noise_scheduler.prediction_type} not yet supported.")

        if self.loss_cfg["loss_type"] == "l1":
            simple = F.l1_loss(pred, target, reduction="mean")
        elif self.loss_cfg["loss_type"] in ["mse", "l2"]:
            simple = F.mse_loss(pred, target, reduction="mean")
        else:
            raise NotImplementedError("Loss Type: %s not yet supported." / (self.loss_cfg["loss_type"]))

        total_loss = simple

        loss_dict = {
            f"{split}/total_loss": total_loss.clone().detach(),
            f"{split}/simple": simple.detach(),
        }

        return total_loss, loss_dict

    def forward(self, pcd: torch.FloatTensor, image: torch.FloatTensor):
        """

        Args:
            pcd:
            image:
            
            #TODO: add text condition

        Returns:

        """


        latents = self.encode_first_stage(pcd)

        # conditions = self.cond_stage_model.encode(batch[self.cond_stage_key])
        conditions = self.cond_stage_model.encode(image)

        # Sample noise that we"ll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_z = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # diffusion model forward
        noise_pred = self.model(noisy_z, timesteps, conditions)

        diffusion_outputs = {
            "x_0": noisy_z,
            "noise": noise,
            "pred": noise_pred
        }

        return diffusion_outputs



    @torch.no_grad()
    def sample(self,
               image: torch.FloatTensor,
               device: torch.device,
               sample_times: int = 1,
               steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               eta: float = 0.0,
               return_intermediates: bool = False, 
                **kwargs):

        if self.first_stage_model is None:
            self.instantiate_first_stage(self.first_stage_config)

        if steps is None:
            steps = self.scheduler_cfg["num_inference_steps"]

        if guidance_scale is None:
            guidance_scale = self.scheduler_cfg["guidance_scale"]
        do_classifier_free_guidance = guidance_scale > 0

        # conditional encode 
        #TODO add text condition
        xc = image
        # cond = self.cond_stage_model[self.cond_stage_key](xc)
        cond = self.cond_stage_model(xc)

        if do_classifier_free_guidance:
            """
            Note: There are two kinds of uncond for text. 
            1: using "" as uncond text; (in SAL diffusion)
            2: zeros_like(cond) as uncond text; (in MDM)
            """
            un_cond = self.cond_stage_model.unconditional_embedding(batch_size=len(xc))
            cond = torch.cat([un_cond, cond], dim=0)

        outputs = []
        latents = None

        if not return_intermediates:
            for _ in range(sample_times):
                sample_loop = ddim_sample(
                    self.denoise_scheduler,
                    self.model,
                    shape=self.first_stage_model.latent_shape,
                    cond=cond,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    device=device,
                    eta=eta,
                    disable_prog=False
                )
                for sample, t in sample_loop:
                    latents = sample
                outputs.append(self.decode_first_stage(latents, **kwargs))
        else:

            sample_loop = ddim_sample(
                self.denoise_scheduler,
                self.model,
                shape=self.first_stage_model.latent_shape,
                cond=cond,
                steps=steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=device,
                eta=eta,
                disable_prog=False
            )

            iter_size = steps // sample_times
            i = 0
            for sample, t in sample_loop:
                latents = sample
                if i % iter_size == 0 or i == steps - 1:
                    outputs.append(self.decode_first_stage(latents, **kwargs))
                i += 1

        return outputs


def build(args) -> Diffuser:
    loss_cfg = args["NN"]["ldm_loss"]
    scheduler_cfg = args["NN"]["schedulers"]
    first_stage_model, criterions = build_first_stage(args)
    cond_stage_model = FrozenCLIPImageGridEmbedder(version=args["NN"]["image_condition_encoder"]["version"], zero_embedding_radio=args["NN"]["image_condition_encoder"]["zero_embedding_radio"])
    denoiser = build_diffusion_denoiser(args["NN"]["denoiser"])
    noise_scheduler = build_noise_scheduler(args["NN"]["schedulers"]["noise"])
    denoise_scheduler = build_denoise_scheduler(args["NN"]["schedulers"]["denoise"])

    model = Diffuser(loss_cfg, scheduler_cfg, first_stage_model, criterions, cond_stage_model, denoiser, noise_scheduler, denoise_scheduler)

    return model