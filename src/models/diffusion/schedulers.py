from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler
)

def build_noise_scheduler(args):
    scheduler = DDPMScheduler(
        num_train_timesteps=args["num_train_timesteps"],
        beta_start=args["beta_start"],
        beta_end=args["beta_end"],
        beta_schedule=args["beta_schedule"],
        variance_type=args["variance_type"],
        clip_sample=args["clip_sample"]
    )
    return scheduler 

def build_denoise_scheduler(args):
    name = args["denoise_scheduler_type"]
    if name == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=args["num_train_timesteps"],
            beta_start=args["beta_start"],
            beta_end=args["beta_end"],
            beta_schedule=args["beta_schedule"],
            clip_sample=args["clip_sample"],
            set_alpha_to_one=args["set_alpha_to_one"],
            steps_offset=args["steps_offset"]
        )
    elif name == "karras_ve":
        scheduler = KarrasVeScheduler(
            sigma_max=args["sigma_max"],
            sigma_min=args["sigma_min"], 
            s_noise=args["s_noise"],
            s_churn=args["s_churn"],
            s_min=args["s_min"],
            s_max=args["s_max"],
        )
    elif name == "dpms_multistep":
        raise NotImplementedError("DPMS multistep scheduler not implemented yet")
    else:
        raise ValueError(f"Unknown denoise scheduler type: {name}")

    return scheduler 