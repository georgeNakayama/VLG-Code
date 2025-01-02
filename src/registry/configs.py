from dataclasses import dataclass, field
from typing import Optional

@dataclass 
class SystemConfig:
    wandb_username: Optional[str] = None
    output: str = "./"
    

@dataclass
class MainConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    experiment: MyExperimentConfig = field(default_factory=MyExperimentConfig)
    data_wrapper: PCDDataWrapperConfig = field(default_factory=PCDDataWrapperConfig)
    trainer: TrainerPCDConfig = field(default_factory=TrainerPCDConfig)
    model: PCDModelConfig = field(default_factory=PCDModelConfig)
    criteria: GarmentPCDLossConfig = field(default_factory=GarmentPCDLossConfig)
    random_seed: Optional[int] = None
    test_only: bool = False
    step_trained: Optional[str] = None
    pre_trained: Optional[str] = None
    storage_dir: Optional[str] = None