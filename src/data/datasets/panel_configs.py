from dataclasses import dataclass, field
from typing import List, Optional
from torch import Tensor
import torch

@dataclass
class StatsConfig():
    scale: List[float]
    shift: List[float]

@dataclass 
class StandardizeConfig():
    rotations: StatsConfig = field(default_factory=StatsConfig)
    translations: StatsConfig = field(default_factory=StatsConfig)
    vertices: StatsConfig = field(default_factory=StatsConfig)