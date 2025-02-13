from dataclasses import dataclass, field


@dataclass
class StatsConfig:
    scale: list[float]
    shift: list[float]

@dataclass
class StandardizeConfig:
    rotations: StatsConfig = field(default_factory=StatsConfig)
    translations: StatsConfig = field(default_factory=StatsConfig)
    vertices: StatsConfig = field(default_factory=StatsConfig)
