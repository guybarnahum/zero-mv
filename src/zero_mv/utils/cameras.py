
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Pose:
    yaw: float
    pitch: float
    roll: float = 0.0
    fov: float = 40.0
    seed: int | None = None
