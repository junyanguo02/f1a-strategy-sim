from __future__ import annotations
from numbers import Integral
from dataclasses import dataclass, field
from .params import COMPOUNDS


@dataclass
class Strategy:
    """start_compound 起步配方；start_age 起步胎龄（新胎为 1）；stops = [(进站圈, 换上的配方), ...]，
    进站圈 = 进站的那一圈（in-lap），下一圈为新胎第 1 圈。"""
    name: str
    start_compound: str
    stops: list = field(default_factory=list)
    start_age: int = 1

    def validate(self, total_laps: int) -> None:
        comps = [self.start_compound] + [c for _, c in self.stops]
        for c in comps:
            if c not in COMPOUNDS:
                raise ValueError(f"{self.name}: 未知配方 {c}")
        if len(set(comps)) < 2:
            raise ValueError(f"{self.name}: 干地比赛至少使用两种配方")
        laps = [l for l, _ in self.stops]
        if any(not isinstance(lap, Integral) or isinstance(lap, bool) for lap in laps):
            raise ValueError(f"{self.name}: 进站圈必须是整数")
        if laps != sorted(laps) or len(set(laps)) != len(laps):
            raise ValueError(f"{self.name}: 进站圈必须严格递增")
        if laps and (laps[0] < 1 or laps[-1] >= total_laps):
            raise ValueError(f"{self.name}: 进站圈须在 1..{total_laps - 1}")

    def label(self) -> str:
        comps = [self.start_compound] + [c for _, c in self.stops]
        return "-".join(c[0] for c in comps) + (" @" + ",".join(str(l) for l, _ in self.stops) if self.stops else "")
