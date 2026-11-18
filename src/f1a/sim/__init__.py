"""F1-A 单车蒙特卡洛比赛模拟器（v0.1，Claude 编写）。

只回答：同一辆车在同一场比赛里，不同进站策略的总用时分布是什么。不模拟对手、超车和交通。
入口：RaceConfig.from_project / TyreParams.from_posterior / Strategy / simulate / evaluate。
"""
from .params import TyreParams, COMPOUNDS, race_index_map, estimated_compounds
from .race import RaceConfig
from .strategy import Strategy
from .engine import simulate, evaluate, common_draws
from .optimize import optimize_stops

__all__ = ["TyreParams", "COMPOUNDS", "race_index_map", "RaceConfig", "Strategy", "simulate", "evaluate", "common_draws", "optimize_stops"]
