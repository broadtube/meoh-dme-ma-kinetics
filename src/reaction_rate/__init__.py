"""reaction_rate — CO/CO₂/H₂ → メタノール/DME/酢酸メチル 合成の反応速度計算。

コードは references/rate_equations.html のセクション順に対応:
  graaf         → §1 メタノール合成 (Graaf 1988) ＋ 平衡定数
  vbf_kogas     → §2 VBF/KOGAS 合成・逆WGS ＋ 脱水(Bercič–Levec)
  carbonylation → §3 モルデナイト カルボニル化 (DTU/Cheng)
共通基盤: units, state (GasState + SRK フガシティ)
反応器  : network (量論→成分速度), reactors (PFR + CatalystBed)
条件定義: case.py（独立ファイル。`python -m reaction_rate.case` で実行）
"""
from . import units, state, graaf, vbf_kogas, carbonylation, network, reactors
from .state import GasState, Component, COMPONENTS
from .reactors import CatalystBed, Geometry, PFRResult, pfr

__version__ = "0.1.0"
__all__ = [
    "units", "state", "graaf", "vbf_kogas", "carbonylation", "network", "reactors",
    "GasState", "Component", "COMPONENTS",
    "CatalystBed", "Geometry", "PFRResult", "pfr",
]
