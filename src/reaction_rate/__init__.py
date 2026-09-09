"""reaction_rate — CO/CO₂/H₂ → メタノール/DME/酢酸メチル 合成の反応速度計算。

コードは references/rate_equations.html のセクション順に対応:
  graaf         → §1 メタノール合成 (Graaf 1988) ＋ 平衡定数
  vbf_kogas     → §2 VBF/KOGAS 合成・逆WGS ＋ 脱水(Bercič–Levec)
  carbonylation → §3 モルデナイト カルボニル化 (DTU/Cheng)
共通基盤: units, state (GasState + SRK フガシティ), thermo (NASA-7 の h/cp・断熱床用)
反応器  : network (量論→成分速度), reactors (等温/断熱 PFR + CatalystBed)
条件定義: case.py（独立ファイル。`python -m reaction_rate.case` で実行）
書き出し: aspen_lhhw（ZSM-5 脱水を Aspen Plus の LHHW 速度式へ厳密変換）
"""
from . import units, state, thermo, graaf, vbf_kogas, carbonylation, network, reactors, aspen_lhhw
from .state import GasState, Component, COMPONENTS
from .reactors import CatalystBed, Geometry, PFRResult, pfr

__version__ = "0.1.0"
__all__ = [
    "units", "state", "thermo", "graaf", "vbf_kogas", "carbonylation", "network", "reactors",
    "aspen_lhhw",
    "GasState", "Component", "COMPONENTS",
    "CatalystBed", "Geometry", "PFRResult", "pfr",
]
