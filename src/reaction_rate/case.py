"""実行ケース（条件）の定義と実行。

初期組成・流量・温度・圧力・触媒量などの「条件」はここに集約する。
下の CASE_* を編集すれば条件を変えられる。`run_case(CASE_*)` で PFR にかける。

  実行例:  PYTHONPATH=src python3 -m reaction_rate.case
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .reactors import CatalystBed, pfr, PFRResult


# ============================================================
#  ケースの型
# ============================================================
@dataclass
class Case:
    """1つの反応器シミュレーション条件。"""
    name: str
    feed: dict[str, float]          # 入口モル流量 [mol/s]（活性反応に現れる種は 0 でも含める）
    T: float                        # 温度 [K]（等温）
    P: float                        # 圧力 [bar]（一定）
    bed: CatalystBed                # 触媒ベッド（各触媒量[kg]）
    models: dict | None = None      # {役割: モデル名}  例) {"synthesis": "KOGAS"}
    k_eq3: str = "KOGAS"            # 脱水平衡: "KOGAS" / "BL" / "thermo"
    n_points: int = 200


def run_case(case: Case) -> PFRResult:
    """Case を等温 PFR にかけて結果を返す。"""
    return pfr(case.feed, case.T, case.P, case.bed,
               models=case.models, k_eq3=case.k_eq3, n_points=case.n_points)


# ============================================================
#  ケース定義（ここを編集して条件を変える）
#  feed はモル流量 [mol/s]。総流量は任意（W/F が接触時間）。
# ============================================================

# --- メタノール合成: 単一触媒 Cu/ZnO/Al2O3, Graaf 1988 ---
CASE_METHANOL = Case(
    name="methanol synthesis (Graaf 1988)",
    feed={"CO": 0.15e-3, "CO2": 0.10e-3, "H2": 0.75e-3, "H2O": 0.0, "CH3OH": 0.0},
    T=523.15,                       # 250 °C
    P=50.0,                         # bar
    bed=CatalystBed(masses={"synthesis": 50.0}),
    models={"synthesis": "Graaf1988"},
)

# --- DME 合成: ハイブリッド触媒 8:2 (合成:脱水), KOGAS ---
#     k_eq3="thermo" にすると脱水平衡が物理的（KOGAS だと過剰生成）。
CASE_DME = Case(
    name="DME synthesis (KOGAS, 8:2)",
    feed={"CO": 0.32e-3, "CO2": 0.03e-3, "H2": 0.65e-3,
          "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0},
    T=523.15,
    P=50.0,
    bed=CatalystBed(masses={"synthesis": 4.0, "dehydration": 1.0}),  # 8:2
    models={"synthesis": "KOGAS", "dehydration": "KOGAS"},
    k_eq3="thermo",
)

# --- DME カルボニル化: 単一触媒 H-MOR, DTU（Fig 7 条件）---
_FLOW = 300.0 / 22414.0 / 60.0      # 300 Nml/min → mol/s
CASE_CARBONYLATION = Case(
    name="DME carbonylation (DTU, Fig 7)",
    feed={"CO": 0.98 * _FLOW, "DME": 0.02 * _FLOW, "MA": 0.0},
    T=438.0,
    P=10.0,
    bed=CatalystBed(masses={"carbonylation": 1.5e-3}, acid_site_density=1.43),
    models={"carbonylation": "DTU"},
)

# 一括実行用
ALL_CASES = [CASE_METHANOL, CASE_DME, CASE_CARBONYLATION]


if __name__ == "__main__":
    for case in ALL_CASES:
        res = run_case(case)
        y = res.mole_fractions()
        outlet = {s: round(float(y[s][-1]), 4) for s in res.F}
        print(f"[{case.name}]  出口モル分率: {outlet}")
