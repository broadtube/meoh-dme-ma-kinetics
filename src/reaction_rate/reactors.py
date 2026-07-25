"""反応器モデル。まずは等温・一定圧の PFR。

PFR 設計式:  dFᵢ/dW = Rᵢ(state(W))     （W = 触媒質量 [kg], Fᵢ = モル流量 [mol/s]）
各 W で局所組成 yᵢ=Fᵢ/ΣF から GasState を作り、network.species_rates で Rᵢ を得る。
（等温=T一定、一定圧=P一定。流量計算は理想気体、レート駆動力は SRK/分圧/濃度。）

触媒は CatalystBed（各触媒量を指定）。総触媒量 W_total = 各触媒量の合計。
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .units import R_BAR_M3
from .state import GasState
from .network import species_rates


@dataclass
class CatalystBed:
    """ハイブリッド触媒ベッド（各触媒の質量を指定）。

    masses : {役割: 質量[kg]}。役割は 'synthesis' / 'dehydration' / 'carbonylation'。
             例) 単一   : {"synthesis": 50.0}
                 DME合成: {"synthesis": 4.0, "dehydration": 1.0}   # 8:2
                 タンデム: {"synthesis": .., "dehydration": .., "carbonylation": ..}
    acid_site_density : mol Al/kg（'carbonylation' の DTU per-mol-Al → per-kg 変換に使用）。
        既定 1.43 = 1.43 mmol/g の全 Al 量。出典: Rasmussen, Christensen ら, Catal. Sci.
        Technol. 2017（DTU mordenite carbonylation）§2 Experimental「Mordenite (SiO2/Al2O3=20)
        obtained from Zeolyst (CBV21A) and all Al sites (1.43×10⁻³ mol Al/g)」。DTU は TOF を
        全 Al 基準で定義しているため、per-mol-Al 速度定数 k1 とこの値は同一基準で整合。
        （公称 SiO2/Al2O3=20 の理想値は ~1.51 mmol/g、報告値 1.43 は実効 ~22 相当）。
    """
    masses: dict[str, float]
    acid_site_density: float = 1.43   # [mol Al/kg] =1.43 mmol/g。DTU2017 §2, Zeolyst CBV21A(SiO2/Al2O3=20)

    def total(self) -> float:
        return sum(self.masses.values())


@dataclass
class Geometry:
    """反応器の幾何。積分後の横軸 W[kg] を実長さ/滞留時間/流速に変換する後処理用。
    （コアPFRは W 基準のまま。ここは軸の付け替えだけを担う。）

    area          : 断面積 A [m²]
    bulk_density  : 充填密度 ρ_bed [kg-cat / m³-reactor]
    void_fraction : 空隙率 ε [-]（滞留時間・空隙内流速に使用。長さ変換には不要）
    """
    area: float
    bulk_density: float
    void_fraction: float = 0.40


@dataclass
class PFRResult:
    W: np.ndarray                 # 触媒質量グリッド [kg]
    F: dict[str, np.ndarray]      # {species: モル流量 [mol/s]}
    T: float
    P: float

    @property
    def F_total(self) -> np.ndarray:
        return sum(self.F.values())

    def mole_fractions(self) -> dict[str, np.ndarray]:
        Ft = self.F_total
        return {s: self.F[s] / Ft for s in self.F}

    def conversion(self, species: str) -> np.ndarray:
        F0 = self.F[species][0]
        return (F0 - self.F[species]) / F0

    def outlet(self) -> dict[str, float]:
        return {s: float(self.F[s][-1]) for s in self.F}

    # --- 実長さ/時間/流速への変換（Geometry を渡したときだけ）---
    def length(self, geom: "Geometry") -> np.ndarray:
        """実長さ z [m] = W/(ρ_bed·A)。W の単純な線形リスケール。"""
        return self.W / (geom.bulk_density * geom.area)

    def volumetric_flow(self, criticals: str = "chemicals") -> np.ndarray:
        """体積流量 Q [m³/s] = F_total·Z·R·T/P。各 W で SRK の圧縮係数 Z を評価
        （モル数変化＝メタノール合成 3→1 mol を反映。criticals='graaf1986' も可）。"""
        y = self.mole_fractions()
        Ft = self.F_total
        Q = np.empty_like(self.W)
        for i in range(len(self.W)):
            yi = {s: float(y[s][i]) for s in self.F}
            Z = GasState(self.T, self.P, yi).Z(criticals)
            Q[i] = Ft[i] * Z * R_BAR_M3 * self.T / self.P
        return Q

    def residence_time(self, geom: "Geometry", criticals: str = "chemicals") -> np.ndarray:
        """滞留時間 t [s] = ∫₀^W ε/(ρ_bed·Q) dW'（入口 0 からの累積・台形則）。
        Q が反応で縮むため W や長さの線形変換にはならない。"""
        Q = self.volumetric_flow(criticals)
        integrand = geom.void_fraction / (geom.bulk_density * Q)   # [s/kg]
        dt = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(self.W)
        return np.concatenate([[0.0], np.cumsum(dt)])

    def velocity(self, geom: "Geometry", criticals: str = "chemicals",
                 interstitial: bool = False) -> np.ndarray:
        """流速 [m/s]。既定は見かけ速度 u_s=Q/A。interstitial=True で空隙内 u_i=Q/(ε·A)。"""
        Q = self.volumetric_flow(criticals)
        return Q / (geom.area * (geom.void_fraction if interstitial else 1.0))


def pfr(F_in: dict[str, float], T: float, P: float, bed: CatalystBed,
        models: dict | None = None, k_eq3: str = "KOGAS",
        n_points: int = 200) -> PFRResult:
    """等温・一定圧 PFR を積分する。

    F_in   : 入口モル流量 {species: mol/s}（活性反応に現れる種は 0 でも含める）
    T, P   : 温度[K], 圧力[bar]（一定）
    bed    : CatalystBed（各触媒量。総量が積分上限）
    models : {役割: モデル名}。k_eq3 : 脱水平衡 'KOGAS'/'BL'/'thermo'。
    """
    species = list(F_in)
    F0 = np.array([F_in[s] for s in species], dtype=float)
    W_total = bed.total()

    def rhs(W, F):
        F_pos = np.maximum(F, 0.0)
        Ftot = F_pos.sum()
        y = {s: F_pos[i] / Ftot for i, s in enumerate(species)}
        R = species_rates(GasState(T, P, y), bed, models=models, k_eq3=k_eq3)
        return np.array([R.get(s, 0.0) for s in species])

    sol = solve_ivp(rhs, (0.0, W_total), F0, method="BDF",
                    t_eval=np.linspace(0.0, W_total, n_points),
                    rtol=1e-8, atol=1e-16)
    if not sol.success:
        raise RuntimeError(f"PFR 積分に失敗: {sol.message}")
    F = {s: sol.y[i] for i, s in enumerate(species)}
    return PFRResult(sol.t, F, T, P)
