"""反応器モデル。等温/断熱 PFR と等温 CSTR（無勾配反応器）。

PFR 設計式:  dFᵢ/dW = Rᵢ(state(W))     （W = 触媒質量 [kg], Fᵢ = モル流量 [mol/s]）
各 W で局所組成 yᵢ=Fᵢ/ΣF から GasState を作り、network.species_rates で Rᵢ を得る。
（一定圧=P一定。流量計算は理想気体、レート駆動力は SRK/分圧/濃度。）

温度モード:
  等温 (adiabatic=False, 既定): T 一定。result.T は float。
  断熱 (adiabatic=True)       : エネルギー収支を連立して T も積分する。result.T は配列。
      dT/dW = −Σᵢ Rᵢ·hᵢ(T) / Σᵢ Fᵢ·cpᵢ(T)
      hᵢ は生成熱込みの絶対エンタルピーなので Σᵢ Rᵢhᵢ = Σⱼ rⱼΔHⱼ（反応ごとの ΔH 不要）。
      h・cp は thermo.py（Cantera NASA-7）。⚠️ 熱損失ゼロ・圧損ゼロ・η=1。

CSTR（cstr）は無勾配循環反応器（gradientless recycle reactor）の理想化。
触媒が見るのは**出口組成**で、Fᵢ_out = Fᵢ_in + Rᵢ(y_out)·W を解く。
Ortega 2018 の速度論測定はこの型の反応器で行われている。

触媒は CatalystBed（各触媒量を指定）。総触媒量 W_total = 各触媒量の合計。
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

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
    T: float | np.ndarray         # 等温なら float、断熱なら W と同長の温度プロファイル [K]
    P: float

    @property
    def F_total(self) -> np.ndarray:
        return sum(self.F.values())

    @property
    def T_profile(self) -> np.ndarray:
        """温度プロファイル [K]（等温でも W と同長の配列にして返す）。"""
        return np.broadcast_to(np.asarray(self.T, dtype=float), self.W.shape).copy()

    def _T_at(self, i: int) -> float:
        return float(self.T[i]) if np.ndim(self.T) else float(self.T)

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
            Ti = self._T_at(i)
            Z = GasState(Ti, self.P, yi).Z(criticals)
            Q[i] = Ft[i] * Z * R_BAR_M3 * Ti / self.P
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
        n_points: int = 200, adiabatic: bool = False) -> PFRResult:
    """一定圧 PFR を積分する（既定は等温、adiabatic=True で断熱）。

    F_in      : 入口モル流量 {species: mol/s}（活性反応に現れる種は 0 でも含める）
    T, P      : 温度[K]（断熱では**入口温度**）, 圧力[bar]（一定）
    bed       : CatalystBed（各触媒量。総量が積分上限）
    models    : {役割: モデル名}。k_eq3 : 脱水平衡 'KOGAS'/'BL'/'thermo'。
    adiabatic : True でエネルギー収支を連立（h,cp は thermo.py = Cantera NASA-7）。
                熱損失ゼロ・圧損ゼロ・有効係数 η=1 の理想断熱床。
    """
    species = list(F_in)
    F0 = np.array([F_in[s] for s in species], dtype=float)
    W_total = bed.total()

    def rates_at(T_loc, F_pos):
        y = {s: F_pos[i] / F_pos.sum() for i, s in enumerate(species)}
        R = species_rates(GasState(T_loc, P, y), bed, models=models, k_eq3=k_eq3)
        return np.array([R.get(s, 0.0) for s in species])

    if not adiabatic:
        def rhs(W, F):
            return rates_at(T, np.maximum(F, 0.0))

        y0 = F0
    else:
        from . import thermo                      # 遅延 import（Cantera 依存はここだけ）
        thermo.enthalpies(T, species)             # 未対応成分があれば入口で即エラー

        def rhs(W, u):
            F_pos = np.maximum(u[:-1], 0.0)
            T_loc = u[-1]
            dF = rates_at(T_loc, F_pos)
            h = thermo.enthalpies(T_loc, species)
            cp = thermo.heat_capacities(T_loc, species)
            q = -sum(dF[i] * h[s] for i, s in enumerate(species))      # 発熱 [W/kg_cat]
            C = sum(F_pos[i] * cp[s] for i, s in enumerate(species))   # 熱容量流量 [W/K]
            return np.append(dF, q / C)

        y0 = np.append(F0, float(T))

    sol = solve_ivp(rhs, (0.0, W_total), y0, method="BDF",
                    t_eval=np.linspace(0.0, W_total, n_points),
                    rtol=1e-8, atol=1e-16)
    if not sol.success:
        raise RuntimeError(f"PFR 積分に失敗: {sol.message}")
    if not adiabatic:
        return PFRResult(sol.t, {s: sol.y[i] for i, s in enumerate(species)}, T, P)
    return PFRResult(sol.t, {s: sol.y[i] for i, s in enumerate(species)}, sol.y[-1], P)


def cstr(F_in: dict[str, float], T: float, P: float, bed: CatalystBed,
         models: dict | None = None, k_eq3: str = "KOGAS") -> dict[str, float]:
    """等温・一定圧 CSTR（無勾配反応器）の出口モル流量 [mol/s] を返す。

    Fᵢ_out = Fᵢ_in + Rᵢ(y_out)·W_total  を解く。触媒が見るのは**出口組成**なので、
    転化率が高いと生成物阻害・逆反応が効く（PFR より必ず遅い）。

    F_in : 入口モル流量 {species: mol/s}    T, P : 温度[K], 圧力[bar]
    bed  : CatalystBed（総量が触媒質量）    models / k_eq3 : pfr と同じ

    出典: Ortega 2018（ZSM-5 脱水）の速度論測定装置がこの型（循環比が十分大きく
    完全混合とみなせる、RTD で検証済 §3.3）。速度式のパラメータもこの反応器で回帰
    されているので、原著の再現には PFR ではなく本関数を使う。
    """
    W = bed.total()
    F_out = dict(F_in)

    def rates_at(F_map):
        tot = sum(max(v, 0.0) for v in F_map.values())
        y = {s: max(v, 0.0) / tot for s, v in F_map.items()}
        return species_rates(GasState(T, P, y), bed, models=models, k_eq3=k_eq3)

    # 反応に現れる種だけを未知数にする（不活性種は素通り＝方程式が自明）
    active = [s for s in F_in if s in rates_at(F_in)]
    if not active:
        return F_out

    def residual(x):
        trial = {**F_in, **{s: max(x[i], 0.0) for i, s in enumerate(active)}}
        R = rates_at(trial)
        return np.array([max(x[i], 0.0) - F_in[s] - W * R.get(s, 0.0)
                         for i, s in enumerate(active)])

    x0 = np.array([F_in[s] for s in active], dtype=float)
    scale = max(sum(F_in.values()), 1e-30)
    sol = least_squares(residual, x0, method="lm", xtol=1e-14, ftol=1e-14,
                        diff_step=1e-7)
    if not sol.success or np.linalg.norm(sol.fun) > 1e-8 * scale:
        raise RuntimeError(f"CSTR の求解に失敗: {sol.message}（残差 {np.linalg.norm(sol.fun):.3e}）")
    F_out.update({s: float(max(sol.x[i], 0.0)) for i, s in enumerate(active)})
    return F_out
