"""気相状態と SRK フガシティ（rate_equations.html 冒頭の凡例に相当）。

`GasState(T, P, y)` が中核。組成から各レート則が要求する駆動量を供給する:
  - partial_pressures() : pᵢ = yᵢ·P            [bar]      → VBF/KOGAS の r_MS/r_RWGS
  - concentrations()    : Cᵢ = yᵢ·P/(R·T)      [kmol/m³]  → Bercič–Levec の脱水 r_MD
  - fugacities()        : fᵢ = φᵢ·yᵢ·P          [bar]      → Graaf 合成・Cheng カルボニル化
  - fugacity_coeffs()   : φᵢ (SRK)

Cantera に SRK は無いため SRK は自前実装（Graaf 1986 eq13–15）。`thermo` で検証可。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .units import R, R_BAR_M3


@dataclass(frozen=True)
class Component:
    """成分の物性。Pc は bar, Tc は K。"""
    name: str
    MW: float      # g/mol
    Tc: float      # K
    Pc: float      # bar
    omega: float   # 偏心因子
    source: str


# 一般用の臨界物性（chemicals 1.5.1 由来, bar 換算・検証済）。
#   カルボニル化(CO フガシティ)・一般計算はこちらを使用。
COMPONENTS: dict[str, Component] = {
    #                MW       Tc[K]     Pc[bar]  omega
    "CO":   Component("CO",   28.010, 132.860,  34.940,  0.0497, "chemicals"),
    "CO2":  Component("CO2",  44.010, 304.128,  73.773,  0.2239, "chemicals"),
    "H2":   Component("H2",    2.016,  33.145,  12.964, -0.2190, "chemicals"),  # 実H2(※合成では使わない)
    "H2O":  Component("H2O",  18.015, 647.096, 220.640,  0.3443, "chemicals"),
    "CH3OH":Component("CH3OH",32.042, 513.380,  82.159,  0.5625, "chemicals"),
    "DME":  Component("DME",  46.068, 400.378,  53.368,  0.1960, "chemicals"),
    "MA":   Component("MA",   74.079, 506.500,  47.500,  0.3200, "chemicals"),  # 酢酸メチル
    "N2":   Component("N2",   28.013, 126.192,  33.958,  0.0372, "chemicals"),
}

# メタノール合成の SRK 用（Graaf 1986 Table 2・原著再現）。
#   ⚠️ H2 は "effective"（Graboski–Daubert）。実H2ではないので必ずこちらを使うこと。
#   値: (Pc[bar], Tc[K], omega)
GRAAF1986_CRITICALS: dict[str, tuple[float, float, float]] = {
    "CO":    (35.0,  132.9, 0.049),
    "CO2":   (73.8,  304.2, 0.255),
    "H2":    (20.5,   43.6, 0.000),   # ← effective H2 (NOT real H2: Tc=33K, ω=-0.22)
    "H2O":   (220.5, 647.3, 0.344),
    "CH3OH": (81.0,  512.6, 0.572),
}


def _criticals(species: str, source: str) -> tuple[float, float, float]:
    """(Tc[K], Pc[bar], omega) を返す。source='chemicals' / 'graaf1986'。
    graaf1986 に無い成分(不活性等)は chemicals にフォールバック。"""
    if source == "graaf1986" and species in GRAAF1986_CRITICALS:
        Pc, Tc, omega = GRAAF1986_CRITICALS[species]
        return Tc, Pc, omega
    c = COMPONENTS[species]
    return c.Tc, c.Pc, c.omega


class GasState:
    """気相の熱力学状態。T[K], P[bar], y=モル分率(dict)。"""

    def __init__(self, T: float, P: float, y: dict[str, float]):
        self.T = float(T)
        self.P = float(P)
        self.y = dict(y)

    # --- 駆動量（レート則が消費） ---
    def partial_pressures(self) -> dict[str, float]:
        """pᵢ = yᵢ·P [bar]（VBF/KOGAS 用）。"""
        return {s: yi * self.P for s, yi in self.y.items()}

    def concentrations(self, unit: str = "kmol/m3") -> dict[str, float]:
        """Cᵢ = yᵢ·P/(R·T)。既定 kmol/m³（Bercič–Levec 脱水 r_MD 用）。unit='mol/m3' も可。"""
        # C[mol/m³] = pᵢ[Pa]/(R·T) = (yᵢ·P[bar]·1e5)/(R·T)
        c_mol_m3 = {s: (yi * self.P * 1.0e5) / (R * self.T) for s, yi in self.y.items()}
        if unit == "mol/m3":
            return c_mol_m3
        if unit == "kmol/m3":
            return {s: c / 1.0e3 for s, c in c_mol_m3.items()}
        raise ValueError(f"unknown unit: {unit!r}")

    def fugacity_coeffs(self, criticals: str = "chemicals") -> dict[str, float]:
        """φᵢ（SRK）。criticals='graaf1986' でメタノール合成用(effective H2)。"""
        return _srk_fugacity_coeffs(self.y, self.T, self.P, criticals)

    def fugacities(self, criticals: str = "chemicals") -> dict[str, float]:
        """fᵢ = φᵢ·yᵢ·P [bar]（Graaf 合成・Cheng 用）。"""
        phi = self.fugacity_coeffs(criticals)
        return {s: phi[s] * self.y[s] * self.P for s in self.y}

    def Z(self, criticals: str = "chemicals") -> float:
        """圧縮係数（SRK 気相根）。"""
        a, b = _srk_pure(self.y, self.T, criticals)
        a_mix, b_mix = _srk_mix(self.y, a, b)
        RT = R_BAR_M3 * self.T
        return _srk_Z(a_mix * self.P / RT ** 2, b_mix * self.P / RT)


# --- SRK EOS 内部（Graaf 1986 eq13–15, 相互作用係数なし） ---
def _srk_ab(Tc: float, Pc: float, omega: float, T: float) -> tuple[float, float]:
    """純成分の a(T)[m⁶·bar/mol²], b[m³/mol]。
    a = 0.42748·R²·Tc²/Pc·α(T),  b = 0.08664·R·Tc/Pc,  α=[1+m(1-√Tr)]²,  m=0.480+1.574ω-0.176ω²。"""
    m = 0.480 + 1.574 * omega - 0.176 * omega ** 2
    alpha = (1.0 + m * (1.0 - (T / Tc) ** 0.5)) ** 2
    a = 0.42748 * R_BAR_M3 ** 2 * Tc ** 2 / Pc * alpha
    b = 0.08664 * R_BAR_M3 * Tc / Pc
    return a, b


def _srk_pure(y: dict, T: float, criticals: str) -> tuple[dict, dict]:
    """各成分の a_i, b_i。"""
    a, b = {}, {}
    for s in y:
        Tc, Pc, omega = _criticals(s, criticals)
        a[s], b[s] = _srk_ab(Tc, Pc, omega, T)
    return a, b


def _srk_mix(y: dict, a: dict, b: dict) -> tuple[float, float]:
    """混合則: a_mix=ΣΣ yᵢyⱼ√(aᵢaⱼ)=(Σ yᵢ√aᵢ)²,  b_mix=Σ yᵢbᵢ（Graaf 1986 eq14,15）。"""
    a_mix = sum(yi * a[s] ** 0.5 for s, yi in y.items()) ** 2
    b_mix = sum(yi * b[s] for s, yi in y.items())
    return a_mix, b_mix


def _srk_Z(A: float, B: float) -> float:
    """Z³ − Z² + (A−B−B²)Z − AB = 0 の気相(最大実)根。"""
    roots = np.roots([1.0, -1.0, A - B - B ** 2, -A * B])
    real = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > B]
    return max(real)


def _srk_fugacity_coeffs(y: dict, T: float, P: float, criticals: str) -> dict[str, float]:
    """SRK フガシティ係数 φᵢ。相互作用係数=0 なので Σⱼyⱼ√(aᵢaⱼ)=√aᵢ·√a_mix。
    ln φᵢ = (bᵢ/b)(Z−1) − ln(Z−B) − (A/B)(2√(aᵢ/a_mix) − bᵢ/b)·ln(1 + B/Z)。"""
    a, b = _srk_pure(y, T, criticals)
    a_mix, b_mix = _srk_mix(y, a, b)
    RT = R_BAR_M3 * T
    A = a_mix * P / RT ** 2
    B = b_mix * P / RT
    Z = _srk_Z(A, B)
    phi = {}
    for s in y:
        term = 2.0 * (a[s] / a_mix) ** 0.5 - b[s] / b_mix
        ln_phi = (b[s] / b_mix) * (Z - 1.0) - np.log(Z - B) - (A / B) * term * np.log(1.0 + B / Z)
        phi[s] = float(np.exp(ln_phi))
    return phi
