"""§1  Graaf 1988 — メタノール合成（rate_equations.html §1 に対応）。

反応（Graaf の番号）:
  A: CO  + 2H2 ⇌ CH3OH
  B: CO2 + H2  ⇌ CO + H2O        (逆WGS)
  C: CO2 + 3H2 ⇌ CH3OH + H2O     (= A + B)

速度式（model A3B2C3, フガシティ基準・デュアルサイトLHHW, H2 解離吸着）:
  r_A = k_A·K_CO [f_CO·f_H2^1.5 − f_CH3OH/(f_H2^0.5·K_pA)] / D
  r_B = k_B·K_CO2[f_CO2·f_H2 − f_H2O·f_CO/K_pB] / D
  r_C = k_C·K_CO2[f_CO2·f_H2^1.5 − f_CH3OH·f_H2O/(f_H2^1.5·K_pC)] / D
  D   = (1 + K_CO·f_CO + K_CO2·f_CO2)(f_H2^0.5 + (K_H2O/√K_H2)·f_H2O)
速度 [mol·kg_cat⁻¹·s⁻¹], フガシティ [bar]。フガシティは state.GasState.fugacities(criticals="graaf1986")。
"""
from __future__ import annotations
import math

from .units import R

# --- 速度・吸着パラメータ（A3B2C3, κ = A·exp(B/RT), R=8.314 J/mol/K） ---
#   出所: Graaf 1988（graaf1988.pdf p.2892）の eqs (46)–(51)。前指数・B とも原著と照合済。
#   ※ k は「pseudo」定数 k'_ps（H2 吸着定数を吸収した簡略形 eqs 43–45 用）。
PARAMS = {
    "k_A":   (2.69e7,  -109900.0),   # 反応A 速度   eq(46)
    "k_B":   (7.31e8,  -123400.0),   # 反応B 速度   eq(47)
    "k_C":   (4.36e2,   -65200.0),   # 反応C 速度   eq(48)
    "K_CO":  (7.99e-7,   58100.0),   # [bar^-1]     eq(49)
    "K_CO2": (1.02e-7,   67400.0),   # [bar^-1]     eq(50)
    "K_H2O_over_sqrtK_H2": (4.13e-11, 104500.0),  # [bar^-0.5]  eq(51)
}

# --- 平衡定数（Graaf 1986 原著, log10, pressures in bar; graaf1986.pdf p.2884） ---
#   log10 K_p = c0/T + c1
EQ_GRAAF1986 = {
    "K_pA": (5139.0, -12.621),   # CO+2H2⇌CH3OH   (eq11)
    "K_pB": (-2073.0,  2.029),   # CO2+H2⇌CO+H2O  (eq12)
    "K_pC": (3066.0, -10.592),   # CO2+3H2⇌CH3OH+H2O (= K_pA·K_pB)
}

# --- 平衡定数（Graaf & Winkelman 2016, 高精度オプション） ---
#   ln Kp_j° = (1/RT)[c1 + c2·T + c3·T² + c4·T³ + c5·T⁴ + c6·T⁵ + c7·T·lnT]
#   Kp3° = Kp1°·Kp2°
EQ_GRAAF2016 = {
    "Kp1": (7.44140e4, 1.89260e2, 3.2443e-2, 7.0432e-6, -5.6053e-9, 1.0344e-12, -6.4364e1),  # CO+2H2⇌CH3OH
    "Kp2": (-3.94121e4, -5.41516e1, -5.5642e-2, 2.5760e-5, -7.6594e-9, 1.0161e-12, 1.8429e1),  # CO2+H2⇌CO+H2O
}


def _arrhenius(name: str, T: float) -> float:
    """κ = A·exp(B/RT)。"""
    A, B = PARAMS[name]
    return A * math.exp(B / (R * T))


def _ln_Kp_2016(coeffs: tuple, T: float) -> float:
    c1, c2, c3, c4, c5, c6, c7 = coeffs
    poly = c1 + c2 * T + c3 * T**2 + c4 * T**3 + c5 * T**4 + c6 * T**5 + c7 * T * math.log(T)
    return poly / (R * T)


def equilibrium(T: float, source: str = "Graaf1986") -> dict[str, float]:
    """K_pA, K_pB, K_pC を返す。source='Graaf1986'(既定) / 'Graaf2016'。"""
    if source == "Graaf1986":
        return {name: 10.0 ** (c0 / T + c1) for name, (c0, c1) in EQ_GRAAF1986.items()}
    if source == "Graaf2016":
        Kp1 = math.exp(_ln_Kp_2016(EQ_GRAAF2016["Kp1"], T))
        Kp2 = math.exp(_ln_Kp_2016(EQ_GRAAF2016["Kp2"], T))
        return {"K_pA": Kp1, "K_pB": Kp2, "K_pC": Kp1 * Kp2}
    raise ValueError(f"unknown source: {source!r}")


def rates(state, model: str = "A3B2C3", eq_source: str = "Graaf1986") -> dict[str, float]:
    """r_A, r_B, r_C [mol·kg⁻¹·s⁻¹] を返す。state=GasState。フガシティ基準(effective H2)。"""
    T = state.T
    f = state.fugacities(criticals="graaf1986")
    Kp = equilibrium(T, source=eq_source)

    k_A = _arrhenius("k_A", T)
    k_B = _arrhenius("k_B", T)
    k_C = _arrhenius("k_C", T)
    K_CO = _arrhenius("K_CO", T)
    K_CO2 = _arrhenius("K_CO2", T)
    K_H2O_s = _arrhenius("K_H2O_over_sqrtK_H2", T)

    f_CO, f_CO2, f_H2 = f["CO"], f["CO2"], f["H2"]
    f_H2O, f_MeOH = f["H2O"], f["CH3OH"]

    # 共通分母 D = (1 + K_CO·f_CO + K_CO2·f_CO2)(√f_H2 + (K_H2O/√K_H2)·f_H2O)
    D = (1.0 + K_CO * f_CO + K_CO2 * f_CO2) * (f_H2 ** 0.5 + K_H2O_s * f_H2O)

    r_A = k_A * K_CO * (f_CO * f_H2 ** 1.5 - f_MeOH / (f_H2 ** 0.5 * Kp["K_pA"])) / D
    r_B = k_B * K_CO2 * (f_CO2 * f_H2 - f_H2O * f_CO / Kp["K_pB"]) / D
    r_C = k_C * K_CO2 * (f_CO2 * f_H2 ** 1.5 - f_MeOH * f_H2O / (f_H2 ** 1.5 * Kp["K_pC"])) / D
    return {"r_A": r_A, "r_B": r_B, "r_C": r_C}
