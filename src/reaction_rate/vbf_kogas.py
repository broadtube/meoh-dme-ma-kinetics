"""§2  Vanden Bussche–Froment & KOGAS（rate_equations.html §2 に対応）。

反応:
  MS  : CO2 + 3H2 ⇌ CH3OH + H2O   (メタノール合成)
  RWGS: CO  + H2O ⇌ CO2 + H2      (逆水性ガスシフト)
  MD  : 2 CH3OH  ⇌ CH3OCH3 + H2O  (メタノール脱水, Bercič–Levec)

速度式（MS/RWGS は分圧[bar]基準・速度[mol·kg⁻¹·s⁻¹]）:
  r_MS   = k1·pH2·pCO2·[1−(1/Keq1)(pCH3OH·pH2O)/(pCO2·pH2³)] / DEN³
  r_RWGS = k5·pCO2·[1−Keq2·(pCO·pH2O)/(pCO2·pH2)] / DEN
  DEN    = 1 + K2·(pH2O/pH2) + K3·√pH2 + K4·pH2O
    ⚠️ 第3項は K3·√pH2（K3 は VBF の √K_H2 に相当する"√pH2の係数"。K3=0.37, B=17197）。
       原論文は √(K3·pH2) と印字するが、それだと B が半分になり VBF と不整合。
  r_MD   = k6·K_CH3OH²·[C_CH3OH² − C_H2O·C_DME/Keq3] / (1 + 2√(K_CH3OH·C_CH3OH) + K_H2O·C_H2O)⁴
    ⚠️ 濃度 C は kmol/m³（state.concentrations の既定）。速度[mol·kg⁻¹·s⁻¹]。

出所: KOGAS(2008/2021) の値は Ng, Chadwick & Toseland 1999 Table 1 と完全一致
      （VBF の B(1–5) ＋ Bercič–Levec の脱水を Ng が自データにフィット）。

⚠️ 速度の単位は原著ごとに異なる（下の換算係数で mol·kg⁻¹·s⁻¹ に統一する）:
      VBF 1996 (p6)          : mol·kg_cat⁻¹·s⁻¹        → ×1（換算不要）
      Ng 1999 (p6, =KOGAS)   : mol·g_cat⁻¹·h⁻¹         → ×1000/3600
      Bercič–Levec 1993 (p6) : kmol·kg_cat⁻¹·h⁻¹       → ×1000/3600
    k1,k5,k6 が速度の[質量·時間]次元を背負う（DEN の吸着定数は無次元で単位に無関係）。
"""
from __future__ import annotations
import math

from .units import R

# --- MS/RWGS パラメータ（A·exp(B/RT), R=8.314）。model= で選択 ---
#   B は VBF と KOGAS で一致、A のみ相違。KOGAS=Ng=デフォルト。
MS_PARAMS = {
    "KOGAS": {  # = Ng 1999 = KOGAS 2008/2021
        "k1": (1.65,     36696.0),   # MS 速度係数
        "K2": (3610.0,       0.0),   # K_H2O/(K8K9K_H2)
        "K3": (0.37,     17197.0),   # √pH2 の係数(= VBF の √K_H2)
        "K4": (7.14e-11, 124119.0),  # K_H2O [bar^-1]
        "k5": (1.09e10,  -94765.0),  # RWGS 速度係数
    },
    "VBF": {    # Vanden Bussche–Froment 1996 原著
        "k1": (1.07,     36696.0),
        "K2": (3453.38,      0.0),
        "K3": (0.499,    17197.0),
        "K4": (6.62e-11, 124119.0),
        "k5": (1.22e10,  -94765.0),
    },
}

# MS/RWGS 速度の単位 → mol·kg_cat⁻¹·s⁻¹ への換算（k1,k5 が背負う次元。§冒頭 docstring 参照）
#   VBF=mol/(kg·s)→×1,  KOGAS(=Ng)=mol/(g·h)→×1000/3600
_MS_UNIT_TO_MOL_KG_S = {"VBF": 1.0, "KOGAS": 1000.0 / 3600.0}

# --- 脱水(MD) パラメータ。source= で選択 ---
MD_PARAMS = {
    "KOGAS": {  # = Ng 1999（KOGAS 掲載値）, C in kmol/m³, K in m³/kmol
        "k6":      (3.7e10,  -105000.0),
        "K_CH3OH": (7.9e-4,    70500.0),
        "K_H2O":   (0.084,     41100.0),
    },
    # Bercič–Levec 1993 原著(intrinsic eq1)は exp(B'/T) 形式（B'[K], 下記は B=B'·R に換算）:
    "BercicLevec1993": {  # k_s, K_M, K_w  (KOGAS/Ng とは不一致: 脱水速度が 6〜16倍差)
        "k6":      (5.35e13, -17280.0 * 8.314),
        "K_CH3OH": (5.39e-4,   8487.0 * 8.314),
        "K_H2O":   (8.47e-2,   5070.0 * 8.314),
    },
}

# 脱水(MD) 速度の単位 → mol·kg_cat⁻¹·s⁻¹（k6 が背負う次元。両源とも同係数だが理由は別）
#   Ng 1999(=KOGAS): mol/(g·h)→×1000/3600,  BL 1993: kmol/(kg·h)→×1000/3600
_MD_UNIT_TO_MOL_KG_S = {"KOGAS": 1000.0 / 3600.0, "BercicLevec1993": 1000.0 / 3600.0}

# --- 合成/WGS の平衡定数（KOGAS 2008 eq3a–3b = Twigg(1986)/Stull(1969)） ---
#   log10 K_eq1 = 3066/T − 10.592
#   log10 (1/K_eq2) = −2073/T + 2.029    → K_eq2 = 10^(2073/T − 2.029)
EQ_KOGAS = {
    "K_eq1": (3066.0, -10.592),
    "K_eq2_inv": (-2073.0, 2.029),
}

# --- 脱水 2CH3OH⇌DME+H2O の平衡定数 K_eq3 = 10^(c0/T + c1)（3種から選択） ---
#   物理的な値は ~10¹オーダー（523Kで ~18）。KOGAS だけが桁違いに過大。
K_EQ3_SOURCES = {
    "KOGAS":  (10194.0, -13.91),   # KOGAS 2008/2021 ⚠️過大（523Kで10^5.6 → 脱水がほぼ不可逆扱い）
    "BL":     (1000.0,   -0.735),  # Bercič–Levec 1992 記載範囲 K=7〜11(290–360°C) への2点fit
    "thermo": (1121.0,   -0.888),  # Cantera(NASA熱力学)由来のfit（K≈18@250°C, BL とほぼ一致）
}


def K_eq3(T: float, source: str = "KOGAS") -> float:
    """脱水 2CH3OH⇌DME+H2O の平衡定数。source='KOGAS'/'BL'/'thermo'。"""
    c0, c1 = K_EQ3_SOURCES[source]
    return 10.0 ** (c0 / T + c1)


def _arrhenius(param: tuple, T: float) -> float:
    """κ = A·exp(B/RT)。"""
    A, B = param
    return A * math.exp(B / (R * T))


def equilibrium(T: float, k_eq3: str = "KOGAS") -> dict[str, float]:
    """K_eq1(MS), K_eq2(RWGS の駆動力係数), K_eq3(MD) を返す。k_eq3='KOGAS'/'BL'/'thermo'。"""
    c0, c1 = EQ_KOGAS["K_eq1"]
    K_eq1 = 10.0 ** (c0 / T + c1)
    d0, d1 = EQ_KOGAS["K_eq2_inv"]           # log10(1/K_eq2) = d0/T + d1
    K_eq2 = 10.0 ** (-(d0 / T + d1))
    return {"K_eq1": K_eq1, "K_eq2": K_eq2, "K_eq3": K_eq3(T, k_eq3)}


def rate_ms_rwgs(state, model: str = "KOGAS") -> dict[str, float]:
    """r_MS, r_RWGS [mol·kg⁻¹·s⁻¹]。model='KOGAS'(=Ng, 既定)/'VBF'。分圧基準。"""
    T = state.T
    p = state.partial_pressures()
    Keq = equilibrium(T)
    par = MS_PARAMS[model]

    k1 = _arrhenius(par["k1"], T)
    K2 = _arrhenius(par["K2"], T)
    K3 = _arrhenius(par["K3"], T)
    K4 = _arrhenius(par["K4"], T)
    k5 = _arrhenius(par["k5"], T)

    p_CO, p_CO2, p_H2 = p["CO"], p["CO2"], p["H2"]
    p_H2O, p_MeOH = p["H2O"], p["CH3OH"]

    # 共通分母（第3項は K3·√pH2）
    DEN = 1.0 + K2 * (p_H2O / p_H2) + K3 * p_H2 ** 0.5 + K4 * p_H2O

    r_MS = (k1 * p_H2 * p_CO2
            * (1.0 - (1.0 / Keq["K_eq1"]) * (p_MeOH * p_H2O) / (p_CO2 * p_H2 ** 3))
            / DEN ** 3)
    r_RWGS = (k5 * p_CO2
              * (1.0 - Keq["K_eq2"] * (p_CO * p_H2O) / (p_CO2 * p_H2))
              / DEN)
    f = _MS_UNIT_TO_MOL_KG_S[model]          # 源の速度単位 → mol·kg⁻¹·s⁻¹
    return {"r_MS": r_MS * f, "r_RWGS": r_RWGS * f}


def rate_dehydration(state, source: str = "KOGAS", k_eq3: str = "KOGAS") -> float:
    """r_MD [mol·kg⁻¹·s⁻¹]。source=速度定数('KOGAS'/'BercicLevec1993')、k_eq3=平衡定数('KOGAS'/'BL'/'thermo')。濃度[kmol/m³]基準。"""
    T = state.T
    C = state.concentrations(unit="kmol/m3")
    Keq3 = K_eq3(T, k_eq3)
    par = MD_PARAMS[source]

    k6 = _arrhenius(par["k6"], T)
    K_M = _arrhenius(par["K_CH3OH"], T)
    K_W = _arrhenius(par["K_H2O"], T)

    C_M, C_W, C_D = C["CH3OH"], C["H2O"], C["DME"]
    num = k6 * K_M ** 2 * (C_M ** 2 - C_W * C_D / Keq3)
    den = (1.0 + 2.0 * (K_M * C_M) ** 0.5 + K_W * C_W) ** 4
    return (num / den) * _MD_UNIT_TO_MOL_KG_S[source]   # 源の速度単位 → mol·kg⁻¹·s⁻¹
