"""単位・定数（rate_equations.html 冒頭の凡例に相当）。

方針: 各レート則は内部で使う単位を固定・明記し、境界でここの関数を使って変換する
（今回最大の落とし穴だった bar/Pa・mol/kmol・mol/m³ vs kmol/m³ をここに集約）。
"""

# 気体定数
R = 8.314          # J mol^-1 K^-1  (Graaf/VBF/KOGAS/Bercic-Levec が使う値)
R_BAR_M3 = 8.314e-5  # m^3 bar mol^-1 K^-1  (C = p/(R_bar_m3 · T) 用)

# --- 圧力 ---
def bar_to_Pa(p_bar: float) -> float:
    return p_bar * 1.0e5

def Pa_to_bar(p_Pa: float) -> float:
    return p_Pa / 1.0e5

# --- 物質量 ---
def mol_to_kmol(n: float) -> float:
    return n / 1.0e3

def kmol_to_mol(n: float) -> float:
    return n * 1.0e3

# TODO(実装フェーズ): 濃度変換 C[mol/m3] = p[Pa]/(R·T), C[kmol/m3]=C[mol/m3]/1000 等の
#   ヘルパをここへ（state.GasState.concentrations から利用）。
