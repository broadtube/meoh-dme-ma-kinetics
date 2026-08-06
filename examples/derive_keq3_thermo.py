"""K_eq3("thermo") = 10^(1121/T − 0.888) の導出（Cantera NASA 熱力学からの再現）。

脱水平衡  2 CH3OH ⇌ CH3OCH3(DME) + H2O   （Δn=0 → 圧力非依存）
の平衡定数を、Cantera 同梱 nasa_gas.yaml（NASA-7 多項式 = Gordon & McBride /
NASA Glenn の標準熱力学）の標準ギブスエネルギーから計算し、
    log10 K = c0/T + c1
の 2 係数に最小二乗フィットする。vbf_kogas.py の K_EQ3_SOURCES["thermo"] =
(1121.0, -0.888) はこの手順で得た値（脱水運転域 250–315℃ でのフィット）。

要 cantera: pip install cantera
実行: python3 examples/derive_keq3_thermo.py
"""
import cantera as ct
import numpy as np

# --- NASA 熱力学（nasa_gas.yaml）から CH3OH / CH3OCH3 / H2O を取り出す ---
SPECIES = ("CH3OH", "CH3OCH3", "H2O")
NU = {"CH3OH": -2, "CH3OCH3": 1, "H2O": 1}          # 2CH3OH ⇌ DME + H2O
_sp = [s for s in ct.Species.list_from_file("nasa_gas.yaml") if s.name in SPECIES]
_gas = ct.Solution(thermo="ideal-gas", species=_sp)
_idx = {n: _gas.species_index(n) for n in SPECIES}


def log10_Keq3(T: float) -> float:
    """log10 K(T)。標準ギブス（理想気体・1 bar 基準, T のみで決まる）から。"""
    _gas.TPX = T, 1.0e5, {n: 1.0 for n in SPECIES}
    g_RT = _gas.standard_gibbs_RT                    # g°_k/(RT)
    dG_RT = sum(NU[n] * g_RT[_idx[n]] for n in SPECIES)   # ΔG°_rxn/(RT)
    return -dG_RT / np.log(10.0)                     # ln K = −ΔG°/RT


def fit(T_lo_C: float = 250.0, T_hi_C: float = 315.0, n: int = 100):
    """log10 K = c0/T + c1 を [T_lo, T_hi]℃ で最小二乗フィット → (c0, c1)。"""
    Ts = np.linspace(T_lo_C + 273.15, T_hi_C + 273.15, n)
    y = np.array([log10_Keq3(T) for T in Ts])
    A = np.vstack([1.0 / Ts, np.ones_like(Ts)]).T
    (c0, c1), *_ = np.linalg.lstsq(A, y, rcond=None)
    return c0, c1


def main():
    c0, c1 = fit()
    print("=== K_eq3('thermo') 導出（Cantera nasa_gas.yaml, 2CH3OH⇌DME+H2O）===")
    print(f"フィット窓: 250–315℃")
    print(f"  再現値      : log10 K = {c0:.1f}/T + ({c1:.3f})")
    print(f"  コード実装値: log10 K = 1121.0/T + (-0.888)   [vbf_kogas K_EQ3_SOURCES['thermo']]")
    print()
    print(f"{'T[°C]':>6} {'K(NASA厳密)':>12} {'K(fit)':>10} {'K(code)':>10}")
    for Tc in (140, 190, 250, 300, 350):
        T = Tc + 273.15
        K_exact = 10 ** log10_Keq3(T)
        K_fit = 10 ** (c0 / T + c1)
        K_code = 10 ** (1121.0 / T - 0.888)
        print(f"{Tc:>6} {K_exact:>12.3f} {K_fit:>10.3f} {K_code:>10.3f}")
    print("\n注: 250℃で K≈18（Bercič–Levec 1992 の K=7–11@290–360℃ 系とも整合）。")


if __name__ == "__main__":
    main()
