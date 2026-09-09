"""ZSM-5 メタノール脱水（Ortega 2018）を Aspen Plus の LHHW 形にして再検証する。

Aspen Plus (Reactions / LHHW):
    r = [kinetic factor] × [driving force] / [adsorption expression]
    k = k₀·T^n·exp(−E/RT),  DF = K₁Π[Cᵢ]^α − K₂Π[Cᵢ]^β,  ADS = [ΣK_k Π[Cᵢ]^ν]^m
    ln K = A + B/T + C·lnT + D·T

──────────────────────────────────────────────────────────────────────
結論を先に
──────────────────────────────────────────────────────────────────────
**フィットは不要。Ortega Eq.(16) は Aspen の LHHW 形に厳密に書き換えられる**
（`aspen_lhhw.ORTEGA_EXACT`）。鍵は Aspen が**吸着項の指数に実数（0.5）を指定できる**こと:
    ・k と K_M をまとめると単一 Arrhenius になり E = E_app + ΔH_M = **39.0 kJ/mol**
    ・分子分母に p_M を掛けると駆動力は依頼どおり **K_f·p_M² − K_b·p_W·p_D** になり、
      吸着項が (√p_M + a·p_M + b·√p_M·p_W)² となる（**負の指数は出ない**）
本スクリプトの検証では倍精度（〜1e-15）で元の実装と一致し、Fig.4/6/7 の再現も**完全に同じ**。

指数を実数にできない運用向けに、依頼された素の形
    r = k·exp(−E/RT)·(p_M² − p_W p_D/K_eq) / (1 + K_M p_M + K_W p_W + K_D p_D)^m
も最小二乗フィットする（`aspen_lhhw.LINEAR_FIT`）。ただし**関数形が本質的に違う**ため
誤差が残る（下の出力参照）。使うなら実験域限定で、外挿は不可。

実行: PYTHONPATH=src python3 examples/ortega_aspen_lhhw.py
"""
import itertools
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ortega_zsm5 as oz                                    # noqa: E402  デジタイズ済データと検証手順

from reaction_rate import aspen_lhhw as al                  # noqa: E402
from reaction_rate import vbf_kogas                         # noqa: E402
from reaction_rate.units import R                           # noqa: E402

#: フィット用グリッド。Ortega 実験域（140–190℃・~1 bar）と
#: タンデム運転域（250℃・全圧 51 bar だが p_MeOH は 0.1–0.5 bar）の両方を含む。
FIT_T = [140.0, 165.0, 190.0, 215.0, 240.0, 260.0]
FIT_PM = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]
FIT_PW = [0.0, 0.05, 0.2, 0.5, 1.0]
FIT_PD = [0.0, 0.05, 0.3, 0.8]


def lhhw_rate(params):
    """(T[℃], p_M, p_W, p_D) → r_MeOH の関数を作る（ortega_zsm5 の検証に差し込む用）。"""
    def f(T_c, p_M, p_W=0.0, p_D=0.0):
        return al.rate(T_c + 273.15, {"CH3OH": p_M, "H2O": p_W, "DME": p_D}, params)
    return f


def fit_linear_form(verbose: bool = True) -> al.LHHW:
    """吸着項を線形に制限した LHHW を Ortega Eq.(16) に最小二乗フィットする。

    駆動力は p_M² − p_W·p_D/K_eq に固定する（K_eq は本プロジェクトの K_eq3("thermo")）。
    こうしておけば平衡点が Ortega と厳密に一致し、**熱力学的整合性が保たれる**
    ——フィットするのは速度の大きさと吸着項だけ。
    """
    c0, c1 = vbf_kogas.K_EQ3_SOURCES["thermo"]
    ln10 = math.log(10.0)
    rows = []
    for T_c, p_M, p_W, p_D in itertools.product(FIT_T, FIT_PM, FIT_PW, FIT_PD):
        T = T_c + 273.15
        if 1.0 - p_D * p_W / (p_M * p_M * 10.0 ** (c0 / T + c1)) < 0.5:
            continue                       # 平衡近傍は両形とも 0 に落ちるので除外
        rows.append((T, p_M, p_W, p_D, oz.rate_meoh(T_c, p_M, p_W, p_D)))
    T, p_M, p_W, p_D, r_ref = np.array(rows).T
    K_eq = 10.0 ** (c0 / T + c1)
    drive = p_M ** 2 - p_W * p_D / K_eq

    def model(q):
        ln_k0, E_kJ, A_M, B_kK_M, A_W, B_kK_W, A_D, B_kK_D, m = q
        kin = np.exp(ln_k0) * np.exp(-E_kJ * 1e3 / (R * T))
        ads = (1.0 + np.exp(A_M + B_kK_M * 1e3 / T) * p_M
               + np.exp(A_W + B_kK_W * 1e3 / T) * p_W
               + np.exp(A_D + B_kK_D * 1e3 / T) * p_D)
        return kin * drive / ads ** m

    resid = lambda q: np.log(np.maximum(model(q), 1e-300)) - np.log(r_ref)
    best = None
    for m0 in (1.6, 2.0, 2.4, 3.0):        # m の初期値を振って局所解を避ける
        q0 = np.array([-6.8, -40.0, -15.0, 9.0, -27.0, 13.0, -6.0, 2.5, m0])
        sol = least_squares(resid, q0, xtol=1e-15, ftol=1e-15, max_nfev=80000)
        if best is None or sol.cost < best.cost:
            best = sol
    ln_k0, E_kJ, A_M, B_kK_M, A_W, B_kK_W, A_D, B_kK_D, m = best.x
    if verbose:
        err = np.exp(np.abs(best.fun)) - 1
        print(f"  グリッド {len(T)} 点  RMS(相対) {np.sqrt(np.mean(best.fun ** 2)) * 100:.2f}%  "
              f"平均 {err.mean() * 100:.2f}%  最大 {err.max() * 100:.1f}%")
    return al.LHHW(
        name="Ortega2018-linear-fit",
        k0=math.exp(ln_k0), E=E_kJ * 1e3,
        forward=al.Term(exponents={"CH3OH": 2.0}),
        reverse=al.Term(A=-ln10 * c1, B=-ln10 * c0, exponents={"H2O": 1.0, "DME": 1.0}),
        adsorption=(al.Term(),
                    al.Term(A=A_M, B=B_kK_M * 1e3, exponents={"CH3OH": 1.0}),
                    al.Term(A=A_W, B=B_kK_W * 1e3, exponents={"H2O": 1.0}),
                    al.Term(A=A_D, B=B_kK_D * 1e3, exponents={"DME": 1.0})),
        m=m)


def verify(rate_fn, label: str) -> dict:
    """ortega_zsm5 と同じ Fig.4 / Fig.6 / Fig.7 / SI アンカー検証を rate_fn で行う。"""
    dev = {"Fig.4": [], "Fig.6": [], "Fig.7": []}
    for T, row in oz.FIG4.items():
        for p_M, r in row:
            dev["Fig.4"].append((T, oz.rate_at_outlet(T, p_M, rate_fn=rate_fn) / r - 1.0))
    for T, p_M, r in oz.FIG6:
        dev["Fig.6"].append((T, oz.rate_at_outlet(T, p_M, rate_fn=rate_fn) / r - 1.0))
    for T, r in oz.FIG7:
        dev["Fig.7"].append((T, oz.rate_wet_feed(T, rate_fn=rate_fn)[0] / r - 1.0))
    si = (oz.rate_at_outlet(oz.SI_ANCHOR["T"], oz.SI_ANCHOR["p_M"], rate_fn=rate_fn)
          / oz.SI_ANCHOR["rate"] - 1.0)
    allv = [abs(d) for v in dev.values() for _, d in v]
    print(f"  [{label}]  SI アンカー {si:+.2%} / 全 25 点 平均 {sum(allv)/len(allv):.1%} "
          f"最大 {max(allv):.1%}")
    for k, v in dev.items():
        e = [abs(d) for _, d in v]
        print(f"      {k}: 平均 {sum(e)/len(e):.1%}  最大 {max(e):.1%}")
    return {"dev": dev, "si": si}


def main():
    print("=== Ortega 2018 (ZSM-5 脱水) → Aspen Plus LHHW 形 ===\n")

    print("■ 形 A: 厳密変換（推奨・フィット不要）")
    print(al.aspen_table(al.ORTEGA_EXACT))
    print("\n  （同値の別表現。駆動力を p_M の 1 次にしたい場合はこちら）")
    print(al.aspen_table(al.ORTEGA_EXACT_NEG))
    worst = 0.0
    for T_c in (140, 165, 190, 215, 250, 300):
        for p_M, p_W, p_D in ((0.95, 0.01, 0.01), (0.3, 0.3, 0.3), (0.12, 0.39, 0.63),
                              (0.05, 1.0, 0.5), (5.0, 2.0, 2.0)):
            a = al.rate(T_c + 273.15, {"CH3OH": p_M, "H2O": p_W, "DME": p_D})
            worst = max(worst, abs(a / oz.rate_meoh(T_c, p_M, p_W, p_D) - 1.0))
    print(f"\n  元実装（vbf_kogas source='ZSM5'）との最大相対差 = {worst:.2e}  → 倍精度で一致\n")

    print("■ 形 B: 依頼された素の形（吸着項は線形のみ）をフィット")
    linear = fit_linear_form()
    print()
    print(al.aspen_table(linear))
    print()

    print("■ Fig.4 / Fig.6 / Fig.7 ＋ SI アンカーによる検証")
    verify(oz.rate_meoh, "元実装 Ortega Eq.(16)")
    verify(lhhw_rate(al.ORTEGA_EXACT), "形 A: LHHW 厳密")
    verify(lhhw_rate(linear), "形 B: LHHW 線形フィット")

    print("\n■ 運転域（タンデム 250℃）での外挿確認")
    print(f"{'条件':<34}{'Eq.(16)':>12}{'形A':>12}{'形B':>12}{'形B誤差':>10}")
    for T_c, p_M, p_W, p_D in ((250, 0.12, 0.39, 0.63), (250, 0.30, 0.30, 0.30),
                               (250, 0.50, 0.20, 0.10), (190, 0.93, 0.04, 0.04)):
        ref = oz.rate_meoh(T_c, p_M, p_W, p_D)
        a = al.rate(T_c + 273.15, {"CH3OH": p_M, "H2O": p_W, "DME": p_D})
        b = al.rate(T_c + 273.15, {"CH3OH": p_M, "H2O": p_W, "DME": p_D}, linear)
        print(f"{T_c}℃ p=({p_M},{p_W},{p_D})".ljust(34)
              + f"{ref:>12.5f}{a:>12.5f}{b:>12.5f}{b/ref-1:>+9.1%}")
    print("\n→ 形 B は原著の点でも平均 20% ずれ、フィット域の外ではさらに外れる")
    print("  （見かけの E が ~0 kJ/mol に落ち、温度依存を吸着項に肩代わりさせているため）。")
    print("  Aspen で指数 0.5 / −1 が指定できるなら形 A を使うこと。")
    return linear


def plot(linear=None):
    """(a) Fig.4 の p_M 依存性で形 A/B を比較 (b) 全点の偏差。"""
    import matplotlib.pyplot as plt
    from reaction_rate import plots

    linear = linear if linear is not None else fit_linear_form(verbose=False)
    f_exact, f_linear = lhhw_rate(al.ORTEGA_EXACT), lhhw_rate(linear)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=130)

    pm = np.linspace(0.25, 1.0, 40)
    plots.lines(pm, {f"{T} °C": [oz.rate_at_outlet(T, p, rate_fn=f_exact) for p in pm]
                     for T in oz.FIG4},
                "Methanol partial pressure [bar]",
                "Rate [mol$_{MeOH}$ kg$^{-1}$ s$^{-1}$]", ax=ax1)
    for i, (T, row) in enumerate(oz.FIG4.items()):
        col = plots.OKABE_ITO[i % len(plots.OKABE_ITO)]
        ax1.plot(pm, [oz.rate_at_outlet(T, p, rate_fn=f_linear) for p in pm],
                 "--", lw=1.4, color=col, zorder=2)
        ax1.plot([p for p, _ in row], [r for _, r in row], "o", ms=5, color=col,
                 mfc="white", mew=1.4, zorder=3)
    ax1.set_ylim(0.0, None)
    ax1.set_title("(a) Fig. 4 — solid: LHHW exact, dashed: LHHW linear fit\n"
                  "circles = digitized from the paper", fontsize=9.5, pad=6)

    for f, col in ((0.20, "#dbe0dd"), (0.10, "#c7d3cd")):
        ax2.axhspan(-f * 100, f * 100, color=col, zorder=0)
    ax2.axhline(0.0, color="#1b242c", lw=1.2, zorder=1)
    for i, (label, fn) in enumerate((("LHHW exact (= Eq. 16)", f_exact),
                                     ("LHHW linear fit", f_linear))):
        pts = []
        for T, row in oz.FIG4.items():
            pts += [(T, oz.rate_at_outlet(T, p, rate_fn=fn) / r - 1.0) for p, r in row]
        pts += [(T, oz.rate_at_outlet(T, p, rate_fn=fn) / r - 1.0) for T, p, r in oz.FIG6]
        pts += [(T, oz.rate_wet_feed(T, rate_fn=fn)[0] / r - 1.0) for T, r in oz.FIG7]
        ax2.plot([T for T, _ in pts], [d * 100 for _, d in pts], "o", ms=6,
                 color=plots.OKABE_ITO[i], mfc="white", mew=1.5, label=label, zorder=3)
    ax2.set_xlabel("Temperature [°C]", color="#1b242c")
    ax2.set_ylabel("(calculated / measured − 1)  [%]", color="#1b242c")
    ax2.set_ylim(-60, 60)
    ax2.legend(frameon=False, labelcolor="#1b242c", fontsize=8, loc="upper left")
    ax2.text(0.98, 0.03, "shaded: ±10% (inner) / ±20% (outer)", transform=ax2.transAxes,
             fontsize=8, color="#5b6873", ha="right", va="bottom")
    ax2.set_title("(b) deviation from the paper (all 25 points)", fontsize=9.5, pad=6)
    plots._style(ax2)

    fig.suptitle("Ortega 2018 as an Aspen Plus LHHW rate expression", fontsize=11)
    fig.tight_layout()
    print("\nsaved", plots.save(fig, "ortega_aspen_lhhw.png"))


if __name__ == "__main__":
    plot(main())
