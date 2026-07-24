"""単管タンデム出口 STY を、脱水触媒 2種で並べて比較する積み上げ棒。

  左: γ-Al2O3  (source="KOGAS")  … 従来の工業触媒。250℃では脱水が速度律速。
  右: ZSM-5    (source="ZSM5")   … 強 Brønsted 酸。Ortega 2018 の厳密 LHHW。

条件は fig_tandem_sty_stack_zsm5.py と同一（RHO_MA=837, STY 基準=全触媒 3.0 mL,
250℃, 51 bar, SV 5000/h）。250℃/51bar では γ-アルミナは脱水しきれず未転化メタノール
（CH3OH）が大量に抜けるのに対し、ZSM-5 はほぼ完全に DME 化し MA まで進む — その差を可視化。

⚠️ ZSM-5(Ortega) のフィット域は 140–190℃・~1 bar。250℃/51bar は外挿だが LHHW は
   飽和して物理的に有界。絶対値は要実験較正。

実行: PYTHONPATH=src python3 examples/fig_tandem_catalyst_compare.py
"""
import numpy as np
import matplotlib.pyplot as plt

from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate import plots

# ── fig_tandem_sty_stack_zsm5 と同一条件 ─────────────────────
T = 523.15
P = 5.0e6 / 1e5 + 1.01325
H2_CO, Y_CO2, SV = 1.5, 0.03, 5000.0
RHO_SYN, RHO_DEH, RHO_MA = 1300.0, 800.0, 837.0
V_HYB, V_MA = 1.0e-6, 2.0e-6
V_TOT = V_HYB + V_MA
f_syn = 1.0 / 1.9
m_syn, m_deh = RHO_SYN * V_HYB * f_syn, RHO_DEH * V_HYB * (1 - f_syn)
m_ma = RHO_MA * V_MA
N = SV / 3600.0 * V_TOT / 22.414e-3

V_STY_L = V_TOT * 1e3          # STY 基準体積 [L] = 全触媒 3.0 mL
MW = {"CO2": 44.01, "CH3OH": 32.04, "DME": 46.07, "MA": 74.08}

ORDER = ["MA", "DME", "CH3OH", "CO2"]                       # 積み上げ順（下→上）
COLORS = {"MA": plots.OKABE_ITO[3], "DME": plots.OKABE_ITO[2],
          "CH3OH": plots.OKABE_ITO[5], "CO2": plots.OKABE_ITO[4]}
CATS = [("KOGAS", "γ-Al2O3\n(KOGAS)"), ("ZSM5", "ZSM-5\n(Ortega LHHW)")]  # 脱水触媒


def feed():
    y_co = (1 - Y_CO2) / (1 + H2_CO)
    return {"CO": y_co * N, "CO2": Y_CO2 * N, "H2": H2_CO * y_co * N,
            "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0, "MA": 0.0}


def sty_for(deh_model):
    f = feed()
    r1 = pfr(f, T, P, CatalystBed({"synthesis": m_syn, "dehydration": m_deh}),
             models={"synthesis": "KOGAS", "dehydration": deh_model}, k_eq3="thermo")
    r2 = pfr(r1.outlet(), T, P, CatalystBed({"carbonylation": m_ma}),
             models={"carbonylation": "DTU-Cheung2007-2"})
    o = r2.outlet()
    return {s: (o[s] - f[s]) * MW[s] * 3600.0 / V_STY_L for s in ORDER}


def main():
    data = {m: sty_for(m) for m, _ in CATS}
    totals = {m: sum(data[m].values()) for m in data}
    ymax = max(totals.values())

    fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=130)
    xs = [0.0, 1.1]
    for x, (m, lab) in zip(xs, CATS):
        sty, total, bottom = data[m], totals[m], 0.0
        for s in ORDER:
            h = sty[s]
            ax.bar(x, h, width=0.62, bottom=bottom,
                   color=COLORS[s], edgecolor="white", linewidth=2.0,
                   label=s if x == xs[0] else None)
            if h > 0.05 * ymax:
                ax.text(x, bottom + h / 2, f"{s}\n{h:.0f}", ha="center", va="center",
                        color="white", fontsize=8.5, fontweight="bold")
            elif h > 0.008 * ymax:
                ax.text(x + 0.36, bottom + h / 2, f"{s} {h:.0f}", ha="left", va="center",
                        color=plots._INK, fontsize=7.5)
            bottom += h
        ax.text(x, bottom + 0.015 * ymax, f"Σ {total:.0f}", ha="center", va="bottom",
                color=plots._INK, fontsize=10, fontweight="bold")

    ax.set_ylabel("STY  [g/(L·h)]  (per 3.0 mL total-catalyst)", color=plots._INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in CATS], color=plots._INK, fontsize=9.5)
    ax.set_xlim(-0.6, 1.9)
    ax.set_ylim(0, ymax * 1.14)
    ax.set_title("Dehydration-catalyst comparison — tandem outlet STY\n(250°C, 51 bar, SV 5000/h)",
                 color=plots._INK, fontsize=10)
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_[::-1], l_[::-1], frameon=False, labelcolor=plots._INK, fontsize=9,
              title="segment", title_fontsize=9, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=plots._MUTED, labelcolor=plots._INK)
    ax.yaxis.grid(True, color=plots._GRID, lw=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    print("saved", plots.save(fig, "tandem_catalyst_compare.png"))
    for m, lab in CATS:
        print(f"[{m:6}]  Σ={totals[m]:.1f}  " +
              "  ".join(f"{s}={data[m][s]:.1f}" for s in ORDER))


if __name__ == "__main__":
    main()
