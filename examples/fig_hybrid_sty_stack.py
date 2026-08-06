"""ハイブリッド床（合成＋脱水、MA 床なし）の出口 STY を、供給 CO₂ 濃度で比較。

fig_tandem_sty_stack_zsm5.py の派生。条件:
  - 260℃ / 51 bar / SV 15000/h、触媒 合成:脱水 = 体積 1:0.9、合計 1.0 mL（MA 床なし）
  - 供給 H₂:CO は 50:15 固定で、CO₂ を N₂ と入替: CO₂ = ~0 / 5 / 10 / 20 / 30 / 35 %
      ~0% は微量 0.1%(1000 ppm)。残りは CO₂+N₂=35 になるよう N₂ で調整。
      （CO₂>35% は N₂ が尽き H₂:CO=50:15 を保てないので扱わない。）

積み上げは元例に倣い MA を抜いた **CO₂(net)・DME・CH₃OH**（下→上, CH₃OH を最上段）。
STY = net生成量[g/h] / 全触媒体積[L]。**入口より減った成分は STY が負**になり、
その場合は 0 より下に描く（符号を分けて積む）。副生 H₂O は元例同様に非表示（数値は出力）。

⚠️ VBF/KOGAS のメタノール合成は CO₂ 基準（CO₂+3H₂→CH₃OH+H₂O）＋WGS。CO₂ 完全ゼロだと
   反応が立ち上がらず全ゼロになる（CO₂ 基準機構の性質）。ので微量 CO₂ を混ぜる。
   ⚠️ ただし CO₂ は WGS で再生され触媒的に効くため微量でも活性がほぼ立ち上がり、"~0%" の
      値は混入量に強く依存する（0.01%→Σ259, 0.1%→Σ920, 1%→Σ1628）。あくまで代表値。

実行: PYTHONPATH=src python3 examples/fig_hybrid_sty_stack.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate import plots

# ── 条件 ─────────────────────────────────────────────
T = 260.0 + 273.15                       # 260℃
P = 5.0e6 / 1e5 + 1.01325                # 51.0 bar
SV = 15000.0
RHO_SYN, RHO_DEH = 1300.0, 800.0
V_HYB = 1.0e-6                           # ハイブリッド床 1.0 mL（MA 床なし）
f_syn = 1.0 / 1.9                        # 合成の体積分率（合成:脱水 = 1:0.9）
m_syn = RHO_SYN * V_HYB * f_syn
m_deh = RHO_DEH * V_HYB * (1.0 - f_syn)
V_TOT = V_HYB
N = SV / 3600.0 * V_TOT / 22.414e-3
V_STY_L = V_TOT * 1e3                     # = 1.0 mL = 0.001 L

# 供給ケース（ラベル = CO₂ mol%）。
#   CO₂=0 は CO₂ 基準機構が立ち上がらず全ゼロになるため、微量 0.1%(1000 ppm) を混ぜる。
#   ※CO₂ は WGS で再生され触媒的に効くので、微量でも活性はほぼ立ち上がる（感度大）。
CO2_TRACE = 0.001                        # 0.1% = 1000 ppm（"~0%" の代表トレース）
CASES = [
    ("~0%",  {"H2": 0.50, "CO": 0.15, "CO2": CO2_TRACE, "N2": 0.35 - CO2_TRACE}),
    ("5%",   {"H2": 0.50, "CO": 0.15, "CO2": 0.05, "N2": 0.30}),
    ("10%",  {"H2": 0.50, "CO": 0.15, "CO2": 0.10, "N2": 0.25}),
    ("20%",  {"H2": 0.50, "CO": 0.15, "CO2": 0.20, "N2": 0.15}),
    ("30%",  {"H2": 0.50, "CO": 0.15, "CO2": 0.30, "N2": 0.05}),
    ("35%",  {"H2": 0.50, "CO": 0.15, "CO2": 0.35, "N2": 0.00}),
]

ORDER = ["CO2", "DME", "CH3OH"]          # 積み上げ 下→上（CO₂ 最下段, CH₃OH 最上段）
COLORS = {"CO2": plots.OKABE_ITO[4], "CH3OH": plots.OKABE_ITO[5], "DME": plots.OKABE_ITO[2]}
MW = {"DME": 46.07, "CH3OH": 32.04, "CO2": 44.01, "H2O": 18.015}


def sty_for(y):
    f = {s: y.get(s, 0.0) * N for s in ("CO", "CO2", "H2", "N2")}
    f.update({"H2O": 0.0, "CH3OH": 0.0, "DME": 0.0})
    o = pfr(f, T, P, CatalystBed({"synthesis": m_syn, "dehydration": m_deh}),
            models={"synthesis": "KOGAS", "dehydration": "ZSM5"}, k_eq3="thermo").outlet()
    sty = {s: (o[s] - f[s]) * MW[s] * 3600.0 / V_STY_L for s in ORDER}
    sty["H2O"] = (o["H2O"] - f["H2O"]) * MW["H2O"] * 3600.0 / V_STY_L
    conv = {s: (f[s] - o[s]) / f[s] * 100.0 if f[s] > 0 else 0.0 for s in ("CO", "H2")}
    return sty, conv


def main():
    data = [(lab, *sty_for(y)) for lab, y in CASES]
    tops = [sum(max(d[1][s], 0.0) for s in ORDER) for d in data]     # 正側の積み上げ高さ
    bots = [sum(min(d[1][s], 0.0) for s in ORDER) for d in data]     # 負側
    ymax = max(tops + [1e-9]); ymin = min(bots + [0.0])

    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=130)
    xs = list(range(len(data)))
    for xi, (lab, sty, conv) in zip(xs, data):
        pos_b, neg_b = 0.0, 0.0
        for s in ORDER:                                   # CO2 → DME → CH3OH
            h = sty[s]
            b = pos_b if h >= 0 else neg_b
            ax.bar(xi, h, width=0.6, bottom=b, color=COLORS[s],
                   edgecolor="white", linewidth=2.0)
            if abs(h) > 0.045 * ymax:                      # 太い→中央に白文字
                ax.text(xi, b + h / 2, f"{h:.0f}", ha="center", va="center",
                        color="white", fontsize=8.5, fontweight="bold")
            elif abs(h) > 0.5:                             # 小→右側に濃色（隠れ防止）
                ax.text(xi + 0.33, b + h / 2, f"{h:.0f}", ha="left", va="center",
                        color=plots._INK, fontsize=7.5)
            if h >= 0:
                pos_b += h
            else:
                neg_b += h
        total = sum(sty[s] for s in ORDER)
        ax.text(xi, pos_b + 0.02 * ymax, f"Σ {total:.0f}", ha="center", va="bottom",
                color=plots._INK, fontsize=10, fontweight="bold")
    ax.text(0.02, 0.02, "~0% = trace 0.1% CO2 (else no reaction; value is trace-sensitive)",
            transform=ax.transAxes, color=plots._MUTED, fontsize=7.5, va="bottom")

    ax.axhline(0.0, color=plots._MUTED, lw=1.0)
    ax.set_ylabel("STY  [g/(L·h)]  (per 1.0 mL total-catalyst)", color=plots._INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([d[0] for d in data], fontsize=11, color=plots._INK)
    ax.set_xlabel("Feed CO2 [mol%]  (H2:CO=50:15 fixed, rest N2)", color=plots._INK, fontsize=9.5)
    ax.set_xlim(-0.7, len(data) - 0.3)
    ax.set_ylim(min(ymin * 1.15, 0.0), ymax * 1.14)
    ax.set_title("Outlet-product STY vs feed CO2 — hybrid bed (no MA)\n(260°C, 51 bar, SV 15000/h)",
                 color=plots._INK, fontsize=10)
    handles = [Patch(facecolor=COLORS[s], edgecolor="white", label=s) for s in reversed(ORDER)]
    ax.legend(handles=handles, frameon=False, labelcolor=plots._INK, fontsize=9,
              title="g/(L·h)", title_fontsize=9, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=plots._MUTED, labelcolor=plots._INK)
    ax.yaxis.grid(True, color=plots._GRID, lw=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    print("saved", plots.save(fig, "hybrid_sty_stack.png"))
    print(f"260℃, 51 bar, SV 15000/h, 触媒 1.0 mL（合成{m_syn*1e3:.3f}g+脱水{m_deh*1e3:.3f}g）")
    for lab, sty, conv in data:
        seg = "  ".join(f"{s}={sty[s]:+.1f}" for s in ORDER)
        print(f"  CO2={lab:>3}: {seg}  Σ={sum(sty[s] for s in ORDER):7.1f}  "
              f"(副生H2O={sty['H2O']:.1f}, CO転化{conv['CO']:.0f}%/H2転化{conv['H2']:.0f}%)")


if __name__ == "__main__":
    main()
