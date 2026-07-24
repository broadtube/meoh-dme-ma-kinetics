"""単管タンデムの出口生成物 STY を積み上げ棒で示す（ZSM-5 脱水・全触媒基準版）。

fig_tandem_sty_stack.py からの変更点:
  1. 脱水触媒を γ-アルミナ(KOGAS/LHHW) → **ZSM-5(可逆2次・高活性)** に変更。
  2. MA(カルボニル化)触媒の充填密度 RHO_MA を 700 → **837 kg/m³**。
  3. STY 基準体積を MA触媒 2.0 mL(V_MA) → **全触媒 3.0 mL(V_TOT)** に変更。

STY(空時収率) = net生成量[g/h] / 基準体積[L]。基準体積 = 全触媒 V_TOT = 3.0 mL。
各セグメント = 主生成物 MA・副生 CH3OH・残 DME・副生 CO2。
CO2 は feed にも 3mol% 含むため net（出口−入口）で計上（他 3 種は feed=0）。

【脱水の反応式はどこが変わるか】
  KOGAS(γ-Al2O3, LHHW):
     r_MD = k6·K_M²·(C_M² − C_W·C_D/Keq3) / (1 + 2·√(K_M·C_M) + K_W·C_W)⁴
  ZSM-5(可逆2次, Fuel2014形):
     r_MD = k(T)·(C_M² − C_W·C_D/Keq3),   k(T)=K0·exp(−Ea/RT), Ea=93.6 kJ/mol
  → 駆動力項 (C_M² − C_W·C_D/Keq3) は両者共通で不変。変わるのは
    ・LHHW 吸着分母 (1+2√(K_M C_M)+K_W C_W)⁴ が消える（=1。水/メタノール吸着阻害なし）
    ・分子の Langmuir 係数 k6·K_M² を単一の Arrhenius k(T) に畳み込む
    つまり「LHHW律速」→「純2次べき乗律速」。高圧では脱水がほぼ平衡律速になる。

⚠️ 値は fig_tandem_labscale.py の仮定（CO2添加/SV基準/触媒別ρ/水阻害なし/
   250℃外挿）を引き継ぐ。ZSM-5 の絶対速度は常圧フィットの高圧外挿で要実験較正。

実行: PYTHONPATH=src python3 examples/fig_tandem_sty_stack_zsm5.py
"""
import numpy as np
import matplotlib.pyplot as plt

from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate import plots

# ── fig_tandem_labscale と同一条件（RHO_MA のみ 700→837）─────
T = 523.15
P = 5.0e6 / 1e5 + 1.01325
H2_CO, Y_CO2, SV = 1.5, 0.03, 5000.0
RHO_SYN, RHO_DEH, RHO_MA = 1300.0, 800.0, 837.0     # ← MA触媒 700→837
V_HYB, V_MA = 1.0e-6, 2.0e-6
V_TOT = V_HYB + V_MA
f_syn = 1.0 / 1.9
m_syn, m_deh = RHO_SYN * V_HYB * f_syn, RHO_DEH * V_HYB * (1 - f_syn)
m_ma = RHO_MA * V_MA
N = SV / 3600.0 * V_TOT / 22.414e-3

V_STY_L = V_TOT * 1e3          # ← STY 基準体積 [L] = 全触媒 3.0 mL = 0.003 L
MW = {"CO2": 44.01, "CH3OH": 32.04, "DME": 46.07, "MA": 74.08}


def feed():
    y_co = (1 - Y_CO2) / (1 + H2_CO)
    return {"CO": y_co * N, "CO2": Y_CO2 * N, "H2": H2_CO * y_co * N,
            "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0, "MA": 0.0}


def outlet():
    f = feed()
    r1 = pfr(f, T, P, CatalystBed({"synthesis": m_syn, "dehydration": m_deh}),
             models={"synthesis": "KOGAS", "dehydration": "ZSM5"}, k_eq3="thermo")  # ← ZSM5
    r2 = pfr(r1.outlet(), T, P, CatalystBed({"carbonylation": m_ma}),
             models={"carbonylation": "DTU-Cheung2007-2"})
    return r2.outlet(), f


def main():
    o, f = outlet()
    # net 生成の STY [g/(L·h)]（積み上げ順 = CH3OH と DME を入替）
    order = ["MA", "DME", "CH3OH", "CO2"]
    # 色は Okabe-Ito パレット内で固定割当（逸脱なし）
    #   MA=朱(旧CO2色) / DME=緑 / CH3OH=空 / CO2=紫
    COLORS = {"MA": plots.OKABE_ITO[3], "DME": plots.OKABE_ITO[2],
              "CH3OH": plots.OKABE_ITO[5], "CO2": plots.OKABE_ITO[4]}
    sty = {s: (o[s] - f[s]) * MW[s] * 3600.0 / V_STY_L for s in order}
    total = sum(sty.values())

    fig, ax = plt.subplots(figsize=(4.6, 6.0), dpi=130)
    x = 0.0
    bottom = 0.0
    for s in order:
        h = sty[s]
        ax.bar(x, h, width=0.5, bottom=bottom,
               color=COLORS[s], edgecolor="white", linewidth=2.0,
               label=f"{s}  ({h:.0f})")
        # セグメント中央に直接ラベル（DME は薄いので上に小さく出す）
        if h > 0.06 * total:
            ax.text(x, bottom + h / 2, f"{s}\n{h:.0f}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        else:
            ax.text(x + 0.30, bottom + h / 2, f"{s} {h:.0f}", ha="left", va="center",
                    color=plots._INK, fontsize=8)
        bottom += h
    ax.text(x, bottom + 0.02 * total, f"Σ {total:.0f}", ha="center", va="bottom",
            color=plots._INK, fontsize=10, fontweight="bold")

    ax.set_ylabel("STY  [g/(L·h)]  (per 3.0 mL total-catalyst)", color=plots._INK)
    ax.set_xlim(-0.6, 1.3)
    ax.set_xticks([])
    ax.set_ylim(0, bottom * 1.12)
    ax.set_title("Outlet-product STY — single-tube tandem (ZSM-5)\n(250°C, 51 bar, SV 5000/h)",
                 color=plots._INK, fontsize=10)
    # 凡例は棒の上→下と同じ並びにする（積み上げは MA が最下段なので反転）
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_[::-1], l_[::-1], frameon=False, labelcolor=plots._INK, fontsize=9,
              title="g/(L·h)", title_fontsize=9, loc="center right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=plots._MUTED, labelcolor=plots._INK)
    ax.yaxis.grid(True, color=plots._GRID, lw=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    print("saved", plots.save(fig, "tandem_sty_stack_zsm5.png"))
    print(f"脱水=ZSM5, RHO_MA={RHO_MA:.0f}, STY 基準 = 全触媒 {V_STY_L*1e3:.1f} mL")
    for s in order:
        note = " (net 出-入)" if s == "CO2" else ""
        print(f"  {s:6s} {sty[s]:7.1f} g/(L·h){note}")
    print(f"  {'Σ':6s} {total:7.1f} g/(L·h)")


if __name__ == "__main__":
    main()
