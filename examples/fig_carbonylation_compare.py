"""carbonylation 比較図: 純カルボニル化 PFR での速度モデル比較（実長さ z 基準）。

DME + CO → 酢酸メチル(MA) の単独反応を PFR に通す。温度依存モデルは 438 K で DTU と
一致するため、差が出る 200 ℃(473 K) で比較。

配色: 色=系統（DTU=青 / Cheung2007系=橙 / Cheng2017系=緑）、線種=温度依存の段階
      （-1=実線 k₁のみ / -2=破線 +K₂ / -3=点線 +K₂K₃）。
      standalone（Cheung2007/Cheng2017, MA阻害なし）は薄い灰で「参考（過大評価）」。

要点:
  ・DTU(青) は 438K 固定なので 200℃ でも遅い（温度依存を入れると速くなる）。
  ・Ea 差: Cheng(88.65, 緑) > Cheung(69.6, 橙)。
  ・段階: -2(+K₂) は高温で MA吸着↓＝阻害弱まり速い。-3(+K₃) は K₃ の ΔH が小さくほぼ -2 と同じ。
  ・standalone(灰) は MA阻害がなく DME を 100% まで暴走 → 反応器(高転化)では過大評価。
    ＝積分条件では DTU系(MA阻害あり)が妥当、standalone は微分条件専用。

条件: feed CO:DME = 0.90:0.10, 全流量 0.05 mol/s, 200℃, 50 bar,
      触媒 3 kg（内径40mm管・ρ_bed=1200・ε=0.40 → 全長 ~2 m）。
⚠️ 200℃ は DTU/Cheung の実験域(150–190℃)をやや外挿。

実行: PYTHONPATH=src python3 examples/fig_carbonylation_compare.py
"""
import numpy as np
import matplotlib.pyplot as plt

from reaction_rate import Geometry, GasState
from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate.network import species_rates
from reaction_rate import plots

FTOT = 0.05
T, P = 473.15, 50.0                       # 200 ℃, 50 bar
W = 3.0
GEOM = Geometry(area=np.pi / 4 * 0.04**2, bulk_density=1200.0, void_fraction=0.40)

C = plots.OKABE_ITO
_GREY = "#8a95a0"
# (model, color, linestyle, lw, label)
SERIES = [
    ("DTU",              C[0], "-",  2.0, "DTU (438K fixed)"),
    ("DTU-Cheung2007-1", C[1], "-",  1.8, "DTU-Cheung2007-1"),
    ("DTU-Cheung2007-2", C[1], "--", 1.8, "DTU-Cheung2007-2"),
    ("DTU-Cheung2007-3", C[1], ":",  2.2, "DTU-Cheung2007-3"),
    ("DTU-Cheng2017-1",  C[2], "-",  1.8, "DTU-Cheng2017-1"),
    ("DTU-Cheng2017-2",  C[2], "--", 1.8, "DTU-Cheng2017-2"),
    ("DTU-Cheng2017-3",  C[2], ":",  2.2, "DTU-Cheng2017-3"),
    ("Cheung2007",       _GREY, "-", 1.0, "Cheung2007 (ref: no MA inhib.)"),
    ("Cheng2017",        _GREY, "--", 1.0, "Cheng2017 (ref: no MA inhib.)"),
]


def feed():
    return {"CO": 0.90 * FTOT, "DME": 0.10 * FTOT, "MA": 0.0}


def ma_rate_profile(result, bed, models):
    y = result.mole_fractions()
    return np.array([
        species_rates(GasState(result.T, result.P, {s: float(y[s][i]) for s in result.F}),
                      bed, models=models).get("MA", 0.0)
        for i in range(len(result.W))
    ])


def main():
    runs = {}
    for name, *_ in SERIES:
        bed = CatalystBed({"carbonylation": W})
        models = {"carbonylation": name}
        runs[name] = (pfr(feed(), T, P, bed, models=models), bed, models)
    z = next(iter(runs.values()))[0].length(GEOM)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 7.2), dpi=130, sharex=True)
    for ax, kind in ((ax1, "conv"), (ax2, "rate")):
        for name, col, ls, lw, lab in SERIES:
            if kind == "conv":
                yv = runs[name][0].conversion("DME") * 100
            else:
                yv = ma_rate_profile(*runs[name])
            ax.plot(z, yv, color=col, ls=ls, lw=lw, label=lab)
        plots._style(ax)
    ax1.set_ylabel("DME conversion [%]", color=plots._INK)
    ax2.set_ylabel("MA formation rate [mol·kg⁻¹·s⁻¹]", color=plots._INK)
    ax2.set_xlabel("Reactor length z [m]", color=plots._INK)
    ax1.set_title(f"Carbonylation models (DME+CO→MA, {int(T-273.15)} °C, {int(P)} bar)",
                  fontsize=11)
    ax2.legend(frameon=False, labelcolor=plots._INK, fontsize=7.5, ncol=1, loc="upper right")

    fig.tight_layout()
    print("saved", plots.save(fig, "carbonylation_compare.png"))
    for name, *_ in SERIES:
        r = runs[name][0]
        print(f"{name:20s} DME conv={r.conversion('DME')[-1]*100:5.1f}%  "
              f"y_MA={r.mole_fractions()['MA'][-1]:.4f}")


if __name__ == "__main__":
    main()
