"""tandem 図: 実長さ z に対する 合成→脱水→カルボニル化 の段階効果（カルボニル化=DTU-Cheung2007-2）。

fig_length_compare.py と同形式（3ケース・3段）だが、カルボニル化に温度依存モデル
DTU-Cheung2007-2 を使う点が違う（438K固定の 'DTU' より 250℃ で桁違いに速い）。
  MeOH only     : synthesis のみ（10 kg）
  DME hybrid    : synthesis:dehydration = 8:2（計 10 kg）
  DME+MA tandem : synthesis:dehydration:carbonylation = 7:2:1、carbonylation=DTU-Cheung2007-2

DTU-Cheung2007-2 とは（rate_equations.html §3）:
  DTU の MA阻害式で、k1 を温度依存(Cheung Ea=69.6)・K2 も van't Hoff(DTU DFT ΔH2)で温度依存化。
  → 250℃ では k1 が ~22 倍、かつ高温で MA吸着(K2)が下がり阻害が緩む。

⚠️ 2 点の外挿注意（概念図として見ること）:
  ① 温度: カルボニル化データは 150–190℃。250℃ は外挿。
  ② 水阻害: 合成/脱水が H2O を生成し、実際は水がカルボニル化を強く阻害するが、
     DTU-Cheung2007-2 は水項を持たない → 単一床タンデムでは MA を過大評価する。
     物理的には「合成/脱水@250℃ → 水分離 → 乾燥DMEをカルボニル化@~180℃」の段階反応器が正当。

共通条件:
  feed CO:CO2:H2 = 0.30:0.05:0.65, 全流量 0.10 mol/s, 250℃, 50 bar, 触媒計 10 kg,
  内径40mm管・ρ_bed=1200・ε=0.40 → 全長 ~6.6 m。

実行: PYTHONPATH=src python3 examples/fig_tandem_carbonylation.py
"""
import numpy as np
import matplotlib.pyplot as plt

from reaction_rate import Geometry, GasState
from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate.network import species_rates
from reaction_rate import plots

FTOT = 0.10
T, P = 523.15, 50.0                       # 250 ℃, 50 bar
W = 10.0
GEOM = Geometry(area=np.pi / 4 * 0.04**2, bulk_density=1200.0, void_fraction=0.40)
CARB = "DTU-Cheung2007-2"                  # カルボニル化モデル（温度依存・MA阻害・K2温度依存）


def feed():
    return {"CO": 0.30 * FTOT, "CO2": 0.05 * FTOT, "H2": 0.65 * FTOT,
            "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0, "MA": 0.0}


def co_rate_profile(result, bed, models, k_eq3):
    y = result.mole_fractions()
    return np.array([
        -species_rates(GasState(result.T, result.P, {s: float(y[s][i]) for s in result.F}),
                       bed, models=models, k_eq3=k_eq3).get("CO", 0.0)
        for i in range(len(result.W))
    ])


def main():
    bed_syn = CatalystBed({"synthesis": W})
    bed_dme = CatalystBed({"synthesis": 0.8 * W, "dehydration": 0.2 * W})
    bed_tan = CatalystBed({"synthesis": 0.7 * W, "dehydration": 0.2 * W, "carbonylation": 0.1 * W})
    mdl_syn = {"synthesis": "KOGAS"}
    mdl_dme = {"synthesis": "KOGAS", "dehydration": "KOGAS"}
    mdl_tan = {"synthesis": "KOGAS", "dehydration": "KOGAS", "carbonylation": CARB}
    res_syn = pfr(feed(), T, P, bed_syn, models=mdl_syn)
    res_dme = pfr(feed(), T, P, bed_dme, models=mdl_dme, k_eq3="thermo")
    res_tan = pfr(feed(), T, P, bed_tan, models=mdl_tan, k_eq3="thermo")
    z = res_syn.length(GEOM)

    LBL = ["MeOH only", "DME hybrid", "DME+MA tandem"]
    C = plots.OKABE_ITO

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.2, 8.8), dpi=130, sharex=True)

    # 上: CO 転化率
    plots.lines(z, {LBL[0]: res_syn.conversion("CO") * 100,
                    LBL[1]: res_dme.conversion("CO") * 100,
                    LBL[2]: res_tan.conversion("CO") * 100},
                "", "CO conversion [%]", ax=ax1, marker=False)
    ax1.set_title(f"Synthesis → dehydration → carbonylation "
                  f"({int(T-273.15)} °C, {int(P)} bar, KOGAS + {CARB})", fontsize=10.5)

    # 中: 主生成物 CH3OH / DME / MA ＋ tandem 中間体 DME（破線）
    y_tan = res_tan.mole_fractions()
    plots.lines(z, {LBL[0]: res_syn.mole_fractions()["CH3OH"],
                    LBL[1]: res_dme.mole_fractions()["DME"],
                    LBL[2]: y_tan["MA"]},
                "", "Main product mole fraction [-]", ax=ax2, marker=False)
    ax2.get_legend().remove()
    ax2.plot(z, y_tan["DME"], ls="--", lw=1.6, color=C[2])   # tandem 中間体 DME
    ax2.text(z[-1] * 0.82, res_syn.mole_fractions()["CH3OH"][-1] + 0.006, "CH3OH", color=C[0], fontsize=9)
    ax2.text(z[-1] * 0.82, res_dme.mole_fractions()["DME"][-1] + 0.008, "DME (hybrid)", color=C[1], fontsize=9)
    ax2.text(z[-1] * 0.28, y_tan["DME"][int(len(z) * 0.5)] + 0.006, "DME (tandem)→MA", color=C[2], fontsize=8.5)
    ax2.text(z[-1] * 0.82, y_tan["MA"][-1] + 0.006, "MA", color=C[2], fontsize=9)

    # 下: CO 消費速度
    plots.lines(z, {LBL[0]: co_rate_profile(res_syn, bed_syn, mdl_syn, "KOGAS"),
                    LBL[1]: co_rate_profile(res_dme, bed_dme, mdl_dme, "thermo"),
                    LBL[2]: co_rate_profile(res_tan, bed_tan, mdl_tan, "thermo")},
                "Reactor length z [m]", "CO consumption rate [mol·kg⁻¹·s⁻¹]",
                ax=ax3, marker=False)
    ax3.axhline(0.0, color=plots._MUTED, lw=0.8, zorder=0)

    fig.tight_layout()
    print("saved", plots.save(fig, "tandem_carbonylation.png"))
    for name, r in [(LBL[0], res_syn), (LBL[1], res_dme), (LBL[2], res_tan)]:
        y = r.mole_fractions()
        print(f"{name:14s} CO conv={r.conversion('CO')[-1]*100:5.1f}%  "
              f"y_CH3OH={y['CH3OH'][-1]:.3f}  y_DME={y['DME'][-1]:.3f}  y_MA={y['MA'][-1]:.4f}")


if __name__ == "__main__":
    main()
