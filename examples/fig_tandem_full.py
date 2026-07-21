"""tandem 総合図: 合成→脱水→カルボニル化（carbonylation=DTU-Cheung2007-2）を実長さ z で総覧。

fig_tandem_carbonylation.py の 3 段（CO転化率・主生成物モル分率・CO消費速度）に、
STY と 各ケースの全成分モル分率を積み増した 7 段構成:
  1 CO conversion（3ケース）
  2 主生成物モル分率（CH3OH/DME/MA ＋ tandem中間体DME 破線）
  3 CO consumption rate（3ケース）
  4 STY [g/(L·h)]（各ケース主生成物・線形軸）= 生成物質量流量 / 累計触媒ベッド体積 W/ρ_bed
  5 MeOH only     の全成分モル分率
  6 DME hybrid    の全成分モル分率
  7 DME+MA tandem の全成分モル分率

条件: feed CO:CO2:H2 = 0.30:0.05:0.65, 全流量 0.10 mol/s, 250℃, 50 bar, 触媒計 10 kg,
      内径40mm管・ρ_bed=1200・ε=0.40 → 全長 ~6.6 m。
⚠️ カルボニル化は 250℃(実験域150–190℃外)・水阻害なしの外挿（tandem のMAは過大評価側）。

実行: PYTHONPATH=src python3 examples/fig_tandem_full.py
"""
import numpy as np
import matplotlib.pyplot as plt

from reaction_rate import Geometry, GasState
from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate.network import species_rates
from reaction_rate.state import COMPONENTS
from reaction_rate import plots

FTOT = 0.10
T, P = 523.15, 50.0                       # 250 ℃, 50 bar
W = 10.0
RHO = 1200.0
GEOM = Geometry(area=np.pi / 4 * 0.04**2, bulk_density=RHO, void_fraction=0.40)
CARB = "DTU-Cheung2007-2"

SPECIES = ["CO", "CO2", "H2", "CH3OH", "DME", "H2O", "MA"]      # 全成分（固定色）
SPCOL = {sp: plots.OKABE_ITO[i] for i, sp in enumerate(SPECIES)}


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


def sty_profile(result, species):
    """STY [g/(L·h)] = F·MW·3600 / (W/ρ_bed·1000)。入口(W=0)は NaN。"""
    Vbed_L = 1000.0 * result.W / RHO
    sty = np.full_like(result.W, np.nan)
    nz = result.W > 0
    sty[nz] = result.F[species][nz] * COMPONENTS[species].MW * 3600.0 / Vbed_L[nz]
    return sty


def main():
    bed_syn = CatalystBed({"synthesis": W})
    bed_dme = CatalystBed({"synthesis": 0.5 * W, "dehydration": 0.5 * W})
    bed_tan = CatalystBed({"synthesis": 0.32 * W, "dehydration": 0.28 * W, "carbonylation": 0.40 * W})
    mdl_syn = {"synthesis": "KOGAS"}
    mdl_dme = {"synthesis": "KOGAS", "dehydration": "KOGAS"}
    mdl_tan = {"synthesis": "KOGAS", "dehydration": "KOGAS", "carbonylation": CARB}
    res_syn = pfr(feed(), T, P, bed_syn, models=mdl_syn)
    res_dme = pfr(feed(), T, P, bed_dme, models=mdl_dme, k_eq3="thermo")
    res_tan = pfr(feed(), T, P, bed_tan, models=mdl_tan, k_eq3="thermo")
    z = res_syn.length(GEOM)

    LBL = ["MeOH only", "DME hybrid", "DME+MA tandem"]
    C = plots.OKABE_ITO
    fig, axs = plt.subplots(7, 1, figsize=(6.4, 15.5), dpi=110, sharex=True)
    a1, a2, a3, a4, a5, a6, a7 = axs

    # 1 CO 転化率
    plots.lines(z, {LBL[0]: res_syn.conversion("CO") * 100,
                    LBL[1]: res_dme.conversion("CO") * 100,
                    LBL[2]: res_tan.conversion("CO") * 100},
                "", "CO conversion [%]", ax=a1)
    a1.set_title(f"Synthesis → dehydration → carbonylation "
                 f"({int(T-273.15)} °C, {int(P)} bar, KOGAS + {CARB})", fontsize=10.5)

    # 2 主生成物モル分率
    y_tan = res_tan.mole_fractions()
    plots.lines(z, {LBL[0]: res_syn.mole_fractions()["CH3OH"],
                    LBL[1]: res_dme.mole_fractions()["DME"],
                    LBL[2]: y_tan["MA"]},
                "", "Main product mole frac. [-]", ax=a2)
    a2.get_legend().remove()
    a2.plot(z, y_tan["DME"], ls="--", lw=1.6, color=C[2])
    a2.text(z[-1] * 0.82, res_syn.mole_fractions()["CH3OH"][-1] + 0.006, "CH3OH", color=C[0], fontsize=8.5)
    a2.text(z[-1] * 0.82, res_dme.mole_fractions()["DME"][-1] + 0.008, "DME (hybrid)", color=C[1], fontsize=8.5)
    a2.text(z[-1] * 0.28, y_tan["DME"][int(len(z) * 0.5)] + 0.006, "DME (tandem)→MA", color=C[2], fontsize=8)
    a2.text(z[-1] * 0.82, y_tan["MA"][-1] + 0.006, "MA", color=C[2], fontsize=8.5)

    # 3 CO 消費速度
    plots.lines(z, {LBL[0]: co_rate_profile(res_syn, bed_syn, mdl_syn, "KOGAS"),
                    LBL[1]: co_rate_profile(res_dme, bed_dme, mdl_dme, "thermo"),
                    LBL[2]: co_rate_profile(res_tan, bed_tan, mdl_tan, "thermo")},
                "", "CO cons. rate [mol·kg⁻¹·s⁻¹]", ax=a3)
    a3.axhline(0.0, color=plots._MUTED, lw=0.8, zorder=0)

    # 4 STY（各ケース主生成物, 線形）
    plots.lines(z, {"MeOH only (CH3OH)": sty_profile(res_syn, "CH3OH"),
                    "DME hybrid (DME)": sty_profile(res_dme, "DME"),
                    "DME+MA tandem (MA)": sty_profile(res_tan, "MA")},
                "", "STY [g/(L·h)]", ax=a4)

    # 5–7 各ケースの全成分モル分率（固定色）
    for ax, res, name, legend in [(a5, res_syn, LBL[0], True),
                                  (a6, res_dme, LBL[1], False),
                                  (a7, res_tan, LBL[2], False)]:
        y = res.mole_fractions()
        for sp in SPECIES:
            ax.plot(z, y[sp], color=SPCOL[sp], lw=1.6, label=sp)
        plots._style(ax)
        ax.set_ylabel("Mole fraction [-]", color=plots._INK)
        ax.text(0.015, 0.9, name, transform=ax.transAxes, fontsize=9.5,
                color=plots._INK, fontweight="bold")
        if legend:
            ax.legend(frameon=False, labelcolor=plots._INK, fontsize=8, ncol=4, loc="upper right")
    a7.set_xlabel("Reactor length z [m]", color=plots._INK)

    fig.tight_layout()
    print("saved", plots.save(fig, "tandem_full.png"))
    for name, r in [(LBL[0], res_syn), (LBL[1], res_dme), (LBL[2], res_tan)]:
        y = r.mole_fractions()
        print(f"{name:14s} CO conv={r.conversion('CO')[-1]*100:5.1f}%  "
              f"y_CH3OH={y['CH3OH'][-1]:.3f}  y_DME={y['DME'][-1]:.3f}  y_MA={y['MA'][-1]:.4f}")


if __name__ == "__main__":
    main()
