"""STY 図: 実長さ z に対する STY [g/(L·h)]（合成→脱水→カルボニル化, carbonylation=DTU-Cheung2007-2）。

fig_tandem_carbonylation.py と同じ 3 ケース・同条件だが、縦軸を STY にする。
  STY(z) = [生成物の質量流量 F·MW·3600]  /  [z 時点の累計触媒ベッド体積 V_bed(z)]   [g/(L·h)]
  分母 V_bed(z) = W(z)/ρ_bed（累計触媒量を体積で表したもの, L）。
  ＝「その長さまでの触媒を使って、単位ベッド体積・単位時間あたり何 g の生成物を作れるか」。

読み取り:
  ・STY は入口側で高く、長くなるほど低下（生成物が平衡・枯渇で頭打ちなのに触媒だけ増える）。
    → 短い床＝高STY(触媒効率良)だが低転化、長い床＝低STY(触媒余り)だが高転化、の設計トレードオフ。
  上段: 各ケースの主生成物 STY（MeOH only=CH3OH / hybrid=DME / tandem=MA）。
  下段: tandem 内の 3 生成物 STY（CH3OH→DME→MA のカスケード）。

条件: feed CO:CO2:H2 = 0.30:0.05:0.65, 全流量 0.10 mol/s, 250℃, 50 bar, 触媒計 10 kg,
      内径40mm管・ρ_bed=1200・ε=0.40 → 全長 ~6.6 m, 全ベッド体積 ~8.3 L。
⚠️ カルボニル化は 250℃(実験域150–190℃外)・水阻害なしの外挿（tandem のMAは過大評価側）。

実行: PYTHONPATH=src python3 examples/fig_tandem_sty.py
"""
import numpy as np
import matplotlib.pyplot as plt

from reaction_rate import Geometry
from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate.state import COMPONENTS
from reaction_rate import plots

FTOT = 0.10
T, P = 523.15, 50.0                       # 250 ℃, 50 bar
W = 10.0
RHO = 1200.0                              # ρ_bed [kg/m³]（累計触媒量→体積の換算）
GEOM = Geometry(area=np.pi / 4 * 0.04**2, bulk_density=RHO, void_fraction=0.40)


def feed():
    return {"CO": 0.30 * FTOT, "CO2": 0.05 * FTOT, "H2": 0.65 * FTOT,
            "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0, "MA": 0.0}


def sty_profile(result, species):
    """STY(z) [g/(L·h)] = F_species·MW·3600 / V_bed(z)[L]、V_bed = W/ρ_bed。入口(W=0)は NaN。"""
    Vbed_L = 1000.0 * result.W / RHO      # 累計触媒ベッド体積 [L]（= A·z）
    mw = COMPONENTS[species].MW
    sty = np.full_like(result.W, np.nan)
    nz = result.W > 0
    sty[nz] = result.F[species][nz] * mw * 3600.0 / Vbed_L[nz]
    return sty


def main():
    bed_syn = CatalystBed({"synthesis": W})
    bed_dme = CatalystBed({"synthesis": 0.8 * W, "dehydration": 0.2 * W})
    bed_tan = CatalystBed({"synthesis": 0.7 * W, "dehydration": 0.2 * W, "carbonylation": 0.1 * W})
    res_syn = pfr(feed(), T, P, bed_syn, models={"synthesis": "KOGAS"})
    res_dme = pfr(feed(), T, P, bed_dme, models={"synthesis": "KOGAS", "dehydration": "KOGAS"}, k_eq3="thermo")
    res_tan = pfr(feed(), T, P, bed_tan,
                  models={"synthesis": "KOGAS", "dehydration": "KOGAS", "carbonylation": "DTU-Cheung2007-2"},
                  k_eq3="thermo")
    z = res_syn.length(GEOM)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 7.2), dpi=130, sharex=True)

    # 上: 各ケースの主生成物 STY（色=ケース）
    plots.lines(z, {"MeOH only (CH3OH)": sty_profile(res_syn, "CH3OH"),
                    "DME hybrid (DME)":  sty_profile(res_dme, "DME"),
                    "DME+MA tandem (MA)": sty_profile(res_tan, "MA")},
                "", "STY [g/(L·h)]", ax=ax1, logy=True)
    ax1.set_title(f"Space-time yield vs reactor length "
                  f"({int(T-273.15)} °C, {int(P)} bar, KOGAS + DTU-Cheung2007-2)", fontsize=10)

    # 下: tandem 内の 3 生成物 STY（CH3OH→DME→MA カスケード）
    plots.lines(z, {"CH3OH": sty_profile(res_tan, "CH3OH"),
                    "DME":   sty_profile(res_tan, "DME"),
                    "MA":    sty_profile(res_tan, "MA")},
                "Reactor length z [m]", "STY in tandem [g/(L·h)]", ax=ax2, logy=True)

    fig.tight_layout()
    print("saved", plots.save(fig, "tandem_sty.png"))
    # 参考: 出口 STY と全ベッド体積
    Vend = 1000.0 * W / RHO
    print(f"全ベッド体積 = {Vend:.1f} L")
    for name, r, sp in [("MeOH only", res_syn, "CH3OH"), ("DME hybrid", res_dme, "DME"),
                        ("DME+MA tandem", res_tan, "MA")]:
        s = sty_profile(r, sp)
        print(f"{name:14s} 主生成物 {sp:5s}: STY(出口)={s[-1]:7.1f}  STY(max)={np.nanmax(s):7.1f} g/(L·h)")


if __name__ == "__main__":
    main()
