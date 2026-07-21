"""真のタンデム（段階反応器）: MeOH/DME ハイブリッド → 水分離 → 乾燥DMEカルボニル化。

これまでの単一床「混合ハイブリッド」（全反応が共存）は物理的にはタンデムでなく、
脱水が生む水がカルボニル化を阻害する（Cheung 2007: 1.1kPa H2O で速度 1/14）。
工業（DMTE）・研究（dual-bed, Nat. Commun. 2020 の triple tandem）はいずれも
「脱水 → 水を分離 → 乾燥DMEをカルボニル化」の段階構成。本図はそれを模擬する:

  段1  MeOH/DME ハイブリッド PFR @250℃（synthesis:dehydration = 5:5, 長さ L1 可変）
  段間 DME を抽出し H2O・CH3OH を除去（＝蒸留/凝縮で乾燥）、新鮮 CO を CO:DME=R で添加
  段2  カルボニル化 PFR @180℃（乾燥 DME + CO, 固定触媒 W2, DTU-Cheung2007-2）

段2は 180℃＝カルボニル化の検証域(150–190℃)内なので、混合ハイブリッド図（250℃外挿）
より現実的。段間で水を抜くため段2は水阻害なしの DTU-Cheung2007-2 が妥当。

横軸 L1 を振り、生成物 MA の収率を見る（＝「どこでハイブリッドを打ち切りカルボニル化に
繋ぐのが良いか」）。収率定義:
  段1 DME 収率 [%] = 2·F_DME/(F_CO0+F_CO2,0)·100   （DME は syngas 由来 C を 2 個含む）
  総合 MA 収率 [%] = 2·F_MA /(F_CO0+F_CO2,0)·100   （MA の 3C のうち 2C が段1 DME 由来）
  段2 DME→MA 転化 [%] = F_MA/F_DME·100
※段2触媒 W2 は固定。L1↑で DME が増えると固定 W2 では捌ききれず段2転化は低下する
  （＝DME を増やすほどカルボニル化触媒も増やす必要がある、という設計トレードオフ）。

条件: feed CO:CO2:H2 = 0.30:0.05:0.65, 全流量 0.10 mol/s, 50 bar,
      内径40mm管・ρ_bed=1200・ε=0.40, 段2 W2=5 kg, CO:DME=20。

実行: PYTHONPATH=src python3 examples/fig_tandem_stages.py
"""
import numpy as np
import matplotlib.pyplot as plt

from reaction_rate import Geometry
from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate import plots

FTOT = 0.10
T1, T2 = 523.15, 453.15                   # 段1 250℃（ハイブリッド）, 段2 180℃（カルボニル化）
P = 50.0
RHO = 1200.0
GEOM = Geometry(area=np.pi / 4 * 0.04**2, bulk_density=RHO, void_fraction=0.40)
A = GEOM.area
W2 = 5.0                                   # 段2 カルボニル化触媒（固定）[kg]
R_CO_DME = 20.0                            # 段2 供給の CO:DME 比（CO 過剰）
CARB = "DTU-Cheung2007-2"

L1_GRID = np.linspace(0.2, 5.0, 25)        # 段1 ハイブリッド長さ [m]


def feed1():
    return {"CO": 0.30 * FTOT, "CO2": 0.05 * FTOT, "H2": 0.65 * FTOT,
            "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0, "MA": 0.0}


def run_tandem(L1):
    """段1(長さ L1) → 段間(乾燥・DME 抽出・新鮮 CO) → 段2 を解いて要約 dict を返す。"""
    W1 = RHO * A * L1
    res1 = pfr(feed1(), T1, P,
               CatalystBed({"synthesis": 0.5 * W1, "dehydration": 0.5 * W1}),
               models={"synthesis": "KOGAS", "dehydration": "KOGAS"}, k_eq3="thermo")
    F_dme = float(res1.F["DME"][-1])                      # 段1 が作った DME
    # 段間: H2O・CH3OH を除去し（乾燥）、DME に新鮮 CO を CO:DME=R で添加
    feed2 = {"DME": F_dme, "CO": R_CO_DME * F_dme, "MA": 0.0}
    res2 = pfr(feed2, T2, P, CatalystBed({"carbonylation": W2}),
               models={"carbonylation": CARB})
    F_ma = float(res2.F["MA"][-1])
    C0 = feed1()["CO"] + feed1()["CO2"]                   # syngas 由来炭素酸化物
    return {"co_conv": res1.conversion("CO")[-1] * 100,
            "dme_yield": 2 * F_dme / C0 * 100,
            "ma_yield": 2 * F_ma / C0 * 100,
            "stage2_conv": (F_ma / F_dme * 100) if F_dme > 0 else 0.0}


def main():
    rows = [run_tandem(L1) for L1 in L1_GRID]
    co_conv = np.array([r["co_conv"] for r in rows])
    dme_yield = np.array([r["dme_yield"] for r in rows])
    ma_yield = np.array([r["ma_yield"] for r in rows])
    stage2_conv = np.array([r["stage2_conv"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 7.4), dpi=130, sharex=True)

    # 上: 段1 CO転化・段1 DME収率・総合 MA収率（すべて %, 単一軸）
    plots.lines(L1_GRID, {"Stage-1 CO conversion": co_conv,
                          "Stage-1 DME yield": dme_yield,
                          "Overall MA yield": ma_yield},
                "", "Yield / conversion [%]", ax=ax1)
    ax1.set_title(f"True tandem: hybrid(250°C, L1) → dry → carbonylation(180°C, {CARB})",
                  fontsize=9.5)
    # 推奨分割域 ~1.5–2 m を淡く網掛け（MA 収率が頭打ちに入る手前）
    ax1.axvspan(1.5, 2.0, color=plots.OKABE_ITO[2], alpha=0.10, zorder=0)
    ax1.text(1.75, 6, "recommended\nL1 ≈ 1.5–2 m", ha="center", va="bottom",
             fontsize=8, color=plots._INK)

    # 下: 段2 DME→MA 転化（固定 W2 では DME 増で低下＝カルボニル化触媒律速）
    plots.lines(L1_GRID, {"Stage-2 DME→MA conversion": stage2_conv},
                "Stage-1 hybrid length  L1 [m]", "Stage-2 conversion [%]", ax=ax2)
    ax2.axvspan(1.5, 2.0, color=plots.OKABE_ITO[2], alpha=0.10, zorder=0)
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    print("saved", plots.save(fig, "tandem_stages.png"))
    print(f"{'L1[m]':>6} {'CO conv%':>9} {'DME yield%':>11} {'MA yield%':>10} {'stg2 conv%':>11}")
    for L1, r in zip(L1_GRID, rows):
        print(f"{L1:6.2f} {r['co_conv']:9.1f} {r['dme_yield']:11.1f} "
              f"{r['ma_yield']:10.1f} {r['stage2_conv']:11.1f}")


if __name__ == "__main__":
    main()
