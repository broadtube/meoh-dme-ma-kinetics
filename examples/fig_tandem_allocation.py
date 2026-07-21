"""触媒配分の最適化: 段1(ハイブリッド)長さ L1 × 段2(カルボニル化)長さ L2 の 2 次元スイープ。

fig_tandem_stages.py の段階タンデム（段1 MeOH/DMEハイブリッド@250℃ → 水/メタノール除去・
乾燥・新鮮CO添加 → 段2 乾燥DMEカルボニル化@180℃, DTU-Cheung2007-2）で、
段1・段2 の触媒量を独立に振り、総合 MA 収率の等高線を描く。

読み取り:
  ・MA 収率は L1・L2 どちらを増やしても上昇（DME 生成量↑ / DME→MA 転化↑）だが逓減。
  ・同じ総触媒（L1+L2 一定＝図の対角線）では最適な配分比がある。
    → 対角線上の最大点（★）を結んだのが最適リッジ。カルボニル化(段2)を厚めにするのが有利
      （段2は 180℃・CO 希釈で相対的に遅く、より多くの触媒を要するため）。
  ・下段: 総長 L1+L2 を固定したときの MA 収率 vs 配分比 f=L1/(L1+L2)。ピークが最適配分。

収率定義（fig_tandem_stages.py と同じ）:
  総合 MA 収率 [%] = 2·F_MA/(F_CO0+F_CO2,0)·100  （MA の 3C のうち 2C が段1 DME 由来）

条件: feed CO:CO2:H2 = 0.30:0.05:0.65, 全流量 0.10 mol/s, 50 bar,
      内径40mm管・ρ_bed=1200・ε=0.40, 段2 供給 CO:DME=20。
      L1,L2 と触媒質量 W は同一管形状で W = ρ_bed·A·L で対応（総触媒 ∝ L1+L2）。

実行: PYTHONPATH=src python3 examples/fig_tandem_allocation.py   （2次元グリッドで ~20s）
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from reaction_rate import Geometry
from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate import plots

FTOT = 0.10
T1, T2 = 523.15, 453.15                   # 段1 250℃, 段2 180℃
P = 50.0
RHO = 1200.0
GEOM = Geometry(area=np.pi / 4 * 0.04**2, bulk_density=RHO, void_fraction=0.40)
A = GEOM.area
R_CO_DME = 20.0
CARB = "DTU-Cheung2007-2"

L1_GRID = np.linspace(0.2, 4.5, 22)        # 段1 ハイブリッド長さ [m]
L2_GRID = np.linspace(0.2, 4.5, 22)        # 段2 カルボニル化長さ [m]
TOTALS = [2.0, 4.0, 6.0]                    # 下段・等総長の対角線 [m]


def feed1():
    return {"CO": 0.30 * FTOT, "CO2": 0.05 * FTOT, "H2": 0.65 * FTOT,
            "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0, "MA": 0.0}


def stage1_dme(L1):
    """段1(長さ L1)の出口 DME モル流量 [mol/s]。"""
    W1 = RHO * A * L1
    res = pfr(feed1(), T1, P,
              CatalystBed({"synthesis": 0.5 * W1, "dehydration": 0.5 * W1}),
              models={"synthesis": "KOGAS", "dehydration": "KOGAS"}, k_eq3="thermo")
    return float(res.F["DME"][-1])


def stage2_ma(F_dme, L2):
    """段間(乾燥・新鮮CO)＋段2(長さ L2)の出口 MA モル流量 [mol/s]。"""
    W2 = RHO * A * L2
    res = pfr({"DME": F_dme, "CO": R_CO_DME * F_dme, "MA": 0.0}, T2, P,
              CatalystBed({"carbonylation": W2}), models={"carbonylation": CARB})
    return float(res.F["MA"][-1])


def main():
    C0 = feed1()["CO"] + feed1()["CO2"]
    F_dme = np.array([stage1_dme(L1) for L1 in L1_GRID])       # 段1 は L1 のみ依存
    Y = np.array([[2 * stage2_ma(fd, L2) / C0 * 100 for L2 in L2_GRID] for fd in F_dme])
    interp = RegularGridInterpolator((L1_GRID, L2_GRID), Y, bounds_error=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 8.6), dpi=130,
                                   gridspec_kw={"height_ratios": [1.35, 1]})

    # 上: MA 収率の充填等高線（sequential 1 色相 Blues）＋等値線ラベル
    X, Yg = np.meshgrid(L2_GRID, L1_GRID)                       # x=L2, y=L1
    levels = np.arange(0, 34, 3)
    cf = ax1.contourf(X, Yg, Y, levels=levels, cmap="Blues")
    cl = ax1.contour(X, Yg, Y, levels=levels, colors=plots._INK, linewidths=0.5, alpha=0.5)
    ax1.clabel(cl, fmt="%d", fontsize=7, colors=plots._INK)
    cb = fig.colorbar(cf, ax=ax1, pad=0.02)
    cb.set_label("Overall MA yield [%]", color=plots._INK, fontsize=9)
    ax1.set_xlabel("Stage-2 carbonylation length  L2 [m]", color=plots._INK)
    ax1.set_ylabel("Stage-1 hybrid length  L1 [m]", color=plots._INK)
    ax1.set_title(f"Catalyst allocation map (hybrid 250°C → dry → carbonylation 180°C, {CARB})",
                  fontsize=9)

    # 等総長の対角線と、その上の最適点（★）。ラベルは対角線中央に白背景で置く
    Lmax = L1_GRID[-1]
    opt = []
    for Lt in TOTALS:
        seg = np.linspace(max(0.2, Lt - Lmax), min(Lmax, Lt - 0.2), 120)
        ys = interp(np.column_stack([seg, Lt - seg]))          # (L1=seg, L2=Lt-seg)
        ax1.plot(Lt - seg, seg, color=plots._INK, lw=0.9, ls="--", alpha=0.6)
        k = int(np.nanargmax(ys))
        opt.append((seg[k], Lt - seg[k], Lt, ys[k]))
        ax1.plot(Lt - seg[k], seg[k], marker="*", ms=13, color=plots.OKABE_ITO[3],
                 mec="white", mew=0.8, zorder=5)
        mid = len(seg) // 2                                     # 対角線中央にラベル
        ax1.text(Lt - seg[mid], seg[mid], f"L1+L2={Lt:.0f}m", fontsize=7.5,
                 color=plots._INK, ha="center", va="center", rotation=-45,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax1.set_xlim(0.2, Lmax); ax1.set_ylim(0.2, Lmax)

    # 下: 固定総長での MA 収率 vs 配分比 f = L1/(L1+L2)
    series = {}
    for Lt in TOTALS:
        f = np.linspace(0.2 / Lt, 1 - 0.2 / Lt, 80)
        L1s = f * Lt
        ys = interp(np.column_stack([L1s, Lt - L1s]))
        series[f"L1+L2 = {Lt:.0f} m"] = ys
    # 3 総長で共通の f 軸に載せるため再サンプル
    faxis = np.linspace(0.05, 0.95, 80)
    ser2 = {}
    for Lt in TOTALS:
        L1s = faxis * Lt
        ok = (L1s >= 0.2) & (Lt - L1s >= 0.2)
        ys = np.full_like(faxis, np.nan)
        ys[ok] = interp(np.column_stack([L1s[ok], Lt - L1s[ok]]))
        ser2[f"L1+L2 = {Lt:.0f} m"] = ys
    plots.lines(faxis, ser2, "Allocation fraction  f = L1 / (L1+L2)  [-]",
                "Overall MA yield [%]", ax=ax2)
    for (l1, l2, Lt, y) in opt:                                # 各総長のピーク
        ax2.plot(l1 / Lt, y, marker="*", ms=12, color=plots.OKABE_ITO[3],
                 mec="white", mew=0.8, zorder=5)

    fig.tight_layout()
    print("saved", plots.save(fig, "tandem_allocation.png"))
    print(f"MA yield range: {Y.min():.1f}–{Y.max():.1f} %")
    for (l1, l2, Lt, y) in opt:
        print(f"L1+L2={Lt:.0f}m: 最適 L1={l1:.2f} L2={l2:.2f} "
              f"(L1:L2={l1/Lt*100:.0f}:{l2/Lt*100:.0f}) → MA {y:.1f}%")


if __name__ == "__main__":
    main()
