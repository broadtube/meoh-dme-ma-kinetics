"""DTU 2017 Fig 8 の再現: モルデナイト カルボニル化の MA 速度・DME 転化率 vs 全圧。

条件: 438 K, 2 vol% DME in CO, 300 Nml/min, 触媒 1.5 g / 0.15 g（DTU 実条件）。
要点: MA 速度は全圧に対し「一次(k1·pCO)より下に外れる(sub-linear)」＝生成物 MA 阻害。
      触媒を減らす(=転化率↓=MA↓)と TOF が上がる。

実行: PYTHONPATH=src python3 examples/dtu_fig8.py
"""
from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate import carbonylation

FLOW = 300.0 / 22414.0 / 60.0        # 300 Nml/min → mol/s（STP 基準・圧力に依らず一定）
ASD = 1.43   # DTU CBV21A: 1.43 mmol Al/g
K1 = carbonylation.DTU_PARAMS["k1"]


def run(P_bar: float, W_kg: float):
    F_in = {"CO": 0.98 * FLOW, "DME": 0.02 * FLOW, "MA": 0.0}
    bed = CatalystBed({"carbonylation": W_kg}, acid_site_density=ASD)
    res = pfr(F_in, T=438.0, P=P_bar, bed=bed, models={"carbonylation": "DTU"})
    X_DME = res.conversion("DME")[-1]
    TOF = res.outlet()["MA"] / (ASD * W_kg) * 3600.0  # mol MA/(mol Al)/h
    return X_DME, TOF


def main():
    print("MA 速度 [mol/(mol Al)/h] と DME 転化率 vs 全圧 (438 K, 2 vol% DME in CO)")
    print(f"{'P[bar]':>7} | {'ideal k1·pCO':>12} | {'1.5g: TOF':>9} {'X':>6} | {'0.15g: TOF':>10} {'X':>6}")
    for P in (10, 20, 40, 60, 80, 100):
        ideal = K1 * 0.98 * P * 3600.0
        X15, T15 = run(P, 1.5e-3)
        X015, T015 = run(P, 0.15e-3)
        print(f"{P:>7} | {ideal:>12.2f} | {T15:>9.3f} {X15*100:>5.1f}% | {T015:>10.3f} {X015*100:>5.1f}%")


def plot():
    """sweep 図: 全圧を振って MA速度(上)・DME転化率(下) を2段プロット（dual-axis 禁止）。"""
    import matplotlib.pyplot as plt
    from reaction_rate import plots
    P = list(range(10, 101, 10))
    d = {W: [run(p, W) for p in P] for W in (1.5e-3, 0.15e-3)}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 6.4), dpi=130, sharex=True)
    # 上: MA速度。ideal(k1·pCO一次) は「系列」ではなく参照線(灰破線)にし、
    #     データ2系列の色(1.5g=青, 0.15g=橙)を下段と一致させる（色=エンティティ固定）。
    ax1.plot(P, [K1 * 0.98 * p * 3600.0 for p in P], ls="--", color="#5b6873",
             lw=1.2, label="ideal (k1·pCO)")
    plots.lines(P, {
        "1.5 g": [t for _, t in d[1.5e-3]],
        "0.15 g": [t for _, t in d[0.15e-3]],
    }, "", "MA rate [mol/(mol Al)/h]", ax=ax1, marker=True)
    plots.lines(P, {
        "1.5 g": [x * 100 for x, _ in d[1.5e-3]],
        "0.15 g": [x * 100 for x, _ in d[0.15e-3]],
    }, "Total pressure [bar]", "DME conversion [%]", ax=ax2, marker=True)
    ax1.set_title("DTU Fig 8 reproduction (438 K, 2 vol% DME in CO)", fontsize=11)
    fig.tight_layout()
    print("saved", plots.save(fig, "dtu_fig8.png"))


if __name__ == "__main__":
    main()
    plot()
