"""carbonylation 比較図: 純カルボニル化 PFR での速度モデル比較（実長さ z 基準）。

DME + CO → 酢酸メチル(MA) の単独反応を PFR に通し、代表 5 モデルを比較する。
温度依存モデルは 438 K では DTU と一致するため、差が出る 200 ℃(473 K) で比較。

代表 5 モデル（詳細は rate_equations.html §3 の 9 モデル表）:
  DTU              : MA阻害・438K固定（温度依存なし＝基準）
  DTU-Cheung2007-1 : DTU式(MA阻害) + k1 温度依存(Cheung Ea=69.6)
  DTU-Cheng2017-1  : 同上だが Cheng Ea=88.65（Ea 大→より速い）
  Cheung2007       : 1次・MA阻害なし・per kg（8.2e-5·exp(−8370/T) 実値, DME正則化付き）
  Cheng2017        : 同 1次形で Ea=88.65（★混成）
読み取り:
  ・DTU(青) は 438K 固定なので 200℃ でも遅い（温度依存を入れると大きく速くなる）。
  ・Ea 差: Cheng(88.65) > Cheung(69.6) で速い。
  ・MA阻害の有無: standalone(Cheung2007/Cheng2017) は MA 阻害がなく DME を 100% まで転化、
    DTU系(MA阻害あり)は MA 蓄積で頭打ち。

条件: feed CO:DME = 0.90:0.10（過剰CO・DME律速）, 全流量 0.05 mol/s, 200℃, 50 bar,
      触媒 3 kg（内径40mm管・ρ_bed=1200・ε=0.40 → 全長 ~2 m）。
⚠️ 200℃ は DTU/Cheung の実験域(150–190℃)をやや外挿。model 選択で MA 収率が大きく変わる点に注意。

実行: PYTHONPATH=src python3 examples/fig_carbonylation_compare.py
"""
import numpy as np
import matplotlib.pyplot as plt

from reaction_rate import Geometry, GasState
from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate.network import species_rates
from reaction_rate import plots

FTOT = 0.05                               # 全モル流量 [mol/s]
T, P = 473.15, 50.0                       # 200 ℃, 50 bar
W = 3.0                                   # カルボニル化触媒 [kg]
GEOM = Geometry(area=np.pi / 4 * 0.04**2, bulk_density=1200.0, void_fraction=0.40)

MODELS = ["DTU", "DTU-Cheung2007-1", "DTU-Cheng2017-1", "Cheung2007", "Cheng2017"]


def feed():
    return {"CO": 0.90 * FTOT, "DME": 0.10 * FTOT, "MA": 0.0}


def ma_rate_profile(result, bed, models):
    """各 z での MA 生成速度 [mol·kg_bed⁻¹·s⁻¹]（局所組成から再評価）。"""
    y = result.mole_fractions()
    return np.array([
        species_rates(GasState(result.T, result.P, {s: float(y[s][i]) for s in result.F}),
                      bed, models=models).get("MA", 0.0)
        for i in range(len(result.W))
    ])


def main():
    runs = {}
    for m in MODELS:
        bed = CatalystBed({"carbonylation": W})
        models = {"carbonylation": m}
        runs[m] = (pfr(feed(), T, P, bed, models=models), bed, models)
    z = next(iter(runs.values()))[0].length(GEOM)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.8), dpi=130, sharex=True)

    # 上: DME 転化率（standalone は MA阻害なしで 100% まで、DTU系は頭打ち）
    plots.lines(z, {m: runs[m][0].conversion("DME") * 100 for m in MODELS},
                "", "DME conversion [%]", ax=ax1)
    ax1.get_legend().remove()             # 凡例は下段のみ（色対応は共通・上段は曲線と重なるため）
    ax1.set_title(f"Carbonylation models (DME+CO→MA, {int(T-273.15)} °C, {int(P)} bar)",
                  fontsize=11)

    # 下: MA 生成速度（温度依存・Ea 差・DME枯渇で速度が落ちる様子）。右上が空くので凡例をここに
    plots.lines(z, {m: ma_rate_profile(*runs[m]) for m in MODELS},
                "Reactor length z [m]", "MA formation rate [mol·kg⁻¹·s⁻¹]", ax=ax2)

    fig.tight_layout()
    print("saved", plots.save(fig, "carbonylation_compare.png"))
    for m in MODELS:
        r = runs[m][0]
        print(f"{m:18s} DME conv={r.conversion('DME')[-1]*100:5.1f}%  "
              f"y_MA={r.mole_fractions()['MA'][-1]:.4f}")


if __name__ == "__main__":
    main()
