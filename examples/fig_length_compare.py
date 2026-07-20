"""length 比較図: 実長さ z [m] に対する 合成 / 脱水 / カルボニル化 の段階効果。

同一 feed・同一温度で、触媒構成だけを変える 3 ケース:
  - MeOH only     : synthesis のみ（10 kg）。CH3OH まで。
  - DME hybrid    : synthesis:dehydration = 8:2（計 10 kg）。CH3OH→DME。
  - DME+MA tandem : synthesis:dehydration:carbonylation = 7:2:1（計 10 kg）。CH3OH→DME→MA。

読み取れること:
  上段 CO 転化率     : 脱水で ~50%→~94%、カルボニル化追加で CO をさらに消費（DME+CO→MA）。
  中段 主生成物      : CH3OH（MeOH only）／DME（hybrid）／MA（tandem, 破線=中間体 DME）。
  下段 CO 消費速度   : 純合成は平衡で 0、脱水・カルボニル化で駆動力が維持され尾を引く。

⚠️ カルボニル化の温度に注意: 合成/脱水は 250℃ が適温だが、DTU は 438 K の単一温度モデル。
   DTU 原著は温度依存を与えていないため、ここでは 438 K の k1 を 250℃ にそのまま適用（=外挿）。
   その結果 MA 生成は小さく（y_MA≈0.002）、定量値ではなく概念的な段階効果として見ること。
   本格的には温度ゾーンを分けた段階反応器が妥当（合成/脱水@250℃ → カルボニル化@165℃）。

共通条件(推奨値):
  feed : CO:CO2:H2 = 0.30:0.05:0.65, 全流量 0.10 mol/s
  T, P : 523.15 K (250 °C), 50 bar
  触媒 : 計 10 kg
  幾何 : 内径 40 mm 管 (A=1.26e-3 m²), ρ_bed=1200 kg/m³, ε=0.40 → 全長 ~6.6 m

実行: PYTHONPATH=src python3 examples/fig_length_compare.py
"""
import numpy as np
import matplotlib.pyplot as plt

from reaction_rate import Geometry, GasState
from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate.network import species_rates
from reaction_rate import plots

# --- 共通条件（推奨値。ここを変えれば条件比較できる）---
FTOT = 0.10                              # 全モル流量 [mol/s]
T, P = 523.15, 50.0                      # 温度[K], 圧力[bar]
W = 10.0                                 # 総触媒質量 [kg]
GEOM = Geometry(area=np.pi / 4 * 0.04**2,  # 内径 40 mm 管
                bulk_density=1200.0,       # Cu/ZnO/Al2O3 充填密度
                void_fraction=0.40)        # 充填層空隙率


def feed():
    return {"CO": 0.30 * FTOT, "CO2": 0.05 * FTOT, "H2": 0.65 * FTOT,
            "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0, "MA": 0.0}


def co_rate_profile(result, bed, models, k_eq3):
    """各 z での CO 消費速度 −R_CO [mol·kg_bed⁻¹·s⁻¹]（局所組成から再評価）。"""
    y = result.mole_fractions()
    return np.array([
        -species_rates(GasState(result.T, result.P,
                                {s: float(y[s][i]) for s in result.F}),
                       bed, models=models, k_eq3=k_eq3).get("CO", 0.0)
        for i in range(len(result.W))
    ])


def main():
    # 3 ケース（合成のみ / 合成+脱水 8:2 / 合成+脱水+カルボニル化 7:2:1）。色=ケースで固定。
    bed_syn = CatalystBed({"synthesis": W})
    bed_dme = CatalystBed({"synthesis": 0.8 * W, "dehydration": 0.2 * W})
    bed_tan = CatalystBed({"synthesis": 0.7 * W, "dehydration": 0.2 * W,
                           "carbonylation": 0.1 * W})
    mdl_syn = {"synthesis": "KOGAS"}
    mdl_dme = {"synthesis": "KOGAS", "dehydration": "KOGAS"}
    mdl_tan = {"synthesis": "KOGAS", "dehydration": "KOGAS", "carbonylation": "DTU"}
    res_syn = pfr(feed(), T, P, bed_syn, models=mdl_syn)
    res_dme = pfr(feed(), T, P, bed_dme, models=mdl_dme, k_eq3="thermo")
    res_tan = pfr(feed(), T, P, bed_tan, models=mdl_tan, k_eq3="thermo")
    z = res_syn.length(GEOM)              # 3 ケースとも同一(同 W・同幾何)

    LBL = ["MeOH only", "DME hybrid", "DME+MA tandem"]
    C = plots.OKABE_ITO                   # [0]=青(MeOH), [1]=橙(DME), [2]=緑(tandem)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.2, 8.8), dpi=130, sharex=True)

    # 上: CO 転化率 — 脱水で頭打ち解消、カルボニル化でさらに上乗せ
    plots.lines(z, {LBL[0]: res_syn.conversion("CO") * 100,
                    LBL[1]: res_dme.conversion("CO") * 100,
                    LBL[2]: res_tan.conversion("CO") * 100},
                "", "CO conversion [%]", ax=ax1, marker=False)
    ax1.set_title(f"Synthesis → dehydration → carbonylation "
                  f"({int(T-273.15)} °C, {int(P)} bar, KOGAS + DTU)", fontsize=11)

    # 中: 各ケースの主生成物（CH3OH / DME / MA）＋ tandem 中間体 DME（破線）
    y_tan = res_tan.mole_fractions()
    plots.lines(z, {LBL[0]: res_syn.mole_fractions()["CH3OH"],
                    LBL[1]: res_dme.mole_fractions()["DME"],
                    LBL[2]: y_tan["MA"]},
                "", "Main product mole fraction [-]", ax=ax2, marker=False)
    ax2.get_legend().remove()          # 色対応は上下段の凡例で足りる。中段は生成物名で直接ラベル
    ax2.plot(z, y_tan["DME"], ls="--", lw=1.6, color=C[2])   # tandem 中間体 DME（→MA へ消費）
    # 凡例は上下段と同じ色対応なので中段は付けず、生成物名を直接ラベル（衝突回避）
    ax2.text(z[-1] * 0.82, res_syn.mole_fractions()["CH3OH"][-1] + 0.006, "CH3OH", color=C[0], fontsize=9)
    ax2.text(z[-1] * 0.82, res_dme.mole_fractions()["DME"][-1] + 0.008, "DME (hybrid)", color=C[1], fontsize=9)
    ax2.text(z[-1] * 0.30, y_tan["DME"][int(len(z) * 0.42)] - 0.024, "DME (tandem)→MA", color=C[2], fontsize=8.5)
    ax2.text(z[-1] * 0.82, y_tan["MA"][-1] + 0.006, "MA", color=C[2], fontsize=9)

    # 下: CO 消費速度 — 段階が増えるほど駆動力が保たれ速度が維持される
    plots.lines(z, {LBL[0]: co_rate_profile(res_syn, bed_syn, mdl_syn, "KOGAS"),
                    LBL[1]: co_rate_profile(res_dme, bed_dme, mdl_dme, "thermo"),
                    LBL[2]: co_rate_profile(res_tan, bed_tan, mdl_tan, "thermo")},
                "Reactor length z [m]", "CO consumption rate [mol·kg⁻¹·s⁻¹]",
                ax=ax3, marker=False)
    ax3.axhline(0.0, color=plots._MUTED, lw=0.8, zorder=0)   # 入口近傍は逆WGSで一瞬 CO 生成(<0)

    fig.tight_layout()
    print("saved", plots.save(fig, "length_compare.png"))
    # 参考値
    for name, r in [(LBL[0], res_syn), (LBL[1], res_dme), (LBL[2], res_tan)]:
        y = r.mole_fractions()
        print(f"{name:14s} CO conv={r.conversion('CO')[-1]*100:5.1f}%  "
              f"y_CH3OH={y['CH3OH'][-1]:.3f}  y_DME={y['DME'][-1]:.3f}  y_MA={y['MA'][-1]:.4f}")


if __name__ == "__main__":
    main()
