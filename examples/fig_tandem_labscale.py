"""ラボスケール単管タンデム: MeOH/DME ハイブリッド触媒 → MA(カルボニル化)触媒を
一本の反応管に直列充填した積層ベッドを等温 PFR で解く。

構成（内径10mm 単管・段間の水分離なし＝実験室の積層ベッドを模擬）:
  段1  MeOH/DME ハイブリッド 1.0 mL（合成:脱水 体積比 1:0.9, KOGAS）
  段2  MA(カルボニル化) 2.0 mL（DTU-Cheung2007-2）
  段1 出口ストリームを **全成分そのまま** 段2へ流す（乾燥・新鮮CO添加なし）。

条件: 250℃, 5 MPaG, H2/CO=1.5(mol/mol), SV=5000/h。

━━━ 明示した仮定（★＝仕様に無かったので置いた値。要確認） ━━━
★1 CO2 = 3 mol%（Y_CO2）を feed に添加。
     KOGAS 合成は CO2 水素化ベース（r_MS ∝ pCO2）。CO/H2 のみ(CO2=0)だと
     r_MS=r_RWGS≡0 で反応器が全く動かない。実 syngas は数 mol% CO2 を含むため添加。
★2 SV=5000/h の基準 = **全触媒体積 3.0 mL・STP(0℃,1atm)** の GHSV とみなした。
     → Q_STP = 5000×3.0 mL/h = 15 L/h, n_dot = 15/22.414 = 1.859e-4 mol/s。
     基準が「ハイブリッドのみ 1.0 mL」や 25℃ STP なら流量が変わる。
★3 充填密度 ρ_bed を触媒別に設定（mL→kg 換算に使用）:
     合成 Cu/ZnO/Al2O3=1300, 脱水 γ-Al2O3=800, カルボニル化 H-MOR=700 kg/m³（概算）。
     床長は z=V/A（充填体積÷断面積）で ρ に依らないため長さ軸は密度非依存。
★4 段2は水阻害なしの DTU-Cheung2007-2 を使用。だが本構成は段間乾燥が無く MA 床に
     H2O が入る（Cheung2007: 1.1kPa H2O で速度 1/14）。→ **MA は過大評価側**。
★5 250℃ はカルボニル化の検証域(150–190℃)外の外挿（Ea≈69.6, 438K の ~22 倍速）。

実行: PYTHONPATH=src python3 examples/fig_tandem_labscale.py
"""
import numpy as np
import matplotlib.pyplot as plt

from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate import plots

# ── 反応条件 ──────────────────────────────────────────────
T = 523.15                       # 250 ℃
P = 5.0e6 / 1e5 + 1.01325        # 5 MPaG → 絶対 [bar] = 51.01 bar
H2_CO = 1.5                      # H2/CO [mol/mol]
Y_CO2 = 0.03                     # ★1 CO2 添加分 [mol%/100]（KOGAS を動かすため）
SV = 5000.0                      # 空間速度 [1/h]

# ── 触媒（体積 → 質量）★3 触媒別 ρ_bed ───────────────────
RHO_SYN, RHO_DEH, RHO_MA = 1300.0, 800.0, 700.0   # [kg/m³] Cu/ZnO, γ-Al2O3, H-MOR
V_HYB = 1.0e-6                   # ハイブリッド 1.0 mL [m³]
V_MA  = 2.0e-6                   # MA 触媒 2.0 mL [m³]
V_TOT = V_HYB + V_MA             # ★2 SV 基準の全触媒体積 3.0 mL
R_SYN_DEH = (1.0, 0.9)          # ハイブリッド内 合成:脱水 体積比 1:0.9
f_syn = R_SYN_DEH[0] / sum(R_SYN_DEH)
V_syn, V_deh = V_HYB * f_syn, V_HYB * (1 - f_syn)
m_syn = RHO_SYN * V_syn                      # 合成触媒 [kg]
m_deh = RHO_DEH * V_deh                      # 脱水触媒 [kg]
m_ma  = RHO_MA * V_MA                         # MA 触媒 [kg]

# ── 反応管幾何（内径10mm）。床長は z=V/A（ρ 非依存）─────────
ID = 0.010
AREA = np.pi / 4 * ID**2
L_HYB = V_HYB / AREA * 1e3        # ハイブリッド床長 [mm]
L_MA  = V_MA / AREA * 1e3         # MA 床長 [mm]

# ── 供給流量（SV → mol/s）★2 ───────────────────────────
VM_STP = 22.414e-3               # STP モル体積 [m³/mol] (0℃,1atm)
Q_STP = SV / 3600.0 * V_TOT      # [m³/s] = 5000/h × 3.0 mL
N_TOT = Q_STP / VM_STP           # 全供給モル流量 [mol/s]


def feed():
    """入口 {species: mol/s}。CO2=Y_CO2、残りを H2/CO=1.5 で配分。"""
    y_co = (1 - Y_CO2) / (1 + H2_CO)
    y_h2 = H2_CO * y_co
    return {"CO": y_co * N_TOT, "CO2": Y_CO2 * N_TOT, "H2": y_h2 * N_TOT,
            "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0, "MA": 0.0}


def run():
    # 段1: ハイブリッド（合成+脱水 共存単床）。脱水平衡は物理的な 'thermo'（KOGAS K_eq3 は過大）
    res1 = pfr(feed(), T, P,
               CatalystBed({"synthesis": m_syn, "dehydration": m_deh}),
               models={"synthesis": "KOGAS", "dehydration": "KOGAS"},
               k_eq3="thermo")
    # 段2: 段1 出口を全成分そのまま供給（段間乾燥なし＝単管）。MA 床 DTU-Cheung2007-2
    res2 = pfr(res1.outlet(), T, P,
               CatalystBed({"carbonylation": m_ma}),
               models={"carbonylation": "DTU-Cheung2007-2"})
    return res1, res2


def main():
    res1, res2 = run()
    # 床長は充填体積で決まる（z=V/A, ρ 非依存）。W グリッドを 0..L に線形マップ
    z1 = res1.W / res1.W[-1] * L_HYB                  # [mm]
    z2 = res2.W / res2.W[-1] * L_MA + L_HYB           # 段2 は段1 長さ分オフセット
    z_if = z1[-1]                                     # 床境界 [mm]

    # 軸方向モル流量 [mmol/s]（段1・段2 を連結）
    SP = ["CO", "H2", "CO2", "CH3OH", "DME", "H2O", "MA"]
    F = {s: np.concatenate([res1.F[s], res2.F.get(s, np.zeros_like(z2))]) * 1e3
         for s in SP}
    z = np.concatenate([z1, z2])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 7.6), dpi=130, sharex=True)

    plots.lines(z, F, "", "Molar flow [mmol/s]", ax=ax1)
    ax1.axvline(z_if, color=plots._MUTED, ls=":", lw=1.2)
    ax1.text(z_if, ax1.get_ylim()[1] * 0.96, " hybrid | MA bed",
             fontsize=8, color=plots._INK, va="top")
    ax1.set_title("Lab-scale single-tube tandem (ID 10mm, 250°C, 51 bar, SV 5000/h)",
                  fontsize=9.5)

    # 下: 積算 CO 転化率と MA/DME 収率（syngas 由来 C 基準, C=CO+CO2）
    C0 = feed()["CO"] + feed()["CO2"]
    co0 = feed()["CO"]
    co_conv = np.concatenate([(co0 - res1.F["CO"]) / co0,
                              (co0 - res2.F["CO"]) / co0]) * 100
    dme_y = np.concatenate([res1.F["DME"], res2.F["DME"]]) * 2 / C0 * 100
    ma_y = np.concatenate([np.zeros_like(z1), res2.F["MA"]]) * 2 / C0 * 100
    plots.lines(z, {"CO conversion": co_conv, "DME yield": dme_y, "MA yield": ma_y},
                "Axial position z [mm]", "Conversion / yield [%]", ax=ax2)
    ax2.axvline(z_if, color=plots._MUTED, ls=":", lw=1.2)

    fig.tight_layout()
    print("saved", plots.save(fig, "tandem_labscale.png"))

    # ── サマリ ───────────────────────────────────────────
    o1, o2 = res1.outlet(), res2.outlet()
    print(f"P={P:.2f} bar  N_feed={N_TOT*1e6:.1f} µmol/s  "
          f"m[syn/deh/MA]={m_syn*1e3:.3f}/{m_deh*1e3:.3f}/{m_ma*1e3:.3f} g")
    print(f"床長 L1={z1[-1]:.1f} mm  L2={z2[-1]-z_if:.1f} mm  合計={z2[-1]:.1f} mm")
    print(f"段1出口: CO転化={100*(co0-o1['CO'])/co0:5.1f}%  "
          f"CH3OH={o1['CH3OH']*1e6:6.2f}  DME={o1['DME']*1e6:6.2f} µmol/s  "
          f"H2O={o1['H2O']*1e6:6.2f}")
    print(f"段2出口: MA={o2['MA']*1e6:6.3f} µmol/s  "
          f"DME残={o2['DME']*1e6:6.2f}  DME→MA転化={100*o2['MA']/o1['DME'] if o1['DME']>0 else 0:5.1f}%")
    print(f"総合 MA収率(C基準)={2*o2['MA']/C0*100:5.2f}%  "
          f"MA STY={o2['MA']*74.08/ (m_ma) *3600:.2f} g/(kg_cat·h)")


if __name__ == "__main__":
    main()
