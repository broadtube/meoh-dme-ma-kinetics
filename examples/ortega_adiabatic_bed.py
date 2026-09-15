"""Ortega 2018 の ZSM-5 速度式で断熱固定床を解く（VBF FIG. 5 と同じ 2 パネル）。

`examples/vbf_fig5.py` が Vanden Bussche–Froment 1996 FIG.5（メタノール合成の断熱ベンチ
反応器）を再現したのと同じ形式で、**メタノール脱水 2 CH3OH → CH3OCH3 + H2O** の断熱床を
解いて組成 mol% と温度のプロファイルを描く。速度式は `vbf_kogas.rate_dehydration(source="ZSM5")`
＝ Ortega 2018 Table 4 Eq.(16)（examples/ortega_zsm5.py で原著 Fig.4/6/7 と照合済み）。

目的は **Aspen Plus に同じ速度式（`aspen_lhhw.ORTEGA_EXACT`）を LHHW として実装した結果との
突き合わせ**。VBF FIG.5 と違い原著に対応する図は無い（Ortega は無勾配循環反応器の微分測定で、
反応器プロファイルの実測は存在しない）。本スクリプトの出力が Aspen 側の照合目標になる。

実行: PYTHONPATH=src python3 examples/ortega_adiabatic_bed.py [validation|industrial]
出力: figures/ortega_adiabatic_bed_<case>.png（図）
      datasets/ortega_adiabatic_bed_<case>.csv（Aspen 突き合わせ用の軸方向プロファイル）
      datasets/ortega_adiabatic_bed_<case>_spec.json（諸元・入口条件・期待結果の機械可読版）

CSV の列: z_mm, z_over_L, W_g, T_C, T_K, X_MeOH, y_CH3OH, y_DME, y_H2O, y_N2（モル分率）,
          F_CH3OH_mol_h, F_DME_mol_h, F_H2O_mol_h, F_N2_mol_h（モル流量）。
Aspen RPlug の Profiles（Length / Temperature / Mole fraction）と列で突き合わせられる。

# Table 3 相当の諸元 — case "validation"（既定・Aspen 突き合わせ用）

床全体が Ortega のフィット域 **140–190 ℃・p_MeOH ≲ 0.1 bar** に収まるように選んだ。
外挿要素が無いので、Aspen と結果が違えば**実装差だけ**を疑えばよい。
表の構成・単位は VBF 1996 Table 3 に揃えた（右列に VBF の値を併記）。

| | Ortega 断熱床 (validation) | VBF Table 3 |
|---|---|---|
| **Catalyst** | | |
| Density (kg/m³_s) | **1300** | 1775 |
| Porosity (m³_g/m³_s) | **0.667** (ε_b = 0.40) | 0.5 (ε_b = 0.333) |
| Mass (g) | **23.52** | 34.8 |
| **Reactor** | | |
| Diameter (m_r) | **0.016** | 0.016 |
| Length (m_r) | **0.15** | 0.15 |
| **Operating conditions** | | |
| T° (K) | **423.15** | 493.2 |
| p_t° (bar) | **1.0** | 50 |
| m (10⁻⁵ kg/s) | **3.342** | 2.8 |
| **Feed composition** | | |
| CH3OH (mol%) | **10.00** | 0.00 |
| H2O (mol%) | 0.00 | 0.00 |
| CH3OCH3 (mol%) | 0.00 | — |
| N2 (mol%) | **90.00** | 11.0 (Inert) |
| 熱条件 | 断熱・圧損なし | 断熱 |

記号の意味（VBF と同じ）: 下付き s = 固体（触媒粒子）、g = 気体、r = 反応器。
- **Density (kg/m³_s)** = 粒子密度 ρ_s（粒子質量 ÷ 粒子外形体積、細孔込み）
- **Porosity (m³_g/m³_s)** = 粒子間の気体体積 ÷ 粒子体積 = ε_b/(1−ε_b)。**床空隙率 ε_b そのものではない**
- 床密度 ρ_bed = ρ_s/(1+Porosity) = ρ_s(1−ε_b) = 780 kg/m³、Mass = ρ_bed × (π/4)D²L
  （VBF で検算: 1775/1.5 × 30.16 mL = 35.7 g ≈ 表の 34.8 g。ε_b=0.5 と読むと 26.8 g で合わない）

ρ_s = 1300 kg/m³ はゼオライト成形体の典型値（Ortega SI Table S7 の「1300」と同値。出典は
Satterfield 1970 の汎用値で Ortega 自身の測定ではない）。ε_b = 0.40 は乱雑充填の一般値。
VBF の Pellet diameter に相当する粒径は**計算に使わないので載せない**（Ortega の実験触媒は 250 µm）。

## 派生量（参考）

| 項目 | 値 |
|---|---|
| 全モル流量 | 1.176 mmol/s = 4.234 mol/h（CH3OH 0.423 / N2 3.811 mol/h） |
| 質量流量 | 120.3 g/h（CH3OH 13.6 / N2 106.8 g/h）、平均分子量 28.42 g/mol |
| W/F_total | 20.0 kg·s/mol |
| WHSV（MeOH 質量基準） | 0.577 h⁻¹ |
| GHSV（入口 150 ℃・1 bar 基準 / NTP 基準） | 4,940 / 3,150 h⁻¹ |
| 入口空塔速度 | 0.206 m/s |

## 期待される結果（本スクリプトの計算値 ＝ Aspen 側の照合目標）

| 項目 | 値 |
|---|---|
| MeOH 転化率 | **0.9244**（出口温度での平衡値と一致） |
| 出口温度 | **182.4 ℃**（ΔT = 32.4 K） |
| 出口組成 [mol%] | CH3OH 0.76 / DME 4.62 / H2O 4.62 / N2 90.00 |
| 転化率 10 / 50 / 90 % 到達位置 | z/L = 0.098 / 0.376 / 0.576（z = 14.8 / 56.3 / 86.4 mm） |

## 計算に効く諸元 / 効かない諸元（重要）

本計算の PFR は触媒質量基準（dFᵢ/dW = Rᵢ）なので、**独立入力は 8 個だけ**:
触媒質量 W、入口温度、圧力、供給モル流量 4 成分、断熱フラグ。

| 諸元 | 扱い |
|---|---|
| W = 23.52 g、T°、p_t、供給モル流量、断熱 | **効く** |
| 内径・長さ・Density・Porosity | **W の算出のみ**（W = ρ_s/(1+Porosity)·A·L）。W が同じなら個別に変えても結果は同一。z[mm] と空塔速度の換算は後処理 |
| 粒径 | **使わない**（圧損なし・粒内拡散なし η=1）ので表に載せない |
| 質量流量・平均分子量・WHSV・GHSV・空塔速度 | **導出量**（モル流量から計算するだけ） |

さらに、組成と温度は **W/F_total の関数**としてしか現れない（dFᵢ/dW を F_tot で割ると
yᵢ と T の式が W/F_tot だけで閉じる）。W と供給流量を同じ比率で変えてもプロファイルは不変
（W 39.21 g → 23.52 g に変えた際、温度 3×10⁻⁷ K・モル分率 7×10⁻¹⁰ の差で一致することを確認済み）。

Aspen RPlug でも「断熱・圧損ゼロ・Rate basis = Cat(wt)」にすれば同様に幾何非依存になる。
**W と供給流量を正確に合わせることが本質**。逆に Aspen で圧損（Ergun）や粒内拡散を ON に
すると、こちらに無い物理が入って一致しなくなる。

## Aspen Plus 側で揃える設定

- **RPlug**、Reactor type = Adiabatic、圧損 0
- Catalyst タブ: Catalyst present、**Catalyst loading = 23.52 g**、Bed voidage = 0.40、
  Particle density = 1300 kg/m³（Aspen が体積×(1−ε_b)×ρ_s で W を出す方式でも 23.52 g になる）
- Reaction: LHHW、**[Ci] basis = Partial pressure（bar）**、Rate basis = **Cat (wt)**
- パラメータは `aspen_lhhw.aspen_table(ORTEGA_EXACT)` の出力。反応を
  `2 CH3OH → CH3OCH3 + H2O` と書くなら **k₀ を 1/2**（進行度基準）、
  `CH3OH → 0.5 CH3OCH3 + 0.5 H2O` と書くなら k₀ そのまま
- 物性法: **IDEAL**（本コードは理想気体・NASA-7 熱力学）。Aspen の cp / ΔH_f はデータ源が
  違うので出口温度に **±1 K 程度**の差は想定内。**組成の一致を主に**見る
- 成分: METHANOL, DIMETHYL-ETHER, WATER, NITROGEN
- LHHW 厳密形は p_MeOH → 0 で吸着項が 0 になるが、本ケースは出口でも p_MeOH = 0.0076 bar
  残るので問題ない

# case "industrial"（応用計算・純メタノール工業条件）

| 項目 | 値 | 根拠 |
|---|---|---|
| 供給 | 純メタノール 100% | DME 間接法 |
| 入口温度 | 200 ℃ | 運転条件まとめの ZSM-5 Normal 200–280 ℃ の下端 |
| 圧力 | 10 bar | 工業 DME 脱水。Δn=0 なので平衡には効かない |
| WHSV | 30 h⁻¹（MeOH 0.71 kg/h） | まとめの Normal 15–40 h⁻¹ |
| 触媒・反応器 | validation と同一 | |

結果: 転化率 0.854（平衡）、200 → 342 ℃（ΔT 142 K）、**z/L ≈ 0.18 に着火フロント**
（E_app = 109 kJ/mol の断熱自己加速。VBF のような滑らかな S 字にならない）。

⚠️ 200→342 ℃・10 bar は Ortega のフィット域（140–190 ℃・~1 bar）の**外挿**。実触媒は
272 ℃ 超で MTH（hydrocarbon pool）副生が始まり DME 選択率が落ちるが、本モデルは脱水
反応しか持たないので選択率 100% と計算する。**床後半は実機では成立しない。**
validation で Aspen 実装が合っていることを確認したあと、強い非線形域（着火位置）でも
一致するかを見る用途。

共通の理想化: 圧損ゼロ・熱損失ゼロ・粒内拡散なし（η=1）・理想気体。
"""
import math

import numpy as np

from reaction_rate.reactors import pfr, CatalystBed

# --- 触媒（VBF Table 3 と同じ定義: 粒子密度 ＋ 空隙比 m³_g/m³_s）---
RHO_S = 1300.0            # 粒子密度 [kg/m³_s]（ゼオライト成形体の典型値。Ortega SI の 1300 と同値）
POROSITY = 0.40 / 0.60    # 粒子間空隙比 [m³_g/m³_s] = ε_b/(1−ε_b)。ε_b=0.40 → 0.667（VBF は 0.5）
VOID = POROSITY / (1.0 + POROSITY)        # 床空隙率 ε_b [-] = 0.40
RHO_BED = RHO_S / (1.0 + POROSITY)        # 床密度 [kg/m³_bed] = ρ_s(1−ε_b) = 780

# --- 反応器（VBF Table 3 と同一寸法）---
DIAMETER, LENGTH = 0.016, 0.150          # [m]
AREA = math.pi / 4 * DIAMETER ** 2
W_CAT = RHO_BED * AREA * LENGTH          # 触媒質量 [kg] = 23.5 g

# --- 運転条件（ケース切替）---
#   "validation": Aspen Plus との突き合わせ用。床全体が Ortega のフィット域 140–190 ℃・
#                 p_MeOH ≲ 0.1 bar に収まり、外挿要素ゼロで滑らかな S 字になる（既定）。
#   "industrial": 純メタノール・工業 DME 条件。E_app=109 kJ/mol の断熱自己加速で
#                 z/L≈0.18 に着火フロントが立つ（滑らかではない。応用計算向け）。
CASES = {
    "validation": dict(T_in=150.0, p_bar=1.0, y_meoh=0.10, w_over_f=20.0),
    "industrial": dict(T_in=200.0, p_bar=10.0, y_meoh=1.00, w_over_f=None, whsv=30.0),
}
CASE = "validation"
MW_MEOH = 32.042e-3                      # [kg/mol]

R_GAS = 8.314
SPECIES = ("CH3OH", "DME", "H2O")


def feed_flows(case: str = CASE) -> dict[str, float]:
    """ケースの入口モル流量 [mol/s]。validation は W/F_total、industrial は WHSV で決める。"""
    c = CASES[case]
    if c["w_over_f"] is not None:
        F_tot = W_CAT / c["w_over_f"]
    else:
        F_tot = W_CAT * c["whsv"] / 3600.0 / MW_MEOH / c["y_meoh"]
    F_M = c["y_meoh"] * F_tot
    feed = {"CH3OH": F_M, "DME": 0.0, "H2O": 0.0}
    if c["y_meoh"] < 1.0:
        feed["N2"] = F_tot - F_M
    return feed


def run(case: str = CASE, n_points: int = 600):
    """断熱 PFR を積分して結果を返す。"""
    c = CASES[case]
    return pfr(feed_flows(case), c["T_in"] + 273.15, c["p_bar"],
               CatalystBed({"dehydration": W_CAT}),
               models={"dehydration": "ZSM5"}, k_eq3="thermo",
               adiabatic=True, n_points=n_points)


def x_eq(T_k: float) -> float:
    """平衡転化率。Δn=0 なので圧力に依らず X = 2√K/(1+2√K)。"""
    K = 10.0 ** (1121.0 / T_k - 0.888)          # K_eq3("thermo")
    return 2.0 * math.sqrt(K) / (1.0 + 2.0 * math.sqrt(K))


def main(case: str = CASE):
    c = CASES[case]
    feed = feed_flows(case)
    F_tot, F_M = sum(feed.values()), feed["CH3OH"]
    print(f"=== Ortega 2018 の ZSM-5 速度式で解く断熱固定床  [case = {case}] ===\n")
    print("--- 諸元（VBF Table 3 相当）---")
    print(f"  触媒   : H-ZSM-5  粒子密度 {RHO_S:.0f} kg/m³_s, 空隙比 {POROSITY:.3f} m³_g/m³_s "
          f"(ε_b={VOID:.2f}, ρ_bed={RHO_BED:.0f})  → W = {W_CAT*1e3:.2f} g")
    print(f"  反応器 : 内径 {DIAMETER*1e3:.0f} mm × 長さ {LENGTH*1e3:.0f} mm（断熱・圧損なし・VBF Table 3 と同寸）")
    print(f"  運転   : 入口 {c['T_in']:.0f} ℃, {c['p_bar']:.1f} bar")
    comp = " / ".join(f"{k} {v/F_tot*100:.0f}%" for k, v in feed.items() if v > 0)
    print(f"  供給   : {comp}   全 {F_tot*1e3:.3f} mmol/s (= {F_tot*3600:.2f} mol/h), "
          f"MeOH {F_M*MW_MEOH*3600*1e3:.1f} g/h")
    print(f"           W/F_total = {W_CAT/F_tot:.2f} kg·s/mol,  WHSV(MeOH) = "
          f"{3600*MW_MEOH*F_M/W_CAT:.2f} h⁻¹\n")

    res = run(case)
    y = res.mole_fractions()
    T = res.T_profile
    zl = res.W / res.W[-1]
    X = (res.F["CH3OH"][0] - res.F["CH3OH"]) / res.F["CH3OH"][0]
    dT = T - T[0]

    print("--- 結果 ---")
    print(f"  メタノール転化率  : {X[-1]:.4f}   （出口温度での平衡値 {x_eq(T[-1]):.4f}）")
    print(f"  温度              : {T[0]-273.15:.1f} → {T[-1]-273.15:.1f} ℃  (ΔT = {dT[-1]:.1f} K)")
    print("  出口 mol%         : " + " / ".join(f"{k} {y[k][-1]*100:.2f}" for k in feed))
    Q = F_tot * R_GAS * np.array([T[0], T[-1]]) / (c["p_bar"] * 1e5)
    print(f"  空塔速度          : 入口 {Q[0]/AREA:.3f} → 出口 {Q[1]/AREA:.3f} m/s")
    for f in (0.10, 0.50, 0.90):
        i = int(np.argmax(X >= f * X[-1]))
        print(f"  転化率 {f*100:.0f}% 到達  : z/L = {zl[i]:.3f}  (z = {zl[i]*LENGTH*1e3:.1f} mm)")
    lo, hi = T.min() - 273.15, T.max() - 273.15
    inside = 140.0 <= lo and hi <= 190.0
    print(f"\n  床内温度 {lo:.0f}–{hi:.0f} ℃ は Ortega のフィット域 140–190 ℃ の"
          f"{'内側 → 外挿なし' if inside else '外側 → 一部外挿'}。")
    if case == "industrial":
        print("  ⚠️ 純メタノール断熱: E_app=109 kJ/mol の自己加速で着火フロントが立つ（滑らかでない）。")
        print("     床後半は 272 ℃ 超で実触媒は MTH 副生が始まるが、本モデルは脱水しか持たない。")


def write_csv(case: str = CASE, outdir: str = "datasets", n_points: int = 601):
    """軸方向プロファイルを CSV に、諸元を JSON に書き出す（Aspen Plus 突き合わせ用）。

    CSV はヘッダ 1 行＋データだけの素の形式（Excel / Aspen の取り込みを邪魔しない）。
    諸元・入口条件・期待結果は同名の *_spec.json に分けて置く。
    """
    import csv
    import json
    import os

    os.makedirs(outdir, exist_ok=True)
    c = CASES[case]
    feed = feed_flows(case)
    res = run(case, n_points=n_points)
    y = res.mole_fractions()
    T = res.T_profile
    zl = res.W / res.W[-1]
    X = (res.F["CH3OH"][0] - res.F["CH3OH"]) / res.F["CH3OH"][0]
    species = list(feed)                                  # CH3OH, DME, H2O, (N2)

    path_csv = os.path.join(outdir, f"ortega_adiabatic_bed_{case}.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")     # csv 既定の \r\n を避ける（.gitattributes と揃える）
        w.writerow(["z_mm", "z_over_L", "W_g", "T_C", "T_K", "X_MeOH"]
                   + [f"y_{s}" for s in species] + [f"F_{s}_mol_h" for s in species])
        for i in range(len(zl)):
            w.writerow([f"{zl[i]*LENGTH*1e3:.4f}", f"{zl[i]:.6f}", f"{res.W[i]*1e3:.5f}",
                        f"{T[i]-273.15:.4f}", f"{T[i]:.4f}", f"{X[i]:.6f}"]
                       + [f"{y[s][i]:.8f}" for s in species]
                       + [f"{res.F[s][i]*3600:.8e}" for s in species])

    MW = {"CH3OH": 32.042, "DME": 46.068, "H2O": 18.015, "N2": 28.013}
    F_tot = sum(feed.values())
    g = lambda x: float(f"{x:.10g}")      # 10 桁に丸める（再生成時の最終桁ノイズで差分が出ないように）
    spec = {
        "case": case,
        "kinetics": "Ortega 2018 Table 4 Eq.(16), Table 5 constants "
                    "(= vbf_kogas source='ZSM5' = aspen_lhhw.ORTEGA_EXACT)",
        "K_eq": "K_eq3('thermo') = 10^(1121/T - 0.888)",
        "reactor": {"type": "adiabatic PFR, isobaric, ideal gas, eta=1",
                    "diameter_m": DIAMETER, "length_m": LENGTH,
                    "bed_volume_mL": g(AREA * LENGTH * 1e6)},
        "catalyst": {"name": "H-ZSM-5",
                     "particle_density_kg_m3_s": RHO_S,
                     "porosity_m3_g_per_m3_s": g(POROSITY),
                     "bed_void_fraction": g(VOID),
                     "bed_density_kg_m3": RHO_BED,
                     "mass_g": g(W_CAT * 1e3)},
        "inlet": {"T_C": c["T_in"], "T_K": c["T_in"] + 273.15, "P_bar": c["p_bar"],
                  "mole_fraction": {s: g(feed[s] / F_tot) for s in species},
                  "molar_flow_mol_h": {s: g(feed[s] * 3600) for s in species},
                  "total_molar_flow_mol_h": g(F_tot * 3600),
                  "mass_flow_kg_s": g(sum(feed[s] * MW[s] * 1e-3 for s in species)),
                  "W_over_F_total_kg_s_per_mol": g(W_CAT / F_tot),
                  "WHSV_MeOH_1_h": g(3600 * MW_MEOH * feed["CH3OH"] / W_CAT)},
        "expected_outlet": {"X_MeOH": g(X[-1]), "X_eq_at_T_out": g(x_eq(float(T[-1]))),
                            "T_C": g(T[-1] - 273.15), "dT_K": g(T[-1] - T[0]),
                            "mole_fraction": {s: g(y[s][-1]) for s in species}},
        "aspen_notes": [
            "RPlug, adiabatic, zero pressure drop, Rate basis = Cat(wt), catalyst loading = mass_g",
            "LHHW on partial-pressure basis [bar]; rate in mol_MeOH/(kg_cat s)",
            "if reaction is written 2 CH3OH -> DME + H2O, halve k0 (extent basis)",
            "property method IDEAL; expect ~±1 K outlet-T difference from cp/dHf data source",
        ],
    }
    path_json = os.path.join(outdir, f"ortega_adiabatic_bed_{case}_spec.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
    print(f"saved {path_csv}  ({len(zl)} rows)")
    print(f"saved {path_json}")
    return path_csv, path_json


def plot(case: str = CASE):
    """VBF FIG.5 と同じ 2 パネル: (a) 組成プロファイル (b) 温度プロファイル。"""
    import matplotlib.pyplot as plt
    from reaction_rate import plots

    c = CASES[case]
    res = run(case)
    y = res.mole_fractions()
    z_l = res.W / res.W[-1]
    T_c = res.T_profile - 273.15

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=130)
    series = {"CH3OH": y["CH3OH"] * 100, "DME": y["DME"] * 100, "H$_2$O": y["H2O"] * 100}
    plots.lines(z_l, series, "Reduced axial distance, z/L [-]", "Concentration [mol%]", ax=ax1)
    # DME と H2O は量論 1:1 で**完全に重なる**。H2O を破線にして両方見えるようにする。
    ax1.lines[2].set_linestyle((0, (4, 3)))
    ax1.lines[2].set_linewidth(2.4)
    ax1.annotate("DME and H$_2$O coincide (1:1)", (0.72, y["DME"][-1] * 100),
                 textcoords="offset points", xytext=(0, 8), fontsize=8, color="#5b6873")
    if "N2" in y:
        ax1.text(0.98, 0.30, f"balance: N$_2$ ({y['N2'][0]*100:.0f} mol% at inlet)",
                 transform=ax1.transAxes, fontsize=8, color="#5b6873", ha="right")
    plots.lines(z_l, {"T": T_c}, "Reduced axial distance, z/L [-]",
                "Temperature [°C]", ax=ax2)
    ax1.set_title("(a) concentration profiles", fontsize=10)
    ax2.set_title("(b) temperature profile", fontsize=10)
    feed_txt = "pure MeOH" if c["y_meoh"] >= 1.0 else f"{c['y_meoh']*100:.0f}% MeOH in N$_2$"
    fig.suptitle(f"Adiabatic ZSM-5 bed, Ortega 2018 kinetics — "
                 f"{feed_txt}, Tin={c['T_in']:.0f} °C, {c['p_bar']:.0f} bar  [{case}]",
                 fontsize=11)
    fig.tight_layout()
    print("\nsaved", plots.save(fig, f"ortega_adiabatic_bed_{case}.png"))


if __name__ == "__main__":
    import sys
    case = sys.argv[1] if len(sys.argv) > 1 else CASE
    main(case)
    write_csv(case)
    plot(case)
