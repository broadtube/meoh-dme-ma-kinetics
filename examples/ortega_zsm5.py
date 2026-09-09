"""Ortega 2018 の ZSM-5 メタノール脱水速度論の再現（Fig. 4 / Fig. 6 / Fig. 7 ＋ SI アンカー）。

原著: C. Ortega, M. Rezaei, V. Hessel, G. Kolb, Chem. Eng. J. 347 (2018) 741–753
      `references/ortega2018_zsm5_dme_intrinsic_LHHW_CEJ.pdf`
      SI: `references/1-s2.0-S1385894718307332-mmc1.docx`

検証対象は `vbf_kogas.rate_dehydration(..., source="ZSM5")`
＝ Table 4 **Eq.(16)**「modified Klusáček & Schneider」＋ Table 5 の定数:
    r_MeOH = k·K_M·p_M·[1 − p_D·p_W/(p_M²·K_eq)] / (1 + 2√(K_M·p_M) + K_W·p_W)²

⚠️ 唯一の論文外入力は K_eq（原著は「ΔG°f から算出」としか書かず閉形式が無い）。ただし
   本再現の条件では駆動力 [·] が 0.99 以上なので、K_eq の選択は結果にほとんど影響しない。

──────────────────────────────────────────────────────────────────────
反応器モデル
──────────────────────────────────────────────────────────────────────
原著の装置は**無勾配循環反応器**（§3.3 で RTD により完全混合を確認）なので、触媒が見るのは
**出口組成**。よって
  ・Fig. 4 / Fig. 6: 図の横軸 p_MeOH は「実際の（＝出口の）メタノール分圧」。SI Table S6 が
      公称 1.0 bar の点を p_MeOH = 0.931 bar と書いているのがその証拠。そこで p_M を出口値に
      固定し、物質収支 r = (F_MeOH/W)·X と p_W = p_D = p_M·X/(2(1−X)) を連立して X を解く。
  ・Fig. 7: 供給組成（70 wt% MeOH / 30 wt% H2O）と WHSV が既知なので、reactors.cstr で
      供給から出口まで解く（転化率が最大 31% あり、水の蓄積が効くため）。

──────────────────────────────────────────────────────────────────────
実験データの出所
──────────────────────────────────────────────────────────────────────
原著にも SI にも**速度の数表は無い**（SI はキャラクタリゼーション・移動現象判定・文献比較）。
そこで下の点は PDF 埋め込み画像（945×907 / 945×966 px）から**ピクセル単位でデジタイズ**した:
軸の目盛ラベル文字の重心から線形較正し、系列色ごとにマーカー円の重心を取っている。
読み取り精度は Fig. 4（線形軸）で低速度側が最も悪く、同じ点を対数軸の Fig. 6 で読むと
最大 13% ずれる（下の FIG6 と FIG4 の p≈1.0 列を比較）。**低温側は Fig. 6 の値の方が信頼できる。**

唯一デジタイズ不要の厳密値が SI にある（SI_ANCHOR）:
  Table S6/S7: 190 °C・p_MeOH=0.931 bar・乾燥・WHSV 100 h⁻¹ で観測速度 89.9 mol·m⁻³cat·s⁻¹、
               希釈前触媒床密度 1300 kg·m⁻³ → 0.069154 mol_MeOH·kg⁻¹·s⁻¹
  Table S3/S4: 同条件の転化率 0.081（MeOH 2.5 g/h, 触媒 25.31 mg → WHSV 98.8 h⁻¹）
               → r = 0.0694 mol·kg⁻¹·s⁻¹。上と 0.3% で一致する独立検算。

実行: PYTHONPATH=src python3 examples/ortega_zsm5.py
"""
import math

from scipy.optimize import brentq

from reaction_rate import vbf_kogas
from reaction_rate.reactors import cstr, CatalystBed
from reaction_rate.state import GasState

MW_MEOH = 32.042e-3          # [kg/mol]
K_EQ3 = "thermo"             # 論文外入力（駆動力 ≈1 なので効かない）

# --- Fig. 4: 速度 vs メタノール分圧（WHSV 100 h⁻¹, 乾燥供給）---
#     {T[℃]: [(p_MeOH[bar], r[mol_MeOH·kg⁻¹·s⁻¹]), ...]}  ← 画像からデジタイズ
FIG4 = {
    140: [(0.317, 0.00399), (0.625, 0.00360), (0.964, 0.00364)],
    152: [(0.322, 0.00706), (0.625, 0.00721), (0.957, 0.00710)],
    165: [(0.312, 0.01478), (0.625, 0.01531), (0.954, 0.01671)],
    177: [(0.312, 0.02810), (0.616, 0.03092), (0.941, 0.03296)],
    190: [(0.299, 0.05443), (0.620, 0.06359), (0.906, 0.06889)],
}
WHSV_FIG4 = 100.0            # [h⁻¹] メタノール基準（SI Table S4: 2.5 g/h ÷ 25.31 mg = 98.8）

# --- Fig. 6: 同じ乾燥・WHSV 100 h⁻¹ の点を対数軸で読んだもの（Fig.4 の p≈1.0 列と同一データ）---
#     縦軸ラベルは "log(rate)" だが**実際は自然対数**（exp を取ると Fig.4 と一致する）。
#     p_MeOH は Fig.4 の第3列（同一実験）の値を使う。高温ほど転化率が上がって下がる。
#     (T[℃], p_MeOH[bar], r[mol_MeOH·kg⁻¹·s⁻¹])
FIG6 = [(139.4, 0.964, 0.003177), (152.5, 0.957, 0.006570), (165.0, 0.954, 0.016769),
        (177.7, 0.941, 0.032190), (190.7, 0.906, 0.070530)]

# --- Fig. 7: 70 wt% MeOH / 30 wt% H2O 供給, WHSV 14 h⁻¹（メタノール基準・図注に明記）---
FIG7 = [(140.0, 0.001013), (164.7, 0.008593), (172.0, 0.016192),
        (176.8, 0.014733), (190.2, 0.037761)]
WHSV_FIG7 = 14.0
WT_MEOH, WT_H2O = 0.70, 0.30

# --- SI の厳密アンカー（デジタイズ不要）---
SI_ANCHOR = {"T": 190.0, "p_M": 0.931, "rate": 89.9 / 1300.0, "X": 0.081}


def rate_meoh(T_c: float, p_M: float, p_W: float = 0.0, p_D: float = 0.0) -> float:
    """r_MeOH [mol_MeOH·kg⁻¹·s⁻¹]。rate_dehydration は DME 生成基準なので ×2 する。"""
    P = p_M + p_W + p_D
    y = {"CH3OH": p_M / P, "H2O": p_W / P, "DME": p_D / P}
    return 2.0 * vbf_kogas.rate_dehydration(GasState(T_c + 273.15, P, y),
                                            source="ZSM5", k_eq3=K_EQ3)


def rate_at_outlet(T_c: float, p_M: float, whsv: float = WHSV_FIG4) -> float:
    """出口 p_MeOH を固定した無勾配反応器の予測速度 [mol_MeOH·kg⁻¹·s⁻¹]。

    r = (F_MeOH/W)·X と p_W = p_D = p_M·X/(2(1−X)) を連立して X を解く（乾燥供給・Δn=0）。
    """
    fw = whsv / 3600.0 / MW_MEOH                      # F_MeOH/W [mol·kg⁻¹·s⁻¹]

    def g(X):
        p_wd = p_M * X / (2.0 * (1.0 - X))
        return rate_meoh(T_c, p_M, p_wd, p_wd) - fw * X

    return fw * brentq(g, 1e-12, 0.9)


def rate_wet_feed(T_c: float, whsv: float = WHSV_FIG7) -> tuple[float, float, dict]:
    """70 wt% MeOH / 30 wt% H2O 供給の CSTR 解。(r_MeOH, 転化率, 出口分圧) を返す。"""
    n_M, n_W = WT_MEOH / 32.042, WT_H2O / 18.015      # 相対モル
    F_M = 1.0e-4                                      # 任意基準 [mol/s]
    F_in = {"CH3OH": F_M, "H2O": F_M * n_W / n_M, "DME": 0.0}
    W = F_M / (whsv / 3600.0 / MW_MEOH)               # F_MeOH/W から触媒質量 [kg]
    out = cstr(F_in, T_c + 273.15, 1.0, CatalystBed({"dehydration": W}),
               models={"dehydration": "ZSM5"}, k_eq3=K_EQ3)
    X = (F_in["CH3OH"] - out["CH3OH"]) / F_in["CH3OH"]
    tot = sum(out.values())
    return (F_in["CH3OH"] - out["CH3OH"]) / W, X, {s: v / tot for s, v in out.items()}


def _apparent_ea(series) -> float:
    """(T[℃], r) の両端 2 点から見かけの活性化エネルギー [kJ/mol]。"""
    (T1, r1), (T2, r2) = series[0], series[-1]
    return 8.314 * math.log(r2 / r1) / (1.0 / (T1 + 273.15) - 1.0 / (T2 + 273.15)) / 1e3


def main():
    print("=== Ortega 2018 ZSM-5 メタノール脱水の再現（modified Klusáček & Schneider, Eq.16）===\n")

    print("--- SI Table S6/S7 の厳密アンカー（デジタイズ不要）---")
    calc = rate_at_outlet(SI_ANCHOR["T"], SI_ANCHOR["p_M"])
    print(f"  190℃ / p_MeOH={SI_ANCHOR['p_M']} bar / 乾燥 / WHSV 100 h⁻¹")
    print(f"    SI 記載   : 89.9 mol·m⁻³·s⁻¹ ÷ 1300 kg·m⁻³ = {SI_ANCHOR['rate']:.5f} mol·kg⁻¹·s⁻¹")
    print(f"    本実装     : {calc:.5f} mol·kg⁻¹·s⁻¹  ({calc/SI_ANCHOR['rate']-1:+.2%})")
    print(f"    SI Table S3 の転化率 0.081 からの独立検算: "
          f"{0.081 * WHSV_FIG4 / 3600.0 / MW_MEOH:.5f} mol·kg⁻¹·s⁻¹\n")

    print("--- Fig. 4: 速度 vs メタノール分圧（乾燥, WHSV 100 h⁻¹）---")
    print(f"{'T[℃]':>5} {'p_M':>6} {'X':>6} {'図(実測)':>10} {'本実装':>10} {'差':>8}")
    errs = []
    for T, row in FIG4.items():
        for p_M, r_exp in row:
            r_calc = rate_at_outlet(T, p_M)
            errs.append(abs(r_calc / r_exp - 1))
            X = r_calc / (WHSV_FIG4 / 3600.0 / MW_MEOH)
            print(f"{T:>5} {p_M:>6.3f} {X:>6.3f} {r_exp:>10.5f} {r_calc:>10.5f} "
                  f"{r_calc/r_exp-1:>+7.1%}")
    print(f"  平均絶対誤差 {sum(errs)/len(errs):.1%} / 最大 {max(errs):.1%}"
          "（低温 2 点は Fig.4 の線形軸読み取り誤差が大きい。下の Fig.6 参照）\n")

    print("--- Fig. 6: 同一データを対数軸で読んだもの（低速度側の読み取り精度が高い）---")
    print(f"{'T[℃]':>6} {'p_M':>6} {'図(実測)':>10} {'本実装':>10} {'差':>8}")
    e6 = []
    for T, p_M, r_exp in FIG6:
        r_calc = rate_at_outlet(T, p_M)
        e6.append(abs(r_calc / r_exp - 1))
        print(f"{T:>6.1f} {p_M:>6.3f} {r_exp:>10.5f} {r_calc:>10.5f} {r_calc/r_exp-1:>+7.1%}")
    print(f"  平均絶対誤差 {sum(e6)/len(e6):.1%} / 最大 {max(e6):.1%}")
    print(f"  見かけの活性化エネルギー（両端 2 点の傾き）: "
          f"実測 {_apparent_ea([(T, r) for T, _, r in FIG6]):.1f} / "
          f"本実装 {_apparent_ea([(T, rate_at_outlet(T, p)) for T, p, _ in FIG6]):.1f} kJ/mol"
          f"（Table 5 の E_app=109.3 より低いのは、分母の K_M·p_M 項も温度で動くため）\n")

    print("--- Fig. 7: 70 wt% MeOH / 30 wt% H2O 供給, WHSV 14 h⁻¹（水阻害の検証）---")
    print(f"{'T[℃]':>6} {'X':>6} {'出口p_W':>8} {'図(実測)':>10} {'本実装':>10} {'差':>8}")
    e7 = []
    for T, r_exp in FIG7:
        r_calc, X, y_out = rate_wet_feed(T)
        e7.append(abs(r_calc / r_exp - 1))
        print(f"{T:>6.1f} {X:>6.3f} {y_out['H2O']:>8.3f} {r_exp:>10.5f} {r_calc:>10.5f} "
              f"{r_calc/r_exp-1:>+7.1%}")
    print(f"  平均絶対誤差 {sum(e7)/len(e7):.1%} / 最大 {max(e7):.1%}")
    print("  ※ 172.0℃ と 176.8℃ は原著でも上下に散っている 2 点（誤差バー最大）。"
          "本実装はその中間を通る。")


def deviation_points():
    """{系列: [(T[℃], 計算/実測 − 1), ...]}。パネル (b) 用。"""
    out = {"Fig. 4 — dry (linear axis)": [], "Fig. 6 — dry (log axis)": [],
           "Fig. 7 — 30 wt% H$_2$O in feed": []}
    for T, row in FIG4.items():
        for p_M, r in row:
            out["Fig. 4 — dry (linear axis)"].append((T, rate_at_outlet(T, p_M) / r - 1.0))
    for T, p_M, r in FIG6:
        out["Fig. 6 — dry (log axis)"].append((T, rate_at_outlet(T, p_M) / r - 1.0))
    for T, r in FIG7:
        out["Fig. 7 — 30 wt% H$_2$O in feed"].append((T, rate_wet_feed(T)[0] / r - 1.0))
    return out


def plot():
    """2 パネル: (a) Fig.4 の p_M 依存性 (b) 全点の偏差（計算/実測 − 1）。

    (b) を対数軸のアレニウス図にすると「線の上に点が乗っている」ようにしか見えず、
    肝心の一致度（何 % ずれているか）が読めない。パリティ図（原著 Fig.8 と同型）も
    対数軸だと帯が潰れるので、**偏差そのもの**を縦軸に取る。0% 線からの距離が誤差、
    網掛けが ±10%/±20%。低温ほど下振れするという残差の構造も一目で分かる。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from reaction_rate import plots

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=130)

    # (a) 速度 vs p_MeOH。線=本実装、点=原著 Fig.4、★=SI の厳密値
    pm = np.linspace(0.25, 1.0, 40)
    plots.lines(pm, {f"{T} °C": [rate_at_outlet(T, p) for p in pm] for T in FIG4},
                "Methanol partial pressure [bar]",
                "Rate [mol$_{MeOH}$ kg$^{-1}$ s$^{-1}$]", ax=ax1)
    for i, (T, row) in enumerate(FIG4.items()):
        ax1.plot([p for p, _ in row], [r for _, r in row], "o", ms=5,
                 color=plots.OKABE_ITO[i % len(plots.OKABE_ITO)],
                 mfc="white", mew=1.4, zorder=3)
    ax1.plot([SI_ANCHOR["p_M"]], [SI_ANCHOR["rate"]], "*", ms=13, color="#1b242c", zorder=4)
    ax1.annotate("SI Table S6 (exact)", (SI_ANCHOR["p_M"], SI_ANCHOR["rate"]),
                 textcoords="offset points", xytext=(-104, -20), fontsize=8, color="#1b242c",
                 arrowprops=dict(arrowstyle="-", color="#5b6873", lw=0.8))
    ax1.set_ylim(0.0, None)
    ax1.set_title("(a) Fig. 4 — rate vs $p_{MeOH}$ (dry, WHSV 100 h$^{-1}$)\n"
                  "lines = this implementation, circles = paper", fontsize=9.5, pad=6)

    # (b) 偏差: (計算/実測 − 1) を温度に対して。パリティより「何 % ずれか」が直読できる
    bands = ((0.20, "#dbe0dd"), (0.10, "#c7d3cd"))
    for f, col in bands:
        ax2.axhspan(-f * 100, f * 100, color=col, zorder=0)
    ax2.axhline(0.0, color="#1b242c", lw=1.2, zorder=1)
    for i, (label, pts) in enumerate(deviation_points().items()):
        ax2.plot([T for T, _ in pts], [d * 100 for _, d in pts], "o", ms=6,
                 color=plots.OKABE_ITO[i], mfc="white", mew=1.5, label=label, zorder=3)
    dev_si = rate_at_outlet(SI_ANCHOR["T"], SI_ANCHOR["p_M"]) / SI_ANCHOR["rate"] - 1.0
    ax2.plot([SI_ANCHOR["T"]], [dev_si * 100], "*", ms=14, color="#1b242c", zorder=4,
             label="SI Table S6 (exact value)")
    ax2.set_xlabel("Temperature [°C]", color="#1b242c")
    ax2.set_ylabel("(calculated / measured − 1)  [%]", color="#1b242c")
    ax2.set_ylim(-40, 40)
    ax2.legend(frameon=False, labelcolor="#1b242c", fontsize=8, loc="upper left")
    ax2.text(0.98, 0.03, "shaded: ±10% (inner) / ±20% (outer)", transform=ax2.transAxes,
             fontsize=8, color="#5b6873", ha="right", va="bottom")
    ax2.set_title("(b) deviation from the paper, point by point\n"
                  "0% line = perfect agreement", fontsize=9.5, pad=6)
    plots._style(ax2)

    fig.suptitle("Ortega 2018 (ZSM-5 methanol dehydration) reproduction", fontsize=11)
    fig.tight_layout()
    print("\nsaved", plots.save(fig, "ortega_zsm5.png"))


if __name__ == "__main__":
    main()
    plot()
