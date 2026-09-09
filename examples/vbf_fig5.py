"""Vanden Bussche–Froment 1996 FIG. 5 の再現（断熱ベンチスケール反応器・VBF 原著パラメータ）。

原著: K. M. Vanden Bussche, G. F. Froment, J. Catal. 161, 1–10 (1996)
      `references/vandenBussche_Froment_1996_JCatal.pdf` p.7–8（Table 3 と FIG. 5）。

条件（Table 3「Operating Conditions for the Simulation of the Bench Scale Reactor」）:
  触媒   密度 1775 kg/m³_s, 空隙率 0.5, 質量 34.8 g, ペレット径 0.5 mm
  反応器 内径 0.016 m, 長さ 0.15 m（断熱）
  運転   T°=493.2 K, p°=50 bar, m=2.8×10⁻⁵ kg/s
  供給   CO 4.00 / H2O 0 / MeOH 0 / H2 82.00 / CO2 3.00 / 不活性 11.0 mol%

⚠️ 不活性ガスの種類は Table 3 に明記がない。原著 §Experimental は GC の内部標準に
   **アルゴン**を使っているので Ar を既定とする。不活性の種類は (a) 質量流量→モル流量の
   換算（平均分子量）と (b) 混合 cp を通じて結果に効くため、Ar/N2/CH4 の感度も出力する。

再現の要点（原著本文 p.7 の記述と FIG. 5 の読み取り値）:
  ・入口では CO2 が消費されて **CO が一旦増える**（RWGS が正方向）
  ・**3 mm** で RWGS が平衡に達し向きが反転（＝CO のピーク位置）
  ・**3 cm**（z/L=0.2）で熱力学平衡に到達、以降フラット
  ・出口 T ≈ 550 K（FIG. 5b）、出口 mol% ≈ CO 2.89 / MeOH 2.33 / CO2 2.07 / H2O 1.06（FIG. 5a）

実行: PYTHONPATH=src python3 examples/vbf_fig5.py
"""
import numpy as np

from reaction_rate.reactors import pfr, CatalystBed

# --- Table 3 ---
T_IN, P_BAR = 493.2, 50.0          # 入口温度 [K], 圧力 [bar]
MASS_FLOW = 2.8e-5                 # [kg/s]
W_CAT = 34.8e-3                    # 触媒質量 [kg]
LENGTH = 0.15                      # 床長 [m]（横軸 z/L = W/W_total の換算用）
FEED_MOLPCT = {"CO": 4.00, "CO2": 3.00, "H2": 82.00, "H2O": 0.0, "CH3OH": 0.0}
INERT_MOLPCT = 11.0

MW = {"CO": 28.010, "CO2": 44.010, "H2": 2.016, "H2O": 18.015, "CH3OH": 32.042,
      "Ar": 39.948, "N2": 28.013, "CH4": 16.043}

# FIG. 5 からの読み取り値（500 dpi 描画・軸目盛からのピクセル換算）
PAPER = {"T_out": 550.4, "CO": 2.89, "CH3OH": 2.33, "CO2": 2.07, "H2O": 1.06,
         "z_rwgs_mm": 3.0, "z_eq_mm": 30.0}


def feed_flows(inert: str = "Ar") -> tuple[dict, float]:
    """Table 3 の mol% と質量流量から入口モル流量 [mol/s] を作る。(feed, 平均MW[g/mol])。"""
    y = {**FEED_MOLPCT, inert: INERT_MOLPCT}
    y = {s: v / 100.0 for s, v in y.items()}
    mw = sum(y[s] * MW[s] for s in y)                 # [g/mol]
    F_total = MASS_FLOW / (mw * 1.0e-3)               # [mol/s]
    return {s: y[s] * F_total for s in y}, mw


def run(inert: str = "Ar", model: str = "VBF", n_points: int = 600):
    """断熱 PFR を積分して (z/L, mol%, T[K], 結果) を返す。"""
    feed, _ = feed_flows(inert)
    res = pfr(feed, T_IN, P_BAR, CatalystBed({"synthesis": W_CAT}),
              models={"synthesis": model}, adiabatic=True, n_points=n_points)
    return res.W / W_CAT, {s: v * 100.0 for s, v in res.mole_fractions().items()}, res.T_profile, res


def _markers(zL, molpct, T):
    """FIG. 5 と比べる特徴量: CO ピーク位置(=RWGS 反転)、ΔT が 95% に達する位置、出口値。"""
    dT = T - T[0]
    return {
        "T_out": T[-1],
        "dT": dT[-1],
        "z_rwgs_mm": zL[int(np.argmax(molpct["CO"]))] * LENGTH * 1e3,
        "z_eq_mm": zL[int(np.argmax(dT >= 0.95 * dT[-1]))] * LENGTH * 1e3,
        **{s: molpct[s][-1] for s in ("CO", "CH3OH", "CO2", "H2O")},
    }


def main():
    print("=== VBF 1996 FIG. 5 の再現（断熱ベンチスケール反応器, VBF 原著パラメータ）===")
    feed, mw = feed_flows("Ar")
    F = sum(feed.values())
    print(f"Table 3: T_in={T_IN} K, P={P_BAR} bar, W={W_CAT*1e3:.1f} g, m={MASS_FLOW:.1e} kg/s")
    print(f"         不活性=Ar → 平均MW={mw:.3f} g/mol, F_total={F:.3e} mol/s, W/F={W_CAT/F:.2f} kg·s/mol\n")

    zL, molpct, T, _ = run("Ar")
    m = _markers(zL, molpct, T)
    rows = [
        ("出口温度 [K]",            m["T_out"], PAPER["T_out"]),
        ("出口 CO [mol%]",          m["CO"],     PAPER["CO"]),
        ("出口 CH3OH [mol%]",       m["CH3OH"],  PAPER["CH3OH"]),
        ("出口 CO2 [mol%]",         m["CO2"],    PAPER["CO2"]),
        ("出口 H2O [mol%]",         m["H2O"],    PAPER["H2O"]),
        ("RWGS 反転位置 [mm]",      m["z_rwgs_mm"], PAPER["z_rwgs_mm"]),
        ("平衡到達位置 [mm]",       m["z_eq_mm"],   PAPER["z_eq_mm"]),
    ]
    print(f"{'項目':<22} {'本実装':>10} {'原著 FIG.5':>12} {'差':>10}")
    for label, calc, ref in rows:
        print(f"{label:<22} {calc:>10.2f} {ref:>12.2f} {(calc/ref-1)*100:>9.1f}%")

    print("\n--- 不活性ガスの感度（Table 3 に種類の記載なし）---")
    print(f"{'不活性':<6} {'平均MW':>7} {'T_out[K]':>9} {'CO':>6} {'MeOH':>6} {'CO2':>6} {'H2O':>6} "
          f"{'反転[mm]':>9} {'平衡[mm]':>9}")
    for inert in ("Ar", "N2", "CH4"):
        zL_i, mp_i, T_i, _ = run(inert)
        mi = _markers(zL_i, mp_i, T_i)
        _, mw_i = feed_flows(inert)
        print(f"{inert:<6} {mw_i:>7.3f} {mi['T_out']:>9.1f} {mi['CO']:>6.2f} {mi['CH3OH']:>6.2f} "
              f"{mi['CO2']:>6.2f} {mi['H2O']:>6.2f} {mi['z_rwgs_mm']:>9.1f} {mi['z_eq_mm']:>9.1f}")
    print(f"{'FIG.5':<6} {'—':>7} {PAPER['T_out']:>9.1f} {PAPER['CO']:>6.2f} {PAPER['CH3OH']:>6.2f} "
          f"{PAPER['CO2']:>6.2f} {PAPER['H2O']:>6.2f} {PAPER['z_rwgs_mm']:>9.1f} {PAPER['z_eq_mm']:>9.1f}")

    print("\n--- 同条件での KOGAS(=Ng 1999) との比較 ---")
    print(f"{'model':<8} {'T_out[K]':>9} {'CO':>6} {'MeOH':>6} {'CO2':>6} {'H2O':>6} {'平衡[mm]':>9}")
    for model in ("VBF", "KOGAS"):
        zL_m, mp_m, T_m, _ = run("Ar", model=model)
        mm = _markers(zL_m, mp_m, T_m)
        print(f"{model:<8} {mm['T_out']:>9.1f} {mm['CO']:>6.2f} {mm['CH3OH']:>6.2f} "
              f"{mm['CO2']:>6.2f} {mm['H2O']:>6.2f} {mm['z_eq_mm']:>9.1f}")
    print("→ 出口は平衡支配なので両者一致。差は「平衡までの距離」＝活性の差だけに出る。")


def plot():
    """FIG. 5 と同じレイアウト: (a) 濃度プロファイル (b) 温度プロファイル vs z/L。"""
    import matplotlib.pyplot as plt
    from reaction_rate import plots

    zL, molpct, T, _ = run("Ar")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=130)
    plots.lines(zL, {s: molpct[s] for s in ("CO", "CH3OH", "CO2", "H2O")},
                "Reduced axial distance, z/L [-]", "Concentration [mol%]", ax=ax1)
    plots.lines(zL, {"T": T}, "Reduced axial distance, z/L [-]", "Temperature [K]", ax=ax2)
    ax1.set_title("(a) concentration profiles", fontsize=10)
    ax2.set_title("(b) temperature profile", fontsize=10)
    fig.suptitle("VBF 1996 FIG. 5 reproduction (adiabatic, 50 bar, Tin=493.2 K)", fontsize=11)
    fig.tight_layout()
    print("\nsaved", plots.save(fig, "vbf_fig5.png"))


if __name__ == "__main__":
    main()
    plot()
