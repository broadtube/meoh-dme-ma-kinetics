"""PFR の論文再現テスト（CatalystBed API）。"""
import math
from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate.state import GasState
from reaction_rate import graaf, vbf_kogas

FLOW = 300.0 / 22414.0 / 60.0          # 300 Nml/min → mol/s


def test_dtu_carbonylation_reproduction():
    # DTU 2017 Fig 7: 438 K, 10 bar, 2 vol% DME in CO, 1.5 g cat, 300 Nml/min
    #   報告値: DME 転化率 9%, MA 速度 0.68 mol/(mol Al)/h
    F_in = {"CO": 0.98 * FLOW, "DME": 0.02 * FLOW, "MA": 0.0}
    bed = CatalystBed(masses={"carbonylation": 1.5e-3}, acid_site_density=1.43)
    res = pfr(F_in, T=438.0, P=10.0, bed=bed, models={"carbonylation": "DTU"})

    X_DME = res.conversion("DME")[-1]
    TOF = res.outlet()["MA"] / (bed.acid_site_density * bed.total()) * 3600.0
    assert math.isclose(X_DME, 0.09, abs_tol=0.02)     # 実装 10.0% ≈ 報告 9%
    assert math.isclose(TOF, 0.68, rel_tol=0.15)       # 実装 0.747 ≈ 報告 0.68（+10%, フィット散布内）


def test_dme_synthesis_synergy():
    # DME合成(ハイブリッド)は methanol 合成のみより CO 転化が進む（脱水がメタノールを除去）。
    feed = {"CO": 0.32, "CO2": 0.03, "H2": 0.65, "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0}
    F_in = {k: v * 1e-3 for k, v in feed.items()}
    T, P = 523.15, 50.0
    rm = pfr({k: v for k, v in F_in.items() if k != "DME"}, T, P,
             CatalystBed({"synthesis": 5.0}), models={"synthesis": "KOGAS"})
    rd = pfr(F_in, T, P, CatalystBed({"synthesis": 4.0, "dehydration": 1.0}),
             models={"synthesis": "KOGAS", "dehydration": "KOGAS"})
    assert rd.mole_fractions()["CO"][-1] < rm.mole_fractions()["CO"][-1]   # 相乗効果
    assert rd.mole_fractions()["DME"][-1] > 0.1                            # DME 生成


def test_catalyst_amount_changes_yield():
    # 各触媒量を変えると（平衡未到達域で）DME 収率が変わる。
    feed = {"CO": 0.32, "CO2": 0.03, "H2": 0.65, "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0}
    F_in = {k: v * 1e-3 for k, v in feed.items()}
    r1 = pfr(F_in, 523.15, 50.0, CatalystBed({"synthesis": 4.5, "dehydration": 0.5}))
    r5 = pfr(F_in, 523.15, 50.0, CatalystBed({"synthesis": 2.5, "dehydration": 2.5}))
    assert r5.mole_fractions()["DME"][-1] > r1.mole_fractions()["DME"][-1]  # 脱水触媒多い→DME多い


def test_k_eq3_option_fixes_overshoot():
    # k_eq3='thermo' は KOGAS より DME 平衡が控えめ（過剰生成が是正される）。
    feed = {"CO": 0.32, "CO2": 0.03, "H2": 0.65, "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0}
    F_in = {k: v * 1e-3 for k, v in feed.items()}
    bed = CatalystBed({"synthesis": 4.0, "dehydration": 1.0})
    y_kogas = pfr(F_in, 523.15, 50.0, bed, k_eq3="KOGAS").mole_fractions()["DME"][-1]
    y_thermo = pfr(F_in, 523.15, 50.0, bed, k_eq3="thermo").mole_fractions()["DME"][-1]
    assert y_thermo < y_kogas
    assert 0.15 < y_thermo < 0.25      # Cantera の ~0.20 付近


def test_methanol_synthesis_reaches_equilibrium():
    # 大きな触媒量で PFR 出口が平衡に到達（駆動力 → 1）。Graaf はフガシティ基準。
    F_in = {s: v * 1e-3 for s, v in {"CO": 0.15, "CO2": 0.10, "H2": 0.75,
                                     "H2O": 0.0, "CH3OH": 0.0}.items()}
    T, P = 523.15, 50.0
    res_g = pfr(F_in, T, P, CatalystBed({"synthesis": 200.0}), models={"synthesis": "Graaf1988"})
    y = {s: res_g.mole_fractions()[s][-1] for s in F_in}
    f = GasState(T, P, y).fugacities(criticals="graaf1986")
    Kp = graaf.equilibrium(T, "Graaf1986")
    beta_C = f["CH3OH"] * f["H2O"] / (f["CO2"] * f["H2"] ** 3 * Kp["K_pC"])
    assert math.isclose(beta_C, 1.0, abs_tol=1e-3)


# ============================================================
#  断熱 PFR（エネルギー収支）
# ============================================================
def test_adiabatic_conserves_total_enthalpy():
    """断熱・定常なので入口と出口の全エンタルピー流量 ΣFᵢhᵢ(T) が保存する。"""
    from reaction_rate import thermo
    from reaction_rate.case import CASE_METHANOL_VBF, run_case

    case = CASE_METHANOL_VBF
    res = run_case(case)
    species = list(case.feed)
    h_in = thermo.enthalpies(case.T, species)
    h_out = thermo.enthalpies(float(res.T_profile[-1]), species)
    H_in = sum(case.feed[s] * h_in[s] for s in species)
    H_out = sum(res.outlet()[s] * h_out[s] for s in species)
    assert math.isclose(H_in, H_out, rel_tol=1e-6)


def test_isothermal_unchanged_by_adiabatic_option():
    """adiabatic=False（既定）は従来どおり T 一定・result.T は float。"""
    F_in = {"CO": 0.98 * FLOW, "DME": 0.02 * FLOW, "MA": 0.0}
    res = pfr(F_in, T=438.0, P=10.0, bed=CatalystBed({"carbonylation": 1.5e-3}))
    assert isinstance(res.T, float) and res.T == 438.0
    assert res.T_profile.shape == res.W.shape and (res.T_profile == 438.0).all()


def test_vbf1996_fig5_reproduction():
    """VBF 1996 FIG. 5（断熱ベンチ反応器・Table 3 条件）の再現。

    原著 FIG. 5 読み取り値: 出口 T≈550.4 K, 出口 mol% CO 2.89 / MeOH 2.33 / CO2 2.07 / H2O 1.06。
    本文記述: RWGS は 3 mm で向きが反転（CO のピーク）、3 cm で熱力学平衡に到達。
    不活性は Ar 仮定（原著は内部標準に Ar を使用。Table 3 に種類の記載なし）。
    """
    import numpy as np
    from reaction_rate.case import CASE_METHANOL_VBF, run_case

    res = run_case(CASE_METHANOL_VBF)
    T = res.T_profile
    mol_pct = {s: v * 100.0 for s, v in res.mole_fractions().items()}
    z_mm = res.W / res.W[-1] * 150.0                      # 床長 0.15 m

    assert math.isclose(float(T[-1]), 550.4, abs_tol=4.0)          # 553.2 K
    assert math.isclose(mol_pct["CO"][-1],    2.89, rel_tol=0.06)  # 3.00
    assert math.isclose(mol_pct["CH3OH"][-1], 2.33, rel_tol=0.06)  # 2.24
    assert math.isclose(mol_pct["CO2"][-1],   2.07, rel_tol=0.03)  # 2.07
    assert math.isclose(mol_pct["H2O"][-1],   1.06, rel_tol=0.03)  # 1.06

    # 入口で CO が一旦増える（CO2 → CO の正方向 RWGS）→ 3 mm 付近でピーク＝反転
    assert mol_pct["CO"].max() > mol_pct["CO"][0]
    assert 2.0 < z_mm[int(np.argmax(mol_pct["CO"]))] < 6.0         # 3.8 mm
    # ΔT の 95% 到達＝熱力学平衡到達 ≈ 3 cm
    dT = T - T[0]
    assert 20.0 < z_mm[int(np.argmax(dT >= 0.95 * dT[-1]))] < 45.0  # 35 mm


def test_vbf_and_kogas_reach_same_equilibrium():
    """出口が平衡支配なら VBF と KOGAS は同じ出口状態（差は活性＝到達距離だけ）。"""
    from reaction_rate.case import CASE_METHANOL_VBF, run_case
    from dataclasses import replace

    r_vbf = run_case(CASE_METHANOL_VBF)
    r_kog = run_case(replace(CASE_METHANOL_VBF, models={"synthesis": "KOGAS"}))
    assert math.isclose(float(r_vbf.T_profile[-1]), float(r_kog.T_profile[-1]), abs_tol=0.5)
    for s in ("CO", "CH3OH", "CO2", "H2O"):
        assert math.isclose(r_vbf.outlet()[s], r_kog.outlet()[s], rel_tol=0.02)
