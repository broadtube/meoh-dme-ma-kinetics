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
    assert math.isclose(X_DME, 0.09, abs_tol=0.02)     # 9.5% ≈ 9%
    assert math.isclose(TOF, 0.68, rel_tol=0.10)       # 0.71 ≈ 0.68


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
