"""§3 モルデナイト カルボニル化の検証。"""
import math
import pytest
from reaction_rate import carbonylation
from reaction_rate.state import GasState


def test_dtu_params():
    assert carbonylation.DTU_PARAMS == {"k1": 2.28e-5, "K2": 4.65, "K3": 1.76}


def test_dtu_differential_first_order_in_CO():
    # 微分条件(pMA=0, DME存在)で r_MA ≈ k1·pCO（DME正則化因子は pDME≳1e-4bar で ~1）
    s = GasState(438.0, 15.0, {"CO": 0.98, "DME": 0.02, "MA": 0.0})
    r = carbonylation.rate(s, model="DTU")
    p = s.partial_pressures()
    expect = carbonylation.DTU_PARAMS["k1"] * p["CO"] * carbonylation._dme_reg(p["DME"])
    assert math.isclose(r, expect, rel_tol=1e-9)
    # DME 実質零次: DME 分圧を変えても微分レートはほぼ不変（正則化による差は ~1e-4 以内）
    s2 = GasState(438.0, 15.0, {"CO": 0.98, "DME": 0.5, "MA": 0.0})
    assert math.isclose(r, carbonylation.rate(s2, "DTU"), rel_tol=1e-3)


def test_dtu_MA_inhibition():
    # MA があると阻害でレート低下
    s_fresh = GasState(438.0, 15.0, {"CO": 0.97, "DME": 0.03, "MA": 0.0})
    s_MA = GasState(438.0, 15.0, {"CO": 0.90, "DME": 0.03, "MA": 0.07})
    assert carbonylation.rate(s_MA, "DTU") < carbonylation.rate(s_fresh, "DTU")


def test_dtu_variants_equal_dtu_at_reference_T():
    # 基準 438 K では DTU-Cheung/Cheng 系はすべて DTU と一致（exp(0)=1）
    s = GasState(438.0, 50.0, {"CO": 0.30, "DME": 0.02, "MA": 0.005})
    r0 = carbonylation.rate(s, "DTU")
    for m in ("DTU-Cheung2007-1", "DTU-Cheung2007-2", "DTU-Cheung2007-3",
              "DTU-Cheng2017-1", "DTU-Cheng2017-2", "DTU-Cheng2017-3"):
        assert math.isclose(carbonylation.rate(s, m), r0, rel_tol=1e-9)


def test_k1_temperature_dependence_two_Ea():
    # k1 は高温で増加（DTU固定より大）。Cheng(Ea大) は Cheung(Ea小) より急峻
    s_hi = GasState(523.15, 50.0, {"CO": 0.30, "DME": 0.02, "MA": 0.005})
    r_dtu = carbonylation.rate(s_hi, "DTU")
    r_cheung = carbonylation.rate(s_hi, "DTU-Cheung2007-1")
    r_cheng = carbonylation.rate(s_hi, "DTU-Cheng2017-1")
    assert r_cheng > r_cheung > r_dtu
    # Ea=88.65 の Arrhenius 比は ~50 倍、Ea=69.6 は ~22 倍
    assert 45 < carbonylation._vant_hoff(1.0, carbonylation.EA_CHENG, 523.15) < 60
    assert 18 < carbonylation._vant_hoff(1.0, carbonylation.EA_CHEUNG, 523.15) < 27


def test_K2_decreases_with_T():
    # -2 は K2 が温度依存（MA吸着↓）→ 高温で阻害が弱まり -1 より大きい
    s_hi = GasState(523.15, 50.0, {"CO": 0.30, "DME": 0.02, "MA": 0.02})
    assert carbonylation.rate(s_hi, "DTU-Cheng2017-2") > carbonylation.rate(s_hi, "DTU-Cheng2017-1")


def test_cheung2007_consistent_with_dtu_at_ref():
    # Cheung2007(per kg 直接) は 438K で DTU(per mol Al×酸点密度) と ~25%以内で整合
    s = GasState(438.0, 50.0, {"CO": 0.30, "DME": 0.02, "MA": 0.0})
    r_kg = carbonylation.rate(s, "Cheung2007")                        # per kg
    r_dtu_kg = carbonylation.rate(s, "DTU") * 1.43                    # per mol Al → per kg
    assert 0.7 < r_kg / r_dtu_kg < 1.5


def test_cheng2017_hybrid_matches_cheung_at_ref():
    # Cheng2017(混成) は 438K で Cheung2007 と一致（大きさを anchor）、523K では Ea大で上回る
    s_ref = GasState(438.0, 50.0, {"CO": 0.30, "DME": 0.02, "MA": 0.0})
    s_hi = GasState(523.15, 50.0, {"CO": 0.30, "DME": 0.02, "MA": 0.0})
    assert carbonylation.rate(s_ref, "Cheng2017") == pytest.approx(carbonylation.rate(s_ref, "Cheung2007"))
    assert carbonylation.rate(s_hi, "Cheng2017") > carbonylation.rate(s_hi, "Cheung2007")


def test_standalone_first_order_no_MA_inhibition():
    # Cheung2007/Cheng2017 は CO 一次・MA 阻害なし: MA を足しても不変
    s0 = GasState(438.0, 50.0, {"CO": 0.30, "DME": 0.02, "MA": 0.0})
    sMA = GasState(438.0, 50.0, {"CO": 0.30, "DME": 0.02, "MA": 0.05})
    for m in ("Cheung2007", "Cheng2017"):
        assert carbonylation.rate(sMA, m) == pytest.approx(carbonylation.rate(s0, m))


def test_unknown_model_raises():
    s = GasState(438.0, 50.0, {"CO": 0.30, "DME": 0.02, "MA": 0.0})
    with pytest.raises(ValueError):
        carbonylation.rate(s, model="nonsense")


def test_dtu_third_term_uses_reciprocal_K3():
    """DTU eqn(11) の第3項は K3⁻¹·pMA/pDME（K3 は eqn(7) の平衡定数）。掛ける実装への回帰防止。"""
    s = GasState(438.0, 50.0, {"CO": 0.60, "DME": 0.15, "MA": 0.25})
    p = s.partial_pressures()
    k1, K2, K3 = (carbonylation.DTU_PARAMS[k] for k in ("k1", "K2", "K3"))
    den = 1.0 + K2 * p["MA"] + p["MA"] / (K3 * p["DME"])
    expect = k1 * p["CO"] / den * carbonylation._dme_reg(p["DME"])
    assert carbonylation.rate(s, "DTU") == pytest.approx(expect, rel=1e-12)


def test_dtu_reproduces_paper_surface_coverages():
    """DTU p8: 100 bar・2 vol% DME in CO・1.5 g 出口で methyl/acetyl/CH3-MA = 21/7/72%。
    PFR 積分＋酸点密度換算まで含めた end-to-end 検証（K3 を掛けると 19/18/63% でずれる）。"""
    from reaction_rate.reactors import pfr, CatalystBed

    flow = 300.0 / 22414.0 / 60.0                      # 300 Nml/min → mol/s
    res = pfr({"CO": 0.98 * flow, "DME": 0.02 * flow, "MA": 0.0}, T=438.0, P=100.0,
              bed=CatalystBed({"carbonylation": 1.5e-3}, acid_site_density=1.43),
              models={"carbonylation": "DTU"})
    y = res.mole_fractions()
    p_MA, p_DME = y["MA"][-1] * 100.0, y["DME"][-1] * 100.0
    K2, K3 = carbonylation.DTU_PARAMS["K2"], carbonylation.DTU_PARAMS["K3"]
    den = 1.0 + K2 * p_MA + p_MA / (K3 * p_DME)        # eqn(8): Θ_CH3 = 1/den
    theta = (100.0 / den, 100.0 * (p_MA / (K3 * p_DME)) / den, 100.0 * K2 * p_MA / den)
    assert theta == pytest.approx((21.0, 7.0, 72.0), abs=0.6)
