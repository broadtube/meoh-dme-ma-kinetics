"""§3 モルデナイト カルボニル化の検証。"""
import math
import pytest
from reaction_rate import carbonylation
from reaction_rate.state import GasState


def test_dtu_params():
    assert carbonylation.DTU_PARAMS == {"k1": 2.28e-5, "K2": 4.65, "K3": 1.76}


def test_dtu_differential_first_order_in_CO():
    # 微分条件(pMA=0)で r_MA = k1·pCO（CO一次・DME零次）
    s = GasState(438.0, 15.0, {"CO": 0.98, "DME": 0.02, "MA": 0.0})
    r = carbonylation.rate(s, model="DTU")
    p_CO = s.partial_pressures()["CO"]
    assert math.isclose(r, carbonylation.DTU_PARAMS["k1"] * p_CO, rel_tol=1e-9)
    # DME 零次: DME 分圧を変えても微分レートは不変
    s2 = GasState(438.0, 15.0, {"CO": 0.98, "DME": 0.5, "MA": 0.0})
    assert math.isclose(r, carbonylation.rate(s2, "DTU"), rel_tol=1e-9)


def test_dtu_MA_inhibition():
    # MA があると阻害でレート低下
    s_fresh = GasState(438.0, 15.0, {"CO": 0.97, "DME": 0.03, "MA": 0.0})
    s_MA = GasState(438.0, 15.0, {"CO": 0.90, "DME": 0.03, "MA": 0.07})
    assert carbonylation.rate(s_MA, "DTU") < carbonylation.rate(s_fresh, "DTU")


def test_cheng_not_implemented():
    s = GasState(478.0, 15.0, {"CO": 0.93, "DME": 0.02, "N2": 0.05})
    with pytest.raises(NotImplementedError):
        carbonylation.rate(s, model="Cheng")
