"""§2 VBF/KOGAS の検証。"""
import math
from reaction_rate import vbf_kogas
from reaction_rate.state import GasState

T = 523.15


def test_equilibrium_literature():
    eq = vbf_kogas.equilibrium(T)
    # log10 K_eq1 = 3066/T − 10.592 → 1.86e-5
    assert math.isclose(eq["K_eq1"], 10 ** (3066 / T - 10.592), rel_tol=1e-9)
    # WGS 正反応 K ≈ 86 @523K
    assert math.isclose(eq["K_eq2"], 85.8, rel_tol=0.02)


def test_kogas_vs_vbf_params():
    # A 比 KOGAS/VBF ≈ 1.54, B は一致（同一 Ea）
    assert math.isclose(
        vbf_kogas.MS_PARAMS["KOGAS"]["k1"][0] / vbf_kogas.MS_PARAMS["VBF"]["k1"][0],
        1.54, rel_tol=0.02,
    )
    for key in ("k1", "K3", "K4", "k5"):
        assert vbf_kogas.MS_PARAMS["KOGAS"][key][1] == vbf_kogas.MS_PARAMS["VBF"][key][1]


def test_dehydration_kogas_vs_bl_gap():
    # KOGAS(=Ng) 脱水速度は Bercič–Levec 原著の 6〜16 倍（本作業で確認済）
    s = GasState(T, 50.0, {"CH3OH": 0.5, "DME": 0.1, "H2O": 0.1, "CO": 0.3})
    r_kogas = vbf_kogas.rate_dehydration(s, source="KOGAS")
    r_bl = vbf_kogas.rate_dehydration(s, source="BercicLevec1993")
    assert 6.0 < r_kogas / r_bl < 16.0


def test_ms_positive():
    s = GasState(T, 50.0, {"CO": 0.15, "CO2": 0.08, "H2": 0.72, "H2O": 0.01, "CH3OH": 0.04})
    assert vbf_kogas.rate_ms_rwgs(s, "KOGAS")["r_MS"] > 0
