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


def test_dehydration_zsm5_in_measured_range():
    # ZSM5=Ortega 2018 LHHW。実測レンジ r_MeOH=0.001–0.07 mol/(kg·s)(140–190℃)に入る（175℃,p_M=0.5bar 純）
    s = GasState(448.15, 0.5, {"CH3OH": 1.0})
    r_meoh = 2.0 * vbf_kogas.rate_dehydration(s, source="ZSM5", k_eq3="thermo")
    assert 0.001 <= r_meoh <= 0.07


def test_dehydration_zsm5_reaction_order_methanol():
    # 論文明記の n_M=0.07–0.45（Eq.20）を再現する = 分母のメタノール項が √(K_M·p_M) である証左。
    # n_M = d ln r / d ln p_M を数値微分（175℃, p_M=0.5bar, 生成物≈0）。
    import math
    T, pM = 448.15, 0.5
    def r(pm):
        return vbf_kogas.rate_dehydration(GasState(T, pm, {"CH3OH": 1.0}), source="ZSM5", k_eq3="thermo")
    n_m = (math.log(r(pM * 1.01)) - math.log(r(pM / 1.01))) / (math.log(pM * 1.01) - math.log(pM / 1.01))
    assert 0.07 <= n_m <= 0.45


def test_dehydration_zsm5_matches_dalena_lowP():
    # 低圧の絶対活性は独立の Dalena 2021（160℃・希薄 p_M≈0.057bar で r_MeOH=5.23e-3）と同オーダー（~30%）
    s = GasState(433.15, 0.057, {"CH3OH": 1.0})
    r_meoh = 2.0 * vbf_kogas.rate_dehydration(s, source="ZSM5", k_eq3="thermo")
    assert math.isclose(r_meoh, 5.23e-3, rel_tol=0.35)


def test_dehydration_zsm5_saturates_at_high_pressure():
    # LHHW は高 p_M で飽和（プラトー漸近）→ 10倍加圧しても速度は 2倍未満（純2次なら ~100倍）
    y = {"CH3OH": 1.0}
    r_lo = vbf_kogas.rate_dehydration(GasState(433.15, 0.5, y), source="ZSM5", k_eq3="thermo")
    r_hi = vbf_kogas.rate_dehydration(GasState(433.15, 5.0, y), source="ZSM5", k_eq3="thermo")
    assert r_hi / r_lo < 2.0


def test_dehydration_zsm5_faster_than_gamma_at_low_T():
    # ZSM-5 は低温で γ-アルミナ(KOGAS)より高活性（190℃・1bar 混合で ~19倍）
    s = GasState(463.15, 1.0, {"CH3OH": 0.5, "H2O": 0.05, "DME": 0.05, "N2": 0.4})
    r_zsm5 = vbf_kogas.rate_dehydration(s, source="ZSM5", k_eq3="thermo")
    r_gamma = vbf_kogas.rate_dehydration(s, source="KOGAS", k_eq3="thermo")
    assert r_zsm5 / r_gamma > 10.0
