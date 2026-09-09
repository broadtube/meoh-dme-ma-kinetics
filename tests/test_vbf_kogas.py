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


# ============================================================
#  Ortega 2018 の再現（examples/ortega_fig4.py と同じ検証を数値だけで）
# ============================================================
def _r_meoh(T_c, p_M, p_W=0.0, p_D=0.0):
    """r_MeOH [mol_MeOH·kg⁻¹·s⁻¹]（rate_dehydration は DME 基準なので ×2）。"""
    P = p_M + p_W + p_D
    y = {"CH3OH": p_M / P, "H2O": p_W / P, "DME": p_D / P}
    return 2.0 * vbf_kogas.rate_dehydration(GasState(T_c + 273.15, P, y),
                                            source="ZSM5", k_eq3="thermo")


def _rate_at_outlet(T_c, p_M, whsv=100.0):
    """出口 p_MeOH を固定した無勾配反応器の速度（examples/ortega_fig4.rate_at_outlet と同一）。"""
    from scipy.optimize import brentq
    fw = whsv / 3600.0 / 32.042e-3
    g = lambda X: _r_meoh(T_c, p_M, p_M * X / (2 * (1 - X)), p_M * X / (2 * (1 - X))) - fw * X
    return fw * brentq(g, 1e-12, 0.9)


def test_ortega_si_anchor():
    """SI Table S6/S7（デジタイズ不要の厳密値）: 190℃, p_MeOH=0.931 bar, 乾燥, WHSV 100 h⁻¹ で
    観測速度 89.9 mol·m⁻³cat·s⁻¹ ÷ 床密度 1300 kg·m⁻³ = 0.06915 mol_MeOH·kg⁻¹·s⁻¹。
    SI Table S3/S4 の転化率 0.081 からの独立検算 (0.0702) とも整合する。"""
    assert math.isclose(_rate_at_outlet(190.0, 0.931), 89.9 / 1300.0, rel_tol=0.05)


def test_ortega_fig4_grid():
    """Fig. 4（速度 vs p_MeOH, 乾燥・WHSV 100 h⁻¹）。152℃ 以上は ±15% 以内で再現する。
    140℃ の 3 点は Fig.4 が線形軸で読み取り誤差が大きいので Fig.6（対数軸）で見る。"""
    fig4 = {152: [(0.322, 0.00706), (0.625, 0.00721), (0.957, 0.00710)],
            165: [(0.312, 0.01478), (0.625, 0.01531), (0.954, 0.01671)],
            177: [(0.312, 0.02810), (0.616, 0.03092), (0.941, 0.03296)],
            190: [(0.299, 0.05443), (0.620, 0.06359), (0.906, 0.06889)]}
    errs = [abs(_rate_at_outlet(T, p) / r - 1) for T, row in fig4.items() for p, r in row]
    assert max(errs) < 0.15
    assert sum(errs) / len(errs) < 0.05


def test_ortega_fig4_nearly_zero_order_at_low_T():
    """論文本文の主張「165℃ 以下では p_MeOH に零次、177/190℃ ではわずかに正」を再現する
    （分母が 2√(K_M·p_M) である証左）。p_M を 0.3→0.95 に振ったときの速度比で見る。"""
    ratio = {T: _rate_at_outlet(T, 0.95) / _rate_at_outlet(T, 0.30) for T in (140, 165, 190)}
    assert 1.0 < ratio[140] < 1.15          # ほぼ零次
    assert ratio[140] < ratio[165] < ratio[190]
    assert 1.20 < ratio[190] < 1.45         # 高温側でははっきり正


def test_ortega_fig7_water_inhibition():
    """Fig. 7（70 wt% MeOH / 30 wt% H2O, WHSV 14 h⁻¹）を CSTR で解いて再現する。
    水阻害は低温ほど強く効く（乾燥比が 140℃ で ~0.35、190℃ で ~0.6）。"""
    from reaction_rate.reactors import cstr, CatalystBed

    def wet(T_c, whsv=14.0):
        F_M = 1e-4
        F_in = {"CH3OH": F_M, "H2O": F_M * (0.30 / 18.015) / (0.70 / 32.042), "DME": 0.0}
        W = F_M / (whsv / 3600.0 / 32.042e-3)
        out = cstr(F_in, T_c + 273.15, 1.0, CatalystBed({"dehydration": W}),
                   models={"dehydration": "ZSM5"}, k_eq3="thermo")
        return (F_in["CH3OH"] - out["CH3OH"]) / W

    for T, r_exp in ((140.0, 0.001013), (164.7, 0.008593), (190.2, 0.037761)):
        assert math.isclose(wet(T), r_exp, rel_tol=0.10)
    assert wet(140.0) / _rate_at_outlet(140.0, 0.95) < 0.45      # 低温は水で強く落ちる
    assert wet(190.0) / _rate_at_outlet(190.0, 0.95) > 0.50      # 高温では緩む


def test_aspen_lhhw_exact_equals_ortega():
    """aspen_lhhw.ORTEGA_EXACT は Ortega Eq.(16) の厳密な書き換え（フィットではない）。
    Aspen LHHW の指数に実数（吸着項 p_M^0.5・逆反応項 p_M^-1）を使えることが前提。"""
    from reaction_rate import aspen_lhhw as al

    for T_c in (140.0, 190.0, 250.0, 300.0):
        for p_M, p_W, p_D in ((0.95, 0.01, 0.01), (0.30, 0.30, 0.30),
                              (0.12, 0.39, 0.63), (5.0, 2.0, 2.0)):
            got = al.rate(T_c + 273.15, {"CH3OH": p_M, "H2O": p_W, "DME": p_D})
            assert math.isclose(got, _r_meoh(T_c, p_M, p_W, p_D), rel_tol=1e-11)


def test_aspen_lhhw_exact_reproduces_si_anchor():
    """厳密形も SI Table S6/S7 のアンカーを再現する（元実装と同じ −1.2%）。"""
    from reaction_rate import aspen_lhhw as al
    from scipy.optimize import brentq

    fw = 100.0 / 3600.0 / 32.042e-3
    p_M = 0.931
    f = lambda T, pm, pw, pd: al.rate(T + 273.15, {"CH3OH": pm, "H2O": pw, "DME": pd})
    g = lambda X: f(190.0, p_M, p_M * X / (2 * (1 - X)), p_M * X / (2 * (1 - X))) - fw * X
    assert math.isclose(fw * brentq(g, 1e-12, 0.9), 89.9 / 1300.0, rel_tol=0.05)


def test_aspen_lhhw_kinetic_factor_is_39kJ():
    """kinetic factor の E は E_app + ΔH_M（吸着熱を吸収した見かけの値）。"""
    from reaction_rate import aspen_lhhw as al
    assert math.isclose(al.ORTEGA_EXACT.E,
                        vbf_kogas.EAPP_ORT + vbf_kogas.DH_M_ORT, rel_tol=1e-12)
    assert math.isclose(al.ORTEGA_EXACT.E / 1e3, 39.0, abs_tol=0.01)


def test_aspen_lhhw_linear_fit_is_worse():
    """線形吸着項に制限した LINEAR_FIT は関数形が違うので誤差が残る（外挿禁止の根拠）。"""
    from reaction_rate import aspen_lhhw as al

    errs = [abs(al.rate(T + 273.15, {"CH3OH": pm, "H2O": pw, "DME": pd}, al.LINEAR_FIT)
                / _r_meoh(T, pm, pw, pd) - 1.0)
            for T, pm, pw, pd in ((140.0, 0.95, 0.01, 0.01), (190.0, 0.93, 0.04, 0.04),
                                  (250.0, 0.30, 0.30, 0.30))]
    assert max(errs) > 0.10          # 厳密形なら 1e-11 で一致する
    assert max(errs) < 0.60          # それでもオーダーは合っている
