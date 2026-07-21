"""§3  モルデナイト カルボニル化: DME + CO → 酢酸メチル（rate_equations.html §3 に対応）。

反応:  CH3OCH3 + CO → CH3COOCH3

確立した速度則（Cheung 2006/07・Bhan 2007, Iglesia）:
  CO に一次・DME に零次。律速 = 8-MR サイドポケットの Brønsted 酸点上 CH3 への CO 挿入。
  速度 ∝ 8-MR の OH 数。MA 選択率 >99%。DTU・Cheng はこの速度則を共有。

温度依存は基準 438 K の Arrhenius/van't Hoff:  x(T)=x(438)·exp(−E/R·(1/T−1/438))。
温度依存 Ea の出所は 2 系統（別論文・値は ~15% 差で整合）:
  Cheung 2007（Iglesia/BP, 基礎速度論）: Ea≈69.6 kJ/mol（2015形 exp(−8370/T)=Cheung データの整形）
  Cheng 2017（天津大, 失活論文）        : Ea=88.65 kJ/mol
K2,K3 の van't Hoff ΔH は DTU 自身の DFT（Table 2/3）由来（両系統で共通）。

model= で選択（9 種）:
  'DTU'               : k1·pCO/(1+K2·pMA+K3·pMA/pDME)。全定数 438 K 固定。 per mol Al。 [DTU 2017]
  'Cheung2007'        : k(T)·pCO。MA阻害なし・1次・乾燥。per kg。            [Cheung→2015形, 実値]
                        k = 8.2e-5·exp(−8370/T) kmol·kg⁻¹·s⁻¹·Pa⁻¹（Ea=69.6）を直接使用。
  'Cheng2017'         : 同上 1次形だが Ea=88.65（Cheng）。per kg。★混成★     [混成]
                        大きさは Cheung(8.2e-5) の 438K 値に anchor、温度依存のみ Cheng Ea。
                        （Cheng 独自の絶対値は接触時間・質量基準で mol·(kg·s)⁻¹ に変換できないため）
  'DTU-Cheung2007-1'  : DTU式で k1 温度依存（Ea=69.6, Cheung）。K2,K3 固定。 per mol Al。
  'DTU-Cheung2007-2'  : + K2 も温度依存（van't Hoff ΔH2, DTU DFT）。
  'DTU-Cheung2007-3'  : + K3 も温度依存（van't Hoff ΔH3, DTU DFT）。
  'DTU-Cheng2017-1/2/3': 上と同じだが k1 の Ea=88.65（Cheng）。
失活（コーク）モデル deactivation() は別途（Cheng 2017 Table 1, 過渡計算）で未実装。
"""
from __future__ import annotations
import math

from .units import R

# --- DTU 2017 パラメータ（Table 4, 438 K, p8。per mol Al）---
DTU_PARAMS = {
    "k1": 2.28e-5,  # mol·(mol Al)⁻¹·s⁻¹·bar⁻¹  (CO カルボニル化速度定数)
    "K2": 4.65,     # bar⁻¹   (MA 阻害, 式6 平衡定数)
    "K3": 1.76,     # 無次元   (MA/DME 阻害)
}
T_REF = 438.0       # 温度依存の基準温度 [K]（DTU 実験温度・Cheung 438K 整合点）

# --- 温度依存 Ea（k1, [J/mol]）2 系統 ---
EA_CHEUNG = 8370.0 * R   # ≈69.6 kJ/mol。2015形 exp(−8370/T)＝Cheung 2007 データ（Cheung Fig8 70–85 と整合）
EA_CHENG = 88.65e3       # Cheng 2017（失活論文）の前進速度 Ea
# --- K2,K3 の van't Hoff ΔH（吸着, ΔH<0。DTU DFT 由来の概算, [J/mol]）---
#   K2: MA 吸着 −1.01 eV(側pocket) ≈ −97 kJ/mol。 K3: MA≈DME 吸着（差小）→ ほぼ温度非依存。
#   ⚠️ 概算（±10–20 kJ/mol、K2/K3 の DFT ステップ対応は DTU eqn 5–10 の精読要）。
DFT_DH_K2 = -97.0e3
DFT_DH_K3 = -8.0e3

# --- Cheung/2015 形（per-kg・Pa 基準の温度依存 k, 水阻害は乾燥前提で無視）---
#   ProductionofMethylAcetate.pdf（2015, Miriyam）が Cheung 2007 データを整形。
#   r = k·(pCO/(1+K_w·p_w))·(K_DME·pDME/(1+K_DME·pDME))
#     乾燥前提 K_w=0。DME 因子は零次形の DME>100% 転化を防ぐ正則化（K_DME=1 Pa⁻¹）。
#     pDME≳1e-4 bar で因子≈1、枯渇で→0。438K・50bar 微分で DTU と ~25% 整合。
CHEUNG2007 = (8.2e-5, -8370.0)   # (A [kmol·kg⁻¹·s⁻¹·Pa⁻¹], −Ea/R [K])
K_DME_REG = 1.0e5                # DME 擬似吸着定数 = 1 Pa⁻¹ = 1e5 bar⁻¹（2015 論文の正則化）


def _dme_reg(p_DME: float) -> float:
    """2015 論文の DME 正則化因子 K_DME·pDME/(1+K_DME·pDME)。DME 枯渇で速度→0（>100%転化を防ぐ）。"""
    x = K_DME_REG * p_DME
    return x / (1.0 + x)

_DTU_MODELS = ("DTU", "DTU-Cheung2007-1", "DTU-Cheung2007-2", "DTU-Cheung2007-3",
               "DTU-Cheng2017-1", "DTU-Cheng2017-2", "DTU-Cheng2017-3")
_PER_KG_MODELS = ("Cheung2007", "Cheng2017")   # network で ×酸点密度 しない（per kg）


def _vant_hoff(x_ref: float, energy: float, T: float) -> float:
    """x(T) = x(438)·exp(−energy/R·(1/T − 1/438))。energy>0=Arrhenius(k), <0=van't Hoff(吸着K)。"""
    return x_ref * math.exp(-energy / R * (1.0 / T - 1.0 / T_REF))


def rate_dtu(state, model: str = "DTU") -> float:
    """DTU eqn(11): r_MA = k1·pCO / (1 + K2·pMA + K3·pMA/pDME) [mol·(mol Al)⁻¹·s⁻¹]。
    model で k1(Ea は Cheung2007/Cheng2017 系で選択)・K2・K3 の温度依存を段階的に有効化。"""
    T = state.T
    p = state.partial_pressures()
    p_CO, p_MA, p_DME = p["CO"], p.get("MA", 0.0), p.get("DME", 0.0)

    k1, K2, K3 = DTU_PARAMS["k1"], DTU_PARAMS["K2"], DTU_PARAMS["K3"]
    if model != "DTU":
        ea = EA_CHEUNG if "Cheung2007" in model else EA_CHENG   # k1 の Ea 系統
        k1 = _vant_hoff(k1, ea, T)
        tier = model[-1]                                        # '1'/'2'/'3'
        if tier in ("2", "3"):
            K2 = _vant_hoff(K2, DFT_DH_K2, T)
        if tier == "3":
            K3 = _vant_hoff(K3, DFT_DH_K3, T)

    inhibition = 1.0 + K2 * p_MA
    if p_DME > 0.0:                       # 微分条件(pMA→0)では第3項は 0
        inhibition += K3 * p_MA / p_DME
    # DME 正則化: DTU 式は DME 零次で、pMA が低いと K3 項が効かず反応器で DME を過剰消費し得る
    # （負の DME）。2015 の因子 K_DME·pDME/(1+K_DME·pDME) を掛け、DME 枯渇で速度→0 にする。
    # pDME≳1e-4 bar で因子≈1 なので微分域(DTU の検証域)の速度は変えない。
    return k1 * p_CO / inhibition * _dme_reg(p_DME)


def rate_cheung2007(state) -> float:
    """Cheung/2015 形: r_MA = k(T)·pCO [mol·kg_cat⁻¹·s⁻¹]（per kg・MA阻害なし・乾燥前提）。
    k=8.2e-5·exp(−8370/T) kmol·kg⁻¹·s⁻¹·Pa⁻¹ を直接使用。network 側で ×酸点密度 しない。"""
    A, mEaR = CHEUNG2007
    p = state.partial_pressures()
    p_CO, p_DME = p["CO"], p.get("DME", 0.0)      # bar
    # k[kmol/(kg·s·Pa)]·pCO[bar]·1e5[Pa/bar]·1000[mol/kmol] = mol/(kg·s)、×DME正則化
    return A * math.exp(mEaR / state.T) * p_CO * 1.0e8 * _dme_reg(p_DME)


def rate_cheng2017(state) -> float:
    """★混成★ 1次形（per kg・MA阻害なし）だが温度依存のみ Cheng Ea=88.65 kJ/mol。
    大きさは Cheung2007 の k(438K) に anchor（Cheng 独自の絶対値は接触時間・質量基準で
    mol·(kg·s)⁻¹ に清潔に変換できないため）。438K では Cheung2007 と一致、他温度で傾きが異なる。"""
    A, mEaR = CHEUNG2007
    k_ref = A * math.exp(mEaR / T_REF)            # Cheung の k(438K) [kmol/(kg·s·Pa)]
    k_T = k_ref * math.exp(-EA_CHENG / R * (1.0 / state.T - 1.0 / T_REF))
    p = state.partial_pressures()
    p_CO, p_DME = p["CO"], p.get("DME", 0.0)
    return k_T * p_CO * 1.0e8 * _dme_reg(p_DME)   # → mol/(kg·s), per kg（×DME正則化）


def rate(state, model: str = "DTU") -> float:
    """r_MA を返す。model は docstring の 9 種。
    DTU 系は per mol Al [mol·(mol Al)⁻¹·s⁻¹]、Cheung2007/Cheng2017 は per kg [mol·kg⁻¹·s⁻¹]。"""
    if model in _DTU_MODELS:
        return rate_dtu(state, model)
    if model == "Cheung2007":
        return rate_cheung2007(state)
    if model == "Cheng2017":
        return rate_cheng2017(state)
    raise ValueError(f"unknown model: {model!r}")


def deactivation(state, a: float) -> float:
    """da/dt（Cheng 2017 失活）。a=相対活性。Cheng Table 1 の k_di 転記＋過渡計算が必要で未実装。"""
    raise NotImplementedError("失活モデル: −da/dt = Σ k_di·y_i·a^d（Cheng 2017 Table 1・過渡計算）")
