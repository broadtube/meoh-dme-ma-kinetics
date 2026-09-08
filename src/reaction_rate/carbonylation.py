"""§3  モルデナイト カルボニル化: DME + CO → 酢酸メチル（rate_equations.html §3 に対応）。

反応:  CH3OCH3 + CO → CH3COOCH3

確立した速度則（Cheung 2006/07・Bhan 2007, Iglesia）:
  CO に一次・DME に零次。律速 = 8-MR サイドポケットの Brønsted 酸点上 CH3 への CO 挿入。
  速度 ∝ 8-MR の OH 数。MA 選択率 >99%。DTU・Cheng はこの速度則を共有。

温度依存は基準 438 K の Arrhenius/van't Hoff:  x(T)=x(438)·exp(−E/R·(1/T−1/438))。
温度依存 Ea の出所は 2 系統（別論文・値は ~15% 差で整合）:
  Cheung 2007（Iglesia/BP, 基礎速度論）: Ea≈69.6 kJ/mol（exp(−8370/T)=Cheung データの整形形。
                                          primary は Diemer & Luyben 2010 IECR 査読 eq6, 非査読 Miriyam2015 も同形）
  Cheng 2017（天津大, 失活論文）        : Ea=88.65 kJ/mol
K2,K3 の van't Hoff ΔH は ⚠️ DTU が報告した値ではなく、DTU の DFT から当実装が当てた推定
（対応づけ・不確かさは下の DFT_DH_K2/K3 のコメント参照）。既定は tier 1（K2,K3 固定）を推奨。

model= で選択（9 種）:
  'DTU'               : k1·pCO/(1+K2·pMA+K3⁻¹·pMA/pDME)。全定数 438 K 固定。 per mol Al。 [DTU 2017]
  'Cheung2007'        : k(T)·pCO。MA阻害なし・1次・乾燥。per kg。   [Cheung→Diemer&Luyben2010形, 実値]
                        k = 8.2e-5·exp(−8370/T) kmol·kg⁻¹·s⁻¹·Pa⁻¹（Ea=69.6）を直接使用。
  'Cheng2017'         : 同上 1次形だが Ea=88.65（Cheng）。per kg。★混成★     [混成]
                        大きさは Cheung(8.2e-5) の 438K 値に anchor、温度依存のみ Cheng Ea。
                        （Cheng 独自の絶対値は接触時間・質量基準で mol·(kg·s)⁻¹ に変換できないため）
  'DTU-Cheung2007-1'  : DTU式で k1 温度依存（Ea=69.6, Cheung）。K2,K3 固定。 per mol Al。
  'DTU-Cheung2007-2'  : + K2 も温度依存（van't Hoff ΔH2 推定値）。★感度解析用★
  'DTU-Cheung2007-3'  : + K3 も温度依存（van't Hoff ΔH3 推定値）。★感度解析用・信頼度最低★
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
EA_CHEUNG = 8370.0 * R   # ≈69.6 kJ/mol。exp(−8370/T)＝Cheung 2007 データの整形（Diemer&Luyben2010, Fig8 70–85 と整合）
EA_CHENG = 88.65e3       # Cheng 2017（失活論文）の前進速度 Ea
# --- K2,K3 の van't Hoff ΔH [J/mol] ---
# ⚠️ DTU は ΔH を一切報告していない。以下は DTU の DFT から当実装が当てた推定値であり、
#    「DFT のどの数値を ΔH に対応させるか」が自明でない点が最大の不確かさ。
#
# ΔH2 = -97 kJ/mol【採用】: DTU Table 2（PDF p.3=誌面1143）の MA on H–Z, T3-O3 = -1.01 eV。
#   ただし K2 の定義 eqn(6) は MA + CH3–Z ⇌ C（メチル化席の閉塞）で、Table 2 は
#   プロトン席(H–Z)への吸着＝別反応。eqn(6) を字義どおり DFT で見ると Table 3 の
#   CH3–MA+ 錯体生成は吸熱 +0.24/+0.48 eV(主channel/側pocket) = +23〜+46 kJ/mol で符号が逆。
#   それでも Table 2 を採るのは DTU 本文(PDF p.7=誌面1147)自身が
#   "the CH3–MA species ... cannot explain the detrimental effect of MA ... However, MA binds
#    to protonated sites with sufficient strength to inhibit methylation (Table 2)" と述べ、
#   阻害の実体をプロトン席吸着に帰しているため。
#   熱力学整合(当実装): K2=4.65 bar^-1 @438K → ΔG°=-5.6 kJ/mol。
#     ΔH=-97 → ΔS°=-210 J/(mol·K)（強い化学吸着として説明可だが経験則 -100〜-190 よりやや大 ⇒ 上限寄り）
#     ΔH=+46 → ΔS°=+118 J/(mol·K)（会合反応でエントロピー増＝物理的に不可）⇒ 発熱側が支持される
#
# ΔH3 = -34 kJ/mol【tier 3 のみ・信頼度低】: DTU Table 3（PDF p.4=誌面1144）の
#   DME+CH3CO–Z→CH3–MA+ + Z- (ΔE=+0.13) と CH3–MA+ + Z-→MA+CH3–Z (ΔE=-0.48) の和
#   = eqn(7) 全体 ΔE = -0.35 eV @T3-O3（主channel は -0.48 eV = -46 kJ/mol）。
#   ただし eqn(7) は両辺の気体分子数が等しい交換反応で ΔS°≈+25 J/(mol·K)(気相 MA-DME)程度のはずで、
#   fit 値 K3=1.76 (ΔG°=-2.1 kJ/mol) と組むと ΔH3≈+9 kJ/mol となり DFT の -34 と符号すら食い違う。
#   ⇒ 既定では K3 を動かさない（tier 3 は上限側の感度確認用）。
#   （旧値 -8.0e3 は Table 2 の MA/DME 吸着差 0.02 eV から取っていたが、これは eqn(7) とは
#     無関係な量の混同だったため撤回）
DFT_DH_K2 = -97.0e3
DFT_DH_K3 = -33.8e3

# --- Cheung 整形形（per-kg・Pa 基準の温度依存 k, 水阻害は乾燥前提で無視）---
#   Diemer & Luyben 2010（IECR 49, 12224, 査読, eq6-7, DOI 10.1021/ie101583j）が
#   Cheung 2007 データを Aspen 向けに整形。同形は非査読 Miriyam 2015 レポートにも掲載
#   （2015 は本式を Diemer&Luyben から転記した可能性が高い）。primary は Diemer&Luyben 2010。
#   r = k·(pCO/(1+K_w·p_w))·(K_DME·pDME/(1+K_DME·pDME))
#     乾燥前提 K_w=0。DME 因子は零次形の DME>100% 転化を防ぐ正則化（K_DME=1 Pa⁻¹）。
#     pDME≳1e-4 bar で因子≈1、枯渇で→0。438K・50bar 微分で DTU と ~25% 整合。
CHEUNG2007 = (8.2e-5, -8370.0)   # (A [kmol·kg⁻¹·s⁻¹·Pa⁻¹], −Ea/R [K]) Diemer&Luyben2010 eq6
K_DME_REG = 1.0e5                # DME 擬似吸着定数 = 1 Pa⁻¹ = 1e5 bar⁻¹（Diemer&Luyben2010 eq7 の正則化）


def _dme_reg(p_DME: float) -> float:
    """Diemer&Luyben2010 の DME 正則化因子 K_DME·pDME/(1+K_DME·pDME)。DME 枯渇で速度→0（>100%転化を防ぐ）。"""
    x = K_DME_REG * p_DME
    return x / (1.0 + x)

_DTU_MODELS = ("DTU", "DTU-Cheung2007-1", "DTU-Cheung2007-2", "DTU-Cheung2007-3",
               "DTU-Cheng2017-1", "DTU-Cheng2017-2", "DTU-Cheng2017-3")
_PER_KG_MODELS = ("Cheung2007", "Cheng2017")   # network で ×酸点密度 しない（per kg）


def _vant_hoff(x_ref: float, energy: float, T: float) -> float:
    """x(T) = x(438)·exp(−energy/R·(1/T − 1/438))。energy>0=Arrhenius(k), <0=van't Hoff(吸着K)。"""
    return x_ref * math.exp(-energy / R * (1.0 / T - 1.0 / T_REF))


def rate_dtu(state, model: str = "DTU") -> float:
    """DTU eqn(11): r_MA = k1·pCO / (1 + K2·pMA + K3⁻¹·pMA/pDME) [mol·(mol Al)⁻¹·s⁻¹]。
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
        # ⚠️ 原著 eqn(10)(11) は K3⁻¹（K3 は eqn(7) DME+CH3CO-Z⇌MA+CH3-Z の平衡定数なので
        #    θ_acetyl/θ_CH3 = K3⁻¹·pMA/pDME）。Table 4 の 1.76 はそのまま「割る」値。
        #    検証: 100 bar 出口被覆率が論文 p8 の methyl/acetyl/CH3-MA = 21/7/72% を再現する。
        inhibition += p_MA / (K3 * p_DME)
    # DME 正則化: DTU 式は DME 零次で、pMA が低いと K3 項が効かず反応器で DME を過剰消費し得る
    # （負の DME）。Diemer&Luyben2010 の因子 K_DME·pDME/(1+K_DME·pDME) を掛け、DME 枯渇で速度→0 にする。
    # pDME≳1e-4 bar で因子≈1 なので微分域(DTU の検証域)の速度は変えない。
    return k1 * p_CO / inhibition * _dme_reg(p_DME)


def rate_cheung2007(state) -> float:
    """Cheung 整形形(Diemer&Luyben2010): r_MA = k(T)·pCO [mol·kg_cat⁻¹·s⁻¹]（per kg・MA阻害なし・乾燥前提）。
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
