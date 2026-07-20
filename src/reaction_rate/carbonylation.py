"""§3  モルデナイト カルボニル化: DME + CO → 酢酸メチル（rate_equations.html §3 に対応）。

反応:  CH3OCH3 + CO → CH3COOCH3

確立した速度則（Cheung 2006/07・Bhan 2007, Iglesia）:
  CO に一次・DME に零次。律速 = 8-MR サイドポケットの Brønsted 酸点上 CH3 への CO 挿入。
  速度 ∝ 8-MR の OH 数。MA 選択率 >99%。DTU・Cheng はこの速度則を共有。

model= で選択:
  'DTU'  : r_MA = k1·pCO / (1 + K2·pMA + K3·pMA/pDME)   [mol·(mol Al)⁻¹·s⁻¹], 圧力[bar]
           k1 は 438 K の実験値。DTU 原著は単一温度 438 K で、温度依存は与えていない。
           微分条件(pMA→0)で r_MA ≈ k1·pCO
  'Cheng': r_MA = dy_MA/d(W/F_DME) = k·f_CO             （モル分率・接触時間・CO フガシティ基準）
"""
from __future__ import annotations

# --- DTU 2017 パラメータ（Table 4, 438 K）---
DTU_PARAMS = {
    "k1": 2.28e-5,  # mol·(mol Al)⁻¹·s⁻¹·bar⁻¹  (CO カルボニル化速度定数)
    "K2": 4.65,     # bar⁻¹   (MA 阻害, 式6 平衡定数)
    "K3": 1.76,     # 無次元   (MA/DME 阻害)
}
# ⚠️ DTU 原著は 438 K の単一温度のみ。温度依存(Arrhenius/活性化エネルギー)は与えていない。
#    438 K 以外での使用は「438 K 実験値の外挿」であり、原著の裏付けはない。

# --- Cheng 2017 パラメータ（本文 Table 1）---
#   r_MA = k·f_CO,  失活: −da/dt = Σ_i k_di·y_i·a^d,  a = r_MA/(r_MA)0
#   TODO(実装フェーズ): cheng2017.pdf Table 1 から k, k_di(DME/CO/MA), d を転記。
#     Cheng は接触時間基準(dy_MA/d(W/F)) で DTU と単位系が異なるため、
#     採用する速度基準を決めてから実装する。
CHENG_PARAMS: dict = {}


def rate_dtu(state) -> float:
    """DTU eqn(11): r_MA = k1·pCO / (1 + K2·pMA + K3·pMA/pDME) [mol·(mol Al)⁻¹·s⁻¹]。
    k1 は 438 K 実験値（原著は単一温度）。他温度で使う場合は外挿である点に注意。"""
    p = state.partial_pressures()
    p_CO = p["CO"]
    p_MA = p.get("MA", 0.0)
    p_DME = p.get("DME", 0.0)

    inhibition = 1.0 + DTU_PARAMS["K2"] * p_MA
    if p_DME > 0.0:                       # 微分条件(pMA→0)では第3項は 0
        inhibition += DTU_PARAMS["K3"] * p_MA / p_DME
    return DTU_PARAMS["k1"] * p_CO / inhibition


def rate(state, model: str = "DTU") -> float:
    """r_MA を返す。model='DTU'(438K実験値, 圧力基準) / 'Cheng'(mol基準, f_CO)。"""
    if model == "DTU":
        return rate_dtu(state)
    if model == "Cheng":
        raise NotImplementedError(
            "Cheng は接触時間基準(r_MA=k·f_CO, dy_MA/d(W/F))で DTU と単位系が異なる。"
            "cheng2017.pdf Table 1 の k を転記し、速度基準を決めてから実装する。"
        )
    raise ValueError(f"unknown model: {model!r}")


def deactivation(state, a: float) -> float:
    """da/dt（Cheng eqn8, 失活）。a=相対活性。Cheng Table 1 の k_di 転記後に実装。"""
    raise NotImplementedError("実装フェーズ: −da/dt = Σ k_di·y_i·a^d（Cheng Table 1）")
