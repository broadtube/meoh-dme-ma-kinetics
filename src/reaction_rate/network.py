"""反応の量論 → 成分ごとの正味生成速度 Rᵢ [mol·kg_cat⁻¹·s⁻¹]。

各レート則(§1–§3)を量論で束ね、反応器が使う「成分ごとの速度」に変換する。

ハイブリッド触媒 = 各触媒の反応が共存するだけ、という考え方（CatalystBed）:
  bed.masses = {"synthesis": kg, "dehydration": kg, "carbonylation": kg}
  各触媒の反応レートを「その触媒の質量分率」で重み付けして合算する。
  → 単一触媒(methanol合成/カルボニル化)も、DME合成(合成+脱水)も、
    さらに合成+脱水+カルボニル化のタンデムも、すべて同じ仕組みで書ける。

触媒の役割(role)と反応:
  'synthesis'     : Cu/ZnO/Al2O3。model='KOGAS'(既定)/'VBF'/'Graaf1988'。MS/RWGS。
                    ※ Graaf はフガシティ基準・純メタノール合成向け（ハイブリッドでは非推奨）。
  'dehydration'   : γ-Al2O3。model='KOGAS'(既定)/'BercicLevec1993'。MD。
  'carbonylation' : H-MOR。model='DTU'。r_MA[mol/(mol Al)/s]×acid_site_density で kg基準へ。
"""
from collections import defaultdict

from . import graaf, vbf_kogas, carbonylation


def _role_rates(state, role, model, bed, k_eq3):
    """1つの触媒(role)の、その触媒 kg あたりの成分速度 [mol·kg⁻¹·s⁻¹]。"""
    if role == "synthesis":
        model = model or "KOGAS"
        if model == "Graaf1988":
            # A: CO+2H2→CH3OH,  B: CO2+H2→CO+H2O,  C: CO2+3H2→CH3OH+H2O
            r = graaf.rates(state)
            rA, rB, rC = r["r_A"], r["r_B"], r["r_C"]
            return {
                "CO":    -rA + rB,
                "CO2":   -rB - rC,
                "H2":    -2 * rA - rB - 3 * rC,
                "CH3OH":  rA + rC,
                "H2O":    rB + rC,
            }
        # VBF / KOGAS:  MS: CO2+3H2→CH3OH+H2O,  RWGS: CO2+H2→CO+H2O
        r = vbf_kogas.rate_ms_rwgs(state, model=model)
        rMS, rRWGS = r["r_MS"], r["r_RWGS"]
        return {
            "CO":     rRWGS,
            "CO2":   -rMS - rRWGS,
            "H2":    -3 * rMS - rRWGS,
            "CH3OH":  rMS,
            "H2O":    rMS + rRWGS,
        }

    if role == "dehydration":
        # MD: 2 CH3OH → DME + H2O
        model = model or "KOGAS"
        rMD = vbf_kogas.rate_dehydration(state, source=model, k_eq3=k_eq3)
        return {"CH3OH": -2 * rMD, "DME": rMD, "H2O": rMD}

    if role == "carbonylation":
        # CH3OCH3 + CO → CH3COOCH3。r_MA[mol/(mol Al)/s] × n_Al → mol·kg⁻¹·s⁻¹
        model = model or "DTU"
        r = carbonylation.rate(state, model=model) * bed.acid_site_density
        return {"DME": -r, "CO": -r, "MA": r}

    raise ValueError(f"unknown catalyst role: {role!r}")


def species_rates(state, bed, models=None, k_eq3="KOGAS") -> dict[str, float]:
    """ハイブリッド触媒ベッド全体の、ベッド kg あたりの成分速度 Rᵢ [mol·kg⁻¹·s⁻¹]。

    各触媒の寄与を「その触媒の質量分率」で重み付けして合算する。
    models = {role: model}（省略時は各 role の既定）。k_eq3 は脱水平衡の選択。
    """
    models = models or {}
    total = bed.total()
    R = defaultdict(float)
    for role, mass in bed.masses.items():
        w = mass / total
        for species, rate in _role_rates(state, role, models.get(role), bed, k_eq3).items():
            R[species] += w * rate
    return dict(R)
