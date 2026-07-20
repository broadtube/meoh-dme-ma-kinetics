"""§1 Graaf の検証。文献値と照合する。"""
import math
from reaction_rate import graaf
from reaction_rate.state import GasState

T = 523.15  # 250 °C


def test_equilibrium_internal_consistency():
    # K_pC = K_pA · K_pB（Graaf 1988 式26）
    eq = graaf.equilibrium(T, source="Graaf1986")
    assert math.isclose(eq["K_pC"], eq["K_pA"] * eq["K_pB"], rel_tol=1e-9)


def test_equilibrium_graaf1986_literature():
    # Graaf 1986 原著値（p.2884）: 523.15 K で K_pB≈1.17e-2, K_pA≈1.6e-3 bar^-2
    eq = graaf.equilibrium(T, source="Graaf1986")
    assert math.isclose(eq["K_pB"], 1.17e-2, rel_tol=0.02)
    assert math.isclose(eq["K_pA"], 1.59e-3, rel_tol=0.02)


def test_equilibrium_graaf2016_literature():
    # Graaf 2016 eq8: Kp1°(250℃)=1.66e-3 bar^-2（論文記載の検算値）
    eq = graaf.equilibrium(T, source="Graaf2016")
    assert math.isclose(eq["K_pA"], 1.66e-3, rel_tol=0.02)


def test_rates_signs():
    # CO/H2 rich・低生成物 → メタノール生成(A,C)は正
    s = GasState(T, 50.0, {"CO": 0.15, "CO2": 0.08, "H2": 0.72, "H2O": 0.01, "CH3OH": 0.04})
    r = graaf.rates(s)
    assert r["r_A"] > 0 and r["r_C"] > 0
