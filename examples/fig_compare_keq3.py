"""compare 図: 脱水平衡 K_eq3(KOGAS/BL/thermo) の比較（DME モル分率 vs 触媒質量）。

KOGAS だけ過剰生成、BL・thermo は Cantera 独立熱力学(~0.20)と一致することを示す。
実行: PYTHONPATH=src python3 examples/fig_compare_keq3.py
"""
import dataclasses
from reaction_rate.case import CASE_DME, run_case
from reaction_rate import plots

series, W = {}, None
for k_eq3 in ("KOGAS", "BL", "thermo"):
    res = run_case(dataclasses.replace(CASE_DME, k_eq3=k_eq3))
    series[k_eq3] = res.mole_fractions()["DME"]
    W = res.W

fig, ax = plots.lines(W, series, "Catalyst mass W [kg]", "DME mole fraction [-]",
                      title="DME synthesis: K_eq3 comparison (KOGAS 8:2)", marker=False)
ax.axhline(0.201, ls="--", color="#5b6873", lw=1.0)                 # Cantera 平衡 参照線
ax.text(W[-1], 0.201, "  Cantera eq.", color="#5b6873", va="center", fontsize=8)
fig.tight_layout()
print("saved", plots.save(fig, "keq3_compare.png"))
