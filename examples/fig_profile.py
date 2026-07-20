"""profile 図: DME 合成の反応器内プロファイル（モル分率 vs 触媒質量 W）。

実行: PYTHONPATH=src python3 examples/fig_profile.py
"""
from reaction_rate.case import CASE_DME, run_case
from reaction_rate import plots

res = run_case(CASE_DME)     # KOGAS 8:2, k_eq3=thermo（物理的な脱水平衡）
fig, ax = plots.profile(res, species=["CO", "CO2", "H2", "CH3OH", "DME", "H2O"])
ax.set_title("DME synthesis profile (KOGAS 8:2, k_eq3=thermo)", fontsize=11)
fig.tight_layout()
print("saved", plots.save(fig, "dme_profile.png"))
