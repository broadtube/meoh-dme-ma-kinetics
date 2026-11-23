"""単管タンデム（合成 VBF/KOGAS ＋ 脱水 ZSM-5 → カルボニル化）の STY を、DME 時点と MA 時点で並べる積み上げ棒。

  左: DME 時点  = ハイブリッド床（合成＋脱水）出口。MA 床に入る直前。
  右: MA 時点   = タンデム出口（MA 床の後）。

条件は fig_tandem_catalyst_compare.py の ZSM-5 側と同一。メタノール合成の速度論を
--synthesis-model で選ぶ: VBF（1996 原著・既定。chemflow2 の example_plant4 既定と揃える）/
KOGAS（Ng 1999 = fig_tandem_catalyst_compare と同じ）。式形と B は共通で前指数 A だけが違う。
250℃, 51 bar, SV 5000/h（全触媒 3.0 mL・新鮮供給基準）, ハイブリッド 1.0 mL + MA 床 2.0 mL。
STY は 2 本とも同じ基準（全触媒 3.0 mL）で割るので、棒の差がそのまま MA 床の寄与になる。
カルボニル化は DTU 式の第3項を K3⁻¹ で使う修正後（cecc1da）の実装。

出力:
（<m> = vbf / kogas。モデルごとに別ファイルなので上書きし合わない）
  figures/tandem_<m>_stages.png            積み上げ棒
  datasets/tandem_<m>_stages.csv           各時点の流量・組成・STY・転化率
  datasets/tandem_<m>_stages_spec.json     全諸元（運転条件・触媒・速度論定数・結果）

⚠️ ZSM-5(Ortega) のフィット域は 140–190℃・~1 bar、カルボニル化の検証域は 150–190℃。
   250℃/51bar はどちらも外挿。絶対値は要実験較正。

  --compare-ma を付けると、VBF と KOGAS の MA 時点（タンデム出口）だけを横に並べた
  1 枚を描く（DME 時点は描かない。datasets は書かない＝各モデル単独実行で出力済みのもの）:
  figures/tandem_vbf_vs_kogas_ma_stage.png

実行: PYTHONPATH=src python3 examples/fig_tandem_zsm5_stages.py                       # VBF
      PYTHONPATH=src python3 examples/fig_tandem_zsm5_stages.py --synthesis-model KOGAS
      PYTHONPATH=src python3 examples/fig_tandem_zsm5_stages.py --compare-ma          # VBF vs KOGAS
"""
import argparse
import csv
import json
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from reaction_rate.reactors import pfr, CatalystBed
from reaction_rate import plots, vbf_kogas, carbonylation

# ── fig_tandem_catalyst_compare と同一条件（合成速度論は main() で選ぶ）─────────
T = 523.15
P = 5.0e6 / 1e5 + 1.01325
H2_CO, Y_CO2, SV = 1.5, 0.03, 5000.0
RHO_SYN, RHO_DEH, RHO_MA = 1300.0, 800.0, 837.0
V_HYB, V_MA = 1.0e-6, 2.0e-6
V_TOT = V_HYB + V_MA
f_syn = 1.0 / 1.9
m_syn, m_deh = RHO_SYN * V_HYB * f_syn, RHO_DEH * V_HYB * (1 - f_syn)
m_ma = RHO_MA * V_MA
VM_STP = 22.414e-3
N = SV / 3600.0 * V_TOT / VM_STP

SYN_MODEL, DEH_MODEL, MA_MODEL, K_EQ3 = "VBF", "ZSM5", "DTU-Cheung2007-2", "thermo"
SYN_SOURCES = {"VBF": "Vanden Bussche & Froment 1996 (original)",
               "KOGAS": "Ng, Chadwick & Toseland 1999 Table 1 (= KOGAS 2008/2021)"}
ACID = CatalystBed({"carbonylation": 1.0}).acid_site_density   # 既定値をそのまま記録

V_STY_L = V_TOT * 1e3          # STY 基準体積 [L] = 全触媒 3.0 mL
SPECIES = ["CO", "CO2", "H2", "H2O", "CH3OH", "DME", "MA"]
MW = {"CO": 28.01, "CO2": 44.01, "H2": 2.016, "H2O": 18.015,
      "CH3OH": 32.04, "DME": 46.07, "MA": 74.08}
N_C = {"CO": 1, "CO2": 1, "CH3OH": 1, "DME": 2, "MA": 3}

ORDER = ["MA", "DME", "CH3OH", "CO2"]                       # 積み上げ順（下→上）
COLORS = {"MA": plots.OKABE_ITO[3], "DME": plots.OKABE_ITO[2],
          "CH3OH": plots.OKABE_ITO[5], "CO2": plots.OKABE_ITO[4]}
STAGES = [("dme", "DME stage\n(hybrid-bed outlet)"), ("ma", "MA stage\n(tandem outlet)")]
STAGE_VOL_ML = {"dme": V_HYB * 1e6, "ma": V_TOT * 1e6}      # その時点までの累積触媒体積

g = lambda x: float(f"{x:.10g}")      # JSON の数値を 10 桁に丸める（再生成ノイズ対策）


def feed():
    y_co = (1 - Y_CO2) / (1 + H2_CO)
    return {"CO": y_co * N, "CO2": Y_CO2 * N, "H2": H2_CO * y_co * N,
            "H2O": 0.0, "CH3OH": 0.0, "DME": 0.0, "MA": 0.0}


def run(syn_model=None):
    """→ {"feed", "dme", "ma"}: 各時点のモル流量 [mol/s]。syn_model 省略時は SYN_MODEL。"""
    f = feed()
    r1 = pfr(f, T, P, CatalystBed({"synthesis": m_syn, "dehydration": m_deh}),
             models={"synthesis": syn_model or SYN_MODEL, "dehydration": DEH_MODEL},
             k_eq3=K_EQ3)
    r2 = pfr(r1.outlet(), T, P, CatalystBed({"carbonylation": m_ma}),
             models={"carbonylation": MA_MODEL})
    return {"feed": f, "dme": r1.outlet(), "ma": r2.outlet()}


def sty(f, o, v_l=V_STY_L):
    """新鮮供給からの正味生成量 → STY [g/(L·h)]。"""
    return {s: (o[s] - f[s]) * MW[s] * 3600.0 / v_l for s in SPECIES}


def metrics(f, o):
    c_in = f["CO"] + f["CO2"]
    return {"X_CO": (f["CO"] - o["CO"]) / f["CO"],
            "X_COx": (c_in - o["CO"] - o["CO2"]) / c_in,
            # 炭素収率は新鮮 CO+CO2 の炭素基準（MA の 3C は全量 syngas 由来）
            "Y_C_CH3OH": o["CH3OH"] / c_in, "Y_C_DME": 2 * o["DME"] / c_in,
            "Y_C_MA": 3 * o["MA"] / c_in,
            "C_balance": sum(N_C[s] * o[s] for s in N_C) / c_in}


def write_csv(res, path):
    f = res["feed"]
    head = (["point", "cum_catalyst_mL"]
            + [f"F_{s}_mol_h" for s in SPECIES] + [f"y_{s}" for s in SPECIES]
            + [f"STY_{s}_g_Lh_total3mL" for s in SPECIES]
            + [f"STY_{s}_g_Lh_cum_bed" for s in SPECIES]
            + ["X_CO", "X_COx", "Y_C_CH3OH", "Y_C_DME", "Y_C_MA", "C_balance"])
    rows = [["feed", 0.0] + [f[s] * 3600 for s in SPECIES]
            + [f[s] / sum(f.values()) for s in SPECIES]
            + [0.0] * (2 * len(SPECIES)) + [0.0] * 5 + [1.0]]
    for key, _ in STAGES:
        o = res[key]
        s3 = sty(f, o)
        sb = sty(f, o, STAGE_VOL_ML[key] * 1e-3)
        m = metrics(f, o)
        rows.append([key, STAGE_VOL_ML[key]] + [o[s] * 3600 for s in SPECIES]
                    + [o[s] / sum(o.values()) for s in SPECIES]
                    + [s3[s] for s in SPECIES] + [sb[s] for s in SPECIES]
                    + [m[k] for k in ("X_CO", "X_COx", "Y_C_CH3OH", "Y_C_DME", "Y_C_MA",
                                      "C_balance")])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(head)
        for r in rows:
            w.writerow([r[0]] + [f"{x:.8g}" for x in r[1:]])


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_spec(res, path):
    f = res["feed"]
    arr = lambda p: {"A": g(p[0]), "B_J_mol": g(p[1])}
    ms = vbf_kogas.MS_PARAMS[SYN_MODEL]
    k1_native = ms["k1"][0] * np.exp(ms["k1"][1] / (8.314 * T))
    spec = {
        "script": "examples/fig_tandem_zsm5_stages.py",
        "synthesis_model": SYN_MODEL,
        "git_head": _git_head(),
        "operating": {"T_C": g(T - 273.15), "T_K": T, "P_bar": g(P), "isothermal": True,
                      "SV_1_h": SV, "SV_basis": "fresh feed (STP 22.414 L/mol) / total catalyst 3.0 mL",
                      "Vm_STP_m3_mol": VM_STP},
        "feed": {"H2_over_CO": H2_CO, "y_CO2": Y_CO2,
                 "mole_fraction": {s: g(f[s] / N) for s in SPECIES},
                 "molar_flow_mol_h": {s: g(f[s] * 3600) for s in SPECIES},
                 "total_mol_h": g(N * 3600), "total_NL_h": g(N * 3600 * VM_STP * 1e3)},
        "beds": [
            {"name": "hybrid (synthesis + dehydration, physically mixed)",
             "volume_mL": V_HYB * 1e6, "volume_fraction_synthesis": g(f_syn),
             "synthesis": {"catalyst": "Cu/ZnO/Al2O3", "bulk_density_kg_m3": RHO_SYN,
                           "mass_g": g(m_syn * 1e3)},
             "dehydration": {"catalyst": "H-ZSM-5", "bulk_density_kg_m3": RHO_DEH,
                             "mass_g": g(m_deh * 1e3)}},
            {"name": "carbonylation", "catalyst": "H-MOR", "volume_mL": V_MA * 1e6,
             "bulk_density_kg_m3": RHO_MA, "mass_g": g(m_ma * 1e3),
             "acid_site_density_mol_Al_kg": ACID}],
        "kinetics": {
            "synthesis": {
                "model": SYN_MODEL, "source": SYN_SOURCES[SYN_MODEL],
                "form": "x = A*exp(B/(R*T)), R=8.314; rates in mol/(kg_cat s)",
                "params": {k: arr(v) for k, v in ms.items()},
                "rate_unit_factor_to_mol_kg_s": vbf_kogas._MS_UNIT_TO_MOL_KG_S[SYN_MODEL],
                "k1_at_T_mol_kg_s_bar2": g(k1_native * vbf_kogas._MS_UNIT_TO_MOL_KG_S[SYN_MODEL]),
                "equilibrium": {"log10_K_eq1": list(vbf_kogas.EQ_KOGAS["K_eq1"]),
                                "log10_K_eq2_inv(RWGS)": list(vbf_kogas.EQ_KOGAS["K_eq2_inv"]),
                                "form": "log10 K = c0/T + c1"}},
            "dehydration": {
                "model": DEH_MODEL, "source": "Ortega et al. 2018 LHHW (Table 5)",
                "T0_K": vbf_kogas.T0_ORT, "kT0_mol_kg_s": vbf_kogas.KT0_ORT,
                "Eapp_J_mol": vbf_kogas.EAPP_ORT,
                "dS_MeOH_J_mol_K": vbf_kogas.DS_M_ORT, "dH_MeOH_J_mol": vbf_kogas.DH_M_ORT,
                "dS_H2O_J_mol_K": vbf_kogas.DS_W_ORT, "dH_H2O_J_mol": vbf_kogas.DH_W_ORT,
                "k_eq3": K_EQ3, "log10_K_eq3": list(vbf_kogas.K_EQ3_SOURCES[K_EQ3]),
                "K_eq3_at_T": g(vbf_kogas.K_eq3(T, K_EQ3)),
                "rate_basis": "DME formation (MeOH consumption / 2)"},
            "carbonylation": {
                "model": MA_MODEL,
                "form": "r = k1*pCO/(1 + K2*pMA + pMA/(K3*pDME)) * K_DME*pDME/(1+K_DME*pDME)",
                "source": "DTU 2017 eqn(11) Table 4 @438 K; k1 Ea from Cheung 2007; "
                          "K2 van't Hoff (DFT estimate); K3 fixed",
                "T_ref_K": carbonylation.T_REF,
                "k1_ref_mol_molAl_s_bar": carbonylation.DTU_PARAMS["k1"],
                "K2_ref_1_bar": carbonylation.DTU_PARAMS["K2"],
                "K3_ref": carbonylation.DTU_PARAMS["K3"],
                "Ea_k1_J_mol": g(carbonylation.EA_CHEUNG), "dH_K2_J_mol": carbonylation.DFT_DH_K2,
                "K3_temperature_dependent": False,
                "K_DME_reg_1_bar": carbonylation.K_DME_REG, "water_inhibition": "none (K_w=0)",
                "K3_inverse_fix": "cecc1da (pMA/(K3*pDME))"}},
        "solver": {"type": "isothermal PFR, isobaric, ideal gas", "method": "BDF",
                   "rtol": 1e-8, "atol": 1e-16},
        "STY_basis": "net production from fresh feed per total catalyst 3.0 mL (both stages)",
        "results": {},
    }
    for key, _ in STAGES:
        o = res[key]
        spec["results"][key] = {
            "cum_catalyst_mL": STAGE_VOL_ML[key],
            "outlet_mol_h": {s: g(o[s] * 3600) for s in SPECIES},
            "STY_g_Lh_total3mL": {s: g(v) for s, v in sty(f, o).items()},
            **{k: g(v) for k, v in metrics(f, o).items()}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def _stacked_bars(ax, bars, title):
    """bars = [(xticklabel, {species: STY})] を積み上げ棒で描き、軸を整える。→ 各棒の Σ。"""
    totals = [sum(d.values()) for _, d in bars]
    ymax = max(totals)
    xs = [1.1 * i for i in range(len(bars))]
    for x, (_, d), tot in zip(xs, bars, totals):
        bottom = 0.0
        for s in ORDER:
            h = d[s]
            ax.bar(x, h, width=0.62, bottom=bottom,
                   color=COLORS[s], edgecolor="white", linewidth=2.0,
                   label=s if x == xs[0] else None)
            if h > 0.05 * ymax:
                ax.text(x, bottom + h / 2, f"{s}\n{h:.0f}", ha="center", va="center",
                        color="white", fontsize=8.5, fontweight="bold")
            elif h > 0.008 * ymax:
                ax.text(x + 0.36, bottom + h / 2, f"{s} {h:.0f}", ha="left", va="center",
                        color=plots._INK, fontsize=7.5)
            bottom += max(h, 0.0)
        ax.text(x, bottom + 0.015 * ymax, f"Σ {tot:.0f}", ha="center", va="bottom",
                color=plots._INK, fontsize=10, fontweight="bold")

    ax.set_ylabel("STY  [g/(L·h)]  (per 3.0 mL total-catalyst)", color=plots._INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for lab, _ in bars], color=plots._INK, fontsize=9.5)
    ax.set_xlim(-0.6, xs[-1] + 0.8)
    ax.set_ylim(0, ymax * 1.22)
    ax.set_title(title, color=plots._INK, fontsize=10)
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_[::-1], l_[::-1], frameon=False, labelcolor=plots._INK, fontsize=9,
              loc="upper center", ncol=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=plots._MUTED, labelcolor=plots._INK)
    ax.yaxis.grid(True, color=plots._GRID, lw=0.8)
    ax.set_axisbelow(True)
    return totals


def compare_ma():
    """VBF と KOGAS の MA 時点（タンデム出口）だけを並べた 1 枚。datasets は書かない。"""
    bars, results = [], {}
    for model in SYN_SOURCES:                       # VBF, KOGAS の順
        res = run(model)
        f = res["feed"]
        results[model] = res
        bars.append((f"{model} synthesis\n(MA stage, tandem outlet)",
                     {s: sty(f, res["ma"])[s] for s in ORDER}))

    fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=130)
    totals = _stacked_bars(
        ax, bars,
        "Tandem STY at MA stage — VBF vs KOGAS synthesis + ZSM-5 (Ortega)\n"
        "(250°C, 51 bar, SV 5000/h, hybrid 1.0 mL + MA bed 2.0 mL)")
    fig.tight_layout()
    print("saved", plots.save(fig, "tandem_vbf_vs_kogas_ma_stage.png"))
    for (model, res), (_, d), tot in zip(results.items(), bars, totals):
        m = metrics(res["feed"], res["ma"])
        print(f"[{model:5}] MA stage  Σ={tot:.1f}  "
              + "  ".join(f"{s}={d[s]:.1f}" for s in ORDER)
              + f"  X_CO={m['X_CO']:.3f}  C_bal={m['C_balance']:.6f}")


def main():
    global SYN_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthesis-model", choices=tuple(SYN_SOURCES), default=SYN_MODEL)
    ap.add_argument("--compare-ma", action="store_true",
                    help="VBF と KOGAS の MA 時点のみを並べた比較図を描く（datasets は書かない）")
    args = ap.parse_args()
    if args.compare_ma:
        compare_ma()
        return
    SYN_MODEL = args.synthesis_model
    tag = SYN_MODEL.lower()

    res = run()
    f = res["feed"]
    data = {k: {s: sty(f, res[k])[s] for s in ORDER} for k, _ in STAGES}
    totals = {k: sum(data[k].values()) for k in data}

    fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=130)
    _stacked_bars(ax, [(lab, data[k]) for k, lab in STAGES],
                  f"Tandem STY at DME and MA stages — {SYN_MODEL} synthesis + ZSM-5 (Ortega)\n"
                  "(250°C, 51 bar, SV 5000/h, hybrid 1.0 mL + MA bed 2.0 mL)")
    fig.tight_layout()
    print("saved", plots.save(fig, f"tandem_{tag}_stages.png"))

    os.makedirs("datasets", exist_ok=True)
    for path, writer in ((f"datasets/tandem_{tag}_stages.csv", write_csv),
                         (f"datasets/tandem_{tag}_stages_spec.json", write_spec)):
        writer(res, path)
        print("saved", path)
    for k, _ in STAGES:
        m = metrics(f, res[k])
        print(f"[{k:3}]  Σ={totals[k]:.1f}  " + "  ".join(f"{s}={data[k][s]:.1f}" for s in ORDER)
              + f"  X_CO={m['X_CO']:.3f}  C_bal={m['C_balance']:.6f}")


if __name__ == "__main__":
    main()
