"""図の出力（matplotlib, 静的PNG）。examples/ から呼ぶ汎用関数。

方針(dataviz): カテゴリ色は Okabe–Ito（色覚安全・検証済）を固定順で使用（循環させない）、
細い線、控えめなグリッド/軸、単位付きラベル、凡例(≥2系列)、dual-axis 禁止。
軸ラベルは英語（matplotlib は日本語フォントが無いと文字化けするため）。
"""
from __future__ import annotations
import os

import matplotlib
matplotlib.use("Agg")             # ヘッドレス（PNG保存）
import matplotlib.pyplot as plt

# Okabe–Ito 色覚安全カテゴリ色（固定順・循環させない。validate_palette.js で PASS 済）
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
_INK, _MUTED, _GRID = "#1b242c", "#5b6873", "#dbe0dd"


def _style(ax):
    """控えめな軸・グリッド、ink 色のテキスト。"""
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_MUTED, labelcolor=_INK)
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def lines(x, series: dict, xlabel: str, ylabel: str, *, ax=None,
          title: str | None = None, marker: bool = False, logy: bool = False):
    """複数系列の折れ線。series={label: y}。fixed-order の色。(fig, ax) を返す。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=130)
    else:
        fig = ax.figure
    for i, (label, y) in enumerate(series.items()):
        ax.plot(x, y, color=OKABE_ITO[i % len(OKABE_ITO)], lw=1.8,
                marker="o" if marker else None, markersize=5, label=label)
    ax.set_xlabel(xlabel, color=_INK)
    ax.set_ylabel(ylabel, color=_INK)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title, color=_INK, fontsize=11)
    if len(series) >= 2:                        # 単一系列はタイトルで名指し、凡例なし
        ax.legend(frameon=False, labelcolor=_INK, fontsize=9)
    _style(ax)
    if ax.figure is fig and len(fig.axes) == 1:
        fig.tight_layout()
    return fig, ax


def profile(result, species: list | None = None, *, ax=None):
    """PFR の反応器内プロファイル: モル分率 vs 触媒質量 W。"""
    y = result.mole_fractions()
    species = species or list(result.F)
    series = {s: y[s] for s in species}
    return lines(result.W, series, "Catalyst mass W [kg]", "Mole fraction [-]",
                 ax=ax, marker=False)


def save(fig, name: str, outdir: str = "figures") -> str:
    """figures/ に PNG 保存してパスを返す。"""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path
