"""標準熱力学（NASA-7 多項式）から h(T)・cp(T) を供給する。断熱反応器のエネルギー収支専用。

出所: Cantera 同梱 `nasa_gas.yaml`（= Gordon & McBride / NASA Glenn の標準熱力学多項式）。
      K_eq3("thermo") の導出（examples/derive_keq3_thermo.py）と同じデータ源。

理想気体なので h・cp は圧力に依らず T のみの関数。h は生成熱込み（絶対エンタルピー）なので、
  Σᵢ Rᵢ·hᵢ(T) = Σⱼ rⱼ·ΔHⱼ(T)
が成り立ち、反応ごとの ΔH を別途持たなくてもエネルギー収支が書ける（reactors._dT_dW）。

Cantera は遅延 import（この modules を使わない限り依存しない）。
"""
from __future__ import annotations
import functools

from .units import R

# プロジェクトの成分名 → nasa_gas.yaml の種名
#   ⚠️ 酢酸メチル(MA) は nasa_gas.yaml に無いため断熱計算は未対応（明示的にエラーにする）。
CT_NAMES = {
    "CO": "CO", "CO2": "CO2", "H2": "H2", "H2O": "H2O",
    "CH3OH": "CH3OH", "DME": "CH3OCH3",
    "N2": "N2", "Ar": "Ar", "He": "He", "CH4": "CH4",   # 不活性・希釈剤
}


@functools.lru_cache(maxsize=1)
def _solution():
    """CT_NAMES の種だけを含む理想気体 Solution と、プロジェクト名→index の対応。"""
    import cantera as ct
    want = set(CT_NAMES.values())
    species = [s for s in ct.Species.list_from_file("nasa_gas.yaml") if s.name in want]
    missing = want - {s.name for s in species}
    if missing:
        raise RuntimeError(f"nasa_gas.yaml に無い種: {sorted(missing)}")
    gas = ct.Solution(thermo="ideal-gas", species=species)
    return gas, {k: gas.species_index(v) for k, v in CT_NAMES.items()}


def _check(species) -> None:
    unknown = [s for s in species if s not in CT_NAMES]
    if unknown:
        raise ValueError(
            f"熱力学データを持たない成分: {unknown}（対応: {sorted(CT_NAMES)}）。"
            "断熱モードはこれらの成分を含む系では使えない。")


def _standard(T: float, species):
    gas, idx = _solution()
    gas.TPX = float(T), 1.0e5, {n: 1.0 for n in CT_NAMES.values()}   # h,cp は P に依らない
    return gas, idx


def enthalpies(T: float, species) -> dict[str, float]:
    """h°ᵢ(T) [J/mol]（生成熱込み・理想気体）。"""
    _check(species)
    gas, idx = _standard(T, species)
    h_RT = gas.standard_enthalpies_RT
    return {s: float(h_RT[idx[s]]) * R * T for s in species}


def heat_capacities(T: float, species) -> dict[str, float]:
    """cp°ᵢ(T) [J·mol⁻¹·K⁻¹]（理想気体）。"""
    _check(species)
    gas, idx = _standard(T, species)
    cp_R = gas.standard_cp_R
    return {s: float(cp_R[idx[s]]) * R for s in species}
