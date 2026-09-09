"""Aspen Plus の LHHW 速度式への変換（ZSM-5 メタノール脱水 = Ortega 2018）。

Aspen Plus (Reactions / LHHW) の式構造:

    r = [kinetic factor] × [driving force expression] / [adsorption expression]

    kinetic factor      k = k₀ · T^{n_T} · exp(−E/(R·T))
    driving force       Σ_j (−1)^j · K_j · Π_i [C_i]^{α_ij}      (j = 1: 正, 2: 逆)
                        ln K_j = A_j + B_j/T + C_j·ln T + D_j·T
    adsorption          [ Σ_k K_k · Π_i [C_i]^{ν_ik} ]^m
                        ln K_k = A_k + B_k/T + C_k·ln T + D_k·T

指数 α・ν は**実数を指定できる**（整数に限らない）。これが効いてくる。

──────────────────────────────────────────────────────────────────────
2 つの形
──────────────────────────────────────────────────────────────────────
**ORTEGA_EXACT（推奨）** — Ortega 2018 Eq.(16) を**厳密に**書き換えたもの。フィット誤差ゼロ
（`rate()` と `vbf_kogas.rate_dehydration(source="ZSM5")` が倍精度で一致する）。鍵は
**吸着項の指数に 0.5 を使えること**（解離吸着の √(K_M p_M) をそのまま表現できる）。
駆動力は依頼どおり `K_f·p_M² − K_b·p_W·p_D` の形になり、負の指数も現れない。

⚠️ p_MeOH → 0 で分母（吸着項）が 0 になり 0 除算する。Aspen 側で反応器入口に
メタノールが無い状態を通る可能性があるなら、下限を入れるか反応を無効化すること
（`vbf_kogas.rate_dehydration` は p_M ≤ 0 で 0 を返すガードを持つが、Aspen には無い）。

**LINEAR_FIT（代替・非推奨）** — 吸着項を「1 + K_M p_M + K_W p_W + K_D p_D」の**線形のみ**に
制限し、駆動力も p_M² − p_W p_D/K_eq の形に固定した最小二乗フィット。指数を実数にできない
運用のための逃げ道だが、**関数形が本質的に違う**ので誤差が残る:
  フィット面(140–260℃) で RMS 18.8%、原著 Fig.4/6/7 の 25 点で平均 20.6%（厳密形は 8.2%）。
  見かけの活性化エネルギーが ~0 kJ/mol に落ちる（吸着項の温度依存との相殺）ため
  **フィット域の外へ外挿してはいけない**。
  → 指数 0.5 が使えるなら必ず ORTEGA_EXACT を使うこと。
フィットの再現は `examples/ortega_aspen_lhhw.py`。

──────────────────────────────────────────────────────────────────────
単位・基準
──────────────────────────────────────────────────────────────────────
  ・[C_i] = **分圧 [bar]**（Aspen の濃度基準を Partial pressure / bar にすること）
  ・r = **mol_MeOH·kg_cat⁻¹·s⁻¹**（メタノール消費速度基準）
  ・反応を「2 CH3OH → CH3OCH3 + H2O」と書く場合、Aspen の反応速度は進行度基準なので
    **k₀ を 1/2 にする**（本モジュールは一貫して r_MeOH を返す。`vbf_kogas` 側は
    r_MD = r_MeOH/2 を返す規約なので混同しないこと）。
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

from .units import R
from . import vbf_kogas as _vk


@dataclass(frozen=True)
class Term:
    """ln K = A + B/T + C·ln T + D·T と、各成分の指数 {種: 指数}。"""
    A: float = 0.0
    B: float = 0.0
    C: float = 0.0
    D: float = 0.0
    exponents: dict[str, float] = field(default_factory=dict)

    def K(self, T: float) -> float:
        return math.exp(self.A + self.B / T + self.C * math.log(T) + self.D * T)

    def value(self, T: float, p: dict[str, float]) -> float:
        out = self.K(T)
        for s, nu in self.exponents.items():
            out *= p[s] ** nu
        return out


@dataclass(frozen=True)
class LHHW:
    """Aspen Plus LHHW の 1 反応ぶんのパラメータ一式。"""
    name: str
    k0: float                     # kinetic factor 前指数
    E: float                      # [J/mol]
    n_T: float = 0.0              # kinetic factor の T^{n_T}
    forward: Term = Term()        # 駆動力 第1項
    reverse: Term = Term()        # 駆動力 第2項（引かれる側）
    adsorption: tuple[Term, ...] = ()
    m: float = 1.0                # adsorption expression の指数

    def kinetic_factor(self, T: float) -> float:
        return self.k0 * T ** self.n_T * math.exp(-self.E / (R * T))

    def driving_force(self, T: float, p: dict[str, float]) -> float:
        return self.forward.value(T, p) - self.reverse.value(T, p)

    def adsorption_expression(self, T: float, p: dict[str, float]) -> float:
        return sum(t.value(T, p) for t in self.adsorption) ** self.m

    def rate(self, T: float, p: dict[str, float]) -> float:
        """r_MeOH [mol·kg⁻¹·s⁻¹]。T [K]、p は分圧 [bar] の dict。"""
        return (self.kinetic_factor(T) * self.driving_force(T, p)
                / self.adsorption_expression(T, p))


def _ortega_exact(negative_exponent: bool = False) -> LHHW:
    """Ortega 2018 Eq.(16) を Aspen LHHW 形に**厳密に**書き換える（フィットではない）。

    Ortega:
        r = k(T)·K_M(T)·p_M·[1 − p_D p_W/(p_M²·K_eq)] / (1 + 2√(K_M p_M) + K_W p_W)²
        k(T)  = k_T0·exp[−(E_app/R)(1/T − 1/T0)],   K_i(T) = exp(ΔS_i/R)·exp(−ΔH_i/(R T))

    ① kinetic factor に k·K_M をまとめる。指数関数の積なので**単一の Arrhenius**になる:
        k·K_M = [k_T0·e^{E_app/(R T0)}·e^{ΔS_M/R}] · exp(−(E_app + ΔH_M)/(R T))
        → E = E_app + ΔH_M = 109.3 + (−70.3) = **39.0 kJ/mol**（吸着熱のぶん見かけが下がる）

    ② 分子・分母に p_M を掛けて負の指数を消す（**既定形**）:
        分子 → p_M² − p_W p_D/K_eq        ← 依頼の {K_f[W][X] − K_b[Y][Z]} の形そのもの
        分母 → p_M·(1 + a√p_M + b p_W)² = (**√p_M** + a·p_M + b·**√p_M**·p_W)²
        すなわち吸着項は
            term1: p_M^0.5（係数 1）
            term2: 2·e^{ΔS_M/(2R)}·e^{−ΔH_M/(2R T)} · p_M^1
            term3: e^{ΔS_W/R}·e^{−ΔH_W/(R T)} · p_M^0.5·p_W^1
        で m = 2。**負の指数が出ない**ので Aspen 側で扱いやすい。

    negative_exponent=True にすると掛けない版を返す（吸着項は 1 + …√p_M + …p_W と素直だが、
    駆動力の逆反応項に p_M^{−1} が出る）。両者は数学的に同一で、数値も倍精度で一致する。

    K_eq は本プロジェクトの K_eq3("thermo") = 10^(c0/T + c1)
    → ln(1/K_eq) = −ln10·c1 − ln10·c0/T（A, B にそのまま入る）。
    """
    c0, c1 = _vk.K_EQ3_SOURCES["thermo"]
    ln10 = math.log(10.0)
    k0 = _vk.KT0_ORT * math.exp(_vk.EAPP_ORT / (R * _vk.T0_ORT)) * math.exp(_vk.DS_M_ORT / R)
    A_M, B_M = math.log(2.0) + _vk.DS_M_ORT / (2 * R), -_vk.DH_M_ORT / (2 * R)
    A_W, B_W = _vk.DS_W_ORT / R, -_vk.DH_W_ORT / R
    reverse_K = dict(A=-ln10 * c1, B=-ln10 * c0)            # K = 1/K_eq
    if negative_exponent:
        return LHHW(
            name="Ortega2018-exact (p_M^-1 form)",
            k0=k0, E=_vk.EAPP_ORT + _vk.DH_M_ORT,
            forward=Term(exponents={"CH3OH": 1.0}),
            reverse=Term(**reverse_K, exponents={"CH3OH": -1.0, "H2O": 1.0, "DME": 1.0}),
            adsorption=(Term(),
                        Term(A=A_M, B=B_M, exponents={"CH3OH": 0.5}),
                        Term(A=A_W, B=B_W, exponents={"H2O": 1.0})),
            m=2.0)
    return LHHW(
        name="Ortega2018-exact",
        k0=k0, E=_vk.EAPP_ORT + _vk.DH_M_ORT,
        forward=Term(exponents={"CH3OH": 2.0}),
        reverse=Term(**reverse_K, exponents={"H2O": 1.0, "DME": 1.0}),
        adsorption=(Term(exponents={"CH3OH": 0.5}),
                    Term(A=A_M, B=B_M, exponents={"CH3OH": 1.0}),
                    Term(A=A_W, B=B_W, exponents={"CH3OH": 0.5, "H2O": 1.0})),
        m=2.0)


#: 推奨。Ortega 2018 Eq.(16) と倍精度で一致（フィット誤差ゼロ）。負の指数を含まない。
ORTEGA_EXACT = _ortega_exact()

#: 上と数学的に同一の別表現（駆動力に p_M^{−1} を使い、吸着項を素直にした形）。
ORTEGA_EXACT_NEG = _ortega_exact(negative_exponent=True)

#: 代替。吸着項を線形に制限した最小二乗フィット（examples/ortega_aspen_lhhw.py が再現する）。
#: フィット域: 140–260℃ × p_M 0.05–1.5 / p_W 0–1.0 / p_D 0–0.8 bar の 767 点、相対誤差最小化。
#: ⚠️ 関数形が違うため誤差が残る: フィット面で RMS 18.8%、原著 Fig.4/6/7 の 25 点で平均 20.6%
#: （厳密形は 8.2%）、SI アンカーで −19.3%。E が ~0 なのは吸着項の温度依存と相殺した
#: 見かけの値で物理的意味は無い＝**フィット域の外に外挿しないこと**。
LINEAR_FIT = LHHW(
    name="Ortega2018-linear-fit",
    k0=217.76788482,
    E=1201.4711,
    forward=Term(exponents={"CH3OH": 2.0}),
    reverse=Term(A=-math.log(10.0) * _vk.K_EQ3_SOURCES["thermo"][1],
                 B=-math.log(10.0) * _vk.K_EQ3_SOURCES["thermo"][0],
                 exponents={"H2O": 1.0, "DME": 1.0}),
    adsorption=(
        Term(),
        Term(A=-9.018538962, B=5988.417628, exponents={"CH3OH": 1.0}),
        Term(A=-22.096694309, B=10996.005439, exponents={"H2O": 1.0}),
        Term(A=-5.855549509, B=2083.796769, exponents={"DME": 1.0}),
    ),
    m=2.02194384,
)


def rate(T: float, p: dict[str, float], params: LHHW = ORTEGA_EXACT) -> float:
    """r_MeOH [mol_MeOH·kg_cat⁻¹·s⁻¹]。T [K]、p は分圧 [bar]（CH3OH/H2O/DME）。"""
    q = {"CH3OH": 0.0, "H2O": 0.0, "DME": 0.0, **p}
    return params.rate(T, q)


def aspen_table(params: LHHW = ORTEGA_EXACT) -> str:
    """Aspen Plus の入力欄に転記できる形でパラメータを整形する。"""
    L = [f"[{params.name}]  r = k·DF/ADS   基準: mol_MeOH·kg⁻¹·s⁻¹, 分圧 bar",
         "",
         "Kinetic factor:  k = k0 · T^n · exp(-E/(R T))",
         f"  k0 = {params.k0:.6e}      n = {params.n_T:g}      E = {params.E/1e3:.4f} kJ/mol",
         "",
         "Driving force:   K1·Π[Ci]^a1 − K2·Π[Ci]^a2     (ln K = A + B/T + C lnT + D T)"]
    for tag, t in (("term 1", params.forward), ("term 2", params.reverse)):
        exps = ", ".join(f"{s}^{v:g}" for s, v in t.exponents.items()) or "(none)"
        L.append(f"  {tag}: A={t.A:+.6f}  B={t.B:+.4f}  C={t.C:g}  D={t.D:g}   exponents: {exps}")
    L += ["", f"Adsorption:      [ Σ K_k·Π[Ci]^v ]^m,   m = {params.m:g}"]
    for i, t in enumerate(params.adsorption, 1):
        exps = ", ".join(f"{s}^{v:g}" for s, v in t.exponents.items()) or "(constant 1)"
        L.append(f"  term {i}: A={t.A:+.6f}  B={t.B:+.4f}  C={t.C:g}  D={t.D:g}   exponents: {exps}")
    L += ["",
          "※ 反応を「2 CH3OH → CH3OCH3 + H2O」と書く場合は進行度基準なので k0 を 1/2 に。",
          "※ [Ci] は分圧 [bar]。Aspen の濃度基準を Partial pressure / bar に設定すること。"]
    return "\n".join(L)
