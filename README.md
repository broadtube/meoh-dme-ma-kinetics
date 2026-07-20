# reaction_rate

CO/CO₂/H₂ からの **メタノール / DME / 酢酸メチル** 合成の反応速度計算。

コードは `references/rate_equations.html` の**セクション順に1対1**で対応させ、HTMLを横に置いて検証できるようにしている。

## 構成

| モジュール | 対応 | 内容 |
|---|---|---|
| `units.py` | HTML 凡例 | R・単位変換（bar↔Pa, mol↔kmol, 分圧/濃度） |
| `state.py` | HTML 凡例 | 気相状態 `GasState`＋SRKフガシティ。組成→分圧/濃度/フガシティを供給 |
| `graaf.py` | **§1** | Graaf 1988 メタノール合成 r_A/r_B/r_C ＋ 平衡定数(Graaf 1986・2016) |
| `vbf_kogas.py` | **§2** | VBF(=KOGAS=Ng) の r_MS/r_RWGS ＋ 脱水 r_MD(Bercič–Levec) ＋ K_eq1/2/3 |
| `carbonylation.py` | **§3** | モルデナイト カルボニル化: 速度則＋DTU＋Cheng(＋失活) |

各反応系は `model=`/`source=` 引数で切替（例: 合成 `Graaf1988`/`VBF`、脱水 `BercicLevec`/`KOGAS`、
カルボニル化 `DTU`/`Cheng`、平衡 `Graaf1986`/`Graaf2016`）。パラメータは各モジュール内の定数。

## 物性・EOS の方針
- 臨界物性(Tc/Pc/ω)は `chemicals` ライブラリ由来（`state.py` の `COMPONENTS`）。
- **フガシティは自前SRK**（Cantera に SRK は無い）。`thermo` ライブラリで検証可能。
- **⚠️ メタノール合成の SRK は Graaf 1986 Table 2 の "effective H2"（Graboski–Daubert: Tc=43.6 K, Pc=20.5 bar, ω=0）を使う。**
  `chemicals` の実H2（Tc=33.1 K, ω=−0.219）を使うと原著と合わない。→ `state.GRAAF1986_CRITICALS` を使用。

## 出典
`references/PAPERS_INDEX.md`（全論文・DOI・入手状況）、`references/rate_equations.html`（式・パラメータ・来歴）。
