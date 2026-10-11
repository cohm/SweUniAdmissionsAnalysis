# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Ad-hoc analysis scripts that turn Swedish national admissions data into plots. Two data sources, one script each (see `README.md`):

- `analyze_geo.py` — per-applicant NyA exports for one admission round: choropleth maps, merit-score distributions and correlations, age, gender through the admission chain, outcome/attrition breakdowns.
- `analyze_uhr.py` + `uhr_data.py` — aggregated UHR statistics: multi-year trends and cross-university comparison.
- `plot_utils.py` — the few helpers both use (`finish_plot`, `apply_grid`, `stats_label`).

## Setup and running

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python analyze_geo.py --help     # must be run from the repo root
python analyze_uhr.py --check    # self-check against hand-verified UHR values
```

Paths inside the scripts are relative to the repo root (`data/postalcode_data.csv`, `plots/...`), so they break if run from elsewhere. `plots/` must exist; its PDFs are gitignored.

`--no-show` saves the PDFs without opening a window (and closes each figure, so a long run does not accumulate them). `MPLBACKEND=Agg` also works but leaves every figure open. Set `PLOTS_DIR=/some/scratch/dir` to write plots somewhere other than `plots/` — useful for checking a change without overwriting the real output.

There is no test suite or linter. `analyze_uhr.py --check` is the only automated verification; don't invent pytest/lint commands.

## Sensitive data — read before touching input files

The NyA Excel exports contain personal data for real applicants: full names, addresses, `Personnummer`, e-mail and phone numbers. They live **outside** the repo (e.g. `../../NyA/createReport-2026-08-25-SCI.xls`) and must stay there — never copy them in, never commit them, and never paste raw dataframe output into commits, logs, scratch files, or anything uploaded externally. `.gitignore` now blocks `*.xls`/`*.xlsx`/`*.csv` except the two datasets that belong in git.

Two mechanisms keep PII out of output, and both should be preserved:

- `preprocess_data(..., drop_pii=True)` (the default) drops the identifier columns once `Ålder`, `Födelseår`, `Kön` and `ApplicantID` have been derived. `--keep-pii` disables it for local debugging only.
- `describe_df()` prints shapes, not contents; contents appear only under `--verbose`.

`--schema-only` is the safe way to inspect a new export: it prints column names and value counts of `Resultat`/`Ja-svar`/`Prio` and nothing else, so its output can be shared.

`ApplicantID` is a salted `blake2b` hash. Without `ADMISSIONS_SALT` in the environment a random salt is used per run, so deduplication works but individuals cannot be followed between years — making cross-year linkage an explicit choice.

Versioned datasets: `data/postalcode_data.csv` (15k rows mapping `Postnummer` → `Ort`/`KnNamn`/`KnKod`/`LnNamn` + coordinates) and the tidy UHR exports under `data/uhr/`. The raw UHR cache in `data/uhr_cache/` is gitignored.

## Pipeline

`analyze_geo.py` is a flat module of functions plus `main()`, which holds the configuration: `SOURCES` (the exports to read, with the selection round stated explicitly because it cannot be inferred from a filename) and `PROGRAMS`. Analyses are added by editing `main()`.

Data flow:

1. `load_applicant_files(SOURCES)` — reads each export and **tags every row** with `Antagningsår`, `Antagningsomgång`, `Etikett`, `Referensdatum`, `Källfil`, so rows from different files stay distinguishable. Returns a file-by-column presence table; a file missing a required column is skipped loudly.
2. `load_postal_mapping()` — reads the postal-code CSV, zero-padding `Postnummer` to 5 digits.
3. `preprocess_data()` — joins geography, derives all the columns below, then drops the identifiers.
4. `aggregate_by_column(df, col)` — `value_counts()` into a dataframe with the grouping column plus a column literally named **`number`**. All `plot_*_map()` functions expect that contract. `aggregate_fraction(df, col, mask, min_n=5)` produces the same contract for shares, suppressing groups below `min_n`.
5. Plot functions — each saves to `plots/<Kind>_<name>.pdf` with spaces replaced by underscores, via `plot_utils.finish_plot(kind, name)`.

Filtering (by gender, `Prio == 1`, admitted, program code) happens on the preprocessed dataframe in `main()` before aggregation — the plot functions themselves take no filters. `--only geo,merit,insight` selects which families to draw.

### Columns

Input columns from NyA are Swedish: `Postnummer`, `Meritvärde`, `Resultat`, `Personnummer`, `Prio`, `Kurs-/programkod`, `Ja-svar`, `Land`, `Reservnummer` (37 columns in the HT26 export). Plot titles, axis labels and legends are in Swedish too — keep that convention in new code. `Kurs-/programkod` values carry trailing spaces in the export and are stripped on load.

`preprocess_data()` adds:

- `BI_score`, `BII_score`, `HP_score`, `MAFY_score` — via the single `extract_merit_score()` helper, from the free-text `Meritvärde` field (`BI (20,50) HP (1,05) ...`); decimal commas become points.
- `Resultat_kategori` — ordered categorical, the real NyA vocabulary (see below). Plus booleans `Antagen`, `Reservplacerad`, `Struken`, `Placerad_annat_val`, `Kvar_i_urval`.
- `Svar_kategori` (`Ja`/`Nej`/`Inget svar`) and `Svarat_ja` from `Ja-svar`.
- `Ålder` (float years) and `Födelseår`, from the first 8 chars of `Personnummer`, measured against the source's `Referensdatum` — **not** `Timestamp.now()`, so re-running reproduces the same ages.
- `Kön` — `"K"`/`"M"` from the second-to-last digit of the personnummer, NaN-safe; the unparseable count is reported (49 of 5142 in HT26).
- `ApplicantID` — salted hash, see the PII section.
- Geography from the postal-code merge: `KnKod` (cast to `Int64`), `KnNamn`, `LnNamn`.

Rows whose postal code is not in the mapping (mostly people living abroad — 233 of 5142 in HT26) are **kept**, labelled `UNKNOWN_COUNTY`, so they still count in merit and age analyses. They fall out of the maps naturally, because the label matches no `lan`/`kommun_kod`.

### `Resultat` and what "admitted" means

`admitted(df)` is the single test; use it rather than any string comparison. `ADMITTED_CATEGORIES` is `("Antagen",)`, which also absorbs `Reservantagen` should it appear (`RESULT_PATTERNS` maps it to `Antagen`, since being admitted off the reserve list is an offer). Standing on the reserve list is not admission.

The actual HT26 vocabulary (`createReport-2026-08-25-SCI.xls`, the HT26 entry in `SOURCES`), which is nothing like a simple Antagen/Ej antagen split:

| `Resultat` | n | meaning |
|---|---|---|
| `Struken - Takvärde för ord. urval/inplacering uppnått` | 1467 | **already holds a place on a higher-priority choice** — admitted somewhere, never a rejection |
| `Reserv` | 1465 | on the waiting list |
| `Struken - Uteblivet svar` | 1354 | removed for answering no / not answering |
| `Antagen` | 374 | |
| `Struken - Obehörighet` | 349 | not eligible |
| `Struken - Återbud` | 107 | withdrew after accepting |
| `Reserv - Studieavgiftsvillkor` / `Antagen - Studieavgiftsvillkor` | 4 / 1 | conditional on paying the tuition fee; the `^reserv` / `^antagen` patterns classify them as reserve / admitted |
| others | ≤11 each | |

This export was taken on 2026-08-25, well after urval 2 was published, so it is a later snapshot than UHR's urval-2 figures: `Återbud` has grown (51 in the earlier 2026-07-25 export, 107 here) and `Antagen` has shrunk (402 → 374) as admitted students declined. Numbers below are from this file.

`audit_admitted_definitions()` prints the canonical count next to the two definitions the code used previously; for HT26 the canonical count equals `contains("Antagen")` for all four programmes and differs from `== "antagen"` by one row (CFATE: the `Antagen - Studieavgiftsvillkor` row).

**`Ja-svar` is not an offer acceptance.** It is the reply to the *first* round's notification, where answering yes is required to stay in the running. In the HT26 export all 375 admitted rows *and* all 1469 reserve rows have `Ja` (as do all 107 `Återbud` rows), because everyone who answered no (1134) or did not answer (220) was already struck. A "share of admitted who accepted" plot would therefore be a flat 100 %; the real signal is attrition between the rounds, which is what `plot_response_attrition()` shows.

**Minimum merit score among admitted ≠ the group's admission threshold.** Admitted students hold scores in several groups but are admitted through one, so `min_BI_antagna` for CTFYS HT26 is 14.22 while UHR's BI threshold is 22.40. The export has no column for the group someone was admitted through, so the threshold has to come from `analyze_uhr.py`. Don't relabel these as cut-offs.

### Geodata and merge keys

Maps come from `swemaps` GeoParquet files (`swemaps.get_path("kommun")` / `("lan")`), reprojected to EPSG:3006 (SWEREF99 TM, Swedish National Grid) so the Stockholm inset can use a hardcoded bounding box in metres. The joins are:

- municipality: `map_df.kommun_kod` ↔ `aggregated_df.KnKod`, both cast to `Int64` (the cast matters — otherwise they're `object` and the merge silently produces all-NaN).
- county: `map_df.lan` ↔ `aggregated_df.LnNamn`, matched on the full name including "län".

`_map_style(values, value_kind)` picks the colour scale: counts keep `LogNorm` (so 0 is off-scale and drawn light grey), fractions get a linear `Normalize` from 0 to the data's rounded-up maximum with a percent-formatted colourbar. Always pass the range as an explicit `norm` — **geopandas ignores `vmin`/`vmax` when a `norm` argument is present**, which silently let fraction colours auto-scale while the labels showed the real values. Per-region labels go at polygon centroids via `_annotate_regions()`.

## Colours

`palette.py` holds every colour decision; no plotting code should contain a literal
colour name. Two palettes for two jobs:

- **`KTH_COLORS`** — the KTH graphic profile, from
  [cohm/kth-doc-templates](https://github.com/cohm/kth-doc-templates). Used by every
  KTH-only figure. Series roles are named (`ALL_APPLICANTS` = kthgray,
  `ADMITTED` = kthblue, `ACCEPTED` = kthgreen, `HIGHLIGHT` = kthbrick), so "antagna"
  is the same blue everywhere. `KTH_SEQUENCE` orders the profile colours for stacked
  charts; `CMAP_COUNT` (white→kthblue→kthnavy) and `CMAP_FRACTION` (kthteal ramp) are
  the choropleth ramps — counts and shares get different hues so they cannot be
  mistaken for each other.
- **`UNIVERSITY_STYLE`** — one `(colour, marker, line style)` per institution code,
  used by the cross-university plots. Group on `larosate_id`, never on `label`: the
  label is display text and the same institution is called "KTH" in three different
  peer sets. `sort_by_identity()` fixes the legend order.

Each colour is that university's own: KTH blue `#004791`, Chalmers `#6550EB` (their
2026 purple, sampled from the logo artwork — their web site still served the old dark
blue in July 2026), Lund bronze `#875E29` (`--bs-primary` on lu.se), Uppsala red
`#990000`, Linköping turquoise `#00A0AF`, Luleå `#286BBD`, Umeå navy `#2A4765`,
Karlstad green `#1A936F`, Mittuniversitetet plum `#AA4A9B`, Linné `#B58900`. Two are
deliberately darkened because the on-brand value cannot reach 3:1 on white:
Linköping's `#03DAF7` (unusable as a line at all) and Linné's `#FFE000` (1.3:1).

**Why marker and line style are part of the identity.** Four of the ten universities
have a blue profile colour and two a red one, so colour alone cannot separate them.
Run the dataviz skill's `validate_palette.js` on the shipped palette and it reports:

| check | adjacent pairs (the rule for line charts) | all pairs |
|---|---|---|
| CVD separation | **8.3** (deutan) — passes the ≥8 target | 5.8 — fails |
| Normal vision | **16.3** — passes the ≥15 gate | 7.7 (KTH vs Umeå) — fails |
| Contrast ≥3:1 | **all 10 pass** | — |

All-pairs cannot be made to pass at nine series with these brands, and forcing it
destroys them — a hue-preserving optimiser turns KTH into `#005EDE` and Uppsala into a
pink. So the colours stay authentic and every series also carries a distinct marker and
line style; Umeå in particular is dotted with `X` markers because its navy is the one
colour that genuinely collides with KTH blue. Do not "fix" this by recolouring.

Two validator FAILs are accepted deliberately, both properties of the brand colours
rather than of the arrangement: KTH `#004791`, Uppsala `#990000` and Umeå `#2A4765` sit
below the 0.43 lightness band (they are simply dark, and their contrast on white is
8.7–9.4:1, so legibility is better not worse), and Lund's bronze and Umeå's navy are
below the 0.10 chroma floor because bronze and desaturated navy are what those brands
are. Umeå's alternative site blue `#006699` would clear both flags but drops its
separation from Luleå from 16.3 to 6.2 — a worse trade, already tested.

## UHR pipeline

`uhr_data.py` has no matplotlib import; it fetches, caches and normalises, and `analyze_uhr.py` plots.

- **Endpoint**: `GET /api/antagningsstatistik/searchTotal?searchterm=<PERIOD>&larosaten=<INST>&utbildningstyp=program&tillfalle=urv2&pagesize=1000`. One request per (period, institution) returns every programme there with its `sok`, `urval1` *and* `urval2` blocks. `tillfalle` must be `urv1|urv2|sok` (empty → HTTP 500) but does not change the payload. An empty `searchterm` returns an error page, so periods are looped. Coverage HT08–HT26.
- **Cache**: `data/uhr_cache/searchTotal_<PERIOD>_<INST>.json.gz` plus `manifest.json`. A warm run makes zero network calls; `--refresh` refetches. Requests are sequential with a 1 s delay and an identifying User-Agent — this is an undocumented internal API on someone's public site.
- **Join key is `(lärosäteId, programId)`, never `utbildningId`.** `utbildningId` is an MD5 of a key containing the display name, so it changes when the name is reworded: KTH teknisk fysik has one id for HT08–14 and another for HT15–26. `programId` *is* the local programme code (`CTFYS`, `TKTFY`, …).
- **Two tidy frames** from `to_tidy()`: `df_prog` (one row per institution × programme × period × urval) and `df_grupp` (one row per selection group). `sok` counts are repeated on both urval rows, so never `groupby().sum()` across `urval` — use `pick_urval()`.
- **Score sentinels**: `lägstaAntagnaPoäng` is a string. `"*"` means everyone eligible was admitted so no threshold exists (not missing data); `"-"` means nobody was admitted. `_parse_score()` is the only place it is interpreted, and `score` is NaN unless `score_status == "value"` (asserted). Plot code must filter on `score_status` first.
- **Scale families** via `score_scale(urvalsgrupp)`; a figure may only mix groups with the same value. Verified ranges: `BG` 0–20 (HT08–09 only), `BI`/`BII`/`BIEX` 0–22.5 from HT10 (CTFYS was already 21.50 in HT10, so there is **no** scale break at 2014 — the break is HT09→HT10, when meritpoäng arrived), `HP` 0–2.0, `BF` 1–4, `BIII` foreign grades to ~490, `MAFY` and `T10`/`T20` on their own integer scales, `SDAU`/`MAAU`/`AU2`/`KPU` in credits (30–999). KTH HT26 alone has 24 distinct groups.
- **Rows that must be filtered**: `antagningsomgångKategori` separates the national round from international master's and course rounds (the latter never have urval 2); `_drop_senare_del()` removes admission into year 2/3, which is a different population carrying credit-scale groups; `inställd` marks cancelled programmes; `_resolve_duplicates(policy="largest")` collapses several campuses/tracks of one programme in one period.
- **Series have gaps, never zero-fill.** `CTMAT` does not exist before HT20, `MAFY` is absent in some years, and `år`/`tillfälleId` are `None` in HT08 rows (the year is derived from `terminId`).
- `validate_tidy()` (`--check`) asserts hand-verified values (CTFYS HT26: 1904 applicants, 672 first-choice, 143 admitted, BI 22.40, HP 1.80, MAFY 41) plus structural facts. It is what will fail if the API changes shape.
- **Do not read the UHR per-programme Excel export** (`/api/excel/utbildning`) for merit scores — its threshold sheets attach values to the wrong groups. See `README.md`. Any `Antagningsstatsistik-*.xlsx` files lying around came from there.
- `CFATE` has no counterpart at any other Swedish university, so cross-university plots are skipped for it. Peer sets for CTFYS/CTMAT/COPEN are hardcoded in `analyze_uhr.py`.

### Cross-checking the two sources

For HT26 the admitted counts agree reasonably and the shares of women among applicants and among admitted agree to within about two percentage points (largest gap 2.3), which is the useful confirmation that the `admitted()` definition is right:

| | NyA antagna | UHR antagna | NyA sökande | UHR sökande |
|---|---|---|---|---|
| CTFYS | 131 | 143 | 1515 | 1904 |
| CTMAT | 71 | 75 | 919 | 1217 |
| CFATE | 95 | 112 | 1150 | 1539 |
| COPEN | 78 | 90 | 1558 | 2093 |

NyA admitted is 4–17 below UHR because the export postdates urval 2: people who declined after admission now show as `Struken - Återbud`, while UHR reports the count at urval 2. With the earlier 2026-07-25 export the gap was 1–9.

The applicant counts do **not** agree: the NyA export holds 74–80 % of UHR's applicant figure, consistently across all four programmes. That is a population difference rather than a bug — treat UHR as the official applicant count and the NyA row count as "applications in this export". Don't try to reconcile them by adjusting the loading code.

## Known rough edges

Genuine quirks, not a cleanup TODO list — check with Christian before "fixing" any of them.

- **Rows are applications, not applicants.** The same person appears once per programme applied to, so raw counts are application counts. `dedupe_applicants(policy=...)` collapses them per person via `ApplicantID`; existing plots deliberately still count applications.
- `plot_municipality_map()` and `plot_county_map()` remain near-duplicates. They now share `_map_style()` and `_annotate_regions()`, but the inset and figure handling are still separate; merging them fully would change map output.
- `plot_merit_correlation()` calls `sns.set(style="whitegrid")`, which mutates global rc state for every plot drawn afterwards in the same run.
- The exports back to 2016 (`../../NyA/createReport-<year>.xls`) are not wired into `SOURCES`, because which selection round each was taken at is unknown and comparing across rounds would be wrong.
