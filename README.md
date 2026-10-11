# SweUniAdmissionsAnalysis
Scripts for making plots of data from Swedish national admissions systems.

The repository works with two kinds of data, and **which one you have decides what you
can run**:

| | UHR admission statistics | NyA applicant exports |
|---|---|---|
| Who can get it | **Anyone** — published openly on uhr.se | Only university staff who need it for their work in admissions |
| Level | Aggregated: one row per programme, round and selection group | One row per application, i.e. per real person |
| Personal data | None | Yes: names, addresses, personnummer, e-mail, phone |
| In this repo | Yes — tidy exports under `data/uhr/` | **Never** — kept outside the repo |
| Script | `analyze_uhr.py` (+ `uhr_data.py`) | `analyze_geo.py` |

If you do not have NyA access, everything in the UHR column is still open to you:
clone, install, and run `analyze_uhr.py`.

# How to install
The scripts use standard packages like `numpy` and `matplotlib`, and some for making maps use `swemaps` and `geopandas`. The needed packages are listed in `requirements.txt`, and can be installed via `pip install -r requirements.txt`

So, e.g. on a Mac you can do something like this to set up a virtual python environment and install the dependencies
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Public data: what UHR's statistics give you

The Swedish Council for Higher Education, [UHR](https://www.uhr.se/en/start/), publishes
admission statistics for every programme at every Swedish university, for every
admission period from **HT08** onwards. `analyze_uhr.py` fetches them from the JSON
backend behind uhr.se's statistics pages (no login, no key), caches the raw responses
in `data/uhr_cache/`, and plots them. For each programme and period, and separately for
selection round 1 and 2, this gives:

* **Applicants** and **first-choice applicants**, split by gender (women/men) and by
  age band (≤ 24, 25–34, ≥ 35).
* **Admitted** and **reserves**, with the same gender and age splits.
* Per **selection group** (BI, BII, HP, BF, MAFY, …): the number admitted and on the
  reserve list, and the **lowest admitted merit score** — the admission threshold.
* Programme metadata: name, campus, whether it was cancelled.

From that, the script draws, over time:

* applicants, first-choice applicants and admitted;
* competition ratios (applicants and first-choice applicants per place);
* share of women among applicants, first-choice applicants and admitted;
* age composition of applicants and admitted;
* the admission threshold per selection group, one scale family per figure;
* how the intake divides between selection groups;

and, across universities, the same programme compared side by side (counts,
thresholds, gender share) — e.g. teknisk fysik at all nine universities that offer it.

**What UHR cannot tell you**: anything below programme level. There is no
geography, no distribution of merit scores (only the lowest admitted), no exact ages,
no combination of an applicant's choices, and no way to follow anyone from round 1
to round 2. Those questions need NyA.

The tidy, already-filtered data is committed as CSV under `data/uhr/`
(`<group>_program.csv`, `<group>_urvalsgrupp.csv`, with a `_provenance.json` recording
what was fetched and when), so it can be used directly in other tools. The raw
cache is not committed; a fresh clone fetches it on first run — one request per
period and university, a second apart — about 200 requests and a few minutes
for all four peer sets.

UHR also offers a per-programme Excel download, but **it should not be used for merit scores**: in its "Lägsta antagningspoäng" sheets the row labels follow one canonical selection-group order while each year's values follow that year's own order, so scores end up attached to the wrong group (demonstrated for KTH CTFYS HT2017). The JSON keeps each score on the same object as its group id, so it cannot desynchronise.

```bash
python analyze_uhr.py --no-show                       # KTH's four SCI programmes over time
python analyze_uhr.py --group peers-ctfys --no-show   # teknisk fysik, all universities
python analyze_uhr.py --programmes KTH:CTFYS,CTH:TKTFY --from-year 2010
python analyze_uhr.py --check                         # verify against hand-checked values
```

Programmes are identified as `<university code>:<local programme code>`; the peer
sets in `analyze_uhr.py` (`kth-sci`, `peers-ctfys`, `peers-ctmat`, `peers-copen`) show
the codes for the universities used so far.

# Restricted data: NyA applicant exports

NyA is the admissions system the universities work in. Staff with a role in admissions
can export one admission round as an Excel file with one row per application: name,
address, personnummer, contact details, the free-text merit field, priority order,
result and reply. **These files contain personal data about real applicants. They
must stay outside this repository, and plots or tables made from them should only be
shared at an aggregated level.** `.gitignore` blocks `*.xls`/`*.xlsx`/`*.csv` (apart
from the committed datasets) as a safety net.

`analyze_geo.py` reads such exports (paths configured in `SOURCES` in the script) and
adds what UHR cannot give:

* **Geography**: applicants and admitted per municipality and county, on maps, via the
  postal-code mapping in `data/postalcode_data.csv`.
* **Merit score distributions** for applicants and admitted, and correlations between
  score types (BI, BII, HP, MAFY) — not just the lowest admitted value.
* **Exact age** and **gender** through the chain applicants → first choice → admitted.
* **Outcomes**: what each `Resultat` means in practice, attrition between the rounds,
  what became of each programme's first-choice applicants, priority order.

To protect the data, `analyze_geo.py` drops the identifying columns as soon as age,
gender and a pseudonymous id have been derived, prints dataframe shapes rather than
contents (unless `--verbose`), and suppresses map shares for groups smaller than
`--min-n` (default 5). `--schema-only` prints a new export's columns and value counts
without any rows, so its output is safe to share. Run `python analyze_geo.py --help`
for the options.

# Making the plots

Run everything from the repository root — the scripts use paths relative to it
(`data/postalcode_data.csv`, `plots/`), and `plots/` has to exist.

Public data only (anyone can run this):

```bash
source venv/bin/activate
export MPLBACKEND=Agg

for grp in kth-sci peers-ctfys peers-ctmat peers-copen; do
  python analyze_uhr.py --group $grp --no-show --export --quiet
done
```

With NyA access, additionally:

```bash
python analyze_geo.py --no-show
python analyze_geo.py --no-show --label HT25 --only insight
```

Together these make about 300 PDFs in `plots/` (171 from UHR, 128 from the NyA
exports at the last full run) and refresh the tidy CSVs under `data/uhr/`.

`--no-show` saves each figure and closes it instead of opening a window. Drop it (and
the `MPLBACKEND` line) to get the windows, but note that the full run is about 300
figures, so narrow it with `--group` or `--only` first.

Smaller pieces:

```bash
python analyze_uhr.py --no-show                     # KTH's four programmes over time
python analyze_uhr.py --group peers-ctfys --no-show # one peer set at a time
python analyze_geo.py --no-show --only insight      # one NyA plot family (needs NyA)
python analyze_geo.py --no-show --label HT25        # the other admission round (needs NyA)
```

Useful flags:

| Flag / variable | What it does |
|---|---|
| `--refresh` | re-fetch from uhr.se instead of the local cache; needed when a new admission round is published. Otherwise a run makes no network calls at all. |
| `--export` | write the tidy UHR data to `data/uhr/*.csv` |
| `--check` | verify the UHR data against hand-verified values, then exit |
| `--schema-only` | (`analyze_geo.py`) print a new export's columns and value counts, no rows and no plots — safe to share |
| `--only geo,merit,insight` | (`analyze_geo.py`) pick which plot families to draw |
| `PLOTS_DIR=/tmp/testplots` | write the PDFs somewhere else, to check a change without overwriting `plots/` |

## Older plots with different file names

Plot file names gained an admission-round label (`CountyMap_CTFYS_HT26_sökande.pdf`)
when multi-round support was added. Files from before that change are not regenerated
and will keep the older colours and the older definition of *antagen*, so they disagree
with current output. To list and then remove them:

The UHR plots have no round label either — they cover many years — so the filter has
to exclude them explicitly. Check the list before deleting anything; it should be 77
files, all from `analyze_geo.py`:

```bash
cd plots
ls *.pdf | grep -v '^UHR' | grep -v 'HT2[56]'            # check what would go
ls *.pdf | grep -v '^UHR' | grep -v 'HT2[56]' | xargs rm
```

# Colours

`palette.py` holds all colour choices. KTH-only figures use the KTH graphic profile
(from [cohm/kth-doc-templates](https://github.com/cohm/kth-doc-templates)); the
cross-university figures give each institution its own profile colour — KTH blue,
Chalmers purple, Lund bronze, Uppsala red, Linköping turquoise, and so on — so a
university looks the same in every plot.

Because four of the ten universities have a blue profile colour, each series also gets
its own marker and line style, and the legend shows both. That way the lines stay
tellable apart without relying on colour, which also makes the figures readable in
greyscale and for colour-blind readers. See `CLAUDE.md` for the measured separation
figures and the two deliberate compromises.

# Merit score scales

Merit scores from different selection groups are never drawn on the same axis: BG (up to HT09) runs 0–20, BI/BII from HT10 run 0–22.5, högskoleprovet 0–2.0, BF 1–4 and KTH's local groups on their own integer scales. Each scale family gets its own figure.

A lowest admitted score of `*` means every eligible applicant in that group was
admitted, so no threshold exists — it is not missing data and not a zero. Those years
are drawn as open circles along the bottom of the axis instead of being left out.

