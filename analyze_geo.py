"""Plots of per-applicant admission data from the NyA admissions portal:
where applicants live, what merit scores they have, how old they are, which
selection group admitted them, and whether they accept the offer.

The input Excel exports contain personal data (names, addresses, personnummer)
for real applicants. They live outside this repository and must stay there.
preprocess_data() drops the direct identifiers as soon as age, gender and a
pseudonymous id have been derived from them, so that later output - including a
stray print of a dataframe - cannot leak them. Pass --keep-pii to disable that
while debugging locally.

For aggregated cohort-level statistics over many years and other universities,
see analyze_uhr.py.
"""

import argparse
import datetime
import hashlib
import os
import re
import secrets

import pandas as pd
import numpy as np

import swemaps
import geopandas as gpd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

import palette as pal
from plot_utils import apply_grid, finish_plot, set_show, stats_label

# Columns every source file must have for the analyses below to work
REQUIRED_COLUMNS = ["Personnummer", "Resultat", "Kurs-/programkod"]
# Columns that are used when present, and whose absence makes a plot skip a year
OPTIONAL_COLUMNS = ["Meritvärde", "Ja-svar", "Prio", "Urvalsgrupp", "Postnummer"]

# The merit types extracted out of the free-text Meritvärde column
MERIT_TYPES = ["BI", "BII", "HP", "MAFY"]

# Direct identifiers, dropped once the derived columns exist. Matched
# case-insensitively against column names.
PII_PATTERNS = re.compile(
    r"personnummer|f[öo]rnamn|efternamn|mellannamn|\bnamn\b|adress|e-?post"
    r"|telefon|mobil|c/o|v[åa]rdnadshavare", re.IGNORECASE)
# Geography columns that must survive the sweep above (KnNamn/LnNamn look like
# name columns but hold municipality and county names)
PII_KEEP = {"KnNamn", "LnNamn", "KnKod", "Ort", "Postnummer",
            "Latitude", "Longitude"}

# Verbosity, set from --verbose. Controls whether dataframe contents are printed.
VERBOSE = False

# Label used for applicants whose postal code is not in the Swedish mapping,
# typically people living abroad. They used to be dropped from every analysis.
UNKNOWN_COUNTY = "Okänd/utanför Sverige"


def describe_df(df, label):
    """Report a dataframe's size, and its contents only when --verbose is given.

    Replaces bare print(df) calls: the applicant frames hold personal data before
    preprocess_data() strips it, and a shape is what is actually informative.
    """
    print(f"{label}: {len(df)} rader x {len(df.columns)} kolumner")
    if VERBOSE:
        with pd.option_context("display.max_columns", 50, "display.width", 200):
            print(df.head(10))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def infer_source_metadata(path):
    """Guess admission year and reference date from a file name such as
    createReport-2026-07-25.xls or createReport-2025.xls.

    Only a default: which selection round an export was taken at cannot be read
    from the name, so SOURCES below states it explicitly.
    """
    name = os.path.basename(path)
    full = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if full:
        year = int(full.group(1))
        return {"Antagningsår": year,
                "Referensdatum": pd.Timestamp(full.group(0)),
                "Etikett": f"HT{year % 100:02d}"}
    year_only = re.search(r"(20\d{2})", name)
    if year_only:
        year = int(year_only.group(1))
        return {"Antagningsår": year,
                "Referensdatum": pd.Timestamp(year=year, month=9, day=1),
                "Etikett": f"HT{year % 100:02d}"}
    return {"Antagningsår": pd.NA, "Referensdatum": pd.NaT, "Etikett": name}


def _normalise_columns(df):
    """Excel exports pick up trailing spaces and non-breaking spaces in headers."""
    df.columns = [str(c).replace(" ", " ").strip() for c in df.columns]
    return df


def load_applicant_files(sources):
    """Load one or more applicant exports, tagging each row with its source.

    sources is a list of dicts with at least a 'path'; 'Antagningsår',
    'Antagningsomgång', 'Etikett' and 'Referensdatum' override what the file name
    implies. Rows from different files are otherwise indistinguishable, which is
    what makes multi-round comparison possible at all.

    Returns (df, schema) where schema is a file-by-column presence table.
    """
    frames, schema_rows = [], {}
    for source in sources:
        path = source["path"]
        meta = infer_source_metadata(path)
        meta.update({k: v for k, v in source.items() if k != "path"})

        if path.lower().endswith(".csv"):
            frame = pd.read_csv(path, dtype={"Personnummer": str,
                                             "Postnummer": str})
        else:
            frame = pd.read_excel(path, dtype={"Personnummer": str,
                                               "Postnummer": str})
        frame = _normalise_columns(frame)
        # the programme code column carries trailing spaces in the exports
        if "Kurs-/programkod" in frame.columns:
            frame["Kurs-/programkod"] = frame["Kurs-/programkod"].astype(str).str.strip()
        label = meta.get("Etikett") or os.path.basename(path)
        schema_rows[label] = set(frame.columns)

        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            print(f"VARNING: hoppar över {path}, saknar kolumn(er): "
                  f"{', '.join(missing)}")
            continue

        frame["Antagningsår"] = meta.get("Antagningsår")
        frame["Antagningsomgång"] = meta.get("Antagningsomgång", "okänd")
        frame["Etikett"] = label
        frame["Referensdatum"] = meta.get("Referensdatum")
        frame["Källfil"] = os.path.basename(path)
        print(f"Läste {len(frame)} rader från {os.path.basename(path)} "
              f"({label}, {meta.get('Antagningsomgång', 'okänd omgång')})")
        frames.append(frame)

    if not frames:
        raise SystemExit("Ingen av källfilerna kunde användas.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    dupes = df.groupby(["Antagningsår", "Antagningsomgång"], dropna=False).size()
    if len(dupes) < len({(f["Antagningsår"].iloc[0], f["Antagningsomgång"].iloc[0])
                         for f in frames}):
        print("VARNING: flera filer har samma (Antagningsår, Antagningsomgång)")
    return df, check_schema(schema_rows)


def check_schema(schema_rows):
    """File-by-column presence table, printed so a missing Ja-svar or Meritvärde
    in an older export is visible rather than surfacing as an empty plot."""
    all_columns = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS
                                      if c not in REQUIRED_COLUMNS]
    extra = sorted({c for cols in schema_rows.values() for c in cols}
                   - set(all_columns))
    table = pd.DataFrame(
        {label: [c in cols for c in all_columns] for label, cols in schema_rows.items()},
        index=all_columns)
    print("\nKolumner per källfil (kända kolumner):")
    print(table.replace({True: "ja", False: "-"}).to_string())
    if extra:
        print(f"Övriga kolumner i filerna: {len(extra)} st")
    return table


def probe_schema(sources):
    """Print what the source files contain, without printing any row.

    Everything here is an aggregate - column names, value counts of result and
    answer codes, and counts of malformed values - so it is safe to read and to
    paste elsewhere. Run this first when a new export arrives: the result and
    answer vocabularies below are built from its output.
    """
    for source in sources:
        path = source["path"]
        print("\n" + "=" * 70)
        print(f"KÄLLFIL: {path}")
        if not os.path.exists(path):
            print("  saknas!")
            continue
        if path.lower().endswith(".csv"):
            frame = pd.read_csv(path, dtype={"Personnummer": str, "Postnummer": str})
        else:
            frame = pd.read_excel(path, dtype={"Personnummer": str, "Postnummer": str})
        frame = _normalise_columns(frame)
        print(f"  form: {frame.shape[0]} rader x {frame.shape[1]} kolumner")
        print("  kolumner:")
        for col in frame.columns:
            print(f"    {col!r} ({frame[col].dtype})")

        for col in ("Resultat", "Ja-svar", "Prio", "Urvalsgrupp"):
            if col in frame.columns:
                print(f"\n  {col} - antal per värde:")
                counts = frame[col].value_counts(dropna=False)
                for value, n in counts.items():
                    print(f"    {value!r}: {n}")

        if "Kurs-/programkod" in frame.columns:
            codes = frame["Kurs-/programkod"].value_counts()
            print(f"\n  Kurs-/programkod: {len(codes)} olika värden, "
                  f"de 10 största: {dict(codes.head(10))}")

        if "Personnummer" in frame.columns:
            pnr = frame["Personnummer"].astype(str)
            bad = ~pnr.str.match(r"^\d{8}[-+]?\d{4}$")
            print(f"\n  Personnummer som inte matchar YYYYMMDD-NNNN: {bad.sum()}")
        if "Meritvärde" in frame.columns:
            print(f"  Meritvärde saknas för: {frame['Meritvärde'].isna().sum()} rader")
        group_like = [c for c in frame.columns
                      if re.search(r"urval|antagningsgrupp|grupp", str(c), re.I)]
        print(f"  kolumner som ser ut som urvalsgrupp: {group_like or 'inga'}")


def load_postal_mapping(mapping_csv):
    """Load postal code to municipality mapping."""
    mapping = pd.read_csv(mapping_csv, dtype={'Postnummer': str})
    mapping['Postnummer'] = mapping['Postnummer'].str.zfill(5)  # Ensure 5-digit codes
    print("Read in data for %d postal codes" % len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# derived columns
# ---------------------------------------------------------------------------

_MERIT_PATTERNS = {}


def extract_merit_score(text, category):
    """Pull one merit score out of the free-text Meritvärde field, which looks
    like 'BI (20,50) HP (1,05) MAFY (34)'. Swedish decimal commas become points.

    The single implementation: plot_score_distribution used to carry its own copy
    of this regex, which meant two places could drift apart.
    """
    if category not in _MERIT_PATTERNS:
        _MERIT_PATTERNS[category] = re.compile(
            rf"{category}\s*\(\s*([\d.,]+)\s*\)", flags=re.IGNORECASE)
    if pd.isna(text):
        return None
    match = _MERIT_PATTERNS[category].search(str(text))
    if match:
        return float(match.group(1).replace(",", "."))
    return None


# Result strings, classified in order so that the more specific patterns win.
# These are the values NyA actually uses, read off the HT26 export:
#   Antagen / Antagen - Studieavgiftsvillkor
#   Reserv / Reserv - Studieavgiftsvillkor
#   Struken - Takvärde för ord. urval/inplacering uppnått
#   Struken - Uteblivet svar
#   Struken - Obehörighet
#   Struken - Återbud
#   Struken med annan orsak
#   Ej behandlat / Ej till urval - Struken -Sen anmälan inte tillåten
# 'Reservantagen' does not occur in this export, but is classified as admitted if
# it ever appears, since being admitted from the reserve list is an offer.
RESULT_PATTERNS = [
    ("Antagen", r"reservantagen"),
    ("Antagen", r"^antagen"),
    ("Reserv", r"^reserv"),
    ("Struken – återbud", r"återbud|aterbud"),
    ("Struken – takvärde", r"takvärde|inplacering"),
    ("Struken – uteblivet svar", r"uteblivet svar"),
    ("Struken – obehörighet", r"obehörig"),
    ("Ej behandlat", r"ej behandlat|sen anmälan|ej till urval"),
    ("Struken – annan orsak", r"struk"),
]

RESULT_CATEGORY_ORDER = ["Antagen", "Reserv",
                         "Struken – takvärde", "Struken – uteblivet svar",
                         "Struken – obehörighet", "Struken – återbud",
                         "Struken – annan orsak", "Ej behandlat", "Övrigt"]

# Canonical definition of "admitted": offered a place in this selection round.
# Being admitted from the reserve list counts; standing on the reserve list does
# not. Previously the code had two competing tests - str.contains("Antagen"),
# which would also match "Reservantagen", and str.lower() == "antagen", which
# would not - so the same data gave different denominators in different plots.
ADMITTED_CATEGORIES = ("Antagen",)

# 'Struken – takvärde' means the application was struck because the applicant
# already holds a place on a higher-priority choice, so these people are admitted
# somewhere - just not to this programme. Never read them as rejections.
PLACED_ELSEWHERE_CATEGORY = "Struken – takvärde"


def classify_result(value):
    if pd.isna(value):
        return "Övrigt"
    text = str(value).strip().lower()
    for category, pattern in RESULT_PATTERNS:
        if re.search(pattern, text):
            return category
    return "Övrigt"


def add_result_categories(df):
    """Add Resultat_kategori plus the booleans derived from it."""
    df = df.copy()
    df["Resultat_kategori"] = pd.Categorical(
        df["Resultat"].apply(classify_result),
        categories=RESULT_CATEGORY_ORDER, ordered=True)

    unmapped = df.loc[df["Resultat_kategori"] == "Övrigt", "Resultat"]
    if len(unmapped):
        print(f"VARNING: {len(unmapped)} rader med okänt Resultat: "
              f"{dict(unmapped.value_counts(dropna=False).head(10))}")

    df["Antagen"] = df["Resultat_kategori"].isin(ADMITTED_CATEGORIES)
    df["Reservplacerad"] = df["Resultat_kategori"] == "Reserv"
    df["Struken"] = df["Resultat_kategori"].astype(str).str.startswith("Struken")
    df["Placerad_annat_val"] = df["Resultat_kategori"] == PLACED_ELSEWHERE_CATEGORY
    # still in the running for this programme when the export was taken
    df["Kvar_i_urval"] = df["Antagen"] | df["Reservplacerad"]
    return df


def admitted(df):
    """The single admitted test. Boolean mask, never NaN."""
    return df["Antagen"].fillna(False).astype(bool)


def add_answer_columns(df):
    """Add Svar_kategori from the Ja-svar column.

    What Ja-svar actually records, verified against the HT26 export: it is the
    reply to the notification from the FIRST selection, where an applicant has to
    answer yes to stay in the running. It is not an acceptance of a place. In the
    HT26 urval 2 export every one of the 403 admitted rows and all 1504 reserve
    rows have 'Ja', because anyone who answered 'Nej' (1139) or did not answer
    (220) had already been struck as 'Struken – uteblivet svar'.

    So this column measures attrition between the rounds, not yield. A plot of
    "share of admitted who accepted" would be a flat 100 %. The genuine signal is
    who was lost before urval 2 and why, which is what the response plots below
    show. Applies to all rows, not only admitted ones.
    """
    df = df.copy()
    if "Ja-svar" not in df.columns:
        df["Svar_kategori"] = pd.Series(pd.NA, index=df.index, dtype="object")
        df["Svarat_ja"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        return df

    def classify(value):
        if pd.isna(value):
            return "Inget svar"
        text = str(value).strip().lower()
        if text.startswith("ja"):
            return "Ja"
        if text.startswith("nej"):
            return "Nej"
        return "Inget svar"

    df["Svar_kategori"] = pd.Categorical(df["Ja-svar"].apply(classify),
                                         categories=["Ja", "Nej", "Inget svar"],
                                         ordered=True)
    df["Svarat_ja"] = (df["Svar_kategori"] == "Ja").astype("boolean")
    return df


def add_applicant_id(df, salt=None):
    """Add a pseudonymous ApplicantID so applications can be grouped per person
    without keeping the personnummer.

    With ADMISSIONS_SALT set in the environment the id is stable between runs,
    which also makes the same person recognisable across years. Without it a
    random salt is used per run: deduplication still works, cross-year linkage
    does not. Linking individuals across years is a deliberate step, so it is
    opt-in rather than the default.
    """
    df = df.copy()
    if salt is None:
        salt = os.environ.get("ADMISSIONS_SALT")
        if not salt:
            salt = secrets.token_hex(16)
            print("ADMISSIONS_SALT är inte satt: slumpmässigt salt används, "
                  "så samma person kan inte följas mellan år.")
    key = salt.encode("utf-8")[:64]

    def digest(value):
        if pd.isna(value):
            return None
        normalised = re.sub(r"\D", "", str(value))
        return hashlib.blake2b(normalised.encode("utf-8"), key=key,
                               digest_size=8).hexdigest()

    df["ApplicantID"] = df["Personnummer"].apply(digest)
    return df


def dedupe_applicants(df, policy="best_outcome", extra_keys=("Antagningsår",)):
    """Collapse the per-application rows to one row per person.

    The export has one row per application, so a person who applied to four
    programmes appears four times. Counting rows therefore counts applications,
    which is the right denominator for "how many applications" and the wrong one
    for "how many students".

      per_application  no-op, the historical behaviour
      top_choice       keep the application with the lowest Prio
      best_outcome     keep the best result, ties broken by Prio
      admitted_only    keep only admitted rows, then best_outcome
    """
    if policy == "per_application":
        return df
    work = df
    if policy == "admitted_only":
        work = work[admitted(work)]

    keys = [k for k in extra_keys if k in work.columns] + ["ApplicantID"]
    rank = work["Resultat_kategori"].cat.codes if "Resultat_kategori" in work else 0
    prio = pd.to_numeric(work.get("Prio"), errors="coerce").fillna(99)
    if policy == "top_choice":
        order = work.assign(_a=prio, _b=rank)
    else:
        order = work.assign(_a=rank, _b=prio)
    out = (order.sort_values(keys + ["_a", "_b"])
                .drop_duplicates(subset=keys, keep="first")
                .drop(columns=["_a", "_b"]))
    print(f"  dedupe ({policy}): {len(df)} ansökningar -> {len(out)} personer")
    return out


def drop_pii_columns(df):
    """Remove the direct identifiers, keeping the derived columns."""
    dropped = [c for c in df.columns
               if c not in PII_KEEP and PII_PATTERNS.search(str(c))]
    if dropped:
        print(f"Tar bort personuppgiftskolumner: {', '.join(dropped)}")
    return df.drop(columns=dropped)


def preprocess_data(df, mapping, postal_code_col="Postnummer", drop_pii=True):
    """Add geography from postal codes, merit score columns, result categories,
    age, gender and a pseudonymous id, then drop the direct identifiers.

    Applicants whose postal code is not in the mapping (people living abroad,
    mostly) are kept and labelled UNKNOWN_COUNTY. They used to be dropped here,
    which quietly removed them from the merit and age analyses too; the two map
    functions drop them instead, since they cannot be placed on a map.
    """
    df = df.copy()

    # geography
    if postal_code_col in df.columns:
        df[postal_code_col] = df[postal_code_col].astype(str).str.zfill(5)
        df = df.merge(mapping, how='left', left_on=postal_code_col,
                      right_on='Postnummer')
        n_unmatched = df['LnNamn'].isna().sum()
        if n_unmatched:
            print(f"{n_unmatched} rader har postnummer utanför registret "
                  f"(märks som '{UNKNOWN_COUNTY}')")
        df['LnNamn'] = df['LnNamn'].fillna(UNKNOWN_COUNTY)
        df['KnNamn'] = df['KnNamn'].fillna(UNKNOWN_COUNTY)
        df = df.astype({'KnKod': 'Int64'})  # this was otherwise of type 'object'

    # merit scores out of the free-text Meritvärde column
    for merit_type in MERIT_TYPES:
        if "Meritvärde" in df.columns:
            df[f"{merit_type}_score"] = df["Meritvärde"].apply(
                extract_merit_score, category=merit_type)
        else:
            df[f"{merit_type}_score"] = np.nan

    # result categories and the canonical Antagen flag
    df = add_result_categories(df)

    # age, relative to each source's reference date rather than "now", so that
    # re-running the script does not shift every age by the elapsed time
    born = pd.to_datetime(df['Personnummer'].astype(str).str[0:8],
                          format='%Y%m%d', errors='coerce')
    reference = df['Referensdatum'] if 'Referensdatum' in df.columns else pd.NaT
    if not isinstance(reference, pd.Series) or reference.isna().all():
        reference = pd.Series(pd.Timestamp.now(), index=df.index)
        print("VARNING: inget Referensdatum - åldrar räknas från dagens datum")
    df['Födelseår'] = born.dt.year.astype('Int64')
    df['Ålder'] = (reference - born) / datetime.timedelta(days=365.2425)

    # legal gender from the second-to-last digit of the personnummer
    def extract_gender(personal_number):
        text = re.sub(r"\D", "", str(personal_number))
        if len(text) < 12:
            return None
        return "K" if int(text[10]) % 2 == 0 else "M"
    df['Kön'] = df['Personnummer'].apply(extract_gender)
    n_unknown = df['Kön'].isna().sum()
    if n_unknown:
        print(f"{n_unknown} rader har personnummer som inte kunde tolkas för kön")

    df = add_applicant_id(df)
    df = add_answer_columns(df)
    if drop_pii:
        df = drop_pii_columns(df)

    describe_df(df, "Förbehandlade data")
    return df


def aggregate_by_column(df, col_name):
    """Count number of rows in df that have the same value for column with name col_name, e.g. to get the number of applicants from each municipality."""
    return df[col_name].value_counts().reset_index(name='number')


def aggregate_fraction(df, col_name, numerator_mask, min_n=5):
    """Fraction of rows satisfying a mask, per value of col_name.

    Returns the 'number' column the map functions expect, plus n_total. Groups
    with fewer than min_n rows get NaN instead of a fraction: with a handful of
    people a share is noise, and a 0 % or 100 % cell in a shared PDF can identify
    individuals.
    """
    work = pd.DataFrame({col_name: df[col_name],
                         "_num": numerator_mask.fillna(False).astype(bool)})
    grouped = work.groupby(col_name, dropna=True, observed=True)["_num"]
    out = grouped.agg(["sum", "size"]).reset_index()
    out.columns = [col_name, "n_numerator", "n_total"]
    out["number"] = (out["n_numerator"] / out["n_total"]).astype("Float64")
    thin = out["n_total"] < min_n
    if thin.any():
        print(f"  {thin.sum()} grupper har färre än {min_n} rader och visas som "
              f"okända")
        out.loc[thin, "number"] = pd.NA
    return out

def _map_style(values, value_kind):
    """Colour scale and label format for a choropleth layer.

    Counts keep the historical logarithmic scale. Fractions must not use it: with
    values below 1 the old LogNorm(vmin=1, vmax=max) had vmin > vmax, which is why
    the existing 'andel kvinnliga sökande' county maps came out with a broken
    colour scale.

    The scale is always given as an explicit norm. geopandas ignores vmin/vmax
    when a norm argument is present, so setting the range that way silently let
    the colours auto-scale to the data while the labels showed the real values.
    """
    top = values.max()
    if value_kind == "fraction":
        # start at zero and round the top up to whole percentage points, so the
        # colours use the range the data actually occupies
        top = 0.05 if pd.isna(top) or top <= 0 else np.ceil(float(top) * 20) / 20
        return {"norm": Normalize(vmin=0.0, vmax=min(top, 1.0)),
                "cmap": pal.CMAP_FRACTION,
                "legend_kwds": {"format": PercentFormatter(xmax=1, decimals=0)}}, \
               (lambda v: f"{v:.0%}")
    if pd.isna(top) or top <= 0:
        top = 1
    return {"norm": LogNorm(vmin=1, vmax=top), "cmap": pal.CMAP_COUNT}, \
           (lambda v: f"{int(round(v))}")


def _annotate_regions(ax, merged, formatter, value_kind):
    """Write each region's value at its centroid; return the total for counts."""
    total = 0
    for _, row in merged.iterrows():
        value = row["number"]
        if pd.isna(value) or value <= 0:
            continue
        if value_kind == "count":
            total += value
        centroid = row["geometry"].centroid
        ax.text(centroid.x, centroid.y, formatter(value), ha="center", va="center",
                fontsize=6, color=pal.KTH_COLORS["darkgray"], alpha=0.85)
    return total


def plot_municipality_map(aggregated_df, group = "", print_total = True,
                          add_stockholm_inset=False, value_kind="count"):
    if aggregated_df.empty or aggregated_df["number"].dropna().empty:
        print(f"  hoppar över kommunkarta '{group}': inga värden")
        return
    # load GeoParquet data via swemaps.get_path("kommun")
    path = swemaps.get_path("kommun")
    map_df = gpd.read_parquet(path).to_crs(epsg=3006)  # Swedish National Grid
    map_df = map_df.astype({'kommun_kod':'Int64'})

    # columns 'kommun_kod' and 'KnKod' should be matched to merge
    merged = map_df.merge(aggregated_df, left_on="kommun_kod", right_on="KnKod", how="left")
    if value_kind == "count":
        merged["number"] = merged["number"].fillna(0)
    merged["number"] = merged["number"].astype(float)

    style, formatter = _map_style(merged["number"], value_kind)
    fig, ax = plt.subplots(1, 1, figsize=(7, 12))
    merged.plot(
        column="number",
        linewidth=0.8,
        edgecolor="0.8",
        ax=ax,
        legend=True,
        missing_kwds={"color": pal.COLOR_NO_DATA, "label": "Inga data"},
        **style
    )

    # add labels with the number for each region
    total = _annotate_regions(ax, merged, formatter, value_kind)

    if add_stockholm_inset:
        # Stockholm bounding box (EPSG:3006)
        # Bounding box from county geometry - turned out to be too big
        #path = swemaps.get_path("lan")
        #counties_df = gpd.read_parquet(path).to_crs(epsg=3006)
        #print(counties_df)
        #stockholm_geom = counties_df.loc[counties_df["lan"] == "Stockholms län"].geometry.values[0]
        #xmin, ymin, xmax, ymax = stockholm_geom.bounds
        #print("Stockholm region defined as:")
        #print(stockholm_geom.bounds)
        # Filter municipalities inside Stockholm County
        #stockholm_df = merged[merged.geometry.intersects(stockholm_geom)]
        
        xmin, ymin, xmax, ymax = 640000, 6550000, 700000, 6610000
        stockholm_df = merged.cx[xmin:xmax, ymin:ymax]

        if not stockholm_df.empty:
            # Create inset axes in the Baltic Sea area
            axins = inset_axes(ax, width="100%", height="100%",
                                #bbox_to_anchor=(0.7, 0.4, 0.35, 0.35),
                                bbox_to_anchor=(-0.1, 0.7, 0.4, 0.4),
                                bbox_transform=ax.transAxes, borderpad=0)

            stockholm_df.plot(
                column="number",
                linewidth=0.6,
                edgecolor="0.8",
                ax=axins,
                # same colour scale as the main map, but the inset has no legend
                **{k: v for k, v in style.items() if k != "legend_kwds"}
            )

            # Labels for inset
            for idx, row in stockholm_df.iterrows():
                if row.geometry.centroid.is_empty or pd.isna(row["number"]) \
                        or row["number"] <= 0:
                    continue
                x, y = row.geometry.centroid.x, row.geometry.centroid.y
                # don't draw numbers outside the inset box (even if part of the region is inside)
                if (x < xmin) or (x > xmax) or (y < ymin) or (y > ymax):
                    continue
                axins.text(x, y, formatter(row["number"]), fontsize=5,
                           ha="center", va="center", color=pal.KTH_COLORS["darkgray"])

            axins.set_xlim(xmin, xmax)
            axins.set_ylim(ymin, ymax)
            axins.axis("off")
            axins.set_title("Stockholmsregionen", fontsize=8)
            
    if print_total:
        print(group + ": %d" % total)

    ax.set_title(group + " per kommun" + (f" (totalt {int(total)})" if print_total else ""), fontsize=16)
    ax.axis("off")
    finish_plot("MunicipalityMap", group)


def plot_county_map(aggregated_df, group = "", print_total = True,
                    value_kind="count"):
    if aggregated_df.empty or aggregated_df["number"].dropna().empty:
        print(f"  hoppar över länskarta '{group}': inga värden")
        return
    # load GeoParquet data via swemaps.get_path("lan")
    path = swemaps.get_path("lan")
    map_df = gpd.read_parquet(path).to_crs(epsg=3006)
    # columns 'lan' and 'LnNamn' should be matched to merge
    merged = map_df.merge(aggregated_df, left_on="lan", right_on="LnNamn", how="left")
    # counts of zero are meaningful (nobody from that county); a missing fraction
    # is not the same as a fraction of zero, so it stays missing
    if value_kind == "count":
        merged["number"] = merged["number"].fillna(0)
    merged["number"] = merged["number"].astype(float)

    style, formatter = _map_style(merged["number"], value_kind)
    fig, ax = plt.subplots(1, 1, figsize=(7, 12))
    merged.plot(
        column="number",
        linewidth=0.8,
        edgecolor="0.8",
        ax=ax,
        legend=True,
        missing_kwds={"color": pal.COLOR_NO_DATA, "label": "Inga data"},
        **style
    )

    # add labels with number for each region
    total = _annotate_regions(ax, merged, formatter, value_kind)
    if print_total:
        print(group + ": %d" % total)
    
    ax.set_title(group + " per län" + (f" (totalt {int(total)})" if print_total else ""), fontsize=16)

    ax.axis("off")
    finish_plot("CountyMap", group)


def plot_score_distribution(df, category, group):
    """
    Plot histogram of a score category ('BI', 'BII', 'HP' or 'MAFY') for all applicants and admitted ones. Bin sizes depends on category. Adds counts to legend.
    """

    bin_sizes = {
        "BI": 0.1,
        "BII": 0.1,
        "HP": 0.05,
        "MAFY": 1
    }

    if category not in bin_sizes:
        raise ValueError(f"Unknown category '{category}'. Must be one of: {list(bin_sizes.keys())}")

    bin_size = bin_sizes[category]

    # uses the score column added by preprocess_data rather than re-deriving it
    has_score = df[df[f"{category}_score"].notna()]
    admitted_rows = has_score[admitted(has_score)]

    if has_score.empty:
        print(f"No valid {category} scores found.")
        return

    total_all = len(has_score)
    total_admitted = len(admitted_rows)

    min_val = has_score[f"{category}_score"].min()
    max_val = has_score[f"{category}_score"].max()
    bins = np.arange(np.floor(min_val)-bin_size/2, np.ceil(max_val)+bin_size/2, bin_size)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)  # Grid behind bars

    ax.hist(
        has_score[f"{category}_score"],
        bins=bins,
        alpha=0.5,
        label=f"Alla sökande (n = {total_all})",
        color=pal.ALL_APPLICANTS,
        edgecolor="white", linewidth=0.5
    )
    ax.hist(
        admitted_rows[f"{category}_score"],
        bins=bins,
        alpha=0.5,
        label=f"Antagna (n = {total_admitted})",
        color=pal.ADMITTED,
        edgecolor="white", linewidth=0.5
    )

    ax.set_xlabel(f"{category}-poäng")
    ax.set_ylabel("Antal sökande")
    ax.set_title(f"Fördelning av {category}-poäng, {group}: alla sökande vs. antagna")

    # Subtle grid
    apply_grid(ax)

    ax.legend()
    finish_plot("GradeDistribution", f"{group}_{category}")

def plot_merit_correlation(df, group, merit1, merit2):
    """
    Plot correlation between merit1 and merit2 scores.
    Draws three regression lines (all, admitted, rejected),
    shows Pearson and Spearman correlations,
    and includes a count-based legend.
    """

    valid = df.dropna(subset=[f"{merit1}_score", f"{merit2}_score"])

    if valid.empty:
        print("No valid data points with both {merit1} and {merit2} scores.")
        return

    # Subsets
    admitted = valid[valid["Antagen"]]
    rejected = valid[~valid["Antagen"]]

    # Counts
    total_all = len(valid)
    total_admitted = len(admitted)
    total_rejected = len(rejected)

    # Correlations (for all data)
    pearson_corr, _ = pearsonr(valid[f"{merit1}_score"], valid[f"{merit2}_score"])
    spearman_corr, _ = spearmanr(valid[f"{merit1}_score"], valid[f"{merit2}_score"])

    # Plot.
    # Note: no sns.set() here. It resets matplotlib's global rcParams to
    # seaborn's own theme, which changes the font family and undoes
    # palette.apply_style() for every figure drawn afterwards in the same run -
    # so the plots produced after the first correlation figure came out in a
    # different font from the ones before it. seaborn's plotting functions do not
    # need it; the grid comes from apply_grid() like everywhere else.
    fig, ax = plt.subplots(figsize=(9, 7))

    # All candidates - solid dark regression line
    sns.regplot(
        data=valid,
        x=f"{merit1}_score",
        y=f"{merit2}_score",
        scatter=False,
        ax=ax,
        line_kws={"color": pal.REJECTED, "label": f"All (n={total_all})",
                      "linestyle": "-"},
        ci=None
    )

    # Admitted - dashed blue
    if not admitted.empty:
        sns.regplot(
            data=admitted,
            x=f"{merit1}_score",
            y=f"{merit2}_score",
            scatter=False,
            ax=ax,
            line_kws={"color": pal.ADMITTED,
                      "label": f"Antagna (n = {total_admitted})", "linestyle": "--"},
            ci=None
        )

    # Rejected - dashed gray
    if not rejected.empty:
        sns.regplot(
            data=rejected,
            x=f"{merit1}_score",
            y=f"{merit2}_score",
            scatter=False,
            ax=ax,
            line_kws={"color": pal.ALL_APPLICANTS,
                      "label": f"Ej antagna (n = {total_rejected})", "linestyle": "--"},
            ci=None
        )

    # Scatter plot (colored by admission)
    sns.scatterplot(
        data=valid,
        x=f"{merit1}_score",
        y=f"{merit2}_score",
        hue="Antagen",
        ax=ax,
        palette={True: pal.ADMITTED, False: pal.ALL_APPLICANTS},
        edgecolor="white", linewidth=0.5,
        alpha=0.6
    )

    ax.set_title(
        f"Korrelation mellan {merit1}- och {merit2}-poäng, sökande {group}\n"
        f"(Pearson r = {pearson_corr:.3f}, Spearman ρ = {spearman_corr:.3f})"
    )
    ax.set_xlabel(f"{merit1}-poäng")
    ax.set_ylabel(f"{merit2}-poäng")
    apply_grid(ax)
    ax.legend(title="Antagen")
    finish_plot("GradeCorrelation", f"{group}_{merit1}-{merit2}")

# ---------------------------------------------------------------------------
# plots about the admitted students
# ---------------------------------------------------------------------------

def _pct_labels(ax, bars, totals):
    """Write a percentage above each bar."""
    for bar, share in zip(bars, totals):
        if pd.isna(share):
            continue
        ax.annotate(f"{share:.0%}", (bar.get_x() + bar.get_width() / 2,
                                     bar.get_height()),
                    ha="center", va="bottom", fontsize=8)


def plot_outcome_breakdown(df, group, code_col="Kurs-/programkod"):
    """Outcome of every application, per programme.

    The audit plot for the whole analysis: it shows how the applications divide
    between admitted, reserve and the various reasons for being struck, so the
    denominators used everywhere else are visible rather than implied.
    """
    counts = (df.groupby([code_col, "Resultat_kategori"], observed=True)
                .size().unstack(fill_value=0))
    counts = counts.loc[:, counts.sum() > 0]
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(counts))
    for i, category in enumerate([c for c in RESULT_CATEGORY_ORDER
                                  if c in counts.columns]):
        values = counts[category].to_numpy(dtype=float)
        ax.bar(counts.index, values, bottom=bottom, label=category,
               color=pal.KTH_SEQUENCE[i % len(pal.KTH_SEQUENCE)],
               edgecolor="white", linewidth=0.6)
        bottom += values
    ax.set_ylabel("Antal ansökningar")
    ax.set_xlabel("Program")
    ax.set_title(f"Utfall för alla ansökningar, {group}")
    apply_grid(ax)
    ax.legend(fontsize=8, title="Resultat")
    finish_plot("Outcome", group)


def plot_response_attrition(df, group):
    """How applications were lost between the two selection rounds.

    Ja-svar is the reply to the first round's notification, not an acceptance of a
    place, so this is an attrition picture: of everyone who applied, who was still
    in the running, who dropped out by answering no or not answering, and who was
    removed for not being eligible.
    """
    stages = {
        "Antagna": admitted(df),
        "Reserver": df["Reservplacerad"],
        "Placerad på högre val": df["Placerad_annat_val"],
        "Svarade nej": df["Resultat_kategori"] == "Struken – uteblivet svar",
        "Obehöriga": df["Resultat_kategori"] == "Struken – obehörighet",
        "Återbud": df["Resultat_kategori"] == "Struken – återbud",
    }
    counts = {name: int(mask.fillna(False).sum()) for name, mask in stages.items()}
    rest = len(df) - sum(counts.values())
    if rest > 0:
        counts["Övrigt"] = rest

    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(counts)
    values = [counts[n] for n in names]
    bars = ax.bar(names, values, color=pal.ADMITTED, edgecolor="white", linewidth=0.6)
    _pct_labels(ax, bars, [v / len(df) for v in values])
    ax.set_ylabel("Antal ansökningar")
    ax.set_title(f"Var ansökningarna tog vägen, {group} "
                 f"(totalt {len(df)} ansökningar)")
    ax.tick_params(axis="x", rotation=20)
    apply_grid(ax)
    finish_plot("Attrition", group)


def plot_first_choice_outcome(df, group, code_col="Kurs-/programkod"):
    """What happened to each programme's first-choice applicants.

    'Antagen till annat program i urvalet' is only visible because the export
    covers several programmes and applications can be linked per person: an
    applicant struck on their second choice because the ceiling was reached is
    admitted somewhere, and counting them as a loss would be wrong.
    """
    if "ApplicantID" not in df.columns:
        return
    admitted_ids = set(df.loc[admitted(df), "ApplicantID"].dropna())
    first = df[pd.to_numeric(df["Prio"], errors="coerce") == 1]
    if first.empty:
        return

    rows = {}
    for code, part in first.groupby(code_col, observed=True):
        is_admitted = admitted(part)
        elsewhere = (~is_admitted) & part["ApplicantID"].isin(admitted_ids)
        rows[code] = {
            "Antagen här": int(is_admitted.sum()),
            "Antagen till annat program i urvalet": int(elsewhere.sum()),
            "Reserv": int((part["Reservplacerad"] & ~elsewhere).sum()),
            "Svarade nej / uteblivet svar": int(
                ((part["Resultat_kategori"] == "Struken – uteblivet svar")
                 & ~elsewhere).sum()),
            "Obehörig": int(((part["Resultat_kategori"] == "Struken – obehörighet")
                             & ~elsewhere).sum()),
        }
    table = pd.DataFrame(rows).T
    table["Övrigt"] = (first.groupby(code_col, observed=True).size()
                       - table.sum(axis=1)).clip(lower=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(table))
    for i, column in enumerate(table.columns):
        values = table[column].to_numpy(dtype=float)
        ax.bar(table.index, values, bottom=bottom, label=column,
               color=pal.KTH_SEQUENCE[i % len(pal.KTH_SEQUENCE)],
               edgecolor="white", linewidth=0.6)
        bottom += values
    ax.set_ylabel("Antal förstahandssökande")
    ax.set_xlabel("Program")
    ax.set_title(f"Utfall för förstahandssökande, {group}")
    apply_grid(ax)
    ax.legend(fontsize=8)
    finish_plot("FirstChoiceOutcome", group)


def plot_selection_group_profile(df, group):
    """Which selection groups applicants and admitted students have scores in.

    The NyA export has no column for the group an applicant was actually admitted
    through, only the merit values they hold, so this compares the composition of
    the two populations rather than attributing admissions to groups. Attributing
    them would mean estimating each cut-off from the admitted set and then using it
    to explain that same set.
    """
    admitted_rows = df[admitted(df)]
    if df.empty or admitted_rows.empty:
        return
    shares_all, shares_adm = [], []
    for merit in MERIT_TYPES:
        shares_all.append(df[f"{merit}_score"].notna().mean())
        shares_adm.append(admitted_rows[f"{merit}_score"].notna().mean())

    x = np.arange(len(MERIT_TYPES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, [100 * s for s in shares_all], width,
           label=f"Alla sökande (n = {len(df)})", color=pal.ALL_APPLICANTS,
           edgecolor="white", linewidth=0.6)
    ax.bar(x + width / 2, [100 * s for s in shares_adm], width,
           label=f"Antagna (n = {len(admitted_rows)})", color=pal.ADMITTED,
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, MERIT_TYPES)
    ax.set_xlabel("Urvalsgrupp")
    ax.set_ylabel("Andel med meritvärde i gruppen [%]")
    ax.set_title(f"Urvalsgrupper: antagna jämfört med alla sökande, {group}")
    apply_grid(ax)
    ax.legend()
    finish_plot("SelectionGroups", group)


def plot_merit_gap(df, group):
    """Merit score of applicants versus admitted, one panel per selection group.

    Separate panels because the scales differ - BI runs to 22.5, HP to 2.0 and
    MAFY is an integer scale - so a single axis would be meaningless.

    The dashed line is the lowest score held by an admitted student, which is NOT
    the same as the group's admission threshold that UHR publishes. Applicants are
    admitted through one group but hold merit values in several, so somebody
    admitted on their högskoleprov result drags the minimum BI value far below the
    BI cut-off. For CTFYS HT26 that difference is 14.22 here against UHR's 22.40.
    The NyA export has no column for the group an applicant was admitted through,
    so the real threshold has to come from analyze_uhr.py.
    """
    admitted_rows = df[admitted(df)]
    if admitted_rows.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    drawn = False
    for ax, merit in zip(axes.flat, MERIT_TYPES):
        col = f"{merit}_score"
        all_scores = df[col].dropna()
        adm_scores = admitted_rows[col].dropna()
        if all_scores.empty:
            ax.set_axis_off()
            continue
        drawn = True
        data = [all_scores, adm_scores] if not adm_scores.empty else [all_scores]
        labels = [f"Alla sökande\n(n = {len(all_scores)})"]
        if not adm_scores.empty:
            labels.append(f"Antagna\n(n = {len(adm_scores)})")
        ax.boxplot(data, showmeans=True, widths=0.5)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        if not adm_scores.empty:
            lowest = adm_scores.min()
            ax.axhline(lowest, color=pal.HIGHLIGHT, linestyle="--", linewidth=1,
                       label=f"Lägsta {merit} bland antagna: {lowest:g}\n"
                             f"(ej gruppens antagningsgräns)")
            ax.legend(fontsize=7)
        ax.set_ylabel(f"{merit}-poäng")
        ax.set_title(merit)
        apply_grid(ax)
    if not drawn:
        plt.close(fig)
        return
    fig.suptitle(f"Meritvärde: alla sökande vs. antagna, {group}")
    finish_plot("MeritGap", group)


def plot_prio_distribution(df, group):
    """How the admitted students had ranked the programme in their application."""
    admitted_rows = df[admitted(df)]
    if admitted_rows.empty or "Prio" not in df.columns:
        return
    prio_all = pd.to_numeric(df["Prio"], errors="coerce")
    prio_adm = pd.to_numeric(admitted_rows["Prio"], errors="coerce")
    levels = sorted(set(prio_all.dropna().astype(int)))

    counts_all = [int((prio_all == p).sum()) for p in levels]
    counts_adm = [int((prio_adm == p).sum()) for p in levels]
    rate = [a / b if b else np.nan for a, b in zip(counts_adm, counts_all)]

    x = np.arange(len(levels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, counts_all, width, label="Alla sökande",
           color=pal.ALL_APPLICANTS, edgecolor="white", linewidth=0.6)
    ax.bar(x + width / 2, counts_adm, width, label="Antagna", color=pal.ADMITTED,
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, [str(p) for p in levels])
    ax.set_xlabel("Prioritet i anmälan")
    ax.set_ylabel("Antal ansökningar")
    apply_grid(ax)

    twin = ax.twinx()
    twin.plot(x, [100 * r for r in rate], "-o", color=pal.HIGHLIGHT,
              label="Andel antagna")
    twin.set_ylabel("Andel antagna [%]", color=pal.HIGHLIGHT)
    twin.set_ylim(0, max(100 * np.nanmax(rate) * 1.3, 5))
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, fontsize=9)
    ax.set_title(f"Antagna per sökandeprioritet, {group}")
    finish_plot("PrioDistribution", group)


def plot_gender_funnel(df, group):
    """Share of women at each stage from applicant to admitted.

    Shows where in the process the gender balance changes, which a set of
    separate counts does not.
    """
    if "Kön" not in df.columns:
        return
    prio = pd.to_numeric(df.get("Prio"), errors="coerce")
    stages = [
        ("Alla sökande", pd.Series(True, index=df.index)),
        ("Förstahandssökande", prio == 1),
        ("Behöriga", df["Resultat_kategori"] != "Struken – obehörighet"),
        ("Antagna", admitted(df)),
    ]
    names, shares, totals = [], [], []
    for name, mask in stages:
        part = df[mask.fillna(False)]
        known = part[part["Kön"].notna()]
        if known.empty:
            continue
        names.append(name)
        shares.append(100 * (known["Kön"] == "K").mean())
        totals.append(len(part))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(names, shares, "-o", color=pal.ADMITTED)
    for x, (share, total) in enumerate(zip(shares, totals)):
        ax.annotate(f"{share:.1f} %\n(n = {total})", (x, share),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9)
    ax.axhline(50, color=pal.REFERENCE_LINE, linewidth=0.8)
    ax.set_ylabel("Andel kvinnor [%]")
    ax.set_ylim(0, max(60, max(shares) * 1.3 if shares else 60))
    ax.set_title(f"Andel kvinnor genom antagningskedjan, {group}")
    apply_grid(ax)
    finish_plot("Funnel", group)


def summarise_admitted(df, code_col="Kurs-/programkod"):
    """One row per admission round and programme, with the headline numbers.

    Printed instead of dumping dataframes, and the natural object to compare with
    UHR's published figures for the same programme and year (see analyze_uhr.py).

    min_<merit>_antagna is the lowest value of that merit type held by an admitted
    student. It is deliberately not called the admission threshold: admitted
    students hold merit values in several groups but are admitted through one, so
    this sits well below the group cut-off UHR publishes. Compare n_antagna and
    the gender shares with UHR; do not compare these minima with UHR's thresholds.
    """
    prio = pd.to_numeric(df.get("Prio"), errors="coerce")
    work = df.assign(_prio=prio, _admitted=admitted(df))
    rows = []
    for (year, round_name, code), part in work.groupby(
            ["Antagningsår", "Antagningsomgång", code_col], dropna=False,
            observed=True):
        adm = part[part["_admitted"]]
        record = {
            "år": year, "omgång": round_name, "program": code,
            "n_ansökningar": len(part),
            "n_sökande": part["ApplicantID"].nunique()
            if "ApplicantID" in part else pd.NA,
            "n_prio1": int((part["_prio"] == 1).sum()),
            "n_antagna": len(adm),
            "n_reserver": int(part["Reservplacerad"].sum()),
            "andel_kvinnor_sökande": (part["Kön"] == "K").mean()
            if "Kön" in part else pd.NA,
            "andel_kvinnor_antagna": (adm["Kön"] == "K").mean()
            if "Kön" in adm and len(adm) else pd.NA,
            "medelålder_antagna": adm["Ålder"].mean() if len(adm) else pd.NA,
        }
        for merit in MERIT_TYPES:
            scores = adm[f"{merit}_score"].dropna()
            record[f"min_{merit}_antagna"] = (scores.min() if not scores.empty
                                              else pd.NA)
        rows.append(record)
    return pd.DataFrame(rows)


def audit_admitted_definitions(df, code_col="Kurs-/programkod"):
    """Compare the canonical admitted count with the two definitions previously
    used in this file, so any change in the plots' numbers is visible."""
    if "Resultat" not in df.columns:
        return None
    rows = []
    for code, part in df.groupby(code_col, observed=True):
        rows.append({
            "program": code,
            "contains('Antagen')": int(part["Resultat"].str.contains(
                "Antagen", na=False).sum()),
            "lower()=='antagen'": int((part["Resultat"].str.lower()
                                       == "antagen").sum()),
            "kanonisk admitted()": int(admitted(part).sum()),
        })
    table = pd.DataFrame(rows)
    print("\n=== Kontroll av 'antagen'-definitioner (antal rader) ===")
    print(table.to_string(index=False))
    changed = table[table["contains('Antagen')"] != table["kanonisk admitted()"]]
    if changed.empty:
        print("Den kanoniska definitionen ger samma antal som contains('Antagen').")
    else:
        print("OBS: antalen skiljer sig från den tidigare contains('Antagen') "
              "för dessa program - äldre PDF:er visar andra tal.")
    return table


def plot_age_distribution(df, group, admitted_df=None, accepted_df=None,
                          bins=np.arange(15, 45, 0.25)):
    """Histogram of the age distribution: all applicants, optionally overlaid with
    admitted and (where 'Ja-svar' is available) those who accepted their offer."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)  # Grid behind bars

    ax.hist(
        df["Ålder"],
        bins=bins,
        alpha=0.5,
        label=stats_label(df["Ålder"], "Alla sökande"),
        color=pal.ALL_APPLICANTS,
        edgecolor="white", linewidth=0.5
    )
    if admitted_df is not None:
        ax.hist(
            admitted_df["Ålder"],
            bins=bins,
            alpha=0.5,
            label=stats_label(admitted_df["Ålder"], "Antagna"),
            color=pal.ADMITTED,
            edgecolor="white", linewidth=0.5
        )
    if accepted_df is not None:
        ax.hist(
            accepted_df["Ålder"],
            bins=bins,
            alpha=0.5,
            label=stats_label(accepted_df["Ålder"], "Tackat ja"),
            color=pal.ACCEPTED,
            edgecolor="white", linewidth=0.5
        )

    ax.set_xlabel("Ålder [år]")
    ax.set_ylabel("Antal sökande")
    ax.set_title(f"Åldersfördelning {group}: alla sökande vs. antagna")
    apply_grid(ax)
    ax.legend()
    finish_plot("AgeDistribution", group)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

# The applicant exports to analyse. Which selection round a file was taken at
# cannot be read from its name, so it is stated here: it decides whether two
# files are comparable at all.
SOURCES = [
    {"path": "../../NyA/createReport-2026-08-25-SCI.xls",
     "Antagningsår": 2026, "Antagningsomgång": "urval 2", "Etikett": "HT26"},
    {"path": "../../NyA/createReport-2025-08-11.xls",
     "Antagningsår": 2025, "Antagningsomgång": "urval 2", "Etikett": "HT25"},
]

POSTALCODE_MAPPING_FILE = "data/postalcode_data.csv"
PROGRAMS = ["CTFYS", "CTMAT", "CFATE", "COPEN"]

PLOTSETS = ("geo", "merit", "insight")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema-only", action="store_true",
                        help="skriv bara ut kolumner och värdefördelningar för "
                             "källfilerna (inga rader, inga diagram)")
    parser.add_argument("--label", default="HT26",
                        help="vilken källfil som analyseras, enligt Etikett "
                             "(standard: HT26). 'alla' använder samtliga.")
    parser.add_argument("--only", default=",".join(PLOTSETS),
                        help=f"vilka diagramgrupper som ritas: {', '.join(PLOTSETS)}")
    parser.add_argument("--programs", default=",".join(PROGRAMS),
                        help="programkoder, komma-separerat")
    parser.add_argument("--keep-pii", action="store_true",
                        help="behåll namn, adress och personnummer i dataramen "
                             "(bara för lokal felsökning)")
    parser.add_argument("--verbose", action="store_true",
                        help="skriv ut dataraminnehåll, inte bara storlekar")
    parser.add_argument("--no-show", action="store_true",
                        help="spara diagram utan att öppna fönster")
    parser.add_argument("--min-n", type=int, default=5,
                        help="minsta antal per grupp för att en andel ska visas "
                             "på kartor (standard: 5)")
    return parser


def main(argv=None):
    global VERBOSE
    args = build_parser().parse_args(argv)
    pal.apply_style()
    VERBOSE = args.verbose
    set_show(not args.no_show)

    if args.schema_only:
        probe_schema(SOURCES)
        return

    plotsets = {p.strip() for p in args.only.split(",") if p.strip()}
    programs = [p.strip() for p in args.programs.split(",") if p.strip()]

    df, _schema = load_applicant_files(SOURCES)
    mapping = load_postal_mapping(POSTALCODE_MAPPING_FILE)
    df = preprocess_data(df, mapping, drop_pii=not args.keep_pii)

    audit_admitted_definitions(df)

    if args.label.lower() != "alla":
        available = sorted(df["Etikett"].dropna().unique())
        if args.label not in available:
            raise SystemExit(f"Etikett '{args.label}' finns inte, välj bland "
                             f"{available} eller 'alla'")
        df = df[df["Etikett"] == args.label]
    round_label = args.label

    summary = summarise_admitted(df)
    print("\n=== Sammanfattning per program ===")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary.to_string(index=False))

    # ---- all programmes together ----
    all_label = f"CING SCI {round_label}"
    if "geo" in plotsets:
        plot_municipality_map(aggregate_by_column(df, "KnKod"),
                              f"{all_label} sökande", add_stockholm_inset=True)
        plot_county_map(aggregate_by_column(df, "LnNamn"),
                        f"{all_label} sökande")
        plot_county_map(aggregate_by_column(df[df["Kön"] == "K"], "LnNamn"),
                        f"{all_label} kvinnliga sökande")
    if "insight" in plotsets:
        plot_outcome_breakdown(df, all_label)
        plot_first_choice_outcome(df, all_label)

    # ---- per programme ----
    for p in programs:
        print(f"\n--- {p} ({round_label}) ---")
        df_program = df[df["Kurs-/programkod"].str.contains(p, na=False)]
        if df_program.empty:
            print(f"  ingen data för {p}")
            continue
        group = f"{p} {round_label}"
        df_female = df_program[df_program["Kön"] == "K"]
        prio = pd.to_numeric(df_program["Prio"], errors="coerce")
        df_prio = df_program[prio == 1]
        df_prio_female = df_prio[df_prio["Kön"] == "K"]
        df_admitted = df_program[admitted(df_program)]
        df_admitted_female = df_admitted[df_admitted["Kön"] == "K"]
        describe_df(df_program, f"  {p}")

        if "geo" in plotsets:
            plot_municipality_map(aggregate_by_column(df_program, "KnKod"),
                                  f"{group} sökande", add_stockholm_inset=True)
            plot_municipality_map(aggregate_by_column(df_prio, "KnKod"),
                                  f"{group} sökande prio 1", add_stockholm_inset=True)
            plot_municipality_map(aggregate_by_column(df_admitted, "KnKod"),
                                  f"{group} antagna", add_stockholm_inset=True)
            plot_county_map(aggregate_by_column(df_program, "LnNamn"),
                            f"{group} sökande")
            plot_county_map(aggregate_by_column(df_female, "LnNamn"),
                            f"{group} kvinnliga sökande")
            plot_county_map(aggregate_by_column(df_prio, "LnNamn"),
                            f"{group} sökande prio 1")
            plot_county_map(aggregate_by_column(df_prio_female, "LnNamn"),
                            f"{group} kvinnliga sökande prio 1")
            # fractions, using the helper instead of a hand-rolled merge
            plot_county_map(aggregate_fraction(df_prio, "LnNamn",
                                               df_prio["Kön"] == "K",
                                               min_n=args.min_n),
                            f"{group} andel kvinnliga sökande prio 1",
                            print_total=False, value_kind="fraction")
            plot_county_map(aggregate_fraction(df_program, "LnNamn",
                                               admitted(df_program),
                                               min_n=args.min_n),
                            f"{group} andel av sökande som antagits",
                            print_total=False, value_kind="fraction")
            plot_county_map(aggregate_by_column(df_admitted, "LnNamn"),
                            f"{group} antagna")
            plot_county_map(aggregate_by_column(df_admitted_female, "LnNamn"),
                            f"{group} kvinnliga antagna")

        if "merit" in plotsets:
            for merit in MERIT_TYPES:
                plot_score_distribution(df_program, merit, group)
            plot_merit_correlation(df_program, group, "HP", "BI")
            plot_merit_correlation(df_program, group, "MAFY", "BI")
            plot_merit_correlation(df_program, group, "BII", "BI")
            plot_merit_correlation(df_program, group, "HP", "MAFY")

        if "insight" in plotsets:
            plot_age_distribution(df_program, group, admitted_df=df_admitted)
            plot_response_attrition(df_program, group)
            plot_selection_group_profile(df_program, group)
            plot_merit_gap(df_program, group)
            plot_prio_distribution(df_program, group)
            plot_gender_funnel(df_program, group)


if __name__ == "__main__":
    main()
