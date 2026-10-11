"""Import aggregated admission statistics from UHR (uhr.se).

UHR publishes per-programme admission statistics — applicants, admitted, reserves
and the lowest admitted merit score per selection group — for every admission
period from HT08 onwards. There is no documented open API, but the statistics
pages are driven by a JSON backend that can be fetched directly:

    GET https://www.uhr.se/api/antagningsstatistik/searchTotal
        ?searchfor=&searchterm=HT26&searchkategori=&pagesize=1000
        &tillfalle=urv2&page=1&larosaten=KTH&utbildningstyp=program

One request per (period, institution) returns every programme at that institution
with its `sok`, `urval1` and `urval2` blocks, so the full KTH history HT08-HT26 is
19 requests. Responses are cached on disk, so a warm run makes no network calls.

Deliberately NOT used: /api/excel/utbildning?utbildningId=... That Excel export has
a value-shift bug in its "Lägsta antagningspoäng" sheets — the row labels follow a
canonical selection-group order while each year's values follow that year's own
order, so scores land on the wrong group (demonstrated for KTH CTFYS HT2017). The
JSON keeps each score on the same object as its urvalsgruppId, so it cannot
desynchronise. Any Antagningsstatsistik-*.xlsx files downloaded from that endpoint
are unreliable for thresholds.

The endpoints are undocumented and unversioned. validate_tidy() checks a handful
of values that were verified by hand, and is what will fail loudly if the shape or
content changes.
"""

import gzip
import json
import os
import re
import time
from dataclasses import dataclass

import pandas as pd
import requests

BASE_URL = "https://www.uhr.se/api/antagningsstatistik"
USER_AGENT = "SweUniAdmissionsAnalysis/0.1 (+chohm@kth.se; teaching/research use)"
CACHE_DIR = os.path.join("data", "uhr_cache")
EXPORT_DIR = os.path.join("data", "uhr")
PAGE_SIZE = 1000          # server caps pagesize at 1000
REQUEST_DELAY_S = 1.0     # be polite: this is someone's public web site
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 3

# The national autumn/spring/summer round. UHR also publishes international
# master's and international course rounds, which are a different population
# (and never have an urval 2). Filter on the *category*, not antagningsomgångId,
# which is inconsistent across years (MASTERH08/MASTERH09 vs MASTERHT10 onwards).
NATIONAL_ROUNDS = {
    "HT": "Hösttermin",
    "VT": "Vårtermin",
    "ST": "Sommartermin",
}

FIRST_PERIOD_YEAR = 2008  # HT08 is the oldest period UHR publishes

# Rows for admission into year 2/3 of a programme ("senare del av program").
# A different population on a different scale — their selection groups are
# measured in credits, not merit scores — so they are dropped by default.
_SENARE_DEL_RE = re.compile(r"senare del|EJ\s*åk\s*1", re.IGNORECASE)
_SENARE_DEL_GROUPS = {"SDAU", "MAAU"}


@dataclass(frozen=True)
class Programme:
    """One programme at one institution, identified the only way that is stable
    across years: the institution code plus the local programme code.

    utbildningId must NOT be used for this - it is an MD5 of a key that includes
    the programme's display name, so it changes whenever the name is reworded.
    KTH teknisk fysik has one id for HT08-HT14 and another for HT15-HT26.
    """
    larosate_id: str            # e.g. "KTH"
    program_id: str             # e.g. "CTFYS" - UHR's programId is the local code
    label: str = None           # legend text; defaults to "<larosate> <program>"

    @property
    def display(self):
        """Human-readable text for titles and legends. Not unique: the same
        institution appears in several peer sets with different programmes."""
        return self.label or f"{self.larosate_id} {self.program_id}"

    @property
    def slug(self):
        """Unique identifier for file names. display() cannot be used for this -
        'KTH' is the legend label for CTFYS, CTMAT and COPEN in their respective
        peer sets, so plots would overwrite each other."""
        return f"{self.larosate_id}-{self.program_id}"

    @property
    def title(self):
        """Heading for a single-programme figure. Includes the programme code, so
        that a figure titled 'KTH' in a peer set still says which programme it is."""
        if self.label and self.program_id not in self.label:
            return f"{self.label} ({self.program_id})"
        return self.display


# ---------------------------------------------------------------------------
# selection group scales
# ---------------------------------------------------------------------------
# Lowest-admitted scores from different selection groups are not comparable, and
# some are not even the same kind of quantity. Verified against live data:
#   BG    0-20      betyg without meritpoäng (HT08-HT09 only)
#   BI    0-22.5    betyg incl. up to 2.5 meritpoäng; already 21.50 for CTFYS in
#                   HT10, so the 22.5 scale applies from HT10 onwards
#   HP    0.0-2.0   högskoleprovet
#   BF    1-4       folkhögskola (4.00 for CTFYS HT20)
#   BIII  ~0-480    foreign upper-secondary grades (480 for CTFYS HT10)
#   MAFY  ~0-50     KTH's local maths/physics group
#   T10/T20         other KTH local groups, in the tens-to-hundreds
#   SDAU/MAAU/AU2/KPU   högskolepoäng (credits): 30, 55, 75, 120, 999 ...
_GROUP_FAMILY = {
    "BG": "betyg_20",
    "BI": "betyg_22_5", "BII": "betyg_22_5", "BIEX": "betyg_22_5",
    "BIAV": "betyg_22_5", "BIBF": "betyg_22_5",
    "BIIAV": "betyg_22_5", "BIIBF": "betyg_22_5",
    "BIII": "utlandska_betyg", "BIV": "utlandska_betyg",
    "HP": "hogskoleprov", "HPAVG": "hogskoleprov", "HPBEF": "hogskoleprov",
    "BF": "folkhogskola",
    "MAFY": "mafy",
    "T1": "kth_lokal", "T2": "kth_lokal", "T10": "kth_lokal", "T20": "kth_lokal",
    "SDAU": "hogskolepoang", "MAAU": "hogskolepoang",
    "AU2": "hogskolepoang", "KPU": "hogskolepoang",
}

# Human-readable axis labels per scale family
SCALE_LABELS = {
    "betyg_20": "Betygspoäng (skala 0–20)",
    "betyg_22_5": "Betygspoäng (skala 0–22,5)",
    "utlandska_betyg": "Poäng, utländska betyg",
    "hogskoleprov": "Högskoleprovspoäng (skala 0–2,0)",
    "folkhogskola": "Folkhögskolebetyg (skala 1–4)",
    "mafy": "MAFY-poäng",
    "kth_lokal": "Poäng, lokal urvalsgrupp",
    "hogskolepoang": "Högskolepoäng",
    "annat": "Poäng",
}

# Groups worth plotting by default: the national ones plus KTH's MAFY.
DEFAULT_GROUPS = ["BI", "BII", "HP", "MAFY"]


def score_scale(urvalsgrupp):
    """Scale family for a selection group, e.g. 'betyg_22_5' for BI.

    Two groups may only share a plot axis if this returns the same value for both.
    """
    return _GROUP_FAMILY.get(urvalsgrupp, "annat")


# ---------------------------------------------------------------------------
# periods
# ---------------------------------------------------------------------------

def periods(from_year=FIRST_PERIOD_YEAR, to_year=None, terms=("HT",)):
    """Period codes for a year range, e.g. ['HT08', 'HT09', ...].

    UHR's codes are HT/VT/ST plus a two-digit year. There is no "all periods"
    query - an empty searchterm returns an error page - so periods are looped.
    """
    if to_year is None:
        to_year = from_year
    out = []
    for year in range(from_year, to_year + 1):
        for term in terms:
            out.append(f"{term}{year % 100:02d}")
    return out


def period_year(period):
    """'HT26' -> 2026. The rows' own 'år' field is None for early periods, so the
    year is always derived from the period code instead."""
    return 2000 + int(period[2:])


# ---------------------------------------------------------------------------
# fetching and caching
# ---------------------------------------------------------------------------

_session = None


def _get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT,
                                 "Accept": "application/json"})
    return _session


def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name + ".json.gz")


def _cache_read(name):
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _cache_write(name, payload, meta=None):
    with gzip.open(_cache_path(name), "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    if meta is not None:
        manifest_path = os.path.join(CACHE_DIR, "manifest.json")
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        manifest[name] = meta
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=1, sort_keys=True)


def _get_json(path, params):
    """GET one JSON endpoint, with retries and a delay between requests."""
    url = f"{BASE_URL}/{path}"
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(REQUEST_DELAY_S)
        try:
            resp = _get_session().get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    # a bad parameter combination returns an HTML error page
                    last_error = ValueError(
                        f"{resp.url} returned {resp.headers.get('content-type')}, "
                        f"not JSON: {exc}")
                    break
            last_error = RuntimeError(f"HTTP {resp.status_code} for {resp.url}")
            if resp.status_code < 500:
                break
        if attempt < MAX_RETRIES:
            time.sleep(REQUEST_DELAY_S * 2 ** attempt)
    raise RuntimeError(f"UHR request failed: {last_error}")


def fetch_period(period, larosate, refresh=False, verbose=True):
    """All programme rows for one institution in one admission period.

    Cached per (period, institution), so adding an institution costs only its own
    periods and changing the programme list costs no requests at all.
    """
    name = f"searchTotal_{period}_{larosate}"
    if not refresh:
        cached = _cache_read(name)
        if cached is not None:
            return cached

    rows = []
    page = 1
    total = None
    while True:
        payload = _get_json("searchTotal", {
            "searchfor": "",
            "searchterm": period,
            "searchkategori": "",
            "pagesize": PAGE_SIZE,
            # tillfalle must be one of urv1/urv2/sok (empty gives HTTP 500) but
            # does not change the payload: every row carries sok, urval1, urval2.
            "tillfalle": "urv2",
            "page": page,
            "larosaten": larosate,
            "utbildningstyp": "program",
        })
        total = payload.get("antalObjekt", 0)
        batch = payload.get("data") or []
        rows.extend(batch)
        if not batch or len(rows) >= total or len(batch) < PAGE_SIZE:
            break
        page += 1

    if verbose:
        print(f"  hämtade {len(rows):4d} rader för {larosate} {period}"
              + (f" (av {total})" if total != len(rows) else ""))
    _cache_write(name, rows, meta={
        "url": f"{BASE_URL}/searchTotal",
        "period": period, "larosate": larosate,
        "antalObjekt": total, "pages": page,
        "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    })
    return rows


def fetch_institutions(period="HT26", refresh=False):
    """All institutions UHR has data for: {'KTH': 'Kungl. Tekniska högskolan', ...}"""
    name = f"larosaten_{period}"
    payload = None if refresh else _cache_read(name)
    if payload is None:
        payload = _get_json("larosaten", {
            "searchfor": "", "searchterm": period,
            "searchkategori": "", "tillfalle": "urv2",
        })
        _cache_write(name, payload)
    return {r["lärosäteId"]: r["lärosäte"] for r in payload.get("data", [])}


def fetch_rows(programmes, from_year=FIRST_PERIOD_YEAR, to_year=None,
               terms=("HT",), refresh=False, verbose=True):
    """Raw rows for every (period, institution) needed by a programme list."""
    institutions = sorted({p.larosate_id for p in programmes})
    codes = periods(from_year, to_year, terms)
    if verbose:
        print(f"UHR: {len(codes)} perioder x {len(institutions)} lärosäten "
              f"({', '.join(institutions)})")
    rows = []
    for period in codes:
        for larosate in institutions:
            rows.extend(fetch_period(period, larosate, refresh=refresh,
                                     verbose=verbose))
    return rows


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def _as_int(value):
    """Counts are missing (None) for some early rows, so they stay nullable."""
    if value is None or value == "":
        return pd.NA
    try:
        return int(value)
    except (TypeError, ValueError):
        return pd.NA


def _parse_score(raw):
    """Interpret one lägstaAntagnaPoäng value.

    This is the only place that string is interpreted. The sentinels are not
    missing data and must never be treated as numbers:
      '*' every eligible applicant in the group was admitted, so no threshold
          exists (verified: antagna > 0 and reserver == 0 in all such rows)
      '-' nobody was admitted in the group
    Returns (score, status) where score is None unless status == 'value'.
    """
    if raw is None:
        return None, "no_group"
    text = str(raw).strip()
    if text == "*":
        return None, "all_admitted"
    if text in ("-", ""):
        return None, "none_admitted"
    try:
        return float(text.replace(",", ".")), "value"
    except ValueError:
        return None, "okant"


_SOK_FIELDS = {
    "n_sokande": "sökande",
    "n_forstahand": "förstahandsökande",
    "n_sokande_k": "kvinnligaSökande",
    "n_sokande_m": "manligaSökande",
    "n_forstahand_k": "kvinnligaFörstahandsökande",
    "n_forstahand_m": "manligaFörstahandsökande",
    "n_sokande_u24": "sökande24ÅrEllerYngre",
    "n_sokande_25_34": "sökande25Till34År",
    "n_sokande_35p": "sökande35ÅrEllerÄldre",
    "n_forstahand_u24": "förstahandsökande24ÅrEllerYngre",
    "n_forstahand_25_34": "förstahandsökande25Till34År",
    "n_forstahand_35p": "förstahandsökande35ÅrEllerÄldre",
}

_URVAL_FIELDS = {
    "n_antagna": "antagna",
    "n_reserver": "reserver",
    "n_antagna_k": "kvinnligaAntagna",
    "n_antagna_m": "manligaAntagna",
    "n_reserver_k": "kvinnligaReserver",
    "n_reserver_m": "manligaReserver",
    "n_antagna_u24": "antagna24ÅrEllerYngre",
    "n_antagna_25_34": "antagna25Till34År",
    "n_antagna_35p": "antagna35ÅrEllerÄldre",
    "n_reserver_u24": "reserver24ÅrEllerYngre",
    "n_reserver_25_34": "reserver25Till34År",
    "n_reserver_35p": "reserver35ÅrEllerÄldre",
}

_COUNT_COLUMNS = list(_SOK_FIELDS) + list(_URVAL_FIELDS)


def _is_senare_del(row):
    """Admission into year 2/3 rather than the first year of the programme."""
    if _SENARE_DEL_RE.search(row.get("anmälningsalternativ") or ""):
        return True
    if (row.get("programinriktning") or "").upper() == "SEDE":
        return True
    groups = set()
    for key in ("urval1", "urval2"):
        block = row.get(key) or {}
        for grp in block.get("urvalsgrupper") or []:
            groups.add(grp.get("urvalsgruppId"))
    groups.discard("-")
    return bool(groups) and groups <= _SENARE_DEL_GROUPS


def to_tidy(rows, round_kategori=None, drop_senare_del=True, drop_cancelled=True,
            duplicate_policy="largest", verbose=True):
    """Turn raw API rows into two tidy frames.

    Returns (df_prog, df_grupp):
      df_prog   one row per (institution, programme, period, urval) with counts
      df_grupp  one row per (..., urval, selection group) with the lowest
                admitted score and its status

    The applicant-side counts (n_sokande etc.) do not depend on the selection
    round, so they are repeated on the urval1 and urval2 rows of df_prog. That
    makes every ratio a single-row computation, but it also means you must never
    groupby().sum() across urval - use pick_urval() to collapse instead.

    round_kategori restricts to one admission round (see NATIONAL_ROUNDS); None
    keeps every round.
    """
    kept, skipped_round, cancelled, senare_del = [], 0, [], 0
    for row in rows:
        if round_kategori is not None and row.get("antagningsomgångKategori") != round_kategori:
            skipped_round += 1
            continue
        if drop_senare_del and _is_senare_del(row):
            senare_del += 1
            continue
        if drop_cancelled and str(row.get("inställd") or "0") not in ("0", "False"):
            cancelled.append(f"{row.get('lärosäteId')} {row.get('programId')} "
                             f"{row.get('terminId')}")
            continue
        kept.append(row)

    if verbose:
        print(f"UHR: {len(rows)} rader in -> {len(kept)} kvar "
              f"(annan antagningsomgång: {skipped_round}, senare del: {senare_del}, "
              f"inställda: {len(cancelled)})")
        if cancelled:
            print("     inställda utbildningar: " + ", ".join(sorted(cancelled)))

    prog_records, grupp_records = [], []
    for row in kept:
        period = row.get("terminId")
        base = {
            "larosate_id": row.get("lärosäteId"),
            "larosate": row.get("lärosäte"),
            "program_id": row.get("programId") or row.get("kursId"),
            "program_namn": row.get("anmälningsalternativ"),
            "anmalningskod": row.get("anmälningskod"),
            "termin_id": period,
            "termin": period[:2] if period else None,
            "ar": period_year(period) if period else pd.NA,
            "omgang_kategori": row.get("antagningsomgångKategori"),
            "senaste_urval": row.get("senasteUrval"),
            "studieort": row.get("studieort"),
            "installd": str(row.get("inställd") or "0") not in ("0", "False"),
        }
        sok = row.get("sok") or {}
        sok_counts = {col: _as_int(sok.get(key)) for col, key in _SOK_FIELDS.items()}

        for urval in ("urval1", "urval2"):
            block = row.get(urval)
            if block is None:
                continue
            record = dict(base, urval=urval, **sok_counts)
            record.update({col: _as_int(block.get(key))
                           for col, key in _URVAL_FIELDS.items()})
            prog_records.append(record)

            for grp in block.get("urvalsgrupper") or []:
                group_id = grp.get("urvalsgruppId")
                score, status = _parse_score(grp.get("lägstaAntagnaPoäng"))
                if group_id in (None, "-"):
                    status = "no_group"
                grupp_records.append({
                    "larosate_id": base["larosate_id"],
                    "program_id": base["program_id"],
                    "anmalningskod": base["anmalningskod"],
                    "termin_id": period,
                    "ar": base["ar"],
                    "urval": urval,
                    "urvalsgrupp": group_id,
                    "score": score,
                    "score_status": status,
                    "score_raw": None if grp.get("lägstaAntagnaPoäng") is None
                                 else str(grp.get("lägstaAntagnaPoäng")),
                    "score_scale": score_scale(group_id),
                    "n_antagna_grupp": _as_int(grp.get("antagna")),
                    "n_reserver_grupp": _as_int(grp.get("reserver")),
                })

    df_prog = pd.DataFrame(prog_records)
    df_grupp = pd.DataFrame(grupp_records)
    if df_prog.empty:
        return df_prog, df_grupp

    df_prog = df_prog.astype({col: "Int64" for col in _COUNT_COLUMNS})
    df_prog["ar"] = df_prog["ar"].astype("Int16")
    df_prog, keep_codes = _resolve_duplicates(df_prog, duplicate_policy, verbose=verbose)

    df_grupp = df_grupp.astype({"n_antagna_grupp": "Int64",
                                "n_reserver_grupp": "Int64",
                                "score": "Float64"})
    df_grupp["ar"] = df_grupp["ar"].astype("Int16")
    if keep_codes is not None:
        # keep only the selection groups belonging to the programme rows that
        # survived deduplication. urval must be part of the key, otherwise each
        # group row matches both the urval1 and the urval2 row and is duplicated.
        keys = ["larosate_id", "program_id", "termin_id", "anmalningskod", "urval"]
        df_grupp = df_grupp.merge(keep_codes[keys].drop_duplicates(), on=keys,
                                  how="inner")
    for col in ("urvalsgrupp", "score_status", "score_scale", "urval"):
        df_grupp[col] = df_grupp[col].astype("category")

    # a sentinel must never reach a plot as a number
    assert df_grupp.loc[df_grupp["score_status"] != "value", "score"].isna().all(), \
        "score set for a non-numeric score_status"
    return df_prog, df_grupp


def _resolve_duplicates(df_prog, policy, verbose=True):
    """Collapse several rows for the same programme in the same period.

    These arise when a programme is offered on more than one campus or as more
    than one track, each with its own anmälningskod. 'largest' keeps the row with
    most applicants, which is the only policy that leaves both the counts and the
    thresholds internally consistent ('sum' would add counts that no single
    threshold belongs to).
    """
    keys = ["larosate_id", "program_id", "termin_id", "urval"]
    counts = df_prog.groupby(keys, dropna=False)["anmalningskod"].transform("size")
    df_prog = df_prog.assign(n_rader_kallor=counts.astype("Int8"))
    duplicated = df_prog[counts > 1]

    if not duplicated.empty and verbose:
        for (lar, prog, termin, urval), grp in duplicated.groupby(keys, dropna=False,
                                                                 observed=True):
            codes = ", ".join(f"{r.anmalningskod}({r.n_sokande})"
                              for r in grp.itertuples())
            print(f"     {lar} {prog} {termin} {urval}: {len(grp)} rader [{codes}]"
                  f" -> policy '{policy}'")

    if policy == "keep" or duplicated.empty:
        return df_prog, None

    if policy == "largest":
        order = df_prog["n_sokande"].fillna(-1)
        idx = order.groupby([df_prog[k] for k in keys], dropna=False).idxmax()
        out = df_prog.loc[sorted(idx.values)].reset_index(drop=True)
    elif policy == "sum":
        agg = {col: "sum" for col in _COUNT_COLUMNS}
        agg.update({col: "first" for col in df_prog.columns
                    if col not in _COUNT_COLUMNS + keys})
        out = df_prog.groupby(keys, dropna=False, observed=True).agg(agg).reset_index()
    else:
        raise ValueError(f"unknown duplicate_policy {policy!r}")
    return out, out


def select(df, programmes):
    """Restrict a frame to a programme list and attach each one's display label."""
    wanted = pd.DataFrame([{"larosate_id": p.larosate_id,
                            "program_id": p.program_id,
                            "label": p.display} for p in programmes])
    return df.merge(wanted, on=["larosate_id", "program_id"], how="inner")


def pick_urval(df_prog, prefer="urval2"):
    """One row per (institution, programme, period): the preferred selection round
    where it exists, otherwise the other one.

    Summer terms and the newest period may only have urval 1, so a straight filter
    on urval == 'urval2' would silently drop those years.
    """
    other = "urval1" if prefer == "urval2" else "urval2"
    keys = ["larosate_id", "program_id", "termin_id"]
    rank = df_prog["urval"].map({prefer: 0, other: 1}).fillna(2)
    order = df_prog.assign(_rank=rank).sort_values(keys + ["_rank"])
    return order.drop_duplicates(subset=keys, keep="first").drop(columns="_rank")


def add_ratios(df_prog):
    """Append the derived ratio and share columns used by the trend plots."""
    df = df_prog.copy()
    antagna = df["n_antagna"].astype("Float64")
    df["sokande_per_plats"] = df["n_sokande"] / antagna
    df["forstahand_per_plats"] = df["n_forstahand"] / antagna
    df["reserver_per_antagen"] = df["n_reserver"] / antagna
    for stem in ("sokande", "forstahand", "antagna"):
        total = df[f"n_{stem}"].astype("Float64")
        df[f"andel_kvinnor_{stem}"] = df[f"n_{stem}_k"] / total
    for stem in ("sokande", "antagna"):
        total = df[f"n_{stem}"].astype("Float64")
        df[f"andel_u24_{stem}"] = df[f"n_{stem}_u24"] / total
    return df


def filter_rows(rows, programmes):
    """Keep only the raw rows belonging to the wanted programmes.

    Applied before to_tidy() so that the diagnostics about dropped rounds and
    collapsed duplicates describe the programmes being analysed rather than every
    programme at the institution.
    """
    wanted = {(p.larosate_id, p.program_id) for p in programmes}
    return [r for r in rows
            if (r.get("lärosäteId"), r.get("programId")) in wanted]


def load_tidy(programmes, from_year=FIRST_PERIOD_YEAR, to_year=None, terms=("HT",),
              refresh=False, duplicate_policy="largest", verbose=True):
    """Fetch, normalise and select in one call. The usual entry point."""
    rows = fetch_rows(programmes, from_year=from_year, to_year=to_year, terms=terms,
                      refresh=refresh, verbose=verbose)
    rows = filter_rows(rows, programmes)
    round_kategori = NATIONAL_ROUNDS.get(terms[0]) if len(set(terms)) == 1 else None
    df_prog, df_grupp = to_tidy(rows, round_kategori=round_kategori,
                                duplicate_policy=duplicate_policy, verbose=verbose)
    df_prog = add_ratios(select(df_prog, programmes))
    df_grupp = select(df_grupp, programmes)
    return df_prog, df_grupp


# ---------------------------------------------------------------------------
# export and self-check
# ---------------------------------------------------------------------------

def export_tidy(df_prog, df_grupp, slug):
    """Write the tidy frames as CSV so the plots stay reproducible even if the
    undocumented API changes. Small and aggregated, so safe to commit."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    paths = []
    for name, frame in (("program", df_prog), ("urvalsgrupp", df_grupp)):
        path = os.path.join(EXPORT_DIR, f"{slug}_{name}.csv")
        frame.to_csv(path, index=False)
        paths.append(path)
    provenance = os.path.join(EXPORT_DIR, f"{slug}_provenance.json")
    manifest_path = os.path.join(CACHE_DIR, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    with open(provenance, "w", encoding="utf-8") as fh:
        json.dump({
            "exported_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "source": f"{BASE_URL}/searchTotal",
            "n_program_rader": len(df_prog),
            "n_urvalsgrupp_rader": len(df_grupp),
            "cache": manifest,
        }, fh, ensure_ascii=False, indent=1)
    paths.append(provenance)
    print("Skrev " + ", ".join(paths))
    return paths


# Values read off uhr.se by hand; if any of these stop matching, the API or the
# parsing has changed and every plot should be treated as suspect.
GOLDEN_CHECKS = [
    # (period, larosate, program, urval, field, expected)
    ("HT26", "KTH", "CTFYS", "urval2", "n_sokande", 1904),
    ("HT26", "KTH", "CTFYS", "urval2", "n_forstahand", 672),
    ("HT26", "KTH", "CTFYS", "urval2", "n_antagna", 143),
    ("HT26", "KTH", "CTMAT", "urval2", "n_sokande", 1217),
    ("HT26", "KTH", "CFATE", "urval2", "n_antagna", 112),
    ("HT08", "KTH", "CTFYS", "urval2", "n_sokande", 836),
]
GOLDEN_SCORES = [
    ("HT26", "KTH", "CTFYS", "urval2", "BI", 22.40),
    ("HT26", "KTH", "CTFYS", "urval2", "HP", 1.80),
    ("HT26", "KTH", "CTFYS", "urval2", "MAFY", 41.0),
    ("HT26", "KTH", "CTMAT", "urval2", "BI", 22.29),
    ("HT10", "KTH", "CTFYS", "urval2", "BI", 21.50),
]


def validate_tidy(df_prog, df_grupp):
    """Assert the tidy frames still reproduce hand-verified values. Raises on the
    first mismatch; prints a summary when everything passes."""
    failures = []

    def one(frame, **filters):
        mask = pd.Series(True, index=frame.index)
        for key, value in filters.items():
            mask &= frame[key] == value
        return frame[mask]

    for period, lar, prog, urval, field, expected in GOLDEN_CHECKS:
        got = one(df_prog, termin_id=period, larosate_id=lar, program_id=prog,
                  urval=urval)
        if len(got) != 1:
            failures.append(f"{lar} {prog} {period} {urval}: {len(got)} rader, väntade 1")
        elif got.iloc[0][field] != expected:
            failures.append(f"{lar} {prog} {period} {urval} {field}: "
                            f"{got.iloc[0][field]} != {expected}")

    for period, lar, prog, urval, group, expected in GOLDEN_SCORES:
        got = one(df_grupp, termin_id=period, larosate_id=lar, program_id=prog,
                  urval=urval, urvalsgrupp=group)
        if len(got) != 1:
            failures.append(f"{lar} {prog} {period} {urval} {group}: "
                            f"{len(got)} rader, väntade 1")
        elif abs(float(got.iloc[0]["score"]) - expected) > 1e-9:
            failures.append(f"{lar} {prog} {period} {urval} {group}: "
                            f"{got.iloc[0]['score']} != {expected}")

    # structural expectations verified against live data
    ht08 = one(df_grupp, termin_id="HT08", larosate_id="KTH", program_id="CTFYS")
    if not ht08.empty:
        groups = set(ht08["urvalsgrupp"].astype(str))
        if "BI" in groups:
            failures.append("HT08 should use BG, not BI")
        if "BG" not in groups:
            failures.append("HT08 CTFYS should have a BG group")

    ctmat_early = one(df_prog, larosate_id="KTH", program_id="CTMAT")
    ctmat_early = ctmat_early[ctmat_early["ar"] < 2020]
    if not ctmat_early.empty:
        failures.append(f"CTMAT should not exist before HT20, found "
                        f"{sorted(ctmat_early['termin_id'].unique())}")

    ctfys = one(df_prog, larosate_id="KTH", program_id="CTFYS")
    dup = ctfys.groupby(["termin_id", "urval"]).size()
    if (dup > 1).any():
        failures.append(f"CTFYS has duplicate rows after dedup: "
                        f"{dup[dup > 1].to_dict()}")

    if failures:
        raise AssertionError("UHR-kontroll misslyckades:\n  " + "\n  ".join(failures))
    print(f"UHR-kontroll OK: {len(GOLDEN_CHECKS)} antalsvärden, "
          f"{len(GOLDEN_SCORES)} poängvärden, 4 strukturkontroller "
          f"({len(df_prog)} programrader, {len(df_grupp)} urvalsgruppsrader)")
