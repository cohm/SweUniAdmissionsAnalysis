"""Plots of aggregated admission statistics from UHR: how applicant numbers,
admitted numbers and merit thresholds have developed over time, and how the same
programme compares between universities.

Complements analyze_geo.py, which works with per-applicant NyA data for a single
admission round. The two answer different questions: NyA knows each applicant's
address, exact age and merit scores for one round; UHR knows the official totals
and thresholds for every round back to HT08 and for every university.

Run `python analyze_uhr.py --help` for the options. Bare `python analyze_uhr.py`
plots the four KTH SCI programmes.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import palette as pal
import uhr_data as u
from plot_utils import apply_grid, finish_plot, set_show

# ---------------------------------------------------------------------------
# configuration: which programmes to plot
# ---------------------------------------------------------------------------
# Peer sets are explicit (institution, programme code) lists rather than a name
# search, because programme names are ambiguous: searching "teknisk fysik" also
# matches master's programmes and later-part admissions. All codes below were
# checked to be present and stable from HT10 to HT26.

KTH_SCI = [
    u.Programme("KTH", "CTFYS", "Teknisk fysik"),
    u.Programme("KTH", "CTMAT", "Teknisk matematik"),
    u.Programme("KTH", "CFATE", "Farkostteknik"),
    u.Programme("KTH", "COPEN", "Öppen ingång"),
]

# Teknisk fysik at every Swedish university that offers it.
# LTU and LIU award "teknisk fysik och elektroteknik"; KAU and MIU are small
# (about 100-220 applicants) and only appear in later years.
PEERS_CTFYS = [
    u.Programme("KTH", "CTFYS", "KTH"),
    u.Programme("CTH", "TKTFY", "Chalmers"),
    u.Programme("LU", "TATFY", "Lund"),
    u.Programme("UU", "TTF2Y", "Uppsala"),
    u.Programme("LIU", "6CYYY", "Linköping (F och EE)"),
    u.Programme("LTU", "TCTEA", "Luleå (F och EE)"),
    u.Programme("UMU", "TYCFT", "Umeå"),
    u.Programme("KAU", "TACTF", "Karlstad"),
    u.Programme("MIU", "TFYSA", "Mittuniversitetet"),
]

PEERS_CTMAT = [
    u.Programme("KTH", "CTMAT", "KTH"),
    u.Programme("CTH", "TKTEM", "Chalmers"),
    u.Programme("LU", "TATPI", "Lund"),
    u.Programme("UU", "TMA2Y", "Uppsala"),
    u.Programme("LIU", "6CTMA", "Linköping"),
    u.Programme("LNU", "CTMAT", "Linnéuniversitetet"),
]

PEERS_COPEN = [
    u.Programme("KTH", "COPEN", "KTH"),
    u.Programme("LTU", "TCÖPA", "Luleå"),
    u.Programme("UMU", "TYCÖP", "Umeå"),
]

# CFATE (farkostteknik) has no counterpart at any other Swedish university, so it
# has no peer set - only its own trend plots.
PROGRAMME_GROUPS = {
    "kth-sci": KTH_SCI,
    "peers-ctfys": PEERS_CTFYS,
    "peers-ctmat": PEERS_CTMAT,
    "peers-copen": PEERS_COPEN,
}

# Merit scores are only comparable within a scale family (see uhr_data), and the
# betyg scale changed when meritpoäng were introduced for admission in HT10: BG
# (HT08-HT09) runs 0-20, BI/BII from HT10 run 0-22.5. Score plots therefore start
# at 2010 by default while count plots use the full history.
COUNTS_FROM_YEAR = 2008
SCORES_FROM_YEAR = 2010

SOURCE_NOTE = "Källa: UHR:s antagningsstatistik"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _year_axis(ax, years):
    """Integer year ticks, with gaps left as gaps."""
    years = sorted({int(y) for y in years})
    if not years:
        return
    step = 1 if len(years) <= 12 else 2
    ax.set_xticks(years[::step])
    ax.set_xlabel("Antagningsår")
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)


def _series(df, value_col):
    """Year-sorted (years, values) with missing years absent rather than zero."""
    ok = df[df[value_col].notna()].sort_values("ar")
    return ok["ar"].astype(int).to_numpy(), ok[value_col].astype(float).to_numpy()


def _preliminary_years(df_prog):
    """Years where UHR reports that the second selection has not been run yet.

    Taken from UHR's own senasteUrval field rather than inferred. Comparing the
    admitted counts of the two rounds does not work: KTH has admitted the same
    headcount in both rounds since 2022, because the number of places is fixed -
    what differs between the rounds is which individuals hold them, and how many
    reserves there are.
    """
    unfinished = df_prog[df_prog["senaste_urval"] == "Urval1"]
    return {int(y) for y in unfinished["ar"].dropna().unique()}


def _mark_all_admitted(ax, years):
    """Mark years where every eligible applicant in a group was admitted, so no
    threshold exists. Drawn as open circles just above the bottom of the axis -
    these are not zeros and not missing data. Call after the data lines so the
    axis limits are already settled."""
    low, high = ax.get_ylim()
    ax.plot(sorted(set(years)), np.full(len(set(years)), low + 0.02 * (high - low)),
            "o", mfc="none", mec=pal.KTH_COLORS["darkgray"], markersize=5,
            linestyle="none")
    ax.set_ylim(low, high)


def _enough_data(df, min_years=3):
    """Whether a scale family has enough real thresholds to be worth a figure."""
    return df.loc[df["score_status"] == "value", "ar"].nunique() >= min_years


def _note(ax, df_prog):
    """Footnote with the source and any preliminary years."""
    prel = _preliminary_years(df_prog)
    text = SOURCE_NOTE
    if prel:
        text += f". {', '.join(str(y) for y in sorted(prel))} preliminärt (urval 2 ej klart)"
    ax.annotate(text, xy=(0, -0.12), xycoords="axes fraction", fontsize=8,
                color=pal.KTH_COLORS["gray"])


# ---------------------------------------------------------------------------
# single-programme trend plots
# ---------------------------------------------------------------------------

def plot_applicants_trend(df_prog, label, slug=None):
    """Applicants, first-choice applicants and admitted over time."""
    one = u.pick_urval(df_prog)
    fig, ax = plt.subplots(figsize=(10, 6))
    series = (("n_sokande", "Sökande", "-o", pal.ALL_APPLICANTS),
              ("n_forstahand", "Förstahandssökande", "--s", pal.KTH_COLORS["teal"]),
              ("n_antagna", "Antagna", ":^", pal.ADMITTED))
    for col, name, style, colour in series:
        years, values = _series(one, col)
        ax.plot(years, values, style, color=colour, markersize=4,
                label=f"{name} ({int(values[-1])} senast)" if len(values) else name)
    ax.set_ylabel("Antal")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Sökande och antagna över tid, {label}")
    _year_axis(ax, one["ar"])
    apply_grid(ax)
    ax.legend()
    _note(ax, df_prog)
    finish_plot("UHRApplicants", slug or label)


def plot_admitted_trend(df_prog, label, slug=None):
    """Admitted and reserves, urval 1 versus urval 2."""
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {("urval1", "n_antagna"): ("-o", pal.KTH_COLORS["skyblue"], "Antagna urval 1"),
              ("urval2", "n_antagna"): ("-s", pal.ADMITTED, "Antagna urval 2"),
              ("urval1", "n_reserver"): ("--o", pal.KTH_COLORS["yellow"], "Reserver urval 1"),
              ("urval2", "n_reserver"): ("--s", pal.KTH_COLORS["darkyellow"], "Reserver urval 2")}
    for (urval, col), (style, colour, name) in styles.items():
        part = df_prog[df_prog["urval"] == urval]
        years, values = _series(part, col)
        if len(years):
            ax.plot(years, values, style, color=colour, label=name, markersize=4)
    ax.set_ylabel("Antal")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Antagna och reserver per urval, {label}")
    _year_axis(ax, df_prog["ar"])
    apply_grid(ax)
    ax.legend()
    _note(ax, df_prog)
    finish_plot("UHRAdmitted", slug or label)


def plot_merit_trend(df_grupp, df_prog, label, scale, urval="urval2", slug=None):
    """Lowest admitted merit score over time, one line per selection group.

    Only groups on the same scale appear in one figure - a lowest admitted score
    of 22.4 (betyg) and 1.8 (högskoleprovet) are not comparable quantities. Years
    where everyone eligible was admitted have no threshold, and are drawn as open
    markers along the bottom rather than omitted or treated as zero.
    """
    part = df_grupp[(df_grupp["score_scale"] == scale)]
    part = part[part["urval"] == urval]
    if part.empty or not _enough_data(part):
        return
    assert part["score_scale"].nunique() == 1, "several scales in one figure"

    fig, ax = plt.subplots(figsize=(10, 6))
    groups = [g for g in u.DEFAULT_GROUPS if g in set(part["urvalsgrupp"].astype(str))]
    groups += sorted(set(part["urvalsgrupp"].astype(str)) - set(groups))
    all_admitted_years = []
    for group in groups:
        rows = part[part["urvalsgrupp"].astype(str) == group]
        years, values = _series(rows[rows["score_status"] == "value"], "score")
        if len(years):
            ax.plot(years, values, "-o", markersize=4, label=group)
        all_admitted_years.extend(
            rows.loc[rows["score_status"] == "all_admitted", "ar"].astype(int))

    # drawn after the lines so that the axis limits are already settled
    plotted_all_admitted = bool(all_admitted_years)
    if plotted_all_admitted:
        _mark_all_admitted(ax, all_admitted_years)

    ax.set_ylabel(u.SCALE_LABELS.get(scale, "Poäng"))
    ax.set_title(f"Lägsta antagningspoäng per urvalsgrupp, {label}\n"
                 f"({u.SCALE_LABELS.get(scale, scale)}, {urval.replace('urval', 'urval ')})")
    _year_axis(ax, part["ar"])
    apply_grid(ax)
    handles, labels = ax.get_legend_handles_labels()
    if plotted_all_admitted:
        handles.append(plt.Line2D([], [], marker="o", mfc="none",
                                  mec=pal.KTH_COLORS["darkgray"], linestyle="none"))
        labels.append("alla behöriga antagna (ingen gräns)")
    ax.legend(handles, labels, title="Urvalsgrupp")
    _note(ax, df_prog)
    finish_plot("UHRMeritTrend", f"{slug or label}_{scale}")


def plot_ratio_trend(df_prog, label, slug=None):
    """Competition for places: applicants per place and reserves per admitted."""
    one = u.pick_urval(df_prog)
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, name, style in (("sokande_per_plats", "Sökande per plats", "-o"),
                             ("forstahand_per_plats", "Förstahandssökande per plats", "--s"),
                             ("reserver_per_antagen", "Reserver per antagen", ":^")):
        years, values = _series(one, col)
        if len(years):
            ax.plot(years, values, style, label=name, markersize=4)
    ax.axhline(1.0, color=pal.REFERENCE_LINE, linewidth=0.8, linestyle="-")
    ax.set_ylabel("Kvot")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Söktryck över tid, {label}")
    _year_axis(ax, one["ar"])
    apply_grid(ax)
    ax.legend()
    _note(ax, df_prog)
    finish_plot("UHRRatios", slug or label)


def plot_gender_share(df_prog, label, slug=None):
    """Share of women among applicants, first-choice applicants and admitted."""
    one = u.pick_urval(df_prog)
    fig, ax = plt.subplots(figsize=(10, 6))
    series = (("andel_kvinnor_sokande", "Bland sökande", "-o", pal.ALL_APPLICANTS),
              ("andel_kvinnor_forstahand", "Bland förstahandssökande", "--s",
               pal.KTH_COLORS["teal"]),
              ("andel_kvinnor_antagna", "Bland antagna", ":^", pal.ADMITTED))
    for col, name, style, colour in series:
        years, values = _series(one, col)
        if len(years):
            ax.plot(years, 100 * values, style, color=colour, label=name, markersize=4)
    ax.axhline(50, color=pal.REFERENCE_LINE, linewidth=0.8)
    ax.set_ylabel("Andel kvinnor [%]")
    ax.set_ylim(0, 100)
    ax.set_title(f"Andel kvinnor över tid, {label}")
    _year_axis(ax, one["ar"])
    apply_grid(ax)
    ax.legend()
    _note(ax, df_prog)
    finish_plot("UHRGenderShare", slug or label)


def plot_age_share(df_prog, label, slug=None):
    """Age composition of applicants and admitted, as stacked shares."""
    one = u.pick_urval(df_prog).sort_values("ar")
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    bands = [("u24", "≤ 24 år", pal.KTH_COLORS["blue"]),
             ("25_34", "25–34 år", pal.KTH_COLORS["teal"]),
             ("35p", "≥ 35 år", pal.KTH_COLORS["yellow"])]
    for ax, stem, heading in ((axes[0], "sokande", "Sökande"),
                              (axes[1], "antagna", "Antagna")):
        total = one[f"n_{stem}"].astype("Float64")
        shares, years = [], one["ar"].astype(int).to_numpy()
        for key, _, _ in bands:
            col = f"n_{stem}_{key}"
            shares.append((100 * one[col].astype("Float64") / total)
                          .astype(float).to_numpy())
        ax.stackplot(years, *shares, labels=[b[1] for b in bands],
                     colors=[b[2] for b in bands], alpha=0.8)
        ax.set_ylabel("Andel [%]")
        ax.set_ylim(0, 100)
        ax.set_title(heading)
        apply_grid(ax)
    axes[0].legend(loc="upper left", ncols=3)
    _year_axis(axes[1], one["ar"])
    fig.suptitle(f"Åldersfördelning över tid, {label}")
    _note(axes[1], df_prog)
    finish_plot("UHRAgeShare", slug or label)


def plot_selection_group_mix(df_grupp, df_prog, label, urval="urval2", slug=None):
    """How the intake divides between selection groups, as stacked counts.

    This is only available from UHR: the NyA export shows which groups an
    applicant has a score in, not which group they were actually admitted through.
    """
    part = df_grupp[(df_grupp["urval"] == urval) &
                    (df_grupp["n_antagna_grupp"].fillna(0) > 0)]
    if part.empty:
        return
    wide = part.pivot_table(index="ar", columns="urvalsgrupp",
                            values="n_antagna_grupp", aggfunc="sum",
                            observed=True).fillna(0)
    wide = wide.loc[:, wide.sum() > 0]
    order = [g for g in u.DEFAULT_GROUPS if g in wide.columns]
    order += [c for c in wide.columns if c not in order]
    wide = wide[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(wide))
    years = wide.index.astype(int).to_numpy()
    for i, group in enumerate(wide.columns):
        values = wide[group].to_numpy(dtype=float)
        ax.bar(years, values, bottom=bottom, label=str(group), width=0.8,
               color=pal.KTH_SEQUENCE[i % len(pal.KTH_SEQUENCE)],
               edgecolor="white", linewidth=0.5)
        bottom += values
    ax.set_ylabel("Antal antagna")
    ax.set_title(f"Antagna per urvalsgrupp, {label} ({urval.replace('urval', 'urval ')})")
    _year_axis(ax, wide.index)
    apply_grid(ax)
    ax.legend(title="Urvalsgrupp", ncols=2, fontsize=8)
    _note(ax, df_prog)
    finish_plot("UHRSelectionMix", slug or label)


# ---------------------------------------------------------------------------
# cross-university comparison
# ---------------------------------------------------------------------------

def _by_institution(df):
    """Yield (larosate_id, label, rows, colour, marker, linestyle) in identity order.

    Grouping on larosate_id rather than the label is what ties a colour to an
    institution: the label is display text and differs between peer sets.
    """
    for larosate_id in pal.sort_by_identity(df["larosate_id"].dropna().unique()):
        rows = df[df["larosate_id"] == larosate_id]
        if rows.empty:
            continue
        colour, marker, linestyle = pal.university_style(larosate_id)
        label = rows["label"].iloc[0] if "label" in rows else larosate_id
        yield larosate_id, label, rows, colour, marker, linestyle



def plot_counts_comparison(df_prog, group_name, metric="n_sokande"):
    """One line per university for a single count metric."""
    titles = {"n_sokande": "Antal sökande",
              "n_forstahand": "Antal förstahandssökande",
              "n_antagna": "Antal antagna",
              "sokande_per_plats": "Sökande per plats",
              "forstahand_per_plats": "Förstahandssökande per plats"}
    one = u.pick_urval(df_prog)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _id, label, part, colour, marker, dash in _by_institution(one):
        years, values = _series(part, metric)
        if len(years):
            ax.plot(years, values, linestyle=dash, marker=marker, color=colour,
                    markersize=4, label=label)
    ax.set_ylabel(titles.get(metric, metric))
    ax.set_ylim(bottom=0)
    ax.set_title(f"{titles.get(metric, metric)} vid olika lärosäten, {group_name}")
    _year_axis(ax, one["ar"])
    apply_grid(ax)
    ax.legend(fontsize=8, ncols=2)
    _note(ax, df_prog)
    finish_plot("UHRCountsCompare", f"{group_name}_{metric}")


def plot_merit_comparison(df_grupp, df_prog, group_name, urvalsgrupp="BI",
                          urval="urval2"):
    """Lowest admitted score in one selection group, one line per university."""
    part = df_grupp[(df_grupp["urvalsgrupp"].astype(str) == urvalsgrupp) &
                    (df_grupp["urval"] == urval)]
    if part.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    all_admitted_years = []
    for _id, label, rows, colour, marker, dash in _by_institution(part):
        years, values = _series(rows[rows["score_status"] == "value"], "score")
        if len(years):
            ax.plot(years, values, linestyle=dash, marker=marker, color=colour,
                    markersize=4, label=label)
        all_admitted_years.extend(
            rows.loc[rows["score_status"] == "all_admitted", "ar"].astype(int))
    if all_admitted_years:
        _mark_all_admitted(ax, all_admitted_years)
    scale = u.score_scale(urvalsgrupp)
    ax.set_ylabel(u.SCALE_LABELS.get(scale, "Poäng"))
    ax.set_title(f"Lägsta antagningspoäng ({urvalsgrupp}) vid olika lärosäten, "
                 f"{group_name}")
    _year_axis(ax, part["ar"])
    apply_grid(ax)
    handles, labels = ax.get_legend_handles_labels()
    if all_admitted_years:
        handles.append(plt.Line2D([], [], marker="o", mfc="none",
                                  mec=pal.KTH_COLORS["darkgray"], linestyle="none"))
        labels.append("alla behöriga antagna")
    ax.legend(handles, labels, fontsize=8, ncols=2)
    _note(ax, df_prog)
    finish_plot("UHRMeritCompare", f"{group_name}_{urvalsgrupp}")


def plot_gender_comparison(df_prog, group_name):
    """Share of women among admitted, one line per university."""
    one = u.pick_urval(df_prog)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _id, label, part, colour, marker, dash in _by_institution(one):
        years, values = _series(part, "andel_kvinnor_antagna")
        if len(years):
            ax.plot(years, 100 * values, linestyle=dash, marker=marker,
                    color=colour, markersize=4, label=label)
    ax.axhline(50, color=pal.REFERENCE_LINE, linewidth=0.8)
    ax.set_ylabel("Andel kvinnor bland antagna [%]")
    ax.set_ylim(0, 100)
    ax.set_title(f"Andel kvinnor bland antagna vid olika lärosäten, {group_name}")
    _year_axis(ax, one["ar"])
    apply_grid(ax)
    ax.legend(fontsize=8, ncols=2)
    _note(ax, df_prog)
    finish_plot("UHRGenderCompare", group_name)


# ---------------------------------------------------------------------------
# summary table
# ---------------------------------------------------------------------------

def summarise(df_prog, df_grupp):
    """Headline table: one row per programme and year, with the BI/HP thresholds."""
    one = u.pick_urval(df_prog)
    out = one[["label", "larosate_id", "program_id", "termin_id", "ar", "urval",
               "n_sokande", "n_forstahand", "n_antagna", "n_reserver",
               "sokande_per_plats", "andel_kvinnor_antagna"]].copy()
    for group in ("BI", "HP"):
        scores = df_grupp[(df_grupp["urvalsgrupp"].astype(str) == group)]
        scores = scores[["larosate_id", "program_id", "termin_id", "urval", "score"]]
        scores = scores.rename(columns={"score": f"lagsta_{group}"})
        out = out.merge(scores, on=["larosate_id", "program_id", "termin_id", "urval"],
                        how="left")
    return out.sort_values(["label", "ar"])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def parse_programmes(spec):
    """'KTH:CTFYS,CTH:TKTFY' -> [Programme('KTH','CTFYS'), Programme('CTH','TKTFY')]"""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        larosate, _, program = item.partition(":")
        if not program:
            raise SystemExit(f"kunde inte tolka '{item}', väntade LÄROSÄTE:PROGRAM")
        out.append(u.Programme(larosate.upper(), program.upper()))
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", default="kth-sci", choices=sorted(PROGRAMME_GROUPS),
                        help="fördefinierad lista av program (standard: kth-sci)")
    parser.add_argument("--programmes", help="egen lista, t.ex. KTH:CTFYS,CTH:TKTFY")
    parser.add_argument("--from-year", type=int, default=COUNTS_FROM_YEAR,
                        help=f"första antagningsår (standard: {COUNTS_FROM_YEAR})")
    parser.add_argument("--to-year", type=int, default=None,
                        help="sista antagningsår (standard: senaste tillgängliga)")
    parser.add_argument("--scores-from-year", type=int, default=SCORES_FROM_YEAR,
                        help="första år för poängdiagram; BG (0-20) och BI (0-22,5) "
                             f"är olika skalor (standard: {SCORES_FROM_YEAR})")
    parser.add_argument("--terms", default="HT",
                        help="terminer, komma-separerat: HT, VT, ST (standard: HT)")
    parser.add_argument("--urval", default="urval2", choices=("urval1", "urval2"),
                        help="vilket urval poängdiagrammen visar (standard: urval2)")
    parser.add_argument("--duplicate-policy", default="largest",
                        choices=("largest", "sum", "keep"),
                        help="hur flera rader för samma program och period slås "
                             "samman (standard: largest)")
    parser.add_argument("--refresh", action="store_true",
                        help="hämta om från uhr.se i stället för att använda cachen")
    parser.add_argument("--export", action="store_true",
                        help="skriv tidy-data som CSV under data/uhr/")
    parser.add_argument("--check", action="store_true",
                        help="kontrollera mot handverifierade värden och avsluta")
    parser.add_argument("--no-show", action="store_true",
                        help="spara diagram utan att öppna fönster")
    parser.add_argument("--quiet", action="store_true", help="mindre utskrifter")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pal.apply_style()
    set_show(not args.no_show)
    verbose = not args.quiet

    if args.programmes:
        programmes = parse_programmes(args.programmes)
        group_name = "urval"
    else:
        programmes = PROGRAMME_GROUPS[args.group]
        group_name = args.group
    terms = tuple(t.strip().upper() for t in args.terms.split(",") if t.strip())

    to_year = args.to_year
    if to_year is None:
        to_year = pd.Timestamp.now().year + (1 if "HT" in terms else 0)

    df_prog, df_grupp = u.load_tidy(programmes, from_year=args.from_year,
                                    to_year=to_year, terms=terms,
                                    refresh=args.refresh,
                                    duplicate_policy=args.duplicate_policy,
                                    verbose=verbose)
    if df_prog.empty:
        raise SystemExit("Ingen UHR-data hittades för de valda programmen.")

    if args.check:
        u.validate_tidy(df_prog, df_grupp)
        return

    summary = summarise(df_prog, df_grupp)
    print("\n=== Sammanfattning per program och år ===")
    with pd.option_context("display.max_rows", 400, "display.width", 200):
        print(summary.to_string(index=False))

    if args.export:
        u.export_tidy(df_prog, df_grupp, group_name)

    scores = df_grupp[df_grupp["ar"] >= args.scores_from_year]

    # per-programme trends
    for programme in programmes:
        # label goes in titles and legends, slug in file names: the same
        # institution is labelled just "KTH" in three different peer sets, so
        # naming files after the label would overwrite the earlier programmes
        label, slug = programme.title, programme.slug
        prog_rows = df_prog[(df_prog["larosate_id"] == programme.larosate_id) &
                            (df_prog["program_id"] == programme.program_id)]
        grupp_rows = scores[(scores["larosate_id"] == programme.larosate_id) &
                            (scores["program_id"] == programme.program_id)]
        if prog_rows.empty:
            print(f"  hoppar över {label}: ingen data")
            continue
        print(f"\n--- diagram för {label} ({slug}) ---")
        plot_applicants_trend(prog_rows, label, slug=slug)
        plot_admitted_trend(prog_rows, label, slug=slug)
        plot_ratio_trend(prog_rows, label, slug=slug)
        plot_gender_share(prog_rows, label, slug=slug)
        plot_age_share(prog_rows, label, slug=slug)
        plot_selection_group_mix(grupp_rows, prog_rows, label,
                                 urval=args.urval, slug=slug)
        for scale in sorted(set(grupp_rows["score_scale"].astype(str))):
            if scale in ("annat", "hogskolepoang"):
                continue  # sentinel-only or credit-based groups
            plot_merit_trend(grupp_rows, prog_rows, label, scale,
                             urval=args.urval, slug=slug)

    # cross-university comparison, only meaningful with more than one institution
    if df_prog["larosate_id"].nunique() > 1:
        print(f"\n--- jämförelse mellan lärosäten ({group_name}) ---")
        for metric in ("n_sokande", "n_forstahand", "n_antagna",
                       "forstahand_per_plats"):
            plot_counts_comparison(df_prog, group_name, metric=metric)
        plot_gender_comparison(df_prog, group_name)
        for group in ("BI", "HP"):
            plot_merit_comparison(scores, df_prog, group_name, urvalsgrupp=group,
                                  urval=args.urval)
    else:
        print("\nEndast ett lärosäte i urvalet - hoppar över jämförelsediagram.")


if __name__ == "__main__":
    main()
