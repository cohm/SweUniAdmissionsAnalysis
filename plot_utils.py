"""Small helpers shared by analyze_geo.py and analyze_uhr.py.

Only things with more than one caller live here. The save-path rule and the grid
style were duplicated across both scripts, so they are collected in one place to
keep filenames and styling consistent.
"""

import os

import matplotlib.pyplot as plt

# output directory, overridable so a test run can write somewhere else than the
# plots/ directory that holds the real output
PLOTS_DIR = os.environ.get("PLOTS_DIR", "plots")

# whether plt.show() is called by finish_plot(); the drivers set this from --no-show
SHOW_PLOTS = True


def set_show(show):
    """Set the global show behaviour used by finish_plot()."""
    global SHOW_PLOTS
    SHOW_PLOTS = show


def plot_path(kind, name):
    """Build the output path, following the convention '<Kind>_<name>.pdf' with
    spaces replaced by underscores, e.g. plots/CountyMap_CTFYS_sökande.pdf"""
    return ("%s/%s_%s.pdf" % (PLOTS_DIR, kind, name)).replace(' ', '_')


def finish_plot(kind, name, show=None):
    """Tighten layout, save as PDF at 300 dpi, and show unless disabled.

    kind is the plot family (e.g. 'CountyMap'), name the rest of the filename,
    composed by the caller so that multi-part names keep their own separators
    ('GradeDistribution_<group>_<category>', 'GradeCorrelation_<group>_<m1>-<m2>').
    """
    path = plot_path(kind, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    if SHOW_PLOTS if show is None else show:
        plt.show()
    else:
        plt.close()
    return path


def apply_grid(ax):
    """The subtle dashed grid used by the histogram-style plots."""
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.3)


def stats_label(series, label):
    """Legend label with count and summary statistics, e.g.
    'Antagna (n: 143, min: 17.71, medel: 19.34, median: 19.12, max: 28.40)'.

    n is the number of rows, not the number of non-missing values, so it stays a
    headcount of the group being described; the statistics themselves skip NaN.
    """
    n = len(series)
    if series.notna().sum() == 0:
        return f"{label} (n: {n:d}, inga värden)"
    return (f"{label} (n: {n:d}, min: {series.min():.2f}, medel: {series.mean():.2f}, "
            f"median: {series.median():.2f}, max: {series.max():.2f})")
