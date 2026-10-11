"""Colours for the plots: the KTH visual identity, and one fixed colour per
university so that a given institution looks the same in every figure.

Two palettes, for two different jobs:

* KTH_COLORS - the KTH graphic profile, used for every plot that describes KTH
  data only (analyze_geo.py, and the single-programme trends in analyze_uhr.py).
  Taken from https://github.com/cohm/kth-doc-templates, which in turn follows
  KTH:s grafiska manual.

* UNIVERSITY_STYLE - one (colour, marker, line style) per institution, used by
  the cross-university comparison plots. The colour is that institution's own
  profile colour, so KTH is KTH blue, Chalmers is the purple of their 2026
  identity, Lund is bronze, and so on.

Why marker and line style are part of the identity, not decoration
------------------------------------------------------------------
Four of the ten institutions have a blue profile colour and two have a red one,
so colour alone cannot separate them. Measured in OKLab (ΔE x100, the metric the
dataviz guidance uses), the unavoidable collisions across all pairs in one figure
are KTH vs Umeå at ΔE 7.7 - both are dark navy - and, under deuteranopia, Lund's
bronze vs a red. Chasing the ΔE >= 15 target instead would mean abandoning the
brand colours: an optimiser that keeps only the hue turns KTH into #005EDE and
Uppsala into a pink. Since recognisable colours are the point here, the colours
stay authentic and every series carries a distinct marker and line style as well,
so identity never rests on colour alone. Legend order is chosen so that
neighbouring entries are the well-separated pairs (adjacent-pair check: ΔE 16.3
normal, 8.3 deuteranopia, 16.2 protanopia - all above target).

Sources for each institution's colour are given per entry below; they were read
from each university's own site CSS, logo file or graphic profile in July 2026.
Two are deliberately darkened, because the on-brand value cannot reach the 3:1
contrast a line needs on a white background: Linköping's turquoise (#03DAF7 is
too light to be a line at all) and Linné's yellow (#FFE000 gives 1.3:1).
"""

from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# KTH graphic profile (github.com/cohm/kth-doc-templates)
# ---------------------------------------------------------------------------
KTH_COLORS = {
    # primary
    "blue": "#004791",
    "skyblue": "#6298D2",
    "navy": "#000061",
    "lightblue": "#DEF0FF",
    "sand": "#EBE5E0",
    "white": "#FFFFFF",
    "digitalblue": "#0029ED",
    # green
    "darkgreen": "#0D4A21",
    "green": "#4DA060",
    "lightgreen": "#C7EBBA",
    # teal
    "darkteal": "#1C434C",
    "teal": "#339C9C",
    "lightteal": "#B2E0E0",
    # brick
    "darkbrick": "#78001A",
    "brick": "#E86A58",
    "lightbrick": "#FFCCC4",
    # yellow
    "darkyellow": "#A65900",
    "yellow": "#FFBE00",
    "lightyellow": "#FFF0B0",
    # gray
    "darkgray": "#323232",
    "gray": "#A5A5A5",
    "lightgray": "#E6E6E6",
}

# Series roles used across the KTH-only plots. Keeping these in one place is what
# makes "antagna" the same blue in every figure.
ALL_APPLICANTS = KTH_COLORS["gray"]
ADMITTED = KTH_COLORS["blue"]
ACCEPTED = KTH_COLORS["green"]
RESERVE = KTH_COLORS["yellow"]
REJECTED = KTH_COLORS["darkgray"]
HIGHLIGHT = KTH_COLORS["brick"]
GRID = KTH_COLORS["gray"]
REFERENCE_LINE = KTH_COLORS["darkgray"]

# Ordered colours for a KTH-only plot that needs several series (for example the
# stacked outcome categories). Ordered so neighbours stay distinguishable.
KTH_SEQUENCE = [
    KTH_COLORS["blue"],
    KTH_COLORS["yellow"],
    KTH_COLORS["teal"],
    KTH_COLORS["brick"],
    KTH_COLORS["green"],
    KTH_COLORS["skyblue"],
    KTH_COLORS["darkbrick"],
    KTH_COLORS["darkteal"],
    KTH_COLORS["gray"],
    KTH_COLORS["navy"],
    KTH_COLORS["darkyellow"],
    KTH_COLORS["darkgreen"],
]

# Sequential ramps for the choropleths: one hue, light to dark, per the
# dataviz guidance. Counts use KTH blue; shares use KTH teal, so the two kinds of
# quantity are not mistaken for each other.
CMAP_COUNT = LinearSegmentedColormap.from_list(
    "kth_blue", [KTH_COLORS["lightblue"], KTH_COLORS["blue"], KTH_COLORS["navy"]])
CMAP_FRACTION = LinearSegmentedColormap.from_list(
    "kth_teal", [KTH_COLORS["lightteal"], KTH_COLORS["teal"], KTH_COLORS["darkteal"]])
COLOR_NO_DATA = KTH_COLORS["lightgray"]


# ---------------------------------------------------------------------------
# per-university identity
# ---------------------------------------------------------------------------
# (colour, marker, line style). Order of this dict is the legend order: it is
# chosen so that adjacent entries are the well-separated pairs.
UNIVERSITY_STYLE = {
    # KTH-blå. kth.se --color-primary and KTH:s grafiska manual.
    "KTH": ("#004791", "o", "-"),
    # Chalmers' 2026 identity purple, sampled from the new logo. Their web site
    # still served the previous dark blue when this was written, so the value is
    # taken from the logo artwork rather than from a published hex code.
    "CTH": ("#6550EB", "s", "-"),
    # Lunds universitet brons. lu.se --bs-primary (the graphic manual quotes
    # #9C6114 for print).
    "LU": ("#875E29", "^", "-"),
    # LiU turkos, darkened: the brand cyan #03DAF7 cannot reach 3:1 on white.
    "LIU": ("#00A0AF", "D", "-"),
    # Uppsala universitet rött. uu.se site red; the logo file uses #BD2B30.
    "UU": ("#990000", "v", "-"),
    # Luleå tekniska universitet, blue from ltu.se.
    "LTU": ("#286BBD", "P", "--"),
    # Umeå universitet, navy from umu.se. Dotted and cross-marked because it is
    # the one colour that genuinely collides with KTH blue (ΔE 7.7).
    "UMU": ("#2A4765", "X", ":"),
    # Karlstads universitet, green from kau.se (their logo is yellow, which is
    # unusable as a line colour on white).
    "KAU": ("#1A936F", "*", "-"),
    # Mittuniversitetet, plum from miun.se.
    "MIU": ("#AA4A9B", "d", "-"),
    # Linnéuniversitetet, brand yellow #FFE000 darkened to clear 3:1. Their red
    # would sit ΔE 4.1 from Lund's bronze under deuteranopia.
    "LNU": ("#B58900", "h", "--"),
}

_FALLBACK = (KTH_COLORS["darkgray"], "o", "-")


def university_style(larosate_id):
    """(colour, marker, line style) for an institution code, e.g. 'KTH'."""
    return UNIVERSITY_STYLE.get(larosate_id, _FALLBACK)


def university_color(larosate_id):
    return university_style(larosate_id)[0]


def sort_by_identity(larosate_ids):
    """Order institution codes as in UNIVERSITY_STYLE, unknown ones last.

    Used so that the legend order is the one the colour separation was checked
    for, whichever order the data happens to arrive in.
    """
    order = list(UNIVERSITY_STYLE)
    return sorted(larosate_ids,
                  key=lambda k: (order.index(k) if k in order else len(order), k))


def apply_style():
    """Matplotlib defaults in line with the KTH profile: recessive grid and axes,
    dark gray text rather than pure black."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=KTH_SEQUENCE),
        "axes.edgecolor": KTH_COLORS["gray"],
        "axes.labelcolor": KTH_COLORS["darkgray"],
        "axes.titlecolor": KTH_COLORS["darkgray"],
        "text.color": KTH_COLORS["darkgray"],
        "xtick.color": KTH_COLORS["darkgray"],
        "ytick.color": KTH_COLORS["darkgray"],
        "grid.color": KTH_COLORS["gray"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
