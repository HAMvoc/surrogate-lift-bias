"""Shared matplotlib style and colour-vision-safe Okabe-Ito palette for diagnostic and analysis plots."""
import matplotlib

FAMILY = {"M2": "C0", "B0": "C1", "B1": "C2", "B2": "C3", "B2_300ep": "C4", "B3": "C5"}
VARIANT = {"dec": "C3", "full": "C2"}
OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
}
CYCLE = [OKABE[c] for c in ("blue", "orange", "green", "vermillion", "purple", "sky", "black")] + [
    "#882255",
    "#999933",
    "#44AA99",
]
TYPE = {"fastback": ("0.55", "--"), "notchback": ("0.3", ":"), "estateback": (OKABE["vermillion"], "-")}


def apply():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
    })
