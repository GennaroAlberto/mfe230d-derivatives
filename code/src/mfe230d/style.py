"""Matplotlib style shared by the figures in the notes and the notebooks."""
import matplotlib as mpl

COBALT = "#2657FF"
GOLD = "#B47B12"
INK = "#0F172A"
GREY = "#5B6472"
PALETTE = [COBALT, GOLD, "#1BAF7A", "#6250D6", "#E34948"]


def use_style():
    mpl.rcParams.update({
        "figure.figsize": (6.0, 3.4),
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E6E8EE",
        "grid.linewidth": 0.6,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })
