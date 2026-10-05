#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

THETA = 0.1
N_HAMILTONIANS = 10
N_CYCLES = 5

# Adaptive infidelity is numerically zero.  This floor is used
# only for displaying exact points on the logarithmic axis.
PLOT_FLOOR = 1e-16


def theta_tag(theta):
    return str(float(theta)).replace(".", "p")


TAG = theta_tag(THETA)

DATA_DIR = Path(
    f"weak_local_theta{TAG}_10H_Lonly"
)

SUMMARY_CSV = DATA_DIR / (
    f"weak_local_theta{TAG}_10H_five_cycle_summary.csv"
)

OUTDIR = DATA_DIR / "plots"

OUT_PNG = OUTDIR / (
    f"weak_local_theta{TAG}_10H_five_cycle_PRR.png"
)

OUT_PDF = OUTDIR / (
    f"weak_local_theta{TAG}_10H_five_cycle_PRR.pdf"
)


# ============================================================
# PUBLICATION STYLE
# ============================================================

# Roughly APS/PRR single-column width.
FIG_WIDTH = 3.40
FIG_HEIGHT = 2.75

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "mathtext.fontset": "cm",

        # Text
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.9,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.8,

        # Axes
        "axes.linewidth": 0.8,

        # Tick marks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,

        # Vector-output fonts remain editable.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# STRATEGIES
#
# Internal CSV name -> paper legend name / marker
# ============================================================

STRATEGIES = [
    {
        "key": "syndrome_adapted",
        "label": "Syndrome adapted",
        "marker": "o",
    },
    {
        "key": "static_reuse",
        "label": "Static",
        "marker": "s",
    },
    {
        "key": "full_bath_optimized",
        "label": "Static with full bath",
        "marker": "^",
    },
    {
        "key": "five_qubit_stabilizer",
        "label": r"Static $[[5,1,3]]$",
        "marker": "D",
    },
]


# ============================================================
# LOAD AND CHECK DATA
# ============================================================

def load_summary():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Could not find summary CSV:\n{SUMMARY_CSV}"
        )

    df = pd.read_csv(SUMMARY_CSV)

    required = {
        "strategy",
        "cycle",
        "n_hamiltonians",
        "infidelity_mean",
        "infidelity_min",
        "infidelity_max",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Summary CSV is missing required columns:\n"
            f"{sorted(missing)}"
        )

    strategies_found = set(df["strategy"].unique())

    expected = {x["key"] for x in STRATEGIES}

    missing_strategies = expected - strategies_found

    if missing_strategies:
        raise RuntimeError(
            "Missing strategies in summary CSV:\n"
            f"{sorted(missing_strategies)}"
        )

    # Check each strategy has exactly cycles 1,...,5.
    for item in STRATEGIES:
        sdf = (
            df[df["strategy"] == item["key"]]
            .sort_values("cycle")
        )

        got_cycles = sdf["cycle"].astype(int).tolist()

        if got_cycles != list(range(1, N_CYCLES + 1)):
            raise RuntimeError(
                f"{item['key']}: expected cycles "
                f"{list(range(1, N_CYCLES + 1))}, "
                f"found {got_cycles}"
            )

        nH = sdf["n_hamiltonians"].to_numpy(dtype=int)

        if not np.all(nH == N_HAMILTONIANS):
            raise RuntimeError(
                f"{item['key']}: not all points contain "
                f"{N_HAMILTONIANS} Hamiltonians. "
                f"Found n_hamiltonians={nH.tolist()}"
            )

    print(f"Loaded: {SUMMARY_CSV}")
    print(
        f"All four strategies contain "
        f"{N_HAMILTONIANS} Hamiltonians "
        f"for all {N_CYCLES} cycles."
    )

    return df


# ============================================================
# PLOT
# ============================================================

def make_plot(df):
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )

    for item in STRATEGIES:
        sdf = (
            df[df["strategy"] == item["key"]]
            .sort_values("cycle")
        )

        x = sdf["cycle"].to_numpy(dtype=float)

        mean = sdf[
            "infidelity_mean"
        ].to_numpy(dtype=float)

        ymin = sdf[
            "infidelity_min"
        ].to_numpy(dtype=float)

        ymax = sdf[
            "infidelity_max"
        ].to_numpy(dtype=float)

        # Log axes cannot display exact zero.
        mean_plot = np.maximum(mean, PLOT_FLOOR)
        ymin_plot = np.maximum(ymin, PLOT_FLOOR)
        ymax_plot = np.maximum(ymax, PLOT_FLOOR)

        # First plot the mean curve so that its colour can be
        # reused for the corresponding min--max band.
        line, = ax.plot(
            x,
            mean_plot,
            linestyle="--",
            linewidth=1.2,
            marker=item["marker"],
            markersize=4.3,
            markeredgewidth=0.8,
            label=item["label"],
            zorder=3,
        )

        # Observed range over the ten Hamiltonians.
        ax.fill_between(
            x,
            ymin_plot,
            ymax_plot,
            alpha=0.14,
            color=line.get_color(),
            linewidth=0.0,
            zorder=1,
        )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_yscale("log")

    ax.set_xlabel("QEC cycle")
    ax.set_ylabel(
        r"Worst-case fidelity loss $1-F_{\rm wc}$"
    )

    ax.set_xlim(0.8, 5.2)
    ax.set_xticks([1, 2, 3, 4, 5])

    # Light grid only in y direction.
    ax.grid(
        axis="y",
        which="both",
        linewidth=0.45,
        alpha=0.22,
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend = ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=2.0,
        handletextpad=0.55,
    )

    legend.get_frame().set_linewidth(0.7)

    # No plot title: information belongs in caption / LaTeX.

    # Compact single-column spacing.
    fig.subplots_adjust(
        left=0.205,
        right=0.975,
        bottom=0.175,
        top=0.975,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved PDF: {OUT_PDF}")
    print(f"Saved PNG: {OUT_PNG}")


# ============================================================
# USEFUL NUMERICAL SUMMARY
# ============================================================

def print_summary(df):
    print("\nCycle-5 results:")

    for item in STRATEGIES:
        row = df[
            (df["strategy"] == item["key"])
            & (df["cycle"] == N_CYCLES)
        ].iloc[0]

        print(
            f"  {item['label']}: "
            f"mean={row['infidelity_mean']:.6e}, "
            f"range=[{row['infidelity_min']:.6e}, "
            f"{row['infidelity_max']:.6e}]"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_summary()

    make_plot(df)

    print_summary(df)


if __name__ == "__main__":
    main()