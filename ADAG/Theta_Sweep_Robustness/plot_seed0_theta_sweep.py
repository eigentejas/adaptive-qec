#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ============================================================
# Paths
# ============================================================

DATA = Path(
    "weak_local_seed0_theta_sweep_Lonly/"
    "weak_local_seed0_theta_sweep_five_cycle_all.csv"
)

OUTDIR = Path(
    "weak_local_seed0_theta_sweep_Lonly/plots"
)

PLOT_FLOOR = 1e-16


# ============================================================
# Strategy definitions
# ============================================================

STRATEGIES = [
    ("static_reuse", "Static reuse"),
    ("full_bath_optimized", "Full-bath optimized"),
    ("five_qubit_stabilizer", r"$[[5,1,3]]$ stabilizer"),
]

ADAPTIVE_KEY = "syndrome_adapted"


# ============================================================
# Load + validate
# ============================================================

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Could not find:\n{DATA}")

    df = pd.read_csv(DATA)

    required = {
        "h_seed",
        "theta",
        "strategy",
        "cycle",
        "F_wc",
        "infidelity",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"CSV is missing required columns: {sorted(missing)}"
        )

    thetas = np.sort(df["theta"].unique())

    print(f"Loaded {len(df)} rows")
    print(f"theta values ({len(thetas)}): {thetas.tolist()}")
    print(f"strategies: {sorted(df['strategy'].unique())}")

    expected_thetas = np.array([
        0.01,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ])

    if len(thetas) != len(expected_thetas) or not np.allclose(
        thetas,
        expected_thetas,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Unexpected theta grid.\n"
            f"Found:    {thetas}\n"
            f"Expected: {expected_thetas}"
        )

    # 11 theta x 4 strategies x 5 cycles
    expected_rows = 11 * 4 * 5

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, found {len(df)}."
        )

    return df, thetas


# ============================================================
# Figure 1:
# five-cycle behavior for each theta
# ============================================================

def plot_five_cycle_theta_sweep(df, thetas):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.5, 3.6),
        sharex=True,
        sharey=True,
    )

    norm = Normalize(
        vmin=float(thetas.min()),
        vmax=float(thetas.max()),
    )

    cmap = plt.cm.viridis

    markers = [
        "o", "s", "^", "v", "D", "P",
        "X", "<", ">", "h", "d",
    ]

    for ax, (strategy, title) in zip(axes, STRATEGIES):
        sdf = df[df["strategy"] == strategy]

        for i, theta in enumerate(thetas):
            tdf = (
                sdf[
                    np.isclose(
                        sdf["theta"],
                        theta,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ]
                .sort_values("cycle")
            )

            if len(tdf) != 5:
                raise RuntimeError(
                    f"{strategy}, theta={theta:g}: "
                    f"expected 5 cycles, found {len(tdf)}"
                )

            x = tdf["cycle"].to_numpy(dtype=int)
            y = tdf["infidelity"].to_numpy(dtype=float)

            ax.plot(
                x,
                np.maximum(y, PLOT_FLOOR),
                marker=markers[i],
                markersize=3.5,
                linewidth=1.25,
                color=cmap(norm(theta)),
            )

        ax.set_title(title)
        ax.set_xlabel("QEC cycle")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yscale("log")
        ax.grid(
            True,
            which="both",
            alpha=0.22,
        )

    axes[0].set_ylabel(
        r"Worst-case fidelity loss $1-F_{\rm wc}$"
    )

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    sm.set_array([])

    # Leave dedicated space to the right of the three panels.
    fig.tight_layout(rect=[0.0, 0.0, 0.91, 1.0])

    # Explicit colorbar axis: [left, bottom, width, height]
    cax = fig.add_axes([
        0.925,
        0.17,
        0.012,
        0.72,
    ])

    cbar = fig.colorbar(
        sm,
        cax=cax,
    )

    cbar.set_label(
        r"Interaction strength $\theta$"
    )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = OUTDIR / "theta_sweep_five_cycle.png"
    pdf = OUTDIR / "theta_sweep_five_cycle.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# ============================================================
# Figure 2:
# cycle-5 performance versus theta
# ============================================================

def plot_cycle5_vs_theta(df):
    fig, ax = plt.subplots(
        figsize=(5.6, 4.0)
    )

    markers = ["s", "^", "D"]

    for marker, (strategy, label) in zip(
        markers,
        STRATEGIES,
    ):
        sdf = (
            df[
                (df["strategy"] == strategy)
                & (df["cycle"] == 5)
            ]
            .sort_values("theta")
        )

        x = sdf["theta"].to_numpy(dtype=float)
        y = sdf["infidelity"].to_numpy(dtype=float)

        ax.plot(
            x,
            np.maximum(y, PLOT_FLOOR),
            marker=marker,
            markersize=5,
            linewidth=1.6,
            label=label,
        )

    ax.set_xlabel(
        r"Interaction strength $\theta$"
    )
    ax.set_ylabel(
        r"Cycle-5 worst-case fidelity loss $1-F_{\rm wc}$"
    )

    ax.set_yscale("log")

    ax.grid(
        True,
        which="both",
        alpha=0.22,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    png = OUTDIR / "theta_sweep_cycle5_vs_theta.png"
    pdf = OUTDIR / "theta_sweep_cycle5_vs_theta.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# ============================================================
# Adaptive diagnostics
# ============================================================

def print_adaptive_summary(df):
    adf = df[
        df["strategy"] == ADAPTIVE_KEY
    ].copy()

    if adf.empty:
        print("WARNING: no adaptive rows found.")
        return

    print("\nAdaptive strategy:")
    print(
        "  maximum saved worst-case fidelity loss = "
        f"{adf['infidelity'].max():.3e}"
    )

    print(
        "  Note: exactness is certified by the KL loss, "
        "not by floating-point values of 1-F_wc."
    )


# ============================================================
# Main
# ============================================================

def main():
    df, thetas = load_data()

    plot_five_cycle_theta_sweep(
        df,
        thetas,
    )

    plot_cycle5_vs_theta(df)

    print_adaptive_summary(df)


if __name__ == "__main__":
    main()