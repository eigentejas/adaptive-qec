#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replot the 10-Hamiltonian single-qubit/qutrit no-QEC sweep "
            "for a maximally mixed full-qutrit bath."
        )
    )

    parser.add_argument(
        "--input",
        default="single_qubit_qutrit_fullbath_10H_theta_sweep.csv",
    )
    parser.add_argument(
        "--output-stem",
        default="single_qubit_qutrit_fullbath_10H_theta_sweep",
    )

    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find sweep CSV:\n{path.resolve()}"
        )

    df = pd.read_csv(path)
    required = {
        "seed",
        "theta",
        "F_wc",
        "fidelity_loss",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Input CSV is missing columns: {sorted(missing)}"
        )

    N_H = df["seed"].nunique()

    print(f"Found {N_H} Hamiltonians")

    summary = (
        df.groupby("theta")["fidelity_loss"]
        .agg(
            mean="mean",
            median="median",
            minimum="min",
            maximum="max",
            std="std",
        )
        .reset_index()
    )

    out = Path(args.output_stem)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # LINEAR PLOT
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.8)
    )

    for seed in sorted(
        df["seed"].unique()
    ):
        x = df[
            df["seed"] == seed
        ]

        ax.plot(
            x["theta"],
            x["fidelity_loss"],
            linewidth=0.6,
            alpha=0.08,
        )

    ax.fill_between(
        summary["theta"],
        summary["minimum"],
        summary["maximum"],
        alpha=0.20,
        label=f"Min--max over {N_H} Hamiltonians",
    )

    ax.plot(
        summary["theta"],
        summary["mean"],
        linewidth=2.2,
        label=f"Mean over {N_H} Hamiltonians",
    )

    # Annotate selected theta values, matching replot_test.py.
    selected_thetas = [
        0.05,
        0.10,
        0.20,
    ]

    offsets = {
        0.05: (0, 12),
        0.10: (0, 12),
        0.20: (0, 12),
    }

    for theta0 in selected_thetas:
        match = summary[
            np.isclose(
                summary["theta"],
                theta0,
                atol=1e-12,
                rtol=0.0,
            )
        ]

        if match.empty:
            continue

        row = match.iloc[0]
        y0 = row["mean"]

        ax.scatter(
            theta0,
            y0,
            s=35,
            zorder=5,
        )

        ax.annotate(
            rf"{y0:.3e}",
            xy=(theta0, y0),
            xytext=offsets[theta0],
            textcoords="offset points",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(
        r"Worst-case fidelity loss $1-F_{\rm wc}$"
    )

    ax.set_xlim(
        df["theta"].min(),
        df["theta"].max(),
    )
    ax.set_ylim(
        bottom=0,
    )

    ax.grid(
        alpha=0.2,
    )
    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    linear_png = out.with_suffix(".png")
    linear_pdf = out.with_suffix(".pdf")

    fig.savefig(
        linear_png,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        linear_pdf,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ============================================================
    # LOG-LOG PLOT
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.8)
    )

    ax.fill_between(
        summary["theta"],
        summary["minimum"],
        summary["maximum"],
        alpha=0.20,
    )

    ax.loglog(
        summary["theta"],
        summary["mean"],
        linewidth=2.2,
        label=f"Mean over {N_H} Hamiltonians",
    )

    # theta^2 reference, anchored to the first mean point.
    c = (
        summary["mean"].iloc[0]
        / summary["theta"].iloc[0] ** 2
    )

    reference = (
        c * summary["theta"] ** 2
    )

    ax.loglog(
        summary["theta"],
        reference,
        "--",
        alpha=0.7,
        label=r"$\propto\theta^2$",
    )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(
        r"Worst-case fidelity loss $1-F_{\rm wc}$"
    )

    ax.grid(
        which="both",
        alpha=0.2,
    )
    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    loglog_png = out.with_name(
        out.name + "_loglog"
    ).with_suffix(".png")
    loglog_pdf = out.with_name(
        out.name + "_loglog"
    ).with_suffix(".pdf")

    fig.savefig(
        loglog_png,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        loglog_pdf,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"saved: {linear_png.resolve()}")
    print(f"saved: {linear_pdf.resolve()}")
    print(f"saved: {loglog_png.resolve()}")
    print(f"saved: {loglog_pdf.resolve()}")


if __name__ == "__main__":
    main()
