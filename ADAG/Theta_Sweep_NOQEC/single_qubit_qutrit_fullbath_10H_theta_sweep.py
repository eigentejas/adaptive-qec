#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

N_GRID_THETA = 41
N_GRID_PHI = 80
N_REFINE = 6


# ============================================================
# RANDOM LOCAL QUBIT-QUTRIT HAMILTONIAN
# ============================================================

def generate_h(seed):
    """
    Same 6x6 local Hamiltonian construction as in the QEC numerics:
      - complex Gaussian matrix
      - Hermitian part
      - remove total identity component
      - rescale to operator norm 1
    """
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((6, 6))
    Y = rng.standard_normal((6, 6))

    A = X + 1j * Y
    h = 0.5 * (A + A.conj().T)

    h -= np.trace(h) / 6.0 * np.eye(6, dtype=complex)

    op_norm = np.max(np.abs(np.linalg.eigvalsh(h)))
    if op_norm < 1e-14:
        raise RuntimeError("Generated essentially zero Hamiltonian.")

    h /= op_norm
    return h


# ============================================================
# EFFECTIVE QUBIT CHANNEL: FULLY MIXED QUTRIT BATH
# ============================================================

def effective_kraus(U):
    r"""
    Initial bath state:
        rho_B = I_3 / 3.

    Basis ordering: system x bath.

    Kraus operators:
        E_{alpha,k} = sqrt(1/3) <alpha| U |k>,
        alpha = 0,1,2; k = 0,1,2.

    Hence there are 9 Kraus operators.
    """
    U4 = U.reshape(2, 3, 2, 3)

    kraus = []

    for k in range(3):
        for alpha in range(3):
            K = U4[:, alpha, :, k] / np.sqrt(3.0)
            kraus.append(K)

    return kraus


def channel(rho, kraus):
    out = np.zeros((2, 2), dtype=complex)

    for K in kraus:
        out += K @ rho @ K.conj().T

    return out


# ============================================================
# WORST-CASE FIDELITY
# ============================================================

def pure_state(vartheta, phi):
    return np.array(
        [
            np.cos(vartheta / 2.0),
            np.exp(1j * phi) * np.sin(vartheta / 2.0),
        ],
        dtype=complex,
    )


def fidelity_angles(x, kraus):
    vartheta, phi = x

    psi = pure_state(vartheta, phi)
    rho = np.outer(psi, psi.conj())

    rho_out = channel(rho, kraus)

    return float(
        np.real(
            psi.conj() @ rho_out @ psi
        )
    )


def worst_case_fidelity(kraus):
    varthetas = np.linspace(
        0.0,
        np.pi,
        N_GRID_THETA,
    )

    phis = np.linspace(
        0.0,
        2.0 * np.pi,
        N_GRID_PHI,
        endpoint=False,
    )

    candidates = []

    for vartheta in varthetas:
        for phi in phis:
            F = fidelity_angles(
                (vartheta, phi),
                kraus,
            )
            candidates.append(
                (F, vartheta, phi)
            )

    candidates.sort(key=lambda x: x[0])

    best_F = np.inf

    for _, vartheta0, phi0 in candidates[:N_REFINE]:
        result = minimize(
            lambda x: fidelity_angles(x, kraus),
            x0=np.array(
                [vartheta0, phi0]
            ),
            method="L-BFGS-B",
            bounds=[
                (0.0, np.pi),
                (0.0, 2.0 * np.pi),
            ],
            options={
                "ftol": 1e-14,
                "gtol": 1e-12,
                "maxiter": 500,
            },
        )

        best_F = min(
            best_F,
            float(result.fun),
        )

    return best_F


# ============================================================
# MAIN SWEEP
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single-qubit/qutrit no-QEC theta sweep with the qutrit bath "
            "initially maximally mixed, rho_B = I_3/3."
        )
    )

    parser.add_argument(
        "--n-hamiltonians",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--theta-min",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--theta-step",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--output",
        default="single_qubit_qutrit_fullbath_10H_theta_sweep.csv",
    )

    args = parser.parse_args()

    if args.n_hamiltonians < 1:
        raise ValueError("--n-hamiltonians must be positive.")
    if args.theta_step <= 0:
        raise ValueError("--theta-step must be positive.")
    if args.theta_max < args.theta_min:
        raise ValueError("--theta-max must be >= --theta-min.")

    # Rounded construction avoids floating-point labels such as
    # 0.20000000000000004 in the saved CSV.
    n_theta = int(
        np.floor(
            (args.theta_max - args.theta_min) / args.theta_step
            + 1e-12
        )
    ) + 1

    thetas = (
        args.theta_min
        + args.theta_step * np.arange(n_theta)
    )
    thetas = np.round(thetas, 12)

    seeds = list(range(args.n_hamiltonians))

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("=" * 78)
    print("SINGLE-QUBIT / QUTRIT NO-QEC THETA SWEEP")
    print("=" * 78)
    print("bath state       : I_3 / 3")
    print(f"Hamiltonian seeds: {seeds}")
    print(
        f"theta grid       : {thetas[0]:.6g} to "
        f"{thetas[-1]:.6g} in steps of {args.theta_step:.6g}"
    )
    print(f"theta points     : {len(thetas)}")
    print(f"output           : {output}")
    print()

    for seed in seeds:
        print(
            f"Hamiltonian {seed + 1}/{args.n_hamiltonians} "
            f"(seed={seed})"
        )

        h = generate_h(seed)

        eigvals = np.linalg.eigvalsh(h)
        rms_eig = float(
            np.sqrt(np.mean(eigvals**2))
        )
        spectral_width = float(
            eigvals[-1] - eigvals[0]
        )

        for i, theta in enumerate(thetas):
            U = expm(
                -1j * theta * h
            )

            kraus = effective_kraus(U)
            F_wc = worst_case_fidelity(kraus)
            fidelity_loss = 1.0 - F_wc

            rows.append(
                {
                    "seed": seed,
                    "theta": float(theta),
                    "F_wc": F_wc,
                    "fidelity_loss": fidelity_loss,
                    "rms_eigenvalue": rms_eig,
                    "spectral_width": spectral_width,
                    "bath_rank": 3,
                }
            )

            if (
                i == 0
                or (i + 1) % 50 == 0
                or i == len(thetas) - 1
            ):
                print(
                    f"  theta={theta:.3f}  "
                    f"1-F_wc={fidelity_loss:.6e}"
                )

        # Save after every Hamiltonian so a partial long run is preserved.
        pd.DataFrame(rows).to_csv(
            output,
            index=False,
        )
        print(f"  checkpoint saved: {output}")

    df = pd.DataFrame(rows)

    summary = (
        df
        .groupby("theta")["fidelity_loss"]
        .agg(
            mean="mean",
            median="median",
            minimum="min",
            maximum="max",
            std="std",
        )
        .reset_index()
    )

    summary_path = output.with_name(
        output.stem + "_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 78)
    print("SWEEP FINISHED")
    print("=" * 78)
    print(f"saved: {output.resolve()}")
    print(f"saved: {summary_path.resolve()}")

    selected = [0.05, 0.10, 0.20]
    print()
    print("Mean worst-case fidelity loss at selected theta:")

    for theta0 in selected:
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

        print(
            f"  theta={theta0:.2f}: "
            f"mean={row['mean']:.6e}, "
            f"range=[{row['minimum']:.6e}, "
            f"{row['maximum']:.6e}]"
        )


if __name__ == "__main__":
    main()
