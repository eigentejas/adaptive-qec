#!/usr/bin/env python3
"""
Theta sweep for Example 2 using the fresh L-only QEC stack.

Default experiment
------------------
* Fixed Hamiltonian seed: h_seed = 0.
* Theta values:
      0.01, 0.05, 0.10, 0.15, ..., 0.50.
* Five QEC cycles.
* Same code-search/certification protocol as run_10_hamiltonians_l_loss.py:
    - unnormalized KL loss L only;
    - numerical exactness L < 1e-24 by default;
    - one cycle-1 exact code V0 chosen without look-ahead;
    - identical V0 for static and adaptive strategies in cycle 1;
    - adaptive continuation codes searched on conditional bath supports;
    - full-bath baseline optimized on full qutrit support;
    - reported performance is worst-case logical fidelity/infidelity.

The sweep is fully resumable. Each theta point has its own setup NPZ,
five-cycle CSV/tree, and log files inside a dedicated output directory.
Completed valid theta points are skipped unless --force is supplied.
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

DEFAULT_THETAS = [
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
]

DEFAULT_OUTPUT_DIR = "weak_local_seed0_theta_sweep_Lonly"
DEFAULT_SETUP_SCRIPT = "weak_local_qec_setup_l_loss.py"
DEFAULT_FIVE_SCRIPT = "weak_local_five_cycle_comparison_l_loss.py"

def prefix_for(seed, theta):
    return f"weak_local_rank2_seed{seed}_theta{theta_tag(theta)}"

def paths_for(outdir, seed, theta):
    prefix = prefix_for(seed, theta)
    return {
        "prefix": prefix,
        "two_npz": outdir / f"{prefix}_data.npz",
        "five_csv": outdir / f"{prefix}_five_cycle_four_strategy.csv",
        "five_tree": outdir / f"{prefix}_five_cycle_adaptive_tree.npz",
    }

def load_npz_safely(path):
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return None


def valid_two_cycle_npz(path, seed, theta, exact_loss_tol):
    if not path.exists():
        return False

    dat = load_npz_safely(path)
    if dat is None:
        return False

    try:
        required = {
            "U",
            "B0",
            "rhoB0",
            "selected_cycle1_V",
            "selected_cycle1_kl_loss",
            "cycle1_n_kl_products",
            "adapted_branch_Vs",
            "adapted_branch_kl_loss",
            "adapted_worst_kl_loss",
            "adapted_all_exact",
            "adapted_branch_ranks",
            "full_bath_V",
            "full_bath_kl_loss",
            "full_n_kl_products",
            "h_seed",
            "theta",
            "exact_loss_tol",
            "full_search_best_adam_kl_loss",
            "full_search_best_direct_kl_loss",
            "full_search_candidate_route",
        }
        if not required.issubset(set(dat.files)):
            return False

        if int(dat["h_seed"]) != int(seed):
            return False
        if not np.isclose(float(dat["theta"]), float(theta), rtol=0.0, atol=1e-14):
            return False
        if not np.isclose(
            float(dat["exact_loss_tol"]),
            float(exact_loss_tol),
            rtol=0.0,
            atol=0.0,
        ):
            return False

        if float(dat["selected_cycle1_kl_loss"]) >= exact_loss_tol:
            return False

        branch_losses = np.asarray(dat["adapted_branch_kl_loss"], dtype=float)
        if branch_losses.size == 0 or np.max(branch_losses) >= exact_loss_tol:
            return False
        if not bool(dat["adapted_all_exact"]):
            return False

        full_loss = float(dat["full_bath_kl_loss"])
        if not np.isfinite(full_loss) or full_loss < exact_loss_tol:
            return False

        routes = set(
            np.asarray(dat["full_search_candidate_route"]).astype(str).tolist()
        )
        if "adam_then_lsq" not in routes or "direct_lsq" not in routes:
            return False

        return True
    except Exception:
        return False
    finally:
        dat.close()


def valid_five_cycle_outputs(
    csv_path,
    tree_path,
    seed,
    theta,
    cycles,
    exact_loss_tol,
):
    if not csv_path.exists() or not tree_path.exists():
        return False

    try:
        df = pd.read_csv(csv_path)
        required_cols = {"strategy", "cycle", "F_wc", "infidelity", "exact"}
        if not required_cols.issubset(df.columns):
            return False

        strategies = {
            "syndrome_adapted",
            "static_reuse",
            "full_bath_optimized",
            "five_qubit_stabilizer",
        }
        if set(df["strategy"].unique()) != strategies:
            return False

        for strategy in strategies:
            got = set(
                df.loc[df["strategy"] == strategy, "cycle"].astype(int).tolist()
            )
            if not set(range(1, cycles + 1)).issubset(got):
                return False

        dat = load_npz_safely(tree_path)
        if dat is None:
            return False
        try:
            required_tree = {
                "h_seed",
                "theta",
                "n_cycles",
                "exact_loss_tol",
                                "adaptive_worst_kl_loss",
                "adaptive_exact",
                "cache_B_padded",
                "cache_Vs",
                "cache_kl_loss",
                "cache_ranks",
                "edge_parent",
                "edge_child",
                "edge_cycle",
                "edge_syndrome",
                "frontier_cycle",
                "frontier_node_ids",
                "frontier_states",
            }
            if not required_tree.issubset(set(dat.files)):
                return False
            if int(dat["h_seed"]) != int(seed):
                return False
            if not np.isclose(
                float(dat["theta"]), float(theta), rtol=0.0, atol=1e-14
            ):
                return False
            if int(dat["n_cycles"]) < cycles:
                return False
            if not np.isclose(
                float(dat["exact_loss_tol"]),
                float(exact_loss_tol),
                rtol=0.0,
                atol=0.0,
            ):
                return False

            worst = np.asarray(dat["adaptive_worst_kl_loss"], dtype=float)
            exact = np.asarray(dat["adaptive_exact"], dtype=bool)
            if worst.size < cycles or np.max(worst[:cycles]) >= exact_loss_tol:
                return False
            if exact.size < cycles or not np.all(exact[:cycles]):
                return False
            return True
        finally:
            dat.close()
    except Exception:
        return False


def stream_subprocess(cmd, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print("\n" + "=" * 88)
    print("RUNNING:")
    print(" ".join(str(x) for x in cmd))
    print(f"log: {log_path}")
    print("=" * 88)

    with log_path.open("a", buffering=1) as log:
        log.write("\n" + "=" * 88 + "\n")
        log.write("COMMAND: " + " ".join(str(x) for x in cmd) + "\n")
        log.write("=" * 88 + "\n")

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()

    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)

def theta_tag(theta):
    return str(float(theta)).replace(".", "p")

def theta_key(theta):
    return f"{float(theta):.12g}"


def write_status(path, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True))
    tmp.replace(path)


def aggregate_fidelity_results(outdir, h_seed, thetas, cycles):
    frames = []

    for theta in thetas:
        csv_path = paths_for(outdir, h_seed, theta)["five_csv"]
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if df.empty:
            continue

        df = df[df["cycle"] <= cycles].copy()
        df.insert(0, "h_seed", int(h_seed))
        df.insert(1, "theta", float(theta))
        frames.append(df)

    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True)
    all_df.sort_values(["theta", "strategy", "cycle"], inplace=True)

    path = outdir / (
        f"weak_local_seed{h_seed}_theta_sweep_five_cycle_all.csv"
    )
    all_df.to_csv(path, index=False)
    return path


def aggregate_setup_results(outdir, h_seed, thetas):
    """Compact certification/search table across completed theta points."""
    rows = []

    for theta in thetas:
        npz_path = paths_for(outdir, h_seed, theta)["two_npz"]
        if not npz_path.exists():
            continue

        try:
            with np.load(npz_path, allow_pickle=False) as dat:
                row = {
                    "h_seed": int(dat["h_seed"]),
                    "theta": float(dat["theta"]),
                    "H_operator_norm": float(dat["H_operator_norm"]),
                    "U_minus_I_operator_norm": float(
                        dat["U_minus_I_operator_norm"]
                    ),
                    "exact_loss_tol": float(dat["exact_loss_tol"]),
                    "cycle1_kl_loss": float(dat["selected_cycle1_kl_loss"]),
                    "adapted_cycle2_worst_kl_loss": float(
                        dat["adapted_worst_kl_loss"]
                    ),
                    "adapted_cycle2_all_exact": bool(dat["adapted_all_exact"]),
                    "full_bath_kl_loss": float(dat["full_bath_kl_loss"]),
                    "cycle1_n_kl_products": int(dat["cycle1_n_kl_products"]),
                    "full_n_kl_products": int(dat["full_n_kl_products"]),
                    "n_cycle2_branches": int(
                        np.asarray(dat["adapted_branch_kl_loss"]).size
                    ),
                }
                rows.append(row)
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("theta")
    path = outdir / (
        f"weak_local_seed{h_seed}_theta_sweep_setup_summary.csv"
    )
    df.to_csv(path, index=False)
    return path


def main(args):
    script_dir = Path(__file__).resolve().parent
    workdir = Path(args.workdir).resolve() if args.workdir else script_dir

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = (workdir / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    setup_script = workdir / args.setup_script
    five_script = workdir / args.five_cycle_script

    for path, label in [
        (setup_script, "QEC search/setup script"),
        (five_script, "five-cycle comparison script"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    thetas = (
        [float(x) for x in args.thetas]
        if args.thetas is not None
        else list(DEFAULT_THETAS)
    )

    # Preserve user order while removing accidental duplicates.
    seen = set()
    thetas = [
        x for x in thetas
        if not (theta_key(x) in seen or seen.add(theta_key(x)))
    ]

    logs_dir = outdir / "production_logs"
    status_path = outdir / (
        f"weak_local_seed{args.h_seed}_theta_sweep_Lonly_status.json"
    )

    status = {
        "h_seed": int(args.h_seed),
        "thetas": thetas,
        "cycles": int(args.cycles),
        "exact_loss_tol": float(args.exact_loss_tol),
        "setup_script": str(setup_script),
        "five_cycle_script": str(five_script),
        "completed": {},
        "failed": {},
    }

    if status_path.exists() and not args.force:
        try:
            old = json.loads(status_path.read_text())
            if (
                int(old.get("h_seed", -1)) == int(args.h_seed)
                and int(old.get("cycles", -1)) == int(args.cycles)
                and float(old.get("exact_loss_tol", np.nan))
                == float(args.exact_loss_tol)
            ):
                status = old
                status["thetas"] = thetas
        except Exception:
            pass

    print("=" * 88)
    print("SEED-0 THETA SWEEP: L-ONLY ADAPTIVE QEC")
    print("=" * 88)
    print(f"workdir       : {workdir}")
    print(f"output dir    : {outdir}")
    print(f"H seed        : {args.h_seed}")
    print(f"theta values  : {thetas}")
    print(f"cycles        : {args.cycles}")
    print(f"exact L tol   : {args.exact_loss_tol:.1e}")
    print(f"python        : {sys.executable}")
    print("\nThis directory is independent of the 10-H run and is safe to run in parallel.")

    for number, theta in enumerate(thetas, start=1):
        key = theta_key(theta)
        paths = paths_for(outdir, args.h_seed, theta)

        print("\n" + "#" * 88)
        print(f"THETA {theta:g} ({number}/{len(thetas)})")
        print("#" * 88)

        try:
            setup_ok = valid_two_cycle_npz(
                paths["two_npz"],
                args.h_seed,
                theta,
                args.exact_loss_tol,
            )

            if setup_ok and not args.force:
                print("Valid L-only setup output already present; skipping setup.")
            else:
                cmd = [
                    sys.executable,
                    str(setup_script),
                    "--h-seed", str(args.h_seed),
                    "--theta", str(theta),
                    "--output-dir", str(outdir),
                    "--exact-loss-tol", str(args.exact_loss_tol),
                ]
                if args.interaction_only:
                    cmd.append("--interaction-only")

                stream_subprocess(
                    cmd,
                    logs_dir / f"theta{theta_tag(theta)}_setup.log",
                    workdir,
                )

                if not valid_two_cycle_npz(
                    paths["two_npz"],
                    args.h_seed,
                    theta,
                    args.exact_loss_tol,
                ):
                    raise RuntimeError(
                        "Setup script returned successfully, but its NPZ "
                        "failed L-only validation."
                    )

            five_ok = valid_five_cycle_outputs(
                paths["five_csv"],
                paths["five_tree"],
                args.h_seed,
                theta,
                args.cycles,
                args.exact_loss_tol,
            )

            if five_ok and not args.force:
                print(f"Valid {args.cycles}-cycle output already present; skipping.")
            else:
                cmd = [
                    sys.executable,
                    str(five_script),
                    "--data", str(paths["two_npz"]),
                    "--cycles", str(args.cycles),
                    "--exact-loss-tol", str(args.exact_loss_tol),
                ]

                stream_subprocess(
                    cmd,
                    logs_dir / f"theta{theta_tag(theta)}_five_cycle.log",
                    workdir,
                )

                if not valid_five_cycle_outputs(
                    paths["five_csv"],
                    paths["five_tree"],
                    args.h_seed,
                    theta,
                    args.cycles,
                    args.exact_loss_tol,
                ):
                    raise RuntimeError(
                        "Five-cycle script returned successfully, but its "
                        "outputs failed L-only validation."
                    )

            status.setdefault("completed", {})[key] = {
                "setup": True,
                "five_cycle": True,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            status.setdefault("failed", {}).pop(key, None)
            print(f"THETA {theta:g}: COMPLETE")

        except Exception as exc:
            status.setdefault("failed", {})[key] = {
                "error": repr(exc),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"\nTHETA {theta:g}: FAILED\n{repr(exc)}")
            write_status(status_path, status)
            aggregate_fidelity_results(
                outdir, args.h_seed, thetas, args.cycles
            )
            aggregate_setup_results(outdir, args.h_seed, thetas)

            if args.stop_on_error:
                raise
            print("Continuing to the next theta value.")

        write_status(status_path, status)

        fidelity_path = aggregate_fidelity_results(
            outdir, args.h_seed, thetas, args.cycles
        )
        setup_path = aggregate_setup_results(
            outdir, args.h_seed, thetas
        )

        if fidelity_path is not None:
            print(f"Updated fidelity aggregate: {fidelity_path}")
        if setup_path is not None:
            print(f"Updated setup summary     : {setup_path}")

    fidelity_path = aggregate_fidelity_results(
        outdir, args.h_seed, thetas, args.cycles
    )
    setup_path = aggregate_setup_results(outdir, args.h_seed, thetas)

    completed = [
        theta
        for theta in thetas
        if theta_key(theta) in status.get("completed", {})
        and status["completed"][theta_key(theta)].get("five_cycle", False)
    ]
    failed = [
        theta
        for theta in thetas
        if theta_key(theta) in status.get("failed", {})
    ]

    print("\n" + "=" * 88)
    print("THETA SWEEP FINISHED")
    print("=" * 88)
    print(f"completed theta: {completed}")
    print(f"failed theta   : {failed}")
    print(f"status file    : {status_path}")
    if fidelity_path is not None:
        print(f"fidelity CSV   : {fidelity_path}")
    if setup_path is not None:
        print(f"setup CSV      : {setup_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--workdir",
        default=None,
        help="Folder containing the L-only experiment scripts. Defaults to this script's folder.",
    )
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--setup-script", default=DEFAULT_SETUP_SCRIPT)
    p.add_argument("--five-cycle-script", default=DEFAULT_FIVE_SCRIPT)
    p.add_argument("--h-seed", type=int, default=0)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--exact-loss-tol", type=float, default=1e-24)
    p.add_argument(
        "--thetas",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional explicit theta list. Default: "
            "0.01, 0.05, 0.10, 0.15, ..., 0.50."
        ),
    )
    p.add_argument("--interaction-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    main(p.parse_args())
