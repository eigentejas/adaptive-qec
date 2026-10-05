#!/usr/bin/env python3
"""
Fresh 10-Hamiltonian production driver for Example 2.

Protocol
--------
* Hamiltonian seeds 0,...,9 by default, theta=0.1, five cycles.
* Code search / ranking / exactness use only the unnormalized KL loss

      L = sum_{mu,nu} ||Delta_{mu,nu}||_F^2.

* Numerical exactness: L < 1e-24 by default.
* One cycle-1 exact code V0 is chosen with no look-ahead and is shared by
  static reuse and syndrome-adapted strategies.
* Static reuse keeps that same V0 and its cycle-1 recovery forever.
* Adaptive QEC searches exact continuation codes on the syndrome-conditioned
  bath supports.
* The full-bath baseline is optimized on full qutrit support, but all plotted
  performance is worst-case logical fidelity obtained by running the physical
  protocol from the actual rank-2 initial bath state.

The driver writes each seed incrementally, validates outputs before skipping
on restart, and continuously rebuilds aggregate CSVs.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CORE_MODULE = "adaptive_qec_l_loss.py"
DEFAULT_TWO_SCRIPT = "weak_local_qec_setup_l_loss.py"
DEFAULT_FIVE_SCRIPT = "weak_local_five_cycle_comparison_l_loss.py"
DEFAULT_OUTPUT_DIR = "weak_local_theta0p1_10H_Lonly"


def theta_tag(theta):
    return str(float(theta)).replace(".", "p")


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


def write_status(path, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True))
    tmp.replace(path)


def aggregate_results(outdir, seeds, theta, cycles):
    frames = []
    for seed in seeds:
        p = paths_for(outdir, seed, theta)["five_csv"]
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "h_seed", int(seed))
        df.insert(1, "theta", float(theta))
        frames.append(df)

    if not frames:
        return None, None

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["cycle"] <= cycles].copy()
    all_df.sort_values(["h_seed", "strategy", "cycle"], inplace=True)

    all_path = outdir / (
        f"weak_local_theta{theta_tag(theta)}_{len(seeds)}H_all_five_cycle.csv"
    )
    all_df.to_csv(all_path, index=False)

    summary = (
        all_df.groupby(["strategy", "cycle"], as_index=False)
        .agg(
            n_hamiltonians=("h_seed", "nunique"),
            F_wc_mean=("F_wc", "mean"),
            F_wc_std=("F_wc", "std"),
            F_wc_min=("F_wc", "min"),
            F_wc_median=("F_wc", "median"),
            F_wc_max=("F_wc", "max"),
            infidelity_mean=("infidelity", "mean"),
            infidelity_std=("infidelity", "std"),
            infidelity_min=("infidelity", "min"),
            infidelity_median=("infidelity", "median"),
            infidelity_max=("infidelity", "max"),
        )
    )

    summary_path = outdir / (
        f"weak_local_theta{theta_tag(theta)}_{len(seeds)}H_five_cycle_summary.csv"
    )
    summary.to_csv(summary_path, index=False)
    return all_path, summary_path


def main(args):
    script_dir = Path(__file__).resolve().parent
    workdir = Path(args.workdir).resolve() if args.workdir else script_dir

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = (workdir / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    core = workdir / CORE_MODULE
    two_script = workdir / args.two_cycle_script
    five_script = workdir / args.five_cycle_script
    for path, label in [
        (core, "KL-loss core module"),
        (two_script, "QEC search/setup script"),
        (five_script, "five-cycle comparison script"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    logs_dir = outdir / "production_logs"
    status_path = outdir / (
        f"weak_local_theta{theta_tag(args.theta)}_Lonly_status.json"
    )

    status = {
        "theta": float(args.theta),
        "cycles": int(args.cycles),
        "seeds": seeds,
        "exact_loss_tol": float(args.exact_loss_tol),
        "core_module": str(core),
        "setup_script": str(two_script),
        "five_cycle_script": str(five_script),
        "completed": {},
        "failed": {},
    }

    if status_path.exists() and not args.force:
        try:
            old = json.loads(status_path.read_text())
            if (
                np.isclose(float(old.get("theta", np.nan)), args.theta)
                and int(old.get("cycles", -1)) == args.cycles
                and float(old.get("exact_loss_tol", np.nan)) == args.exact_loss_tol
            ):
                status = old
        except Exception:
            pass

    print("=" * 88)
    print("10-HAMILTONIAN L-ONLY PRODUCTION RUN")
    print("=" * 88)
    print(f"workdir       : {workdir}")
    print(f"output dir    : {outdir}")
    print(f"theta         : {args.theta}")
    print(f"cycles        : {args.cycles}")
    print(f"exact L tol   : {args.exact_loss_tol:.1e}")
    print(f"H seeds       : {seeds}")
    print(f"python        : {sys.executable}")
    print("\nMethod:")
    print("  search/ranking : unnormalized KL loss L only")
    print("  exactness      : L < exact-loss-tol")
    print("  cycle-1 code   : first exact seeded solution, no future look-ahead")
    print("  static/adapt   : identical V0 in cycle 1")
    print("  plotted metric : worst-case logical fidelity / infidelity")

    for number, seed in enumerate(seeds, start=1):
        print("\n" + "#" * 88)
        print(f"HAMILTONIAN SEED {seed} ({number}/{len(seeds)})")
        print("#" * 88)
        paths = paths_for(outdir, seed, args.theta)
        key = str(seed)

        try:
            two_ok = valid_two_cycle_npz(
                paths["two_npz"], seed, args.theta, args.exact_loss_tol
            )
            if two_ok and not args.force:
                print("Valid L-only setup output already present; skipping setup.")
            else:
                cmd = [
                    sys.executable,
                    str(two_script),
                    "--h-seed", str(seed),
                    "--theta", str(args.theta),
                    "--output-dir", str(outdir),
                    "--exact-loss-tol", str(args.exact_loss_tol),
                ]
                if args.interaction_only:
                    cmd.append("--interaction-only")
                stream_subprocess(
                    cmd,
                    logs_dir / f"seed{seed}_setup.log",
                    workdir,
                )
                if not valid_two_cycle_npz(
                    paths["two_npz"], seed, args.theta, args.exact_loss_tol
                ):
                    raise RuntimeError(
                        "Setup script returned successfully, but its NPZ failed L-only validation."
                    )

            five_ok = valid_five_cycle_outputs(
                paths["five_csv"],
                paths["five_tree"],
                seed,
                args.theta,
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
                    logs_dir / f"seed{seed}_five_cycle.log",
                    workdir,
                )
                if not valid_five_cycle_outputs(
                    paths["five_csv"],
                    paths["five_tree"],
                    seed,
                    args.theta,
                    args.cycles,
                    args.exact_loss_tol,
                ):
                    raise RuntimeError(
                        "Five-cycle script returned successfully, but its outputs failed L-only validation."
                    )

            status.setdefault("completed", {})[key] = {
                "setup": True,
                "five_cycle": True,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            status.setdefault("failed", {}).pop(key, None)
            print(f"SEED {seed}: COMPLETE")

        except Exception as exc:
            status.setdefault("failed", {})[key] = {
                "error": repr(exc),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"\nSEED {seed}: FAILED\n{repr(exc)}")
            write_status(status_path, status)
            aggregate_results(outdir, seeds, args.theta, args.cycles)
            if args.stop_on_error:
                raise
            print("Continuing to the next Hamiltonian.")

        write_status(status_path, status)
        all_path, summary_path = aggregate_results(
            outdir, seeds, args.theta, args.cycles
        )
        if all_path is not None:
            print(f"Updated aggregate: {all_path}")
            print(f"Updated summary  : {summary_path}")

    all_path, summary_path = aggregate_results(outdir, seeds, args.theta, args.cycles)
    completed = [
        s for s in seeds
        if str(s) in status.get("completed", {})
        and status["completed"][str(s)].get("five_cycle", False)
    ]
    failed = [s for s in seeds if str(s) in status.get("failed", {})]

    print("\n" + "=" * 88)
    print("BATCH FINISHED")
    print("=" * 88)
    print(f"completed seeds: {completed}")
    print(f"failed seeds   : {failed}")
    print(f"status file    : {status_path}")
    if all_path is not None:
        print(f"aggregate CSV  : {all_path}")
        print(f"summary CSV    : {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--workdir",
        default=None,
        help="Folder containing this experiment bundle. Defaults to this script's folder.",
    )
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--two-cycle-script", default=DEFAULT_TWO_SCRIPT)
    p.add_argument("--five-cycle-script", default=DEFAULT_FIVE_SCRIPT)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--theta", type=float, default=0.1)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--exact-loss-tol", type=float, default=1e-24)
    p.add_argument("--interaction-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    main(p.parse_args())
