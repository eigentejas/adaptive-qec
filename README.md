# MAQEC: syndrome-driven code adaptation for multi-cycle quantum error correction

Numerical code accompanying **“Syndrome-driven code adaptation for multi cycle quantum error correction.”**

This repository implements examples of syndrome-driven code adaptation for repeated quantum error correction in the presence of bath memory. The accompanying paper provides the theoretical framework and interpretation of the results.

The repository contains three types of calculation: a persistent-bath amplitude-damping/amplitude-gain example, numerical code searches for five qubits coupled to a common qutrit bath, and a single-qubit no-QEC reference calculation.

## Requirements

The scripts use Python 3 with:

- NumPy
- SciPy
- PyTorch
- pandas
- Matplotlib

From the repository root, an example setup on macOS/Linux is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy torch pandas matplotlib
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Exact package versions are not currently pinned. The numerical code-search routines use CPU calculations with double and complex-double precision.

## Repository structure

```text
ADAG/
    MergedCode.py
    ADAGWorstCaseFidelity.py
    Saved PDF and PNG figures

Numerical_Code_Search/
    Fixed Theta = 0.1/
        adaptive_qec_l_loss.py
        weak_local_qec_setup_l_loss.py
        weak_local_five_cycle_comparison_l_loss.py
        run_10_hamiltonians_l_loss.py
        plot_10H_l_loss_fidelity_mean.py

    Theta_Sweep_Robustness/
        adaptive_qec_l_loss.py
        weak_local_qec_setup_l_loss.py
        weak_local_five_cycle_comparison_l_loss.py
        run_seed0_theta_sweep_l_loss.py
        plot_seed0_theta_sweep.py

    Theta_Sweep_NOQEC/
        single_qubit_qutrit_fullbath_10H_theta_sweep.py
        replot_fullbath_10H.py
        Saved individual and summary CSVs
        Saved PDF and PNG figures
```

Each experiment folder contains its companion modules. Run the commands from the directories indicated below.

## Running and resuming calculations

For each independent QEC calculation, use a fresh copy of the repository containing no previously generated QEC output directories. This is particularly important after changing source code, Hamiltonian models, search settings or numerical tolerances: the current restart checks do not compare every setting.

Run the production drivers with `--stop-on-error`, as shown below. Generate plots only after every requested Hamiltonian seed or theta point has completed successfully. Aggregate CSVs from an incomplete or failed run should not be treated as final results.

An interrupted calculation may be resumed in its existing folder provided that the source files and all settings remain unchanged. For a changed calculation, start in a fresh folder without copying previous generated outputs into it.

The numerical code searches can be computationally expensive.

## 1. Persistent-bath AD/AG example

From the repository root:

```bash
cd ADAG
python ADAGWorstCaseFidelity.py
cd ..
```

This compares syndrome-adapted four-qubit recovery, static four-qubit recovery, and the standard `[[5,1,3]]` stabilizer code.

The default calculation produces:

- Two-cycle worst-case fidelity loss conditioned on exactly one physical jump in cycle 1.
- Unconditioned two-cycle worst-case fidelity as a function of noise strength.
- Unconditioned worst-case fidelity over 50 cycles at `p = 0.01`.

PDF and PNG figures are saved in `ADAG/` with the following stems:

```text
ADAG_two_cycle_worst_case_fidelity_loss
ADAG_two_cycle_unconditioned_fidelity
ADAG_worst_case_fidelity_vs_cycles
```

The underlying model and entanglement-fidelity calculations are implemented in `MergedCode.py`. To display its entanglement-fidelity plots:

```bash
cd ADAG
python MergedCode.py
cd ..
```

This script displays interactive figures rather than automatically saving them.

The bath is tracked as a persistent classical configuration. Physical jumps flip the corresponding bath bits, while the controller maintains a separate history inferred from syndrome outcomes. For the four-qubit completion outcome `R5`, QEC stops on that trajectory and subsequent physical noise evolution continues.

## 2. Numerical code search at fixed interaction strength

From the repository root:

```bash
cd "Numerical_Code_Search/Fixed Theta = 0.1"
python run_10_hamiltonians_l_loss.py --stop-on-error
python plot_10H_l_loss_fidelity_mean.py
cd ../..
```

The default experiment uses Hamiltonian seeds `0` through `9`, interaction strength `theta = 0.1`, and five QEC cycles.

Five physical qubits interact with a common qutrit bath. The initial bath state is `diag(1/2, 1/2, 0)`, and the joint system–bath state is retained between cycles.

The calculation compares four strategies:

1. Syndrome-adapted codes and recoveries.
2. Static reuse of the exact cycle-1 code and recovery.
3. A code optimized using full qutrit bath support, with fixed transpose recovery.
4. The standard `[[5,1,3]]` stabilizer code with Pauli recovery.

The static-reuse and syndrome-adapted strategies share the same initial code. This code is selected by the seeded cycle-1 search without using future-cycle performance.

The workflow is:

```text
Search the initial and continuation codes
    → save setup NPZ
    → evaluate repeated-cycle performance
    → save strategy CSV and adaptive tree
    → aggregate Hamiltonians
    → generate plots
```

Default outputs are written under:

```text
Numerical_Code_Search/Fixed Theta = 0.1/
    weak_local_theta0p1_10H_Lonly/
```

This directory contains per-Hamiltonian setup NPZs, performance CSVs, adaptive support/code trees, aggregate CSVs, production logs and a status JSON. The plotting script saves PDF and PNG figures in its `plots/` subdirectory.

The plot shows mean worst-case fidelity loss, with shaded minimum–maximum ranges across the ten Hamiltonians. It expects the default ten-Hamiltonian, five-cycle dataset.

## 3. Robustness sweep over interaction strength

From the repository root:

```bash
cd Numerical_Code_Search/Theta_Sweep_Robustness
python run_seed0_theta_sweep_l_loss.py --stop-on-error
python plot_seed0_theta_sweep.py
cd ../..
```

This uses the same four-strategy comparison, fixing Hamiltonian seed `0` and varying interaction strength over:

```text
0.01, 0.05, 0.10, 0.15, 0.20, 0.25,
0.30, 0.35, 0.40, 0.45, 0.50
```

The default calculation uses five QEC cycles. Outputs are written under:

```text
Numerical_Code_Search/Theta_Sweep_Robustness/
    weak_local_seed0_theta_sweep_Lonly/
```

The plotting script expects the complete default theta grid and five cycles. It plots the three static baselines and prints an adaptive-performance diagnostic.

## 4. Single-qubit no-QEC reference

This calculation uses one qubit coupled to a qutrit bath initially in the maximally mixed state `I_3/3`, for one interaction interval. It provides a separate noise-strength reference, with different system size and bath initialization from the repeated-cycle QEC calculations.

The individual and summary CSVs are included. To regenerate figures from the supplied data:

```bash
cd Numerical_Code_Search/Theta_Sweep_NOQEC
python replot_fullbath_10H.py
cd ../..
```

To recompute the data and then regenerate the figures:

```bash
cd Numerical_Code_Search/Theta_Sweep_NOQEC
python single_qubit_qutrit_fullbath_10H_theta_sweep.py
python replot_fullbath_10H.py
cd ../..
```

Defaults are Hamiltonian seeds `0` through `9`, with theta from `0.001` to `0.5` in steps of `0.001`.

Recomputing writes over the default CSV files. The plotting script saves linear and logarithmic figures as PDF and PNG files.

## Parameters and outputs

Scripts with command-line interfaces describe their available options through:

```bash
python SCRIPT_NAME.py --help
```

Replace `SCRIPT_NAME.py` with the relevant script name.

The production drivers support custom output directories through `--output-dir`. The plotting scripts use paths defined near the top of their files; update those paths if using a custom output location.

Code searches use the summed, unnormalized Knill–Laflamme residual loss. Reported performance in the numerical-search workflows is worst-case pure-logical-state fidelity. `MergedCode.py` evaluates entanglement fidelity, which is a separate metric.

Some logarithmic figures display zero or very small losses at a plotting floor of `1e-16`.

The repository includes saved AD/AG figures and the single-qubit no-QEC data and figures. The numerical QEC setup NPZs, adaptive trees and performance CSVs are generated by the workflows above.

## Citation

If you use this code, please cite the accompanying paper and the version-specific Zenodo DOI of the software release used for your calculations.

Paper: **“Syndrome-driven code adaptation for multi cycle quantum error correction.”**

Repository: [eigentejas/adaptive-qec](https://github.com/eigentejas/adaptive-qec)
Software DOI: https://doi.org/10.5281/zenodo.24579444
