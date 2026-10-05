import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from matplotlib.ticker import MaxNLocator

import MergedCode as base


# ============================================================
# PUBLICATION PLOT STYLE
# ============================================================

# The p-sweep is slightly wider because it has a log axis and
# benefits from more horizontal room.
FIG_WIDTH_P = 3.70

# The other two plots are slightly narrower.
FIG_WIDTH_OTHER = 3.30

# Common height so the eventual panels can be aligned cleanly.
FIG_HEIGHT = 2.55


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",

        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.0,

        "axes.linewidth": 0.8,

        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# Same strategy colours / markers as Figure 2.
STYLE_ADAPT = {
    "color": "C0",
    "marker": "o",
    "label": "Syndrome adapted",
}

STYLE_STATIC = {
    "color": "C1",
    "marker": "s",
    "label": "Static",
}

STYLE_FIVE = {
    "color": "C3",
    "marker": "D",
    "label": r"Static $[[5,1,3]]$",
}


# ============================================================
# LOGICAL PROBE BASIS
#
# Any logical input can be written as
#
#   rho(r) = I/2 + r_x X/2 + r_y Y/2 + r_z Z/2.
#
# We propagate these four operators through the full protocol.
# This reconstructs the fidelity for every pure logical input
# without propagating thousands of Bloch-sphere states.
# ============================================================

PROBES = np.stack(
    [
        base.I2 / 2,
        base.X2 / 2,
        base.Y2 / 2,
        base.Z2 / 2,
    ]
)

PAULIS = [
    base.X2,
    base.Y2,
    base.Z2,
]


def sandwich_batch(A, rhos):
    """
    Apply

        rho -> A rho A^\dagger

    to a stack of matrices rhos[k].
    """
    return np.einsum(
        "ab,kbc,dc->kad",
        A,
        rhos,
        A.conj(),
        optimize=True,
    )


# ============================================================
# WORST-CASE FIDELITY FROM THE PROPAGATED PROBES
# ============================================================

def channel_coefficients(
    logical_outputs,
    total_traces,
):
    """
    logical_outputs[k] is the effective logical output of

        I/2, X/2, Y/2, Z/2.

    For the four-qubit R5 failure sector, the physical output
    has already been projected onto the original code FOUR_V0.
    Thus the overlap computed below is exactly the overlap with
    the desired encoded logical input.

    total_traces[k] is the trace of the complete physical output.

    For an unconditioned protocol:
        total_trace(r) = 1.

    For the conditioned protocol:
        total_trace(r)
    is the input-dependent probability of exactly one jump in
    cycle 1.
    """

    O0 = logical_outputs[0]
    Oxyz = logical_outputs[1:]

    alpha = float(
        np.real(np.trace(O0))
    )

    a = np.array(
        [
            np.real(np.trace(O))
            for O in Oxyz
        ]
    )

    t = np.array(
        [
            np.real(np.trace(P @ O0))
            for P in PAULIS
        ]
    )

    M = np.array(
        [
            [
                np.real(
                    np.trace(P @ Oxyz[j])
                )
                for j in range(3)
            ]
            for P in PAULIS
        ]
    )

    # Full physical trace:
    #
    # d(r) = d0 + d.r
    #
    # This is identically 1 for an unconditioned CPTP map,
    # but is the probability of the conditioning event for
    # a conditioned instrument.

    d0 = float(total_traces[0])

    d = np.asarray(
        total_traces[1:],
        dtype=float,
    )

    return {
        "alpha": alpha,
        "a": a,
        "t": t,
        "M": M,
        "d0": d0,
        "d": d,
    }


def fidelity_from_bloch(
    r,
    coeffs,
):
    """
    Fidelity for a pure logical input with Bloch vector r.

    For an unconditioned protocol:

        F(r) = <psi_r | rho_out(r) | psi_r>.

    For a conditioned protocol we divide by the probability of
    the conditioning event for that same input.
    """

    r = np.asarray(
        r,
        dtype=float,
    )

    alpha = coeffs["alpha"]
    a = coeffs["a"]
    t = coeffs["t"]
    M = coeffs["M"]
    d0 = coeffs["d0"]
    d = coeffs["d"]

    # Only the symmetric part contributes to r^T M r.
    Q = 0.5 * (
        M + M.T
    )

    numerator = 0.5 * (
        alpha
        + (a + t) @ r
        + r @ Q @ r
    )

    denominator = (
        d0 + d @ r
    )

    if denominator <= 0:
        return np.inf

    return float(
        numerator / denominator
    )


def worst_case_fidelity(
    coeffs,
    n_theta=91,
    n_phi=180,
    n_starts=12,
):
    """
    Minimize the pure-state fidelity over the Bloch sphere.

    First use a deterministic spherical grid, then refine the
    best n_starts points using constrained SLSQP.
    """

    theta = np.linspace(
        0,
        np.pi,
        n_theta,
    )

    phi = np.linspace(
        0,
        2 * np.pi,
        n_phi,
        endpoint=False,
    )

    st = np.sin(theta)[:, None]
    ct = np.cos(theta)[:, None]

    cp = np.cos(phi)[None, :]
    sp = np.sin(phi)[None, :]

    rx = st * cp
    ry = st * sp

    rx = np.broadcast_to(
        rx,
        (n_theta, n_phi),
    )

    ry = np.broadcast_to(
        ry,
        (n_theta, n_phi),
    )

    rz = np.broadcast_to(
        ct,
        (n_theta, n_phi),
    )

    R = np.stack(
        [rx, ry, rz],
        axis=-1,
    ).reshape(-1, 3)

    alpha = coeffs["alpha"]
    a = coeffs["a"]
    t = coeffs["t"]
    M = coeffs["M"]
    d0 = coeffs["d0"]
    d = coeffs["d"]

    Q = 0.5 * (
        M + M.T
    )

    numerators = 0.5 * (
        alpha
        + R @ (a + t)
        + np.einsum(
            "ni,ij,nj->n",
            R,
            Q,
            R,
        )
    )

    denominators = (
        d0 + R @ d
    )

    values = (
        numerators
        / denominators
    )

    best_indices = np.argsort(
        values
    )[:n_starts]

    def objective(r):
        return fidelity_from_bloch(
            r,
            coeffs,
        )

    constraint = {
        "type": "eq",
        "fun": lambda r:
            np.dot(r, r) - 1.0,
    }

    best_f = float(
        values[best_indices[0]]
    )

    best_r = R[
        best_indices[0]
    ].copy()

    for idx in best_indices:

        result = minimize(
            objective,
            R[idx],
            method="SLSQP",
            constraints=constraint,
            options={
                "ftol": 1e-14,
                "maxiter": 1000,
            },
        )

        r = np.asarray(
            result.x,
            dtype=float,
        )

        nr = np.linalg.norm(r)

        if nr == 0:
            continue

        r /= nr

        f = objective(r)

        if f < best_f:
            best_f = f
            best_r = r

    # Remove tiny numerical excursions.
    best_f = float(
        np.clip(
            best_f,
            0.0,
            1.0,
        )
    )

    return best_f, best_r


# ============================================================
# FIVE-QUBIT STABILIZER: LOGICAL-STATE PROPAGATION
# ============================================================

def five_transition_cache_wc(
    p,
    jump_weight=None,
):
    cache = {}

    for bath in base.FIVE_BATHS:

        transitions = []

        for jumps in base.FIVE_BATHS:

            if (
                jump_weight is not None
                and sum(jumps) != jump_weight
            ):
                continue

            E, next_bath = (
                base.system_jump_operator(
                    bath,
                    jumps,
                    p,
                )
            )

            for R in base.FIVE_R:

                L = (
                    base.FIVE_V.conj().T
                    @ R
                    @ E
                    @ base.FIVE_V
                )

                transitions.append(
                    (
                        next_bath,
                        L,
                    )
                )

        cache[bath] = transitions

    return cache


def five_qubit_worst_case_history(
    p,
    n_max,
    conditioned_first_cycle=False,
):
    full_transitions = (
        five_transition_cache_wc(
            p,
            jump_weight=None,
        )
    )

    if conditioned_first_cycle:

        first_transitions = (
            five_transition_cache_wc(
                p,
                jump_weight=1,
            )
        )

    else:
        first_transitions = None

    ZERO5 = (
        0, 0, 0, 0, 0
    )

    sectors = {
        ZERO5:
        PROBES.copy()
    }

    fidelities = []
    worst_vectors = []

    for cycle in range(n_max):

        transitions = (
            first_transitions
            if (
                conditioned_first_cycle
                and cycle == 0
            )
            else full_transitions
        )

        new_sectors = {}

        for bath, rhos in sectors.items():

            for (
                next_bath,
                L,
            ) in transitions[bath]:

                branch = (
                    sandwich_batch(
                        L,
                        rhos,
                    )
                )

                if next_bath in new_sectors:
                    new_sectors[
                        next_bath
                    ] += branch

                else:
                    new_sectors[
                        next_bath
                    ] = branch

        sectors = new_sectors

        logical_outputs = sum(
            sectors.values()
        )

        total_traces = np.zeros(
            4,
            dtype=float,
        )

        for rhos in sectors.values():

            total_traces += np.real(
                np.trace(
                    rhos,
                    axis1=1,
                    axis2=2,
                )
            )

        coeffs = channel_coefficients(
            logical_outputs,
            total_traces,
        )

        f_wc, r_wc = (
            worst_case_fidelity(
                coeffs
            )
        )

        fidelities.append(
            f_wc
        )

        worst_vectors.append(
            r_wc
        )

    return (
        np.asarray(fidelities),
        np.asarray(worst_vectors),
    )


# ============================================================
# FOUR-QUBIT ADAPTIVE / FIXED TRANSITIONS
# ============================================================

def four_transition_cache_wc(
    p,
    adaptive,
    jump_weight=None,
):
    arrs = (
        base.FOUR_ARRS
        if adaptive
        else [base.ZERO4]
    )

    cache = {}

    for arr in arrs:

        for bath in base.FOUR_ARRS:

            if adaptive:

                V_in = (
                    base.FOUR_V[arr]
                )

                recs = (
                    base.FOUR_R_ADAPT[arr]
                )

            else:

                V_in = base.FOUR_V0
                recs = base.FOUR_R_FIXED

            success = []
            failure = []

            for jumps in base.FOUR_ARRS:

                if (
                    jump_weight is not None
                    and sum(jumps)
                    != jump_weight
                ):
                    continue

                E, next_bath = (
                    base.system_jump_operator(
                        bath,
                        jumps,
                        p,
                    )
                )

                # R0,...,R4
                for i in range(5):

                    if adaptive:

                        next_arr = list(
                            arr
                        )

                        if i > 0:
                            next_arr[
                                i - 1
                            ] ^= 1

                        next_arr = tuple(
                            next_arr
                        )

                        V_out = (
                            base.FOUR_V[
                                next_arr
                            ]
                        )

                    else:

                        next_arr = (
                            base.ZERO4
                        )

                        V_out = (
                            base.FOUR_V0
                        )

                    L = (
                        V_out.conj().T
                        @ recs[i]
                        @ E
                        @ V_in
                    )

                    success.append(
                        (
                            next_arr,
                            next_bath,
                            L,
                        )
                    )

                # R5 failure branch remains physical.
                Fsys = (
                    recs[5]
                    @ E
                    @ V_in
                )

                failure.append(
                    (
                        next_bath,
                        Fsys,
                    )
                )

            cache[
                (arr, bath)
            ] = (
                success,
                failure,
            )

    return cache


def propagate_failed_four_wc(
    failed,
    p,
):
    new_failed = {}

    for bath, rhos in failed.items():

        for jumps in base.FOUR_ARRS:

            E, next_bath = (
                base.system_jump_operator(
                    bath,
                    jumps,
                    p,
                )
            )

            branch = sandwich_batch(
                E,
                rhos,
            )

            if next_bath in new_failed:
                new_failed[
                    next_bath
                ] += branch

            else:
                new_failed[
                    next_bath
                ] = branch

    return new_failed


def four_qubit_worst_case_history(
    p,
    n_max,
    adaptive,
    conditioned_first_cycle=False,
):
    full_transitions = (
        four_transition_cache_wc(
            p,
            adaptive,
            jump_weight=None,
        )
    )

    if conditioned_first_cycle:

        first_transitions = (
            four_transition_cache_wc(
                p,
                adaptive,
                jump_weight=1,
            )
        )

    else:
        first_transitions = None

    # Successful branches:
    #
    # (inferred bath, actual bath)
    #     -> four logical probe operators.

    sectors = {
        (
            base.ZERO4,
            base.ZERO4,
        ):
        PROBES.copy()
    }

    # R5 branches:
    #
    # actual bath
    #     -> four physical 16x16 probe outputs.

    failed = {}

    fidelities = []
    worst_vectors = []

    for cycle in range(n_max):

        # Previously failed states evolve physically.
        if failed:

            failed = (
                propagate_failed_four_wc(
                    failed,
                    p,
                )
            )

        new_sectors = {}

        new_failed = {
            key: value.copy()
            for key, value
            in failed.items()
        }

        transitions = (
            first_transitions
            if (
                conditioned_first_cycle
                and cycle == 0
            )
            else full_transitions
        )

        for (
            arr,
            bath,
        ), rhos in sectors.items():

            success, failure = (
                transitions[
                    (arr, bath)
                ]
            )

            # Successful syndrome branches.
            for (
                next_arr,
                next_bath,
                L,
            ) in success:

                branch = (
                    sandwich_batch(
                        L,
                        rhos,
                    )
                )

                key = (
                    next_arr,
                    next_bath,
                )

                if key in new_sectors:
                    new_sectors[
                        key
                    ] += branch

                else:
                    new_sectors[
                        key
                    ] = branch

            # R5 failure branches.
            for (
                next_bath,
                Fsys,
            ) in failure:

                branch = (
                    sandwich_batch(
                        Fsys,
                        rhos,
                    )
                )

                if next_bath in new_failed:
                    new_failed[
                        next_bath
                    ] += branch

                else:
                    new_failed[
                        next_bath
                    ] = branch

        sectors = new_sectors
        failed = new_failed

        # ----------------------------------------------------
        # Effective logical output used for fidelity.
        #
        # Successful sectors are already represented logically.
        #
        # For failed physical sectors we use
        #
        #     V0^\dagger rho_phys V0,
        #
        # because
        #
        # <psi_L|rho_phys|psi_L>
        # =
        # <psi|V0^\dagger rho_phys V0|psi>.
        # ----------------------------------------------------

        logical_outputs = np.zeros(
            (4, 2, 2),
            dtype=complex,
        )

        for rhos in sectors.values():
            logical_outputs += rhos

        V0dag = (
            base.FOUR_V0.conj().T
        )

        for rhos in failed.values():

            logical_outputs += (
                sandwich_batch(
                    V0dag,
                    rhos,
                )
            )

        # Complete physical trace.
        #
        # For the conditioned calculation this is the
        # probability of exactly one jump in cycle 1.

        total_traces = np.zeros(
            4,
            dtype=float,
        )

        for rhos in sectors.values():

            total_traces += np.real(
                np.trace(
                    rhos,
                    axis1=1,
                    axis2=2,
                )
            )

        for rhos in failed.values():

            total_traces += np.real(
                np.trace(
                    rhos,
                    axis1=1,
                    axis2=2,
                )
            )

        coeffs = channel_coefficients(
            logical_outputs,
            total_traces,
        )

        f_wc, r_wc = (
            worst_case_fidelity(
                coeffs
            )
        )

        fidelities.append(
            f_wc
        )

        worst_vectors.append(
            r_wc
        )

    return (
        np.asarray(fidelities),
        np.asarray(worst_vectors),
    )


# ============================================================
# p SWEEPS
# ============================================================

def sweep_p_unconditioned_wc(
    p_values,
    n_cycles=2,
):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    f5 = np.zeros_like(
        p_values
    )

    f4_fixed = np.zeros_like(
        p_values
    )

    f4_adapt = np.zeros_like(
        p_values
    )

    for j, p in enumerate(
        p_values
    ):

        print(
            f"[unconditioned WC] "
            f"p={p:.6g} "
            f"({j + 1}/{len(p_values)})"
        )

        f5[j] = (
            five_qubit_worst_case_history(
                p,
                n_cycles,
                conditioned_first_cycle=False,
            )[0][-1]
        )

        f4_fixed[j] = (
            four_qubit_worst_case_history(
                p,
                n_cycles,
                adaptive=False,
                conditioned_first_cycle=False,
            )[0][-1]
        )

        f4_adapt[j] = (
            four_qubit_worst_case_history(
                p,
                n_cycles,
                adaptive=True,
                conditioned_first_cycle=False,
            )[0][-1]
        )

    return {
        "p": p_values,
        "five": f5,
        "fixed": f4_fixed,
        "adaptive": f4_adapt,
    }


def sweep_p_conditioned_wc(
    p_values,
    n_cycles=2,
):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    f5 = np.zeros_like(
        p_values
    )

    f4_fixed = np.zeros_like(
        p_values
    )

    f4_adapt = np.zeros_like(
        p_values
    )

    for j, p in enumerate(
        p_values
    ):

        print(
            f"[conditioned WC] "
            f"p={p:.6g} "
            f"({j + 1}/{len(p_values)})"
        )

        f5[j] = (
            five_qubit_worst_case_history(
                p,
                n_cycles,
                conditioned_first_cycle=True,
            )[0][-1]
        )

        f4_fixed[j] = (
            four_qubit_worst_case_history(
                p,
                n_cycles,
                adaptive=False,
                conditioned_first_cycle=True,
            )[0][-1]
        )

        f4_adapt[j] = (
            four_qubit_worst_case_history(
                p,
                n_cycles,
                adaptive=True,
                conditioned_first_cycle=True,
            )[0][-1]
        )

    return {
        "p": p_values,
        "five": f5,
        "fixed": f4_fixed,
        "adaptive": f4_adapt,
    }


# ============================================================
# LOG-LOG SLOPE
# ============================================================

def fit_loglog_slope(
    p,
    y,
    fit_max=None,
):
    p = np.asarray(p)
    y = np.asarray(y)

    mask = (
        (p > 0)
        & (y > 1e-15)
    )

    if fit_max is not None:
        mask &= (
            p <= fit_max
        )

    coeff = np.polyfit(
        np.log10(p[mask]),
        np.log10(y[mask]),
        1,
    )

    return coeff[0]


# ============================================================
# FIGURE 1
#
# CONDITIONED TWO-CYCLE WORST-CASE INFIDELITY VS p
#
# No legend.  No inset.
# ============================================================

def plot_two_cycle_conditioned_wc(
    conditioned,
):
    p = conditioned["p"]

    inf_adapt = np.maximum(
        1.0 - conditioned["adaptive"],
        1e-16,
    )

    inf_fixed = np.maximum(
        1.0 - conditioned["fixed"],
        1e-16,
    )

    inf_five = np.maximum(
        1.0 - conditioned["five"],
        1e-16,
    )

    # Slopes retained for the paper caption / diagnostics.
    slope_adapt = fit_loglog_slope(
        p,
        inf_adapt,
        fit_max=0.02,
    )

    slope_fixed = fit_loglog_slope(
        p,
        inf_fixed,
        fit_max=0.02,
    )

    slope_five = fit_loglog_slope(
        p,
        inf_five,
        fit_max=0.02,
    )

    print("\nSmall-p fitted exponents:")
    print(
        f"  Syndrome adapted: alpha = "
        f"{slope_adapt:.6f}"
    )
    print(
        f"  Static:           alpha = "
        f"{slope_fixed:.6f}"
    )
    print(
        f"  [[5,1,3]] code:   alpha = "
        f"{slope_five:.6f}"
    )

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH_P,
            FIG_HEIGHT,
        )
    )

    # Solid lines: p is continuous and the interpolation is
    # physically meaningful here.
    for y, style in [
        (inf_adapt, STYLE_ADAPT),
        (inf_fixed, STYLE_STATIC),
        (inf_five, STYLE_FIVE),
    ]:
        ax.loglog(
            p,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.25,
            markersize=4.2,
            markeredgewidth=0.7,
            zorder=3,
        )

    ax.set_xlabel(
        r"Noise strength $p$"
    )

    ax.set_ylabel(
        "Worst-case fidelity loss"
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
    )

    ax.grid(
        which="both",
        linewidth=0.4,
        alpha=0.18,
    )

    # Intentionally no legend and no title.

    fig.subplots_adjust(
        left=0.20,
        right=0.985,
        bottom=0.18,
        top=0.975,
    )

    fig.savefig(
        "ADAG_two_cycle_worst_case_fidelity_loss.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        "ADAG_two_cycle_worst_case_fidelity_loss.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "alpha_adaptive": slope_adapt,
        "alpha_static": slope_fixed,
        "alpha_five": slope_five,
    }


# ============================================================
# FIGURE 2
#
# UNCONDITIONED TWO-CYCLE WORST-CASE FIDELITY VS p
#
# This is the old inset, now promoted to its own figure.
# No legend.
# ============================================================

def plot_two_cycle_unconditioned_wc(
    unconditioned,
):
    p = unconditioned["p"]

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH_OTHER,
            FIG_HEIGHT,
        )
    )

    for y, style in [
        (
            unconditioned["adaptive"],
            STYLE_ADAPT,
        ),
        (
            unconditioned["fixed"],
            STYLE_STATIC,
        ),
        (
            unconditioned["five"],
            STYLE_FIVE,
        ),
    ]:
        ax.plot(
            p,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.2,
            markersize=4.0,
            markeredgewidth=0.7,
            zorder=3,
        )

    ax.set_xlabel(
        r"Noise strength $p$"
    )

    ax.set_ylabel(
        "Worst-case fidelity"
    )

    ax.set_xlim(
        float(np.min(p)),
        float(np.max(p)),
    )

    ax.set_xticks(
        [0.0, 0.05, 0.10]
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=4)
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
    )

    ax.grid(
        linewidth=0.4,
        alpha=0.18,
    )

    # Intentionally no legend and no title.

    fig.subplots_adjust(
        left=0.19,
        right=0.975,
        bottom=0.18,
        top=0.975,
    )

    fig.savefig(
        "ADAG_two_cycle_unconditioned_fidelity.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        "ADAG_two_cycle_unconditioned_fidelity.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# FIGURE 3
#
# UNCONDITIONED WORST-CASE FIDELITY VS CYCLES
# ============================================================

def plot_worst_case_vs_cycles(
    p=0.01,
    n_max=50,
):
    cycles = np.arange(
        1,
        n_max + 1,
    )

    print(
        f"Calculating worst-case cycle sweep "
        f"at p={p}, n_max={n_max}"
    )

    f5, r5 = (
        five_qubit_worst_case_history(
            p,
            n_max,
            conditioned_first_cycle=False,
        )
    )

    f4_fixed, r_fixed = (
        four_qubit_worst_case_history(
            p,
            n_max,
            adaptive=False,
            conditioned_first_cycle=False,
        )
    )

    f4_adapt, r_adapt = (
        four_qubit_worst_case_history(
            p,
            n_max,
            adaptive=True,
            conditioned_first_cycle=False,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH_OTHER,
            FIG_HEIGHT,
        )
    )

    # Dashed joining lines because the physically evaluated
    # points occur at integer QEC cycles.
    for y, style in [
        (f4_adapt, STYLE_ADAPT),
        (f4_fixed, STYLE_STATIC),
        (f5, STYLE_FIVE),
    ]:
        ax.plot(
            cycles,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle="--",
            linewidth=1.2,
            markersize=4.0,
            markeredgewidth=0.7,
            markevery=3,
            label=style["label"],
            zorder=3,
        )

    ax.set_xlabel(
        r"Number of QEC cycles $n$"
    )

    ax.set_ylabel(
        "Worst-case fidelity"
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
    )

    ax.grid(
        linewidth=0.4,
        alpha=0.18,
    )

    # The only legend among the three plots.
    legend = ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        fontsize=7.0,
        borderpad=0.25,
        labelspacing=0.15,
        handlelength=1.45,
        handletextpad=0.4,
    )

    legend.get_frame().set_linewidth(
        0.7
    )

    # No title.

    fig.subplots_adjust(
        left=0.19,
        right=0.975,
        bottom=0.18,
        top=0.975,
    )

    fig.savefig(
        "ADAG_worst_case_fidelity_vs_cycles.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        "ADAG_worst_case_fidelity_vs_cycles.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "cycles": cycles,
        "five": f5,
        "fixed": f4_fixed,
        "adaptive": f4_adapt,
        "worst_r_five": r5,
        "worst_r_fixed": r_fixed,
        "worst_r_adaptive": r_adapt,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Two-cycle p sweeps
    # --------------------------------------------------------

    # Log-spaced because this is the log-log scaling plot.
    P_CONDITIONED = np.logspace(
        -3,
        -1,
        12,
    )

    # Linear spacing for the ordinary fidelity plot.
    P_UNCONDITIONED = np.linspace(
        0.0,
        0.1,
        11,
    )

    conditioned = (
        sweep_p_conditioned_wc(
            P_CONDITIONED,
            n_cycles=2,
        )
    )

    unconditioned = (
        sweep_p_unconditioned_wc(
            P_UNCONDITIONED,
            n_cycles=2,
        )
    )

    slope_results = (
        plot_two_cycle_conditioned_wc(
            conditioned,
        )
    )

    plot_two_cycle_unconditioned_wc(
        unconditioned,
    )

    # --------------------------------------------------------
    # Multicycle plot
    # --------------------------------------------------------

    cycle_results = (
        plot_worst_case_vs_cycles(
            p=0.01,
            n_max=50,
        )
    )

    print("\nFinished.")
    print(
        "Saved:\n"
        "  ADAG_two_cycle_worst_case_fidelity_loss.pdf\n"
        "  ADAG_two_cycle_unconditioned_fidelity.pdf\n"
        "  ADAG_worst_case_fidelity_vs_cycles.pdf"
    )