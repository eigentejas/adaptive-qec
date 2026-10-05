import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from functools import reduce
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================================
# BASIC MATRICES / HELPERS
# ============================================================

I2 = np.eye(2, dtype=complex)

X2 = np.array([
    [0, 1],
    [1, 0],
], dtype=complex)

Y2 = np.array([
    [0, -1j],
    [1j, 0],
], dtype=complex)

Z2 = np.array([
    [1, 0],
    [0, -1],
], dtype=complex)


def kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def pauli_string(s):
    table = {
        "I": I2,
        "X": X2,
        "Y": Y2,
        "Z": Z2,
    }
    return kron_all([table[c] for c in s])


def one_qubit_pauli(P, q, n):
    """
    P on qubit q of an n-qubit system.
    Qubit ordering is big-endian:
        |q0 q1 ... q_{n-1}>
    """
    ops = [I2] * n
    ops[q] = P
    return kron_all(ops)


# ============================================================
# INCOHERENT AD / AG PERSISTENT-BATH MODEL
#
# Bath is classical in the computational basis.
#
# bath bit 0:
#       AD on system
#
# bath bit 1:
#       AG on system
#
# A jump flips the corresponding bath bit.
# No-jump and jump are DIFFERENT KRAUS BRANCHES.
# ============================================================


RAISE = np.array([
    [0, 0],
    [1, 0],
], dtype=complex)


def initial_logical_ref_density_small():
    """
    Bell state on

        [logical | reference]

    only.

    Bath configurations are tracked separately as classical labels.
    """
    return P_BELL.copy()


def logical_ref_fidelity_small(rho):
    """
    Fidelity numerator for a subnormalized density matrix on

        [logical | reference].
    """
    return float(
        np.real(
            np.trace(
                P_BELL @ rho
            )
        )
    )


def physical_ref_fidelity_small(
    rho,
    V,
):
    """
    Fidelity numerator for a subnormalized density matrix on

        [physical system | reference].
    """
    P_ideal = encoded_bell_projector(V)

    return float(
        np.real(
            np.trace(
                P_ideal @ rho
            )
        )
    )


def xor_bits(a, b):
    return tuple(
        x ^ y
        for x, y in zip(a, b)
    )


def system_jump_operator(
    bath,
    jumps,
    p,
):
    """
    System Kraus operator for a specified bath configuration and
    specified jump pattern.

    bath[q] = 0:
        no jump = AD no-jump operator
        jump    = sqrt(p) |0><1|

    bath[q] = 1:
        no jump = AG no-jump operator
        jump    = sqrt(p) |1><0|

    jumps[q] = 1 means that pair undergoes the jump Kraus event.

    Returns:
        E          system-only Kraus operator
        next_bath  bath configuration after the event
    """

    s = np.sqrt(1 - p)

    AD_NO = np.array([
        [1, 0],
        [0, s],
    ], dtype=complex)

    AG_NO = np.array([
        [s, 0],
        [0, 1],
    ], dtype=complex)

    ops = []

    for b, j in zip(
        bath,
        jumps,
    ):

        if b == 0:

            if j == 0:
                op = AD_NO
            else:
                op = np.sqrt(p) * LOWER

        else:

            if j == 0:
                op = AG_NO
            else:
                op = np.sqrt(p) * RAISE

        ops.append(op)

    E = kron_all(ops)

    next_bath = xor_bits(
        bath,
        jumps,
    )

    return E, next_bath


def check_system_jump_channel(
    bath,
    p,
):
    """
    Sanity check:
        sum_J E_J^\dagger E_J = I
    """

    n = len(bath)

    completeness = np.zeros(
        (2**n, 2**n),
        dtype=complex,
    )

    for jumps in product(
        [0, 1],
        repeat=n,
    ):

        E, _ = system_jump_operator(
            bath,
            jumps,
            p,
        )

        completeness += (
            E.conj().T @ E
        )

    assert np.allclose(
        completeness,
        np.eye(2**n),
        atol=1e-10,
    )



# ============================================================
# ENTANGLEMENT-FIDELITY HELPERS
# ============================================================

def initial_logical_env_ref_density(n_env):
    """
    Initial state

        (|0_L 0_R> + |1_L 1_R>)/sqrt(2)

    represented in the compressed basis

        [logical qubit | environment | reference]

    with environment initially |00...0>.
    """

    env_dim = 2**n_env

    psi = np.zeros(
        (2, env_dim, 2),
        dtype=complex,
    )

    psi[0, 0, 0] = 1 / np.sqrt(2)
    psi[1, 0, 1] = 1 / np.sqrt(2)

    psi = psi.reshape(-1)

    return np.outer(
        psi,
        psi.conj(),
    )


BELL = np.zeros(
    (2, 2),
    dtype=complex,
)

BELL[0, 0] = 1 / np.sqrt(2)
BELL[1, 1] = 1 / np.sqrt(2)

BELL = BELL.reshape(-1)

P_BELL = np.outer(
    BELL,
    BELL.conj(),
)


def logical_ref_fidelity_numerator(
    rho,
    n_env,
):
    """
    Fidelity numerator for a state represented as

        [logical | environment | reference].
    """

    env_dim = 2**n_env

    r = rho.reshape(
        2,
        env_dim,
        2,
        2,
        env_dim,
        2,
    )

    # trace environment
    reduced = np.trace(
        r,
        axis1=1,
        axis2=4,
    )

    reduced = reduced.reshape(
        4,
        4,
    )

    return float(
        np.real(
            np.trace(
                P_BELL @ reduced
            )
        )
    )


def encoded_bell_projector(V):
    """
    V is a physical-code isometry:

        physical dimension x 2 logical dimensions.

    Returns the encoded Bell-state projector
    on [physical system | reference].
    """

    d_sys = V.shape[0]

    psi = np.zeros(
        (d_sys, 2),
        dtype=complex,
    )

    psi[:, 0] = V[:, 0] / np.sqrt(2)
    psi[:, 1] = V[:, 1] / np.sqrt(2)

    psi = psi.reshape(-1)

    return np.outer(
        psi,
        psi.conj(),
    )


def physical_ref_fidelity_numerator(
    rho,
    V,
    n_env,
):
    """
    Fidelity numerator when the state is still in the
    full physical-system basis:

        [physical system | environment | reference].
    """

    d_sys = V.shape[0]
    d_env = 2**n_env

    r = rho.reshape(
        d_sys,
        d_env,
        2,
        d_sys,
        d_env,
        2,
    )

    reduced = np.trace(
        r,
        axis1=1,
        axis2=4,
    )

    reduced = reduced.reshape(
        2 * d_sys,
        2 * d_sys,
    )

    P_ideal = encoded_bell_projector(V)

    return float(
        np.real(
            np.trace(
                P_ideal @ reduced
            )
        )
    )


# ============================================================
# ============================================================
#
#               FIVE-QUBIT STABILIZER CODE
#
# ============================================================
# ============================================================


FIVE_STABS = [
    pauli_string("XZZXI"),
    pauli_string("IXZZX"),
    pauli_string("XIXZZ"),
    pauli_string("ZXIXZ"),
]


def projector_vector(P):
    """
    Extract the normalized vector from a rank-1 projector.
    """
    P = (P + P.conj().T) / 2

    vals, vecs = np.linalg.eigh(P)

    v = vecs[:, np.argmax(vals)]

    # Fix arbitrary phase for reproducibility.
    nz = np.flatnonzero(
        np.abs(v) > 1e-12
    )

    if len(nz):
        v *= np.exp(
            -1j * np.angle(v[nz[0]])
        )

    return v / np.linalg.norm(v)


def build_five_qubit_isometry():
    """
    Construct |0_L>, |1_L> from the stabilizer code.

    Returns V with shape (32,2).
    """

    I32 = np.eye(
        32,
        dtype=complex,
    )

    P_code = I32.copy()

    for g in FIVE_STABS:
        P_code = P_code @ (
            (I32 + g) / 2
        )

    # Logical Z
    ZL = pauli_string("ZZZZZ")

    P0 = P_code @ (
        (I32 + ZL) / 2
    )

    P1 = P_code @ (
        (I32 - ZL) / 2
    )

    ket0 = projector_vector(P0)
    ket1 = projector_vector(P1)

    return np.column_stack(
        [ket0, ket1]
    )


FIVE_V = build_five_qubit_isometry()


# Exact same correction ordering as old code.
FIVE_ERRS = [
    np.eye(32, dtype=complex),

    one_qubit_pauli(X2, 1, 5),
    one_qubit_pauli(Z2, 4, 5),

    one_qubit_pauli(X2, 2, 5),
    one_qubit_pauli(Z2, 2, 5),
    one_qubit_pauli(Z2, 0, 5),

    one_qubit_pauli(X2, 3, 5),
    one_qubit_pauli(Y2, 2, 5),
    one_qubit_pauli(X2, 0, 5),

    one_qubit_pauli(Z2, 3, 5),
    one_qubit_pauli(Z2, 1, 5),
    one_qubit_pauli(Y2, 1, 5),

    one_qubit_pauli(X2, 4, 5),
    one_qubit_pauli(Y2, 0, 5),
    one_qubit_pauli(Y2, 4, 5),
    one_qubit_pauli(Y2, 3, 5),
]


def build_five_qubit_recoveries():
    """
    The 16 standard stabilizer syndrome recovery Kraus operators
    acting on the five physical qubits.

    Each has dimension 32 x 32.
    """

    recs = []

    I32 = np.eye(
        32,
        dtype=complex,
    )

    # Matches old ordering:
    # product gives (s3,s2,s1,s0)
    for idx, svals in enumerate(
        product([0, 1], repeat=4)
    ):

        s3, s2, s1, s0 = svals

        syndrome_bits = [
            s0,
            s1,
            s2,
            s3,
        ]

        P = I32.copy()

        for bit, g in zip(
            syndrome_bits,
            FIVE_STABS,
        ):
            P = P @ (
                (
                    I32
                    + ((-1) ** bit) * g
                )
                / 2
            )

        R = FIVE_ERRS[idx] @ P

        recs.append(R)

    return recs


FIVE_R = build_five_qubit_recoveries()


# Verify recovery is trace preserving.
_five_completeness = sum(
    R.conj().T @ R
    for R in FIVE_R
)

assert np.allclose(
    _five_completeness,
    np.eye(32),
    atol=1e-10,
)

# ============================================================
# FIVE-QUBIT CODE — NEW JUMP MODEL
# ============================================================


FIVE_BATHS = list(
    product(
        [0, 1],
        repeat=5,
    )
)


def five_qubit_transition_cache_jump(
    p,
    jump_weight=None,
):
    """
    One-cycle transitions.

    Bath is tracked by its actual classical configuration.

    jump_weight=None:
        full channel

    jump_weight=1:
        retain exactly-one-jump Kraus patterns only.
        This gives a trace-decreasing conditioned instrument.
    """

    cache = {}

    for bath in FIVE_BATHS:

        transitions = []

        for jumps in FIVE_BATHS:

            if (
                jump_weight is not None
                and sum(jumps) != jump_weight
            ):
                continue

            E, next_bath = (
                system_jump_operator(
                    bath,
                    jumps,
                    p,
                )
            )

            for R in FIVE_R:

                # logical -> physical error -> recovery -> logical
                L = (
                    FIVE_V.conj().T
                    @ R
                    @ E
                    @ FIVE_V
                )

                # Reference untouched.
                K = np.kron(
                    L,
                    I2,
                )

                transitions.append(
                    (
                        next_bath,
                        K,
                    )
                )

        cache[bath] = transitions

    return cache


def five_qubit_fidelities(
    p,
    n_max,
):
    """
    Completely unconditioned persistent-bath evolution.
    """

    transitions = (
        five_qubit_transition_cache_jump(
            p,
            jump_weight=None,
        )
    )

    ZERO5 = (0, 0, 0, 0, 0)

    sectors = {
        ZERO5:
        initial_logical_ref_density_small()
    }

    fidelities = []

    for cycle in range(n_max):

        new_sectors = {}

        for bath, rho in sectors.items():

            for next_bath, K in transitions[bath]:

                branch = (
                    K
                    @ rho
                    @ K.conj().T
                )

                if next_bath in new_sectors:
                    new_sectors[next_bath] += branch
                else:
                    new_sectors[next_bath] = branch

        sectors = new_sectors

        total_trace = sum(
            np.real(np.trace(rho))
            for rho in sectors.values()
        )

        numerator = sum(
            logical_ref_fidelity_small(rho)
            for rho in sectors.values()
        )

        fidelities.append(
            numerator / total_trace
        )

    return np.asarray(
        fidelities
    )


def five_qubit_fidelity_exact_one_env_flip_first_cycle(
    p,
    n_cycles=2,
):
    """
    Condition cycle 1 on EXACTLY ONE physical jump.

    From cycle 2 onward the full channel is used.

    The old function name is retained so your sweep code
    does not need to change.
    """

    full_transitions = (
        five_qubit_transition_cache_jump(
            p,
            jump_weight=None,
        )
    )

    one_jump_transitions = (
        five_qubit_transition_cache_jump(
            p,
            jump_weight=1,
        )
    )

    ZERO5 = (0, 0, 0, 0, 0)

    sectors = {
        ZERO5:
        initial_logical_ref_density_small()
    }

    fidelities = []

    for cycle in range(n_cycles):

        transitions = (
            one_jump_transitions
            if cycle == 0
            else full_transitions
        )

        new_sectors = {}

        for bath, rho in sectors.items():

            for next_bath, K in transitions[bath]:

                branch = (
                    K
                    @ rho
                    @ K.conj().T
                )

                if next_bath in new_sectors:
                    new_sectors[next_bath] += branch
                else:
                    new_sectors[next_bath] = branch

        sectors = new_sectors

        if cycle == 0:

            p_event = sum(
                np.real(np.trace(rho))
                for rho in sectors.values()
            )

            print(
                "P(exactly one jump in cycle 1) =",
                p_event,
            )

            for bath in sectors:
                sectors[bath] /= p_event

        total_trace = sum(
            np.real(np.trace(rho))
            for rho in sectors.values()
        )

        numerator = sum(
            logical_ref_fidelity_small(rho)
            for rho in sectors.values()
        )

        fidelities.append(
            numerator / total_trace
        )

    return np.asarray(
        fidelities
    )


def environment_weight_projector(n_env, k):
    dim = 2**n_env
    P = np.zeros((dim, dim), dtype=complex)

    for x in range(dim):
        if bin(x).count("1") == k:
            P[x, x] = 1.0

    return P


# ============================================================
# ============================================================
#
#                    FOUR-QUBIT CODE
#
#       fixed recovery AND adaptive recovery
#
# ============================================================
# ============================================================


# Original logical codewords:
#
# |0L> = (|0000> + |1111>)/sqrt(2)
# |1L> = (|0011> + |1100>)/sqrt(2)

FOUR_V0 = np.zeros(
    (16, 2),
    dtype=complex,
)

FOUR_V0[0, 0] = 1 / np.sqrt(2)
FOUR_V0[15, 0] = 1 / np.sqrt(2)

FOUR_V0[3, 1] = 1 / np.sqrt(2)
FOUR_V0[12, 1] = 1 / np.sqrt(2)


ZERO4 = (0, 0, 0, 0)

FOUR_ARRS = list(
    product(
        [0, 1],
        repeat=4,
    )
)


def x_pattern_operator(arr):
    """
    X^{arr[0]} tensor ... tensor X^{arr[3]}.
    """

    return kron_all([
        X2 if bit else I2
        for bit in arr
    ])


# The 16 adaptive codes.
FOUR_V = {
    arr:
    x_pattern_operator(arr) @ FOUR_V0

    for arr in FOUR_ARRS
}


# Same amplitude-damping / gain operators
# from the old code.

LOWER = np.array([
    [0, 1],
    [0, 0],
], dtype=complex)

RAISE_MINUS = np.array([
    [0, 0],
    [-1, 0],
], dtype=complex)


def four_error_operators(arr):
    """
    Correctable error operators for the current bath/history state.

    Returns:
        E0 = identity
        E1,...,E4 = one error on each physical qubit
    """

    errors = [
        np.eye(
            16,
            dtype=complex,
        )
    ]

    for q in range(4):

        ops = [I2] * 4

        if arr[q] == 0:
            ops[q] = LOWER
        else:
            ops[q] = RAISE_MINUS

        errors.append(
            kron_all(ops)
        )

    return errors


def build_four_qubit_recoveries(
    arr,
    adaptive,
):
    """
    Build the six recovery Kraus operators.

    adaptive=True:
        use the history-dependent code and recovery.

    adaptive=False:
        always use the original fixed code.
    """

    if adaptive:
        V = FOUR_V[arr]
        error_arr = arr

    else:
        V = FOUR_V0
        error_arr = ZERO4

    recs = []

    errors = four_error_operators(
        error_arr
    )

    # Correctable outcomes:
    # no error + one error on each qubit.
    for i, E in enumerate(errors):

        corrupted = E @ V

        norms = np.linalg.norm(
            corrupted,
            axis=0,
        )

        if np.any(norms < 1e-14):
            raise RuntimeError(
                "Unexpected zero corrupted logical state."
            )

        corrupted = (
            corrupted
            / norms[None, :]
        )

        # Maps corrupted logical basis
        # back to current code basis.
        R = (
            V
            @ corrupted.conj().T
        )

        if adaptive and i > 0:

            # After detecting an error on qubit i-1,
            # move to the corresponding new code.
            R = (
                one_qubit_pauli(
                    X2,
                    i - 1,
                    4,
                )
                @ R
            )

        recs.append(R)

    # Complete the channel with R5.
    Q = np.eye(
        16,
        dtype=complex,
    )

    for R in recs:
        Q -= R.conj().T @ R

    # Numerical Hermitian cleanup.
    Q = (
        Q
        + Q.conj().T
    ) / 2

    vals, vecs = np.linalg.eigh(Q)

    if np.min(vals) < -1e-10:
        raise RuntimeError(
            "Recovery complement is not positive semidefinite."
        )

    vals[
        np.abs(vals) < 1e-12
    ] = 0

    vals = np.clip(
        vals,
        0,
        None,
    )

    sqrtQ = (
        vecs
        * np.sqrt(vals)
    ) @ vecs.conj().T

    if adaptive:
        R5 = (
            x_pattern_operator(arr)
            @ sqrtQ
        )
    else:
        R5 = sqrtQ

    recs.append(R5)

    # Sanity check.
    completeness = sum(
        R.conj().T @ R
        for R in recs
    )

    assert np.allclose(
        completeness,
        np.eye(16),
        atol=1e-9,
    )

    return recs


# Recovery matrices do NOT depend on p.
# Build them once.

FOUR_R_ADAPT = {
    arr:
    build_four_qubit_recoveries(
        arr,
        adaptive=True,
    )

    for arr in FOUR_ARRS
}


FOUR_R_FIXED = (
    build_four_qubit_recoveries(
        ZERO4,
        adaptive=False,
    )
)

# ============================================================
# FOUR-QUBIT CODE — NEW JUMP MODEL
# ============================================================


def four_transition_data_jump(
    arr,
    bath,
    p,
    adaptive,
    jump_weight=None,
):
    """
    One-cycle transitions.

    arr:
        bath configuration INFERRED by the controller from
        syndrome history.

    bath:
        ACTUAL bath configuration.

    Keeping these as separate labels is important:
    higher-order events can cause arr != bath.

    jump_weight=None:
        full channel.

    jump_weight=1:
        exactly one physical jump only.
    """

    if adaptive:

        V_in = FOUR_V[arr]
        recs = FOUR_R_ADAPT[arr]

    else:

        V_in = FOUR_V0
        recs = FOUR_R_FIXED

    success = []
    failure = []

    for jumps in FOUR_ARRS:

        if (
            jump_weight is not None
            and sum(jumps) != jump_weight
        ):
            continue

        E, next_bath = (
            system_jump_operator(
                bath,
                jumps,
                p,
            )
        )

        # ----------------------------------------------------
        # Successful syndrome outcomes R0,...,R4
        # ----------------------------------------------------

        for i in range(5):

            if adaptive:

                next_arr = list(arr)

                if i > 0:
                    next_arr[i - 1] ^= 1

                next_arr = tuple(
                    next_arr
                )

                V_out = FOUR_V[
                    next_arr
                ]

            else:

                next_arr = ZERO4
                V_out = FOUR_V0

            # logical -> physical noise -> recovery -> logical
            L = (
                V_out.conj().T
                @ recs[i]
                @ E
                @ V_in
            )

            K = np.kron(
                L,
                I2,
            )

            success.append(
                (
                    next_arr,
                    next_bath,
                    K,
                )
            )

        # ----------------------------------------------------
        # R5 failure branch:
        #
        # logical input -> physical output.
        # ----------------------------------------------------

        Fsys = (
            recs[5]
            @ E
            @ V_in
        )

        F = np.kron(
            Fsys,
            I2,
        )

        failure.append(
            (
                next_bath,
                F,
            )
        )

    return success, failure


def four_transition_cache_jump(
    p,
    adaptive,
    jump_weight=None,
):
    """
    Cache all controller/actual-bath sectors.
    """

    arrs = (
        FOUR_ARRS
        if adaptive
        else [ZERO4]
    )

    cache = {}

    for arr in arrs:

        for bath in FOUR_ARRS:

            cache[
                (arr, bath)
            ] = (
                four_transition_data_jump(
                    arr,
                    bath,
                    p,
                    adaptive,
                    jump_weight=jump_weight,
                )
            )

    return cache


def propagate_failed_four_qubit_jump_model(
    fail,
    p,
):
    """
    Once R5 occurs, QEC stops.

    fail is a dictionary

        bath -> rho

    where rho acts on

        [4 physical system qubits | reference].

    Bath remains a classical persistent label.
    """

    new_fail = {}

    for bath, rho in fail.items():

        for jumps in FOUR_ARRS:

            E, next_bath = (
                system_jump_operator(
                    bath,
                    jumps,
                    p,
                )
            )

            K = np.kron(
                E,
                I2,
            )

            branch = (
                K
                @ rho
                @ K.conj().T
            )

            if next_bath in new_fail:
                new_fail[next_bath] += branch
            else:
                new_fail[next_bath] = branch

    return new_fail


def _four_qubit_fidelities_jump_model(
    p,
    n_max,
    adaptive,
    first_cycle_jump_weight=None,
):
    """
    Shared implementation.

    first_cycle_jump_weight=None:
        completely unconditioned.

    first_cycle_jump_weight=1:
        condition first cycle on exactly one physical jump.
    """

    full_transitions = (
        four_transition_cache_jump(
            p,
            adaptive,
            jump_weight=None,
        )
    )

    if first_cycle_jump_weight is not None:

        first_transitions = (
            four_transition_cache_jump(
                p,
                adaptive,
                jump_weight=first_cycle_jump_weight,
            )
        )

    else:

        first_transitions = None

    # Active sectors are labelled by
    #
    #     (controller arr, actual bath)
    #
    # and contain a 4x4 density matrix on
    #
    #     [logical | reference].
    sectors = {
        (
            ZERO4,
            ZERO4,
        ):
        initial_logical_ref_density_small()
    }

    # Failed branches:
    #
    #     actual bath -> physical-system/reference density matrix
    fail = {}

    fidelities = []

    for cycle in range(n_max):

        # ----------------------------------------------------
        # Previously failed trajectories evolve physically
        # with no further QEC.
        # ----------------------------------------------------

        if fail:

            fail = (
                propagate_failed_four_qubit_jump_model(
                    fail,
                    p,
                )
            )

        new_sectors = {}
        new_fail = dict(fail)

        transitions = (
            first_transitions
            if (
                cycle == 0
                and first_transitions
                is not None
            )
            else full_transitions
        )

        # ----------------------------------------------------
        # Active trajectories.
        # ----------------------------------------------------

        for (
            arr,
            bath,
        ), rho in sectors.items():

            success, failure = (
                transitions[
                    (arr, bath)
                ]
            )

            # Successful R0,...,R4 outcomes.
            for (
                next_arr,
                next_bath,
                K,
            ) in success:

                branch = (
                    K
                    @ rho
                    @ K.conj().T
                )

                key = (
                    next_arr,
                    next_bath,
                )

                if key in new_sectors:
                    new_sectors[key] += branch
                else:
                    new_sectors[key] = branch

            # R5 failure outcomes.
            for (
                next_bath,
                F,
            ) in failure:

                branch = (
                    F
                    @ rho
                    @ F.conj().T
                )

                if next_bath in new_fail:
                    new_fail[next_bath] += branch
                else:
                    new_fail[next_bath] = branch

        sectors = new_sectors
        fail = new_fail

        # ----------------------------------------------------
        # If the first cycle was conditioned, normalize now.
        # ----------------------------------------------------

        if (
            cycle == 0
            and first_cycle_jump_weight
            is not None
        ):

            p_event = sum(
                np.real(np.trace(rho))
                for rho in sectors.values()
            )

            p_event += sum(
                np.real(np.trace(rho))
                for rho in fail.values()
            )

            print(
                f"P(exactly "
                f"{first_cycle_jump_weight} "
                f"jump(s) in cycle 1) =",
                p_event,
            )

            for key in sectors:
                sectors[key] /= p_event

            for bath in fail:
                fail[bath] /= p_event

        # ----------------------------------------------------
        # Fidelity after this cycle.
        # ----------------------------------------------------

        p_active = sum(
            np.real(np.trace(rho))
            for rho in sectors.values()
        )

        p_fail = sum(
            np.real(np.trace(rho))
            for rho in fail.values()
        )

        numerator_active = sum(
            logical_ref_fidelity_small(
                rho
            )
            for rho in sectors.values()
        )

        numerator_fail = sum(
            physical_ref_fidelity_small(
                rho,
                FOUR_V0,
            )
            for rho in fail.values()
        )

        total_trace = (
            p_active
            + p_fail
        )

        fidelity = (
            numerator_active
            + numerator_fail
        ) / total_trace

        fidelities.append(
            fidelity
        )

    return np.asarray(
        fidelities
    )


def four_qubit_fidelities(
    p,
    n_max,
    adaptive,
):
    """
    Completely unconditioned evolution.
    """

    return (
        _four_qubit_fidelities_jump_model(
            p,
            n_max,
            adaptive,
            first_cycle_jump_weight=None,
        )
    )


def four_qubit_fidelity_conditioned_first_cycle(
    p,
    n_cycles=2,
    adaptive=True,
):
    """
    Condition cycle 1 on exactly ONE physical jump.
    """

    return (
        _four_qubit_fidelities_jump_model(
            p,
            n_cycles,
            adaptive,
            first_cycle_jump_weight=1,
        )
    )

# ============================================================
# SWEEPS
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
        mask &= p <= fit_max

    coeff = np.polyfit(
        np.log10(p[mask]),
        np.log10(y[mask]),
        1,
    )

    return coeff[0]

def sweep_p_unconditioned(
    p_values,
    n_cycles=2,
):
    """
    Completely unconditioned fidelity after n_cycles.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    f5 = np.zeros_like(p_values)
    f4_fixed = np.zeros_like(p_values)
    f4_adapt = np.zeros_like(p_values)

    for j, p in enumerate(p_values):

        print(
            f"[unconditioned] "
            f"p={p:.6g} "
            f"({j+1}/{len(p_values)})"
        )

        f5[j] = five_qubit_fidelities(
            p,
            n_cycles,
        )[-1]

        f4_fixed[j] = four_qubit_fidelities(
            p,
            n_cycles,
            adaptive=False,
        )[-1]

        f4_adapt[j] = four_qubit_fidelities(
            p,
            n_cycles,
            adaptive=True,
        )[-1]

    return {
        "p": p_values,
        "five": f5,
        "fixed": f4_fixed,
        "adaptive": f4_adapt,
    }


def sweep_p_conditioned_single_jump(
    p_values,
    n_cycles=2,
):
    """
    Condition cycle 1 on exactly one physical jump.

    Returns the fidelity after n_cycles.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    f5 = np.zeros_like(p_values)
    f4_fixed = np.zeros_like(p_values)
    f4_adapt = np.zeros_like(p_values)

    for j, p in enumerate(p_values):

        print(
            f"[conditioned] "
            f"p={p:.6g} "
            f"({j+1}/{len(p_values)})"
        )

        f5[j] = (
            five_qubit_fidelity_exact_one_env_flip_first_cycle(
                p,
                n_cycles=n_cycles,
            )[-1]
        )

        f4_fixed[j] = (
            four_qubit_fidelity_conditioned_first_cycle(
                p,
                n_cycles=n_cycles,
                adaptive=False,
            )[-1]
        )

        f4_adapt[j] = (
            four_qubit_fidelity_conditioned_first_cycle(
                p,
                n_cycles=n_cycles,
                adaptive=True,
            )[-1]
        )

    return {
        "p": p_values,
        "five": f5,
        "fixed": f4_fixed,
        "adaptive": f4_adapt,
    }


# ============================================================
# FIGURE 1
#
# CONDITIONED TWO-CYCLE INFIDELITY VS p
# WITH UNCONDITIONED FIDELITY INSET
# ============================================================


def plot_two_cycle_comparison(
    conditioned,
    unconditioned,
):

    p = conditioned["p"]

    inf_adapt = np.maximum(
        1 - conditioned["adaptive"],
        1e-16,
    )

    inf_fixed = np.maximum(
        1 - conditioned["fixed"],
        1e-16,
    )

    inf_five = np.maximum(
        1 - conditioned["five"],
        1e-16,
    )
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
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    # --------------------------------------------------------
    # Main plot
    #
    # Legend order:
    # adaptive, fixed, stabilizer
    #
    # Stabilizer plotted last and with hollow squares so that
    # it remains visible when it overlaps the adaptive curve.
    # --------------------------------------------------------

    ax.loglog(
        p,
        inf_adapt,
        linestyle="-",
        linewidth=2,
        marker="^",
        markersize=7,
        label=(
            "4-qubit adaptive recovery "
            rf"($\alpha\approx{slope_adapt:.2f}$)"
        ),
        zorder=2,
    )

    ax.loglog(
        p,
        inf_fixed,
        linestyle="-",
        linewidth=2,
        marker="o",
        markersize=6,
        label=(
            "4-qubit fixed recovery "
            rf"($\alpha\approx{slope_fixed:.2f}$)"
        ),
        zorder=1,
    )

    ax.loglog(
        p,
        inf_five,
        linestyle="--",
        linewidth=2,
        marker="s",
        markerfacecolor="none",
        markeredgewidth=1.5,
        markersize=7,
        label=(
            "5-qubit stabilizer code "
            rf"($\alpha\approx{slope_five:.2f}$)"
        ),
        zorder=4,
    )

    ax.set_xlabel(
        r"$p$"
    )

    ax.set_ylabel(
        r"Entanglement infidelity $1-F_e$"
    )

    ax.set_title(
        "Two-cycle performance conditioned on "
        "exactly one jump in cycle 1"
    )

    ax.legend(
        loc="upper left"
    )

    # --------------------------------------------------------
    # Inset:
    # completely unconditioned two-cycle fidelity vs p
    # --------------------------------------------------------

    axins = inset_axes(
        ax,
        width="43%",
        height="43%",
        loc="lower right",
        borderpad=1.5,
    )

    pu = unconditioned["p"]

    axins.plot(
        pu,
        unconditioned["adaptive"],
        linestyle="-",
        linewidth=1.5,
        marker="^",
        markersize=4,
    )

    axins.plot(
        pu,
        unconditioned["fixed"],
        linestyle="-",
        linewidth=1.5,
        marker="o",
        markersize=4,
    )

    axins.plot(
        pu,
        unconditioned["five"],
        linestyle="--",
        linewidth=1.5,
        marker="s",
        markerfacecolor="none",
        markeredgewidth=1.0,
        markersize=4.5,
        zorder=4,
    )

    axins.set_title(
        "Unconditioned",
        fontsize=10,
    )

    axins.set_xlabel(
        r"$p$",
        fontsize=9,
    )

    axins.set_ylabel(
        r"$F_e$",
        fontsize=9,
    )

    axins.tick_params(
        axis="both",
        labelsize=8,
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# FIGURE 2
#
# UNCONDITIONED FIDELITY VS NUMBER OF CYCLES
# ============================================================


def plot_fidelity_vs_cycles(
    p=0.01,
    n_max=50,
):

    cycles = np.arange(
        1,
        n_max + 1,
    )

    print(
        f"Calculating cycle sweep at "
        f"p={p}, n_max={n_max}"
    )

    f5 = five_qubit_fidelities(
        p,
        n_max,
    )

    f4_fixed = four_qubit_fidelities(
        p,
        n_max,
        adaptive=False,
    )

    f4_adapt = four_qubit_fidelities(
        p,
        n_max,
        adaptive=True,
    )

    plt.figure(
        figsize=(9, 6)
    )

    # Adaptive first for desired legend ordering.
    plt.plot(
        cycles,
        f4_adapt,
        linestyle="-",
        linewidth=2,
        marker="^",
        markersize=6,
        markevery=3,
        label="4-qubit adaptive recovery",
        zorder=2,
    )

    plt.plot(
        cycles,
        f4_fixed,
        linestyle="-",
        linewidth=2,
        marker="o",
        markersize=5,
        markevery=3,
        label="4-qubit fixed recovery",
        zorder=1,
    )

    # Stabilizer last so that it remains visible when it nearly
    # coincides with the adaptive curve.
    plt.plot(
        cycles,
        f5,
        linestyle="--",
        linewidth=2,
        marker="s",
        markerfacecolor="none",
        markeredgewidth=1.5,
        markersize=6,
        markevery=3,
        label="5-qubit stabilizer code",
        zorder=4,
    )

    plt.xlabel(
        "Number of QEC cycles"
    )

    plt.ylabel(
        r"Entanglement fidelity $F_e$"
    )

    plt.title(
        rf"Entanglement fidelity over repeated QEC cycles at $p={p}$"
    )

    plt.legend(
        loc="lower left"
    )

    plt.tight_layout()

    plt.show()

    return {
        "cycles": cycles,
        "five": f5,
        "fixed": f4_fixed,
        "adaptive": f4_adapt,
    }

# ============================================================
# MULTICYCLE SCALING CHECK
# ============================================================

def multicycle_scaling_check(
    p_values=(0.005, 0.01, 0.02),
    n_max=40,
):
    """
    Test the expected multicycle scalings:

        5-qubit:
            1-F ~ C_5 n p^2

        4-qubit adaptive:
            1-F ~ C_A n p^2

        4-qubit fixed:
            1-F ~ C_F n^2 p^2

    If these scalings hold, the normalized quantities plotted
    below should be approximately constant in n and approximately
    independent of p in the perturbative regime.
    """

    cycles = np.arange(
        1,
        n_max + 1,
        dtype=float,
    )

    all_results = {}

    for p in p_values:

        print()
        print("=" * 60)
        print(f"SCALING CHECK: p = {p}")
        print("=" * 60)

        f5 = five_qubit_fidelities(
            p,
            n_max,
        )

        f4_fixed = four_qubit_fidelities(
            p,
            n_max,
            adaptive=False,
        )

        f4_adapt = four_qubit_fidelities(
            p,
            n_max,
            adaptive=True,
        )

        r5 = (
            (1 - f5)
            / (cycles * p**2)
        )

        r4_adapt = (
            (1 - f4_adapt)
            / (cycles * p**2)
        )

        r4_fixed = (
            (1 - f4_fixed)
            / (cycles**2 * p**2)
        )

        all_results[p] = {
            "cycles": cycles,
            "five_qubit": f5,
            "four_fixed": f4_fixed,
            "four_adaptive": f4_adapt,
            "ratio_five": r5,
            "ratio_fixed": r4_fixed,
            "ratio_adaptive": r4_adapt,
        }

        # Print a few representative values.
        for n in [1, 2, 5, 10, 20, n_max]:

            if n > n_max:
                continue

            i = n - 1

            print(
                f"n={n:3d} | "
                f"5q/(np^2)={r5[i]:.6f} | "
                f"adapt/(np^2)={r4_adapt[i]:.6f} | "
                f"fixed/(n^2p^2)={r4_fixed[i]:.6f}"
            )

    # --------------------------------------------------------
    # 5-qubit normalized scaling
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    for p in p_values:

        r = all_results[p]

        plt.plot(
            r["cycles"],
            r["ratio_five"],
            marker="o",
            markevery=4,
            label=rf"$p={p}$",
        )

    plt.xlabel(
        "Number of QEC cycles"
    )

    plt.ylabel(
        r"$(1-F_e)/(np^2)$"
    )

    plt.title(
        "5-qubit stabilizer: test of "
        r"$1-F_e \sim np^2$"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------
    # Adaptive 4-qubit normalized scaling
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    for p in p_values:

        r = all_results[p]

        plt.plot(
            r["cycles"],
            r["ratio_adaptive"],
            marker="o",
            markevery=4,
            label=rf"$p={p}$",
        )

    plt.xlabel(
        "Number of QEC cycles"
    )

    plt.ylabel(
        r"$(1-F_e)/(np^2)$"
    )

    plt.title(
        "4-qubit adaptive: test of "
        r"$1-F_e \sim np^2$"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------
    # Fixed 4-qubit normalized scaling
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    for p in p_values:

        r = all_results[p]

        plt.plot(
            r["cycles"],
            r["ratio_fixed"],
            marker="o",
            markevery=4,
            label=rf"$p={p}$",
        )

    plt.xlabel(
        "Number of QEC cycles"
    )

    plt.ylabel(
        r"$(1-F_e)/(n^2p^2)$"
    )

    plt.title(
        "4-qubit fixed: test of "
        r"$1-F_e \sim n^2p^2$"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    return all_results

# ============================================================
# MAIN
# ============================================================

# Multicycle scalign checks 
# if __name__ == "__main__":
#     scaling_results = multicycle_scaling_check(
#         p_values=(0.0025, 0.005, 0.01),
#         n_max=50,
#     )
# if __name__ == "__main__":
#     # ============================================================
#     # SANITY CHECKS
#     # ============================================================

#     plot_unconditioned_vs_cycles(
#         p=0.01,
#         n_max=50,
#     )
#Below is proper run 
if __name__ == "__main__":

    # ========================================================
    # FIGURE 1:
    #
    # Conditioned two-cycle log-log plot,
    # with unconditioned two-cycle inset.
    # ========================================================

    P_CONDITIONED = np.logspace(
        -3,
        -1,
        12,
    )

    P_UNCONDITIONED = np.linspace(
        0.0,
        0.1,
        11,
    )

    conditioned = (
        sweep_p_conditioned_single_jump(
            P_CONDITIONED,
            n_cycles=2,
        )
    )

    unconditioned = (
        sweep_p_unconditioned(
            P_UNCONDITIONED,
            n_cycles=2,
        )
    )

    plot_two_cycle_comparison(
        conditioned,
        unconditioned,
    )


    # ========================================================
    # FIGURE 2:
    #
    # Unconditioned multicycle performance.
    # ========================================================

    cycle_results = (
        plot_fidelity_vs_cycles(
            p=0.01,
            n_max=50,
        )
    )
