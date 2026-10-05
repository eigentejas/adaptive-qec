import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

from adaptive_qec_l_loss import (
    effective_errors,
    search_exact_codes_raw,
    search_approx_codes_raw,
    canonical_syndromes,
    conditional_bath_supports,
)


# ============================================================

# Basic matrices

# ============================================================


I2 = np.eye(2, dtype=complex)

X2 = np.array([[0, 1], [1, 0]], dtype=complex)

Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)

Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = [X2, Y2, Z2]


# Linear logical probes: rho(r) = I/2 + sum_i r_i sigma_i/2.

PROBES = np.stack([I2 / 2, X2 / 2, Y2 / 2, Z2 / 2])


# ============================================================

# Weak random local Hamiltonian

# ============================================================


def random_local_hamiltonian(rng, interaction_only=False):

    """Random Hermitian on qubit x qutrit, normalized to op norm 1."""

    dQ, dB = 2, 3

    d = dQ * dB


    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))

    h = (A + A.conj().T) / 2


    # Remove only the irrelevant total identity component by default.

    h -= np.trace(h) / d * np.eye(d, dtype=complex)


    if interaction_only:

        T = h.reshape(dQ, dB, dQ, dB)

        trB = np.einsum("abcb->ac", T)  # trace over bath

        trQ = np.einsum("abad->bd", T)  # trace over qubit

        tr = np.trace(h)


        h = (

            h

            - np.kron(trB / dB, np.eye(dB))

            - np.kron(np.eye(dQ), trQ / dQ)

            + tr / (dQ * dB) * np.eye(d)

        )

        h = (h + h.conj().T) / 2


    opnorm = np.max(np.abs(np.linalg.eigvalsh(h)))

    if opnorm < 1e-14:

        raise RuntimeError("Generated essentially zero local Hamiltonian.")


    return h / opnorm


def embed_qubit_bath_term(h_local, qubit, n_qubits=5, dB=3):

    """Embed h_local on qubit `qubit` and the common qutrit bath."""

    dS = 2 ** n_qubits

    T = h_local.reshape(2, dB, 2, dB)

    H = np.zeros((dS * dB, dS * dB), dtype=complex)


    for a in range(2):

        for b in range(2):

            qab = np.zeros((2, 2), dtype=complex)

            qab[a, b] = 1


            S = np.array([[1.0 + 0j]])

            for q in range(n_qubits):

                S = np.kron(S, qab if q == qubit else I2)


            H += np.kron(S, T[a, :, b, :])


    return H


def weak_local_unitary(seed, theta, interaction_only=False):

    rng = np.random.default_rng(seed)

    locals_ = []

    embedded = []


    for j in range(5):

        h = random_local_hamiltonian(rng, interaction_only=interaction_only)

        locals_.append(h)

        embedded.append(embed_qubit_bath_term(h, j))


    H = sum(embedded)

    H = (H + H.conj().T) / 2

    U = expm(-1j * theta * H)

    return U, H, np.stack(locals_)


# ============================================================
# KL-loss searches
# ============================================================

def find_one_exact_cycle1_code(
    U,
    B0,
    rng,
    exact_loss_tol,
    batches=8,
    starts=128,
    ls_max_nfev=300,
):
    """
    Return the first numerically exact cycle-1 code found by the deterministic
    seeded search.  Crucially, no future-cycle score is consulted.
    """
    errors, _ = effective_errors(U, B0, 32, 3)

    for b in range(int(batches)):
        result = search_exact_codes_raw(
            errors=errors,
            n_codes=1,
            rng=rng,
            loss_exact_tol=exact_loss_tol,
            max_attempts=min(int(starts), 16),
            max_nfev=int(ls_max_nfev),
        )
        if result.exact:
            cand = result.exact[0]
            print(
                f"    cycle-1 batch {b+1}/{batches}: "
                f"exact code found, L={cand.kl_loss:.3e}"
            )
            return {
                "V": cand.V,
                "kl_loss": float(cand.kl_loss),
            }, int(result.n_kl_products)

        best = min(
            (c.kl_loss for c in result.candidates),
            default=np.inf,
        )
        print(
            f"    cycle-1 batch {b+1}/{batches}: "
            f"no exact code; best L={best:.3e}"
        )

    raise RuntimeError(
        f"No cycle-1 code with L < {exact_loss_tol:.1e} found "
        f"within the search budget."
    )


def full_bath_search(
    U,
    rng,
    exact_loss_tol,
    batches=8,
    starts=128,
    steps=1600,
    adam_polish=4,
    direct_starts=4,
    ls_max_nfev=300,
):
    """
    Optimize the code assuming full qutrit bath support, using only the
    unnormalized physical KL loss L.  This loss is a design/search quantity;
    logical fidelity below is always evaluated by actually running the code
    from the physical rank-2 initial bath state.
    """
    errors, _ = effective_errors(U, np.eye(3, dtype=complex), 32, 3)
    best = None

    batch_best_kl_loss = []
    batch_best_route = []
    candidate_kl_loss = []
    candidate_route = []
    candidate_batch = []
    candidate_nfev = []

    route_best = {"adam_then_lsq": None, "direct_lsq": None}
    n_kl_products = len(errors) ** 2

    for b in range(int(batches)):
        cands, _ = search_approx_codes_raw(
            errors=errors,
            rng=rng,
            adam_starts=starts,
            adam_steps=steps,
            adam_polish=adam_polish,
            direct_starts=direct_starts,
            ls_max_nfev=ls_max_nfev,
        )
        if not cands:
            continue

        cand = cands[0]
        batch_best_kl_loss.append(cand["kl_loss"])
        batch_best_route.append(cand["route"])

        for c in cands:
            candidate_kl_loss.append(c["kl_loss"])
            candidate_route.append(c["route"])
            candidate_batch.append(b)
            candidate_nfev.append(c["nfev"])
            old = route_best[c["route"]]
            if old is None or c["kl_loss"] < old["kl_loss"]:
                route_best[c["route"]] = c

        if best is None or cand["kl_loss"] < best["kl_loss"]:
            best = cand

        print(
            f"    full-bath batch {b+1}/{batches}: "
            f"route={cand['route']}, L={cand['kl_loss']:.3e}"
        )

    if best is None:
        raise RuntimeError("Full-bath search produced no candidate.")
    if best["kl_loss"] < exact_loss_tol:
        raise RuntimeError(
            "A numerically exact full-bath code was found "
            f"(L={best['kl_loss']:.3e}); this instance is unsuitable for "
            "the intended Example-2 comparison."
        )

    adam_best = route_best["adam_then_lsq"]
    direct_best = route_best["direct_lsq"]
    print(
        f"    full-bath overall: route={best['route']}, "
        f"L={best['kl_loss']:.3e}"
    )

    diagnostics = {
        "batch_best_kl_loss": np.asarray(batch_best_kl_loss, dtype=float),
        "batch_best_route": np.asarray(batch_best_route, dtype="U32"),
        "candidate_kl_loss": np.asarray(candidate_kl_loss, dtype=float),
        "candidate_route": np.asarray(candidate_route, dtype="U32"),
        "candidate_batch": np.asarray(candidate_batch, dtype=int),
        "candidate_nfev": np.asarray(candidate_nfev, dtype=int),
        "best_adam_kl_loss": (
            np.nan if adam_best is None else float(adam_best["kl_loss"])
        ),
        "best_direct_kl_loss": (
            np.nan if direct_best is None else float(direct_best["kl_loss"])
        ),
    }
    return best, diagnostics, n_kl_products


def cycle2_supports(U, B0, V):
    errors0, _ = effective_errors(U, B0, 32, 3)
    syndromes, _ = canonical_syndromes(V, errors0)
    supports = conditional_bath_supports(
        U=U,
        V=V,
        bath_basis=B0,
        syndromes=syndromes,
        dS=32,
        dB=3,
        support_tol=1e-9,
    )
    return syndromes, supports


def search_one_exact_code(
    errors,
    rng,
    exact_loss_tol,
    batches=4,
    starts=128,
    ls_max_nfev=300,
):
    """Find the first continuation code with L below the exactness threshold."""
    best_loss = np.inf
    for b in range(int(batches)):
        result = search_exact_codes_raw(
            errors=errors,
            n_codes=1,
            rng=rng,
            loss_exact_tol=exact_loss_tol,
            max_attempts=min(int(starts), 16),
            max_nfev=int(ls_max_nfev),
        )
        if result.candidates:
            best_loss = min(best_loss, result.candidates[0].kl_loss)
        if result.exact:
            cand = result.exact[0]
            print(
                f"      branch search {b+1}/{batches}: "
                f"exact, L={cand.kl_loss:.3e}"
            )
            return {"V": cand.V, "kl_loss": float(cand.kl_loss)}

    raise RuntimeError(
        f"No continuation code with L < {exact_loss_tol:.1e}; "
        f"best L={best_loss:.3e}."
    )


def pad_bath_bases(supports, dB=3):
    """
    Store variable-rank bath bases without object arrays.

    Output has shape (n_supports, dB, dB); for support m, the first
    adapted_branch_ranks[m] columns are the actual orthonormal bath basis.
    Remaining columns are zero.
    """
    out = np.zeros(
        (len(supports), dB, dB),
        dtype=complex,
    )

    for m, support in enumerate(supports):
        B = support["bath_basis"]
        out[m, :, :B.shape[1]] = B

    return out


# ============================================================

# Recoveries

# ============================================================


def psd_sqrt(A, tol=1e-10):

    """

    PSD square root with tolerance for floating-point

    negative eigenvalues.


    Eigenvalues below zero by less than tol * scale are

    interpreted as numerical roundoff and clipped to zero.

    """

    A = (A + A.conj().T) / 2


    w, Q = np.linalg.eigh(A)


    scale = max(

        np.max(np.abs(w)),

        1.0,

    )


    min_w = np.min(w)


    if min_w < -tol * scale:

        raise RuntimeError(

            "Matrix is genuinely non-PSD: "

            f"min eigenvalue={min_w:.3e}, "

            f"tolerance={tol * scale:.3e}"

        )


    w = np.clip(

        w,

        0.0,

        None,

    )


    return (

        Q

        @ np.diag(np.sqrt(w))

        @ Q.conj().T

    )


def complete_recovery(recovery, dS=32):

    S = sum(R.conj().T @ R for R in recovery)

    C = np.eye(dS, dtype=complex) - S

    C = (C + C.conj().T) / 2

    D = psd_sqrt(C)

    if np.linalg.norm(D, "fro") > 1e-12:

        return recovery + [D]

    return recovery


def canonical_recovery(V_in, errors, V_outs=None):

    """

    Canonical exact-QEC recovery for an exact code.


    If V_outs[m] differs with syndrome m, the recovery simultaneously

    corrects the error and adapts into the selected next code.


    The syndrome isometries returned by canonical_syndromes have already

    undergone one global polar orthonormalization.  Using those same

    isometries here keeps the syndrome projectors used for bath-support

    propagation and the recovery Kraus operators numerically consistent.

    """

    syndromes, _ = canonical_syndromes(V_in, errors)


    if V_outs is None:

        V_outs = [V_in] * len(syndromes)

    if len(V_outs) != len(syndromes):

        raise ValueError("V_outs must match the number of syndrome subspaces.")


    recovery = []

    success = []


    for syn, V_out in zip(syndromes, V_outs):

        W = syn["W"]


        R = V_out @ W.conj().T

        recovery.append(R)

        success.append(R)


    full = complete_recovery(recovery, dS=V_in.shape[0])

    failure = full[len(success):]

    return success, failure, syndromes


def transpose_recovery(V, kraus, tol=1e-12):

    P = V @ V.conj().T

    NP = sum(E @ P @ E.conj().T for E in kraus)

    NP = (NP + NP.conj().T) / 2


    w, Q = np.linalg.eigh(NP)

    scale = max(np.max(np.abs(w)), 1e-30)

    keep = w > tol * scale


    invsqrt = (

        Q[:, keep]

        @ np.diag(1.0 / np.sqrt(w[keep]))

        @ Q[:, keep].conj().T

    )


    R = [P @ E.conj().T @ invsqrt for E in kraus]

    return complete_recovery(R, dS=V.shape[0])


# ============================================================

# Standard [[5,1,3]] code and Pauli recovery

# ============================================================


P1 = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}


def pauli(word):

    M = np.array([[1.0 + 0j]])

    for c in word:

        M = np.kron(M, P1[c])

    return M


def commute_bit(A, B, tol=1e-9):

    if np.linalg.norm(A @ B - B @ A, "fro") < tol:

        return 0

    if np.linalg.norm(A @ B + B @ A, "fro") < tol:

        return 1

    raise RuntimeError("Paulis neither commute nor anticommute.")


def five_qubit_code_and_recovery():

    G = [

        pauli("XZZXI"),

        pauli("IXZZX"),

        pauli("XIXZZ"),

        pauli("ZXIXZ"),

    ]


    I = np.eye(32, dtype=complex)

    P = I.copy()

    for g in G:

        P = P @ ((I + g) / 2)

    P = (P + P.conj().T) / 2


    w, Q = np.linalg.eigh(P)

    V0 = Q[:, w > 0.5]

    if V0.shape != (32, 2):

        raise RuntimeError("Five-qubit stabilizer projector is not rank 2.")


    ZL = pauli("ZZZZZ")

    Zr = V0.conj().T @ ZL @ V0

    z, W = np.linalg.eigh((Zr + Zr.conj().T) / 2)

    V = V0 @ W[:, np.argsort(z)[::-1]]


    err = {(0, 0, 0, 0): I}

    for q in range(5):

        for p in "XYZ":

            word = ["I"] * 5

            word[q] = p

            E = pauli("".join(word))

            s = tuple(commute_bit(E, g) for g in G)

            err[s] = E


    recovery = []

    for s in sorted(err):

        Pi = I.copy()

        for bit, g in zip(s, G):

            sign = 1 if bit == 0 else -1

            Pi = Pi @ ((I + sign * g) / 2)

        recovery.append(err[s] @ Pi)


    completeness = sum(R.conj().T @ R for R in recovery)

    if np.linalg.norm(completeness - I, "fro") > 1e-9:

        raise RuntimeError("Five-qubit recovery is not complete.")


    return V, recovery


# ============================================================

# Joint-state propagation

# ============================================================


def encode_joint_probes(V, rhoB):

    return np.stack([

        np.kron(V @ probe @ V.conj().T, rhoB)

        for probe in PROBES

    ])


def apply_noise(rhos, U):

    return np.einsum("ab,kbc,dc->kad", U, rhos, U.conj(), optimize=True)


def apply_one_system_kraus(rhos, R, dB=3):

    K = np.kron(R, np.eye(dB, dtype=complex))

    return np.einsum("ab,kbc,dc->kad", K, rhos, K.conj(), optimize=True)


def apply_recovery(rhos, recovery, dB=3):

    out = np.zeros_like(rhos)

    for R in recovery:

        out += apply_one_system_kraus(rhos, R, dB=dB)

    return out


def partial_trace_bath(X, dS=32, dB=3):

    T = X.reshape(dS, dB, dS, dB)

    return np.einsum("ibjb->ij", T)


def decode_joint_probes(rhos, V, dS=32, dB=3):

    out = []

    for X in rhos:

        rhoS = partial_trace_bath(X, dS=dS, dB=dB)

        out.append(V.conj().T @ rhoS @ V)

    return np.stack(out)


def max_event_probability(event_probes):

    """Maximum trace of a CP event over pure logical qubit inputs."""

    d0 = float(np.real(np.trace(event_probes[0])))

    d = np.array([float(np.real(np.trace(event_probes[j]))) for j in range(1, 4)])

    return max(0.0, d0 + np.linalg.norm(d))


# ============================================================

# Worst-case fidelity on the logical Bloch sphere

# ============================================================


def logical_coefficients(outputs):

    O0 = outputs[0]

    Oxyz = outputs[1:]


    alpha = float(np.real(np.trace(O0)))

    a = np.array([np.real(np.trace(O)) for O in Oxyz])

    t = np.array([np.real(np.trace(P @ O0)) for P in PAULIS])

    M = np.array([

        [np.real(np.trace(P @ Oxyz[j])) for j in range(3)]

        for P in PAULIS

    ])
    return alpha, a, t, M


def worst_case_fidelity(outputs, n_theta=91, n_phi=180, n_starts=12):

    alpha, a, t, M = logical_coefficients(outputs)

    Q = 0.5 * (M + M.T)

    linear = a + t


    theta = np.linspace(0, np.pi, n_theta)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    st = np.sin(theta)[:, None]

    ct = np.cos(theta)[:, None]

    cp = np.cos(phi)[None, :]

    sp = np.sin(phi)[None, :]


    rx = np.broadcast_to(st * cp, (n_theta, n_phi))

    ry = np.broadcast_to(st * sp, (n_theta, n_phi))

    rz = np.broadcast_to(ct, (n_theta, n_phi))

    R = np.stack([rx, ry, rz], axis=-1).reshape(-1, 3)


    vals = 0.5 * (

        alpha

        + R @ linear

        + np.einsum("ni,ij,nj->n", R, Q, R)

    )


    best_idx = np.argsort(vals)[:n_starts]


    def objective(r):

        return float(0.5 * (alpha + linear @ r + r @ Q @ r))


    constraint = {"type": "eq", "fun": lambda r: np.dot(r, r) - 1.0}

    best_f = float(vals[best_idx[0]])

    best_r = R[best_idx[0]].copy()


    for idx in best_idx:

        res = minimize(

            objective,

            R[idx],

            method="SLSQP",

            constraints=constraint,

            options={"ftol": 1e-14, "maxiter": 1000},

        )

        r = np.asarray(res.x, dtype=float)

        nr = np.linalg.norm(r)

        if nr == 0:

            continue

        r /= nr

        f = objective(r)

        if f < best_f:

            best_f, best_r = f, r


    return float(np.clip(best_f, 0.0, 1.0)), best_r


# ============================================================

# Two-cycle protocols

# ============================================================


def static_protocol_fidelities(U, rhoB0, V, recovery):

    state = encode_joint_probes(V, rhoB0)

    F = []

    Rvec = []


    for _ in range(2):

        state = apply_noise(state, U)

        state = apply_recovery(state, recovery)

        logical = decode_joint_probes(state, V)

        f, r = worst_case_fidelity(logical)

        F.append(f)

        Rvec.append(r)


    return np.asarray(F), np.asarray(Rvec)


def adaptive_protocol_fidelities(U, rhoB0, B0, V0, supports, branch_Vs):

    errors0, _ = effective_errors(U, B0, 32, 3)


    # Cycle-1 recovery corrects and simultaneously maps syndrome m into V_m.

    success_R, failure_R, syndromes = canonical_recovery(

        V0,

        errors0,

        V_outs=branch_Vs,

    )


    if len(success_R) != len(supports):

        raise RuntimeError("Syndrome/support count mismatch.")


    initial = encode_joint_probes(V0, rhoB0)

    noisy1 = apply_noise(initial, U)


    branch_states = []

    logical1 = np.zeros((4, 2, 2), dtype=complex)


    for m, (R, Vm) in enumerate(zip(success_R, branch_Vs)):

        state_m = apply_one_system_kraus(noisy1, R)

        branch_states.append(state_m)

        logical1 += decode_joint_probes(state_m, Vm)


    failure1 = np.zeros_like(noisy1)

    for R in failure_R:

        failure1 += apply_one_system_kraus(noisy1, R)


    # Count the formal completion branch as logical failure. It should have

    # numerical-zero weight for an exact cycle-1 code on the stated support.

    p_fail1_max = max_event_probability(failure1)

    if p_fail1_max > 1e-8:

        raise RuntimeError(

            f"Unexpected non-negligible adaptive cycle-1 complement branch: "

            f"p_fail <= {p_fail1_max:.3e}"

        )


    F1, r1 = worst_case_fidelity(logical1)


    # Cycle 2: each cycle-1 branch has its own incoming bath support and code.

    logical2 = np.zeros((4, 2, 2), dtype=complex)


    for m, (state_m, Vm, support) in enumerate(zip(branch_states, branch_Vs, supports)):

        errors_m, _ = effective_errors(U, support["bath_basis"], 32, 3)

        success2, failure2, _ = canonical_recovery(Vm, errors_m, V_outs=None)

        recovery2 = success2 + failure2


        out = apply_noise(state_m, U)

        out = apply_recovery(out, recovery2)

        logical2 += decode_joint_probes(out, Vm)


    # Cycle-1 failure branch is conservatively counted as zero fidelity at cycle 2.

    F2, r2 = worst_case_fidelity(logical2)


    return (

        np.asarray([F1, F2]),

        np.asarray([r1, r2]),

        float(p_fail1_max),

    )


# ============================================================
# Main production run
# ============================================================

def main(args):
    dS, dB = 32, 3

    print("=" * 78)
    print("WEAK LOCAL HAMILTONIAN: QEC SEARCH SETUP")
    print("=" * 78)

    U, H, local_terms = weak_local_unitary(
        args.h_seed,
        args.theta,
        interaction_only=args.interaction_only,
    )

    uhash = hashlib.sha256(
        np.ascontiguousarray(U).view(np.uint8)
    ).hexdigest()

    Hnorm = np.max(
        np.abs(np.linalg.eigvalsh(H))
    )
    Udist = np.linalg.norm(
        U - np.eye(dS * dB),
        2,
    )

    print(f"Hamiltonian seed     = {args.h_seed}")
    print(f"theta                = {args.theta}")
    print(f"interaction only     = {args.interaction_only}")
    print(f"||H||_op             = {Hnorm:.6f}")
    print(f"||U-I||_op           = {Udist:.6f}")
    print(f"U SHA-256            = {uhash}")

    B0 = np.eye(
        dB,
        dtype=complex,
    )[:, :2]

    rhoB0 = 0.5 * (
        B0 @ B0.conj().T
    )

    # --------------------------------------------------------
    # Full-bath optimized code
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print("FULL-BATH OPTIMIZED CODE")
    print("=" * 78)

    (
        full,
        full_search_diagnostics,
        full_n_kl_products,
    ) = full_bath_search(
        U,
        np.random.default_rng(args.full_search_seed),
        args.exact_loss_tol,
        batches=args.full_batches,
        starts=args.full_starts,
        steps=args.full_steps,
        adam_polish=args.full_adam_polish,
        direct_starts=args.full_direct_starts,
        ls_max_nfev=args.full_lsq_nfev,
    )
    full_kl_loss_history = full_search_diagnostics["batch_best_kl_loss"]
    V_full = full["V"]

    # --------------------------------------------------------
    # One exact rank-2 cycle-1 code, chosen with no look-ahead
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print("COMMON EXACT RANK-2 CYCLE-1 CODE")
    print("=" * 78)

    cycle1_code, cycle1_n_kl_products = find_one_exact_cycle1_code(
        U,
        B0,
        np.random.default_rng(args.code_search_seed),
        args.exact_loss_tol,
        batches=args.code_batches,
        starts=args.code_starts,
        ls_max_nfev=args.code_lsq_nfev,
    )
    V0 = cycle1_code["V"]
    selected_cycle1_kl_loss = float(cycle1_code["kl_loss"])

    # The adaptive and static strategies use this identical V0 in cycle 1.
    # No cycle-2 or later information enters its selection.
    syndromes0, supports = cycle2_supports(U, B0, V0)

    print(f"cycle-1 L = {selected_cycle1_kl_loss:.6e}")
    print(
        "This same V0 is used for both static reuse and the adaptive protocol."
    )

    # --------------------------------------------------------
    # One exact adapted code per cycle-2 branch
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print("ADAPTED CYCLE-2 CODES")
    print("=" * 78)

    branch_Vs = []
    branch_kl_loss = []
    branch_ranks = []
    rng_branch = np.random.default_rng(args.branch_search_seed)

    for m, support in enumerate(supports):
        errors_m, _ = effective_errors(
            U,
            support["bath_basis"],
            dS,
            dB,
        )
        cand = search_one_exact_code(
            errors_m,
            rng_branch,
            args.exact_loss_tol,
            batches=args.branch_batches,
            starts=args.branch_starts,
            ls_max_nfev=args.branch_lsq_nfev,
        )
        branch_Vs.append(cand["V"])
        branch_kl_loss.append(cand["kl_loss"])
        branch_ranks.append(support["bath_rank"])
        print(
            f"    branch {m}: rank={support['bath_rank']}, "
            f"L={cand['kl_loss']:.3e}"
        )

    branch_kl_loss = np.asarray(branch_kl_loss, dtype=float)
    branch_ranks = np.asarray(branch_ranks, dtype=int)
    branch_n_kl_products = (3 * branch_ranks) ** 2

    # --------------------------------------------------------
    # Save all search outputs needed by the five-cycle fidelity run
    # --------------------------------------------------------
    theta_tag = str(args.theta).replace(".", "p")
    prefix = (
        f"weak_local_rank2_"
        f"seed{args.h_seed}_"
        f"theta{theta_tag}"
    )
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / f"{prefix}_data.npz"

    # Fixed-shape storage of the actual conditional bath bases.
    # For branch m, keep columns :adapted_branch_ranks[m].
    adapted_branch_bath_bases = (
        pad_bath_bases(
            supports,
            dB=dB,
        )
    )

    adapted_branch_projectors = np.stack(
        [
            support["bath_basis"]
            @ support["bath_basis"].conj().T
            for support in supports
        ]
    )

    adapted_branch_Omega = np.stack(
        [
            support["Omega"]
            for support in supports
        ]
    )

    adapted_branch_bath_eigenvalues = np.stack(
        [
            support["bath_eigenvalues"]
            for support in supports
        ]
    )

    adapted_branch_syndrome_indices = np.asarray(
        [
            support["syndrome_index"]
            for support in supports
        ],
        dtype=int,
    )

    adapted_branch_alpha_eigenvalues = np.asarray(
        [
            support["alpha_eigenvalue"]
            for support in supports
        ],
        dtype=float,
    )

    np.savez_compressed(
        npz_path,

        # Physical instance
        U=U,
        H=H,
        local_terms=local_terms,
        U_sha256=uhash,
        theta=args.theta,
        h_seed=args.h_seed,
        interaction_only=args.interaction_only,
        H_operator_norm=Hnorm,
        U_minus_I_operator_norm=Udist,

        # Search settings / reproducibility
        exact_loss_tol=args.exact_loss_tol,
        full_search_seed=args.full_search_seed,
        full_batches=args.full_batches,
        full_starts=args.full_starts,
        full_steps=args.full_steps,
        full_adam_polish=args.full_adam_polish,
        full_direct_starts=args.full_direct_starts,
        full_lsq_nfev=args.full_lsq_nfev,
        code_search_seed=args.code_search_seed,
        code_batches=args.code_batches,
        code_starts=args.code_starts,
        code_lsq_nfev=args.code_lsq_nfev,
        branch_search_seed=args.branch_search_seed,
        branch_batches=args.branch_batches,
        branch_starts=args.branch_starts,
        branch_lsq_nfev=args.branch_lsq_nfev,

        # Initial bath state/support
        B0=B0,
        rhoB0=rhoB0,

        # Selected common cycle-1 code (no look-ahead)
        selected_cycle1_V=V0,
        selected_cycle1_kl_loss=selected_cycle1_kl_loss,
        cycle1_n_kl_products=cycle1_n_kl_products,

        # Selected code's cycle-1 syndromes / conditional bath supports
        adapted_branch_syndrome_indices=(
            adapted_branch_syndrome_indices
        ),
        adapted_branch_alpha_eigenvalues=(
            adapted_branch_alpha_eigenvalues
        ),
        adapted_branch_bath_bases=(
            adapted_branch_bath_bases
        ),
        adapted_branch_projectors=(
            adapted_branch_projectors
        ),
        adapted_branch_Omega=(
            adapted_branch_Omega
        ),
        adapted_branch_bath_eigenvalues=(
            adapted_branch_bath_eigenvalues
        ),
        adapted_branch_ranks=branch_ranks,

        # Adapted cycle-2 codes
        adapted_branch_Vs=np.stack(
            branch_Vs
        ),
        adapted_branch_kl_loss=branch_kl_loss,
        adapted_branch_n_kl_products=branch_n_kl_products,
        adapted_worst_kl_loss=float(np.max(branch_kl_loss)),
        adapted_all_exact=bool(np.all(branch_kl_loss < args.exact_loss_tol)),

        # Full-bath optimized code/search
        full_bath_V=V_full,
        full_bath_kl_loss=full["kl_loss"],
        full_n_kl_products=full_n_kl_products,
        full_search_kl_loss_history=(
            full_kl_loss_history
        ),

        full_search_batch_best_route=(
            full_search_diagnostics[
                "batch_best_route"
            ]
        ),
        full_search_candidate_kl_loss=(
            full_search_diagnostics[
                "candidate_kl_loss"
            ]
        ),
        full_search_candidate_route=(
            full_search_diagnostics[
                "candidate_route"
            ]
        ),
        full_search_candidate_batch=(
            full_search_diagnostics[
                "candidate_batch"
            ]
        ),
        full_search_candidate_nfev=(
            full_search_diagnostics[
                "candidate_nfev"
            ]
        ),
        full_search_best_adam_kl_loss=(
            full_search_diagnostics[
                "best_adam_kl_loss"
            ]
        ),
        full_search_best_direct_kl_loss=(
            full_search_diagnostics[
                "best_direct_kl_loss"
            ]
        ),

    )

    print("\nSaved search setup:")
    print(npz_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--h-seed",
        type=int,
        default=0,
    )
    p.add_argument(
        "--theta",
        type=float,
        default=0.2,
    )
    p.add_argument(
        "--interaction-only",
        action="store_true",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=".",
    )

    p.add_argument(
        "--exact-loss-tol",
        type=float,
        default=1e-24,
        help="Numerical exactness criterion: unnormalized KL loss L below this value.",
    )

    p.add_argument(
        "--full-search-seed",
        type=int,
        default=123456,
    )
    p.add_argument(
        "--full-batches",
        type=int,
        default=8,
    )
    p.add_argument(
        "--full-starts",
        type=int,
        default=128,
    )
    p.add_argument(
        "--full-steps",
        type=int,
        default=1600,
    )

    p.add_argument(
        "--full-adam-polish",
        type=int,
        default=4,
    )
    p.add_argument(
        "--full-direct-starts",
        type=int,
        default=4,
    )
    p.add_argument(
        "--full-lsq-nfev",
        type=int,
        default=220,
    )

    p.add_argument(
        "--code-search-seed",
        type=int,
        default=24681357,
    )
    p.add_argument(
        "--code-batches",
        type=int,
        default=8,
    )
    p.add_argument(
        "--code-starts",
        type=int,
        default=128,
    )
    p.add_argument(
        "--code-lsq-nfev",
        type=int,
        default=300,
    )

    p.add_argument(
        "--branch-search-seed",
        type=int,
        default=97531,
    )
    p.add_argument(
        "--branch-batches",
        type=int,
        default=4,
    )
    p.add_argument(
        "--branch-starts",
        type=int,
        default=128,
    )
    p.add_argument(
        "--branch-lsq-nfev",
        type=int,
        default=300,
    )

    main(
        p.parse_args()
    )
