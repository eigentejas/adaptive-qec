from dataclasses import dataclass

import numpy as np
import torch

from scipy.linalg import null_space
from scipy.optimize import least_squares


# ============================================================
# Global numerical settings
# ============================================================

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")   # complex128 is safest on CPU


# ============================================================
# Raw physical KL objective
# ============================================================

def raw_kl_products(errors):
    """
    Return all physical products K_mu^dagger K_nu.

    No normalization.
    No SVD.
    No removal of linear dependence.
    """
    errors = list(errors)

    return np.stack([
        A.conj().T @ B
        for A in errors
        for B in errors
    ])


def raw_kl_loss_and_grad(V, products):
    """
    Raw physical KL loss

        L(V) = sum_a || V^dagger A_a V
                         - Tr(V^dagger A_a V)/k I ||_F^2,

    where A_a runs over every physical K_mu^dagger K_nu product.

    Returns the loss and Euclidean gradient with respect to V.
    """
    k = V.shape[1]
    Ik = np.eye(k, dtype=complex)

    # A_a V
    AV = np.einsum(
        "nij,jb->nib",
        products,
        V,
        optimize=True,
    )

    # V^dagger A_a V
    M = np.einsum(
        "ia,nib->nab",
        V.conj(),
        AV,
        optimize=True,
    )

    tr = (
        np.trace(M, axis1=1, axis2=2)
        / k
    )

    R = (
        M
        - tr[:, None, None]
        * Ik[None, :, :]
    )

    loss = float(
        np.sum(np.abs(R) ** 2).real
    )

    # Euclidean gradient:
    #   G = sum_a [A_a V R_a^dagger + A_a^dagger V R_a].
    Rdag = R.conj().transpose(0, 2, 1)

    G1 = np.einsum(
        "nia,nab->ib",
        AV,
        Rdag,
        optimize=True,
    )

    AadjV = np.einsum(
        "nji,jb->nib",
        products.conj(),
        V,
        optimize=True,
    )

    G2 = np.einsum(
        "nia,nab->ib",
        AadjV,
        R,
        optimize=True,
    )

    grad = G1 + G2

    return loss, grad


def raw_kl_loss(V, errors):
    """
    Convenience wrapper for the summed raw physical KL loss.
    """
    products = raw_kl_products(errors)
    loss, _ = raw_kl_loss_and_grad(V, products)
    return loss


def raw_kl_losses_torch_batch(V, products):
    """
    Batched raw physical KL loss.

    V:
        shape (batch, d, k), complex128

    products:
        shape (n_products, d, d), complex128

    Returns one summed raw KL loss per candidate code.
    """
    batch, _, k = V.shape

    if products.shape[0] == 0:
        return torch.zeros(
            batch,
            dtype=torch.float64,
            device=V.device,
        )

    M = torch.einsum(
        "bdi,ade,bej->baij",
        V.conj(),
        products,
        V,
    )

    tr = (
        torch.diagonal(
            M,
            dim1=-2,
            dim2=-1,
        ).sum(dim=-1)
        / k
    )

    I = torch.eye(
        k,
        dtype=torch.complex128,
        device=V.device,
    )

    R = (
        M
        - tr[..., None, None] * I
    )

    return (
        R.abs() ** 2
    ).sum(dim=(1, 2, 3))


# ============================================================
# Basic linear algebra / Stiefel helpers
# ============================================================

def haar_unitary(n, rng):
    """
    Haar-random n x n unitary using complex Gaussian QR.
    """
    z = (
        rng.normal(size=(n, n))
        + 1j * rng.normal(size=(n, n))
    ) / np.sqrt(2.0)

    q, r = np.linalg.qr(z)

    diag = np.diag(r)
    phases = np.ones_like(diag, dtype=complex)

    mask = np.abs(diag) > 0
    phases[mask] = diag[mask] / np.abs(diag[mask])

    q = q * phases[None, :]
    return q


def random_isometry(rng, d, k=2):
    X = (
        rng.normal(size=(d, k))
        + 1j * rng.normal(size=(d, k))
    )

    Q, _ = np.linalg.qr(X)
    return Q[:, :k]


def random_isometries(n_codes, d, k, rng):
    """
    Generate n_codes random d x k complex isometries.
    """
    out = []

    for _ in range(n_codes):
        out.append(
            random_isometry(rng, d, k)
        )

    return np.stack(out, axis=0)


def polar_retract(X):
    """
    Smooth Stiefel retraction

        X -> X (X^dagger X)^(-1/2).
    """
    G = X.conj().T @ X
    G = (G + G.conj().T) / 2

    w, U = np.linalg.eigh(G)
    w = np.maximum(w, 1e-14)

    invsqrt = (
        U
        @ np.diag(1.0 / np.sqrt(w))
        @ U.conj().T
    )

    return X @ invsqrt


def stiefel_gradient(V, G):
    """
    Project Euclidean gradient onto the Stiefel tangent space.
    """
    X = V.conj().T @ G
    sym = (X + X.conj().T) / 2

    return G - V @ sym


def projector(V):
    return V @ V.conj().T


def projector_distance(V1, V2):
    """
    Chordal distance between two rank-k subspaces.

        d^2 = k - Tr(P1 P2).
    """
    k = V1.shape[1]
    P1 = projector(V1)
    P2 = projector(V2)

    overlap = np.real(np.trace(P1 @ P2))
    return np.sqrt(max(0.0, k - overlap))


# ============================================================
# Raw-loss optimizer
# ============================================================

def batch_adam_raw_kl(
    products,
    d,
    k,
    n_starts,
    rng,
    n_steps=1200,
    lr=0.03,
):
    """
    Optimize many random code starts simultaneously using the raw
    physical KL loss, with QR retraction after every Adam step.
    """
    initial = random_isometries(
        n_starts,
        d,
        k,
        rng,
    )

    x_np = np.stack(
        [initial.real, initial.imag],
        axis=-1,
    )

    x = torch.tensor(
        x_np,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )

    products_t = torch.tensor(
        products,
        dtype=torch.complex128,
        device=DEVICE,
    )

    opt = torch.optim.Adam(
        [x],
        lr=lr,
    )

    for step in range(n_steps):
        opt.zero_grad()

        V = (
            x[..., 0]
            + 1j * x[..., 1]
        )

        losses = raw_kl_losses_torch_batch(
            V,
            products_t,
        )

        # Mean only across independent candidate starts.
        # Each candidate loss itself is the SUM over all raw KL products.
        loss = losses.mean()
        loss.backward()
        opt.step()

        with torch.no_grad():
            V = (
                x[..., 0]
                + 1j * x[..., 1]
            )

            Q, _ = torch.linalg.qr(
                V,
                mode="reduced",
            )

            x[..., 0].copy_(Q.real)
            x[..., 1].copy_(Q.imag)

        # Mild learning-rate decay, retained from the previous search.
        if step == n_steps // 2:
            for group in opt.param_groups:
                group["lr"] *= 0.3

        if step == 3 * n_steps // 4:
            for group in opt.param_groups:
                group["lr"] *= 0.3

    with torch.no_grad():
        V = (
            x[..., 0]
            + 1j * x[..., 1]
        )

        losses = raw_kl_losses_torch_batch(
            V,
            products_t,
        )

    return (
        V.cpu().numpy(),
        losses.cpu().numpy(),
    )


def raw_kl_residual_vector(V, products):
    """
    Real residual vector whose squared Euclidean norm is exactly

        L(V) = sum_a || V^dagger A_a V
                       - Tr(V^dagger A_a V)/k I ||_F^2.

    No individual KL product is normalized, removed, whitened, or rescaled.
    For k=2, tracelessness gives R_11=-R_00, so we retain three complex
    entries and weight R_00 by sqrt(2).  This makes ||r||_2^2 exactly equal
    to the full Frobenius-squared KL loss, rather than merely sharing its
    zero set.
    """
    k = V.shape[1]

    AV = np.einsum(
        "nij,jb->nib",
        products,
        V,
        optimize=True,
    )
    M = np.einsum(
        "ia,nib->nab",
        V.conj(),
        AV,
        optimize=True,
    )
    tr = np.trace(M, axis1=1, axis2=2) / k
    R = M - tr[:, None, None] * np.eye(k, dtype=complex)[None, :, :]

    if k == 2:
        z = np.stack(
            [
                np.sqrt(2.0) * R[:, 0, 0],
                R[:, 0, 1],
                R[:, 1, 0],
            ],
            axis=1,
        )
    else:
        z = R.reshape(R.shape[0], -1)

    return np.concatenate([z.real.ravel(), z.imag.ravel()])


def least_squares_polish_raw_kl(
    V0,
    products,
    max_nfev=300,
    xtol=1e-14,
    ftol=1e-14,
    gtol=1e-14,
):
    """
    High-precision minimization of the physical KL loss in a local
    Grassmann chart.  The solver may use one common scalar internally to
    avoid premature small-theta termination; the returned loss is always
    evaluated in the original physical units.
    """
    V0 = np.asarray(V0, dtype=complex)
    d, k = V0.shape
    Vperp = null_space(V0.conj().T)

    expected = (d, d - k)
    if Vperp.shape != expected:
        raise RuntimeError(
            f"Unexpected orthogonal-complement shape {Vperp.shape}; "
            f"expected {expected}."
        )

    n_complex = (d - k) * k
    n_real = 2 * n_complex

    def chart_to_isometry(x):
        x = np.asarray(x, dtype=float)
        Z = (x[:n_complex] + 1j * x[n_complex:]).reshape(d - k, k)
        return polar_retract(V0 + Vperp @ Z)

    def physical_residual(x):
        return raw_kl_residual_vector(chart_to_isometry(x), products)

    x0 = np.zeros(n_real, dtype=float)
    r0 = physical_residual(x0)
    m = r0.size

    # A single common scale leaves the minimizer and zero set unchanged.
    solver_scale = np.linalg.norm(r0) / np.sqrt(max(m, 1))
    if not np.isfinite(solver_scale) or solver_scale == 0.0:
        solver_scale = 1.0

    def residual(x):
        return physical_residual(x) / solver_scale

    method = "lm" if m >= n_real else "trf"
    result = least_squares(
        residual,
        x0,
        method=method,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        max_nfev=int(max_nfev),
    )

    V = chart_to_isometry(result.x)
    loss, _ = raw_kl_loss_and_grad(V, products)

    # Guard against any accidental mismatch between the vector objective and
    # the Frobenius definition of L.
    check = float(np.dot(
        raw_kl_residual_vector(V, products),
        raw_kl_residual_vector(V, products),
    ))
    diff = abs(loss - check)
    scale = max(abs(loss), abs(check))

    # Development consistency check only.  The two expressions are
    # algebraically identical; tiny differences arise from floating-point
    # cancellation in the compressed traceless residual representation.
    if diff > 1e-26 and diff > 1e-3 * scale:
        raise RuntimeError(
            f"KL loss/residual mismatch: "
            f"loss={loss:.6e}, vector^2={check:.6e}, diff={diff:.3e}"
        )

    return V, float(loss), int(result.nfev)


def polish_raw_kl(V, products, max_steps=300, initial_step=0.2, tol=0.0):
    """Backward-compatible wrapper around the KL least-squares polish."""
    V, loss, _ = least_squares_polish_raw_kl(
        V,
        products,
        max_nfev=max_steps,
    )
    return V, loss


@dataclass
class KLCodeCandidate:
    V: np.ndarray
    kl_loss: float


@dataclass
class KLCodeSearchResult:
    candidates: list
    exact: list
    n_kl_products: int


def search_exact_codes_raw(
    errors,
    n_codes,
    rng,
    loss_exact_tol=1e-24,
    max_attempts=None,
    max_nfev=300,
    uniqueness_tol=1e-5,
):
    """
    Search directly for numerically exact codes using only

        L = sum_{mu,nu} ||Delta_{mu,nu}||_F^2.

    A code is certified numerically exact when L < loss_exact_tol.  No
    No separate maximum-residual criterion is computed or used.
    """
    errors = list(errors)
    if not errors:
        raise ValueError("errors must contain at least one operator.")

    d = errors[0].shape[0]
    k = 2
    products = raw_kl_products(errors)

    if max_attempts is None:
        max_attempts = max(4 * int(n_codes), int(n_codes) + 8)

    candidates = []
    for _ in range(int(max_attempts)):
        V0 = random_isometry(rng, d, k)
        V, loss, _ = least_squares_polish_raw_kl(
            V0,
            products,
            max_nfev=max_nfev,
        )
        if loss >= loss_exact_tol:
            continue

        P = V @ V.conj().T
        duplicate = any(
            np.linalg.norm(P - old.V @ old.V.conj().T, "fro") < uniqueness_tol
            for old in candidates
        )
        if duplicate:
            continue

        candidates.append(KLCodeCandidate(V=V, kl_loss=float(loss)))
        if len(candidates) >= int(n_codes):
            break

    # Ranking is only by the same objective L.  For n_codes=1, this is simply
    # the first exact solution found by the deterministic seeded search.
    candidates.sort(key=lambda c: c.kl_loss)
    return KLCodeSearchResult(
        candidates=candidates,
        exact=list(candidates),
        n_kl_products=len(products),
    )


def search_approx_codes_raw(
    errors,
    rng,
    adam_starts=128,
    adam_steps=1600,
    adam_polish=4,
    direct_starts=4,
    ls_max_nfev=300,
    adam_lr=3e-2,
):
    """Robust approximate-code search, ranked only by physical KL loss L."""
    errors = list(errors)
    if not errors:
        raise ValueError("errors must contain at least one operator.")

    d = errors[0].shape[0]
    k = 2
    products = raw_kl_products(errors)
    candidates = []

    if int(adam_starts) > 0:
        Vs, coarse_losses = batch_adam_raw_kl(
            products=products,
            d=d,
            k=k,
            n_starts=int(adam_starts),
            rng=rng,
            n_steps=int(adam_steps),
            lr=adam_lr,
        )
        order = np.argsort(coarse_losses)
        for idx in order[:min(int(adam_polish), len(order))]:
            idx = int(idx)
            V, loss, nfev = least_squares_polish_raw_kl(
                Vs[idx],
                products,
                max_nfev=int(ls_max_nfev),
            )
            candidates.append(
                {
                    "V": V,
                    "kl_loss": float(loss),
                    "route": "adam_then_lsq",
                    "nfev": int(nfev),
                    "coarse_kl_loss": float(coarse_losses[idx]),
                }
            )

    for _ in range(int(direct_starts)):
        V0 = random_isometry(rng, d, k)
        V, loss, nfev = least_squares_polish_raw_kl(
            V0,
            products,
            max_nfev=int(ls_max_nfev),
        )
        candidates.append(
            {
                "V": V,
                "kl_loss": float(loss),
                "route": "direct_lsq",
                "nfev": int(nfev),
                "coarse_kl_loss": np.nan,
            }
        )

    candidates.sort(key=lambda c: c["kl_loss"])
    return candidates, len(products)


def search_codes_raw(
    errors,
    n_starts,
    rng,
    adam_steps=2000,
    n_polish=16,
    loss_exact_tol=0.0,
    polish_if_below=0.0,
    adam_lr=3e-2,
    polish_steps=300,
    max_least_squares_candidates=None,
):
    """Legacy Adam+LSQ search, made fully L-based for compatibility."""
    errors = list(errors)
    if not errors:
        raise ValueError("errors must contain at least one operator.")

    d = errors[0].shape[0]
    k = 2
    products = raw_kl_products(errors)
    Vs, coarse_losses = batch_adam_raw_kl(
        products=products,
        d=d,
        k=k,
        n_starts=n_starts,
        rng=rng,
        n_steps=adam_steps,
        lr=adam_lr,
    )
    order = np.argsort(coarse_losses)
    requested = min(n_polish, n_starts)
    if max_least_squares_candidates is None:
        n_lsq = min(requested, 40) if requested >= 40 else min(requested, 1)
    else:
        n_lsq = min(requested, int(max_least_squares_candidates))

    selected = list(order[:n_lsq])
    polished = []
    for i in selected:
        V, loss, _ = least_squares_polish_raw_kl(
            Vs[int(i)],
            products,
            max_nfev=polish_steps,
        )
        polished.append(KLCodeCandidate(V=V, kl_loss=float(loss)))

    polished.sort(key=lambda c: c.kl_loss)
    exact = [c for c in polished if c.kl_loss < loss_exact_tol]
    return KLCodeSearchResult(
        candidates=polished,
        exact=exact,
        n_kl_products=len(products),
    )


# ============================================================
# From joint U_SB to effective system errors
# ============================================================

def effective_errors(U, bath_basis, dS, dB):
    """
    For a bath input subspace with orthonormal basis vectors beta_a,
    construct

        A_{k,a} = <k|_B U |beta_a>_B

    as operators on S.

    Basis convention:
        joint basis index = s * dB + b

    so U reshapes as

        U[s_out, b_out, s_in, b_in].
    """
    bath_basis = np.asarray(
        bath_basis,
        dtype=complex,
    )

    r = bath_basis.shape[1]

    gram = bath_basis.conj().T @ bath_basis
    if not np.allclose(
        gram,
        np.eye(r),
        atol=1e-10,
    ):
        raise ValueError(
            "bath_basis must have orthonormal columns."
        )

    U4 = U.reshape(
        dS,
        dB,
        dS,
        dB,
    )

    errors = []
    labels = []

    for a in range(r):
        beta = bath_basis[:, a]

        for k in range(dB):
            A = np.tensordot(
                U4[:, k, :, :],
                beta,
                axes=([2], [0]),
            )

            errors.append(A)
            labels.append((k, a))

    return errors, labels


# ============================================================
# Pick geometrically diverse exact codes
# ============================================================

def diverse_subset(candidates, n_keep):
    """
    Farthest-point sampling in Grassmann/projector distance.
    """
    if len(candidates) <= n_keep:
        return candidates

    selected = [candidates[0]]
    remaining = candidates[1:].copy()

    while len(selected) < n_keep:
        best_index = None
        best_distance = -1.0

        for i, cand in enumerate(remaining):
            dmin = min(
                projector_distance(
                    cand.V,
                    s.V,
                )
                for s in selected
            )

            if dmin > best_distance:
                best_distance = dmin
                best_index = i

        selected.append(
            remaining.pop(best_index)
        )

    return selected


# ============================================================
# Canonical KL syndrome decomposition
# ============================================================

def canonical_syndromes(
    V,
    errors,
    eig_tol=1e-10,
):
    """
    Construct the KL matrix

        alpha_{mu,nu}
        = (1/k) Tr[V^dagger A_mu^dagger A_nu V],

    diagonalize it, and construct the canonical syndrome subspaces.

    For a numerically exact code the normalized syndrome isometries are
    mutually orthogonal analytically.  At very small interaction strengths,
    tiny absolute KL errors can be amplified by division by small syndrome
    eigenvalues.  We therefore apply a symmetric global polar cleanup to the
    concatenated syndrome isometries.  This is the identity in exact
    arithmetic and enforces mutual orthogonality at machine precision.
    """
    n = len(errors)
    k = V.shape[1]

    alpha = np.zeros(
        (n, n),
        dtype=complex,
    )

    for mu in range(n):
        for nu in range(n):
            M = (
                V.conj().T
                @ errors[mu].conj().T
                @ errors[nu]
                @ V
            )

            alpha[mu, nu] = (
                np.trace(M) / k
            )

    alpha = (
        alpha + alpha.conj().T
    ) / 2.0

    evals, evecs = np.linalg.eigh(alpha)

    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    max_eval = max(
        1.0,
        np.max(np.abs(evals)),
    )

    raw = []

    for r, lam in enumerate(evals):
        if lam <= eig_tol * max_eval:
            continue

        coeffs = evecs[:, r]

        B = np.zeros_like(
            errors[0]
        )

        for mu, c in enumerate(coeffs):
            B += c * errors[mu]

        # On an exact code, (B V)^dagger(B V) = lam I.
        # The first polar cleanup preserves the logical column
        # correspondence within this syndrome.
        W = B @ V / np.sqrt(lam)
        W = polar_retract(W)

        raw.append(
            {
                "alpha_eigenvalue": float(
                    np.real(lam)
                ),
                "B": B,
                "W": W,
            }
        )

    if not raw:
        return [], alpha

    ncols = len(raw) * k
    if ncols > V.shape[0]:
        raise RuntimeError(
            "Canonical syndrome spaces contain more columns than the "
            "physical system dimension; exact orthogonal syndromes are "
            "therefore impossible."
        )

    # Symmetric (Lowdin) orthonormalization of all syndrome columns at once.
    # In exact arithmetic Wcat^dagger Wcat = I, so this changes nothing.
    Wcat = np.concatenate(
        [item["W"] for item in raw],
        axis=1,
    )
    Wcat = polar_retract(Wcat)

    gram_err = np.linalg.norm(
        Wcat.conj().T @ Wcat
        - np.eye(ncols, dtype=complex),
        "fro",
    )
    if gram_err > 1e-10:
        raise RuntimeError(
            "Global syndrome orthonormalization failed: "
            f"||W^dagger W-I||_F={gram_err:.3e}"
        )

    syndromes = []

    for m, item in enumerate(raw):
        W = Wcat[:, m * k:(m + 1) * k]
        Pi = W @ W.conj().T

        syndromes.append(
            {
                "index": m,
                "alpha_eigenvalue": item["alpha_eigenvalue"],
                "B": item["B"],
                "W": W,
                "Pi": Pi,
            }
        )

    return syndromes, alpha


# ============================================================
# Partial trace and conditional bath supports
# ============================================================

def partial_trace_system(X, dS, dB):
    """
    X acts on S tensor B, with joint index s*dB+b.
    """
    T = X.reshape(
        dS,
        dB,
        dS,
        dB,
    )

    return np.einsum(
        "ibic->bc",
        T,
    )


def conditional_bath_supports(
    U,
    V,
    bath_basis,
    syndromes,
    dS,
    dB,
    support_tol=1e-9,
):
    """
    Implements the bath-support update

        Omega_m = Tr_S[(Pi_m tensor I)
                       U(P tensor Q)U^dagger
                       (Pi_m tensor I)],

        H_{B,m} = supp(Omega_m).
    """
    P = projector(V)

    Qb = (
        bath_basis
        @ bath_basis.conj().T
    )

    incoming_support = np.kron(
        P,
        Qb,
    )

    after_noise = (
        U
        @ incoming_support
        @ U.conj().T
    )

    results = []

    IB = np.eye(
        dB,
        dtype=complex,
    )

    for syn in syndromes:
        Pi = syn["Pi"]

        K = np.kron(
            Pi,
            IB,
        )

        branch = (
            K
            @ after_noise
            @ K
        )

        Omega = partial_trace_system(
            branch,
            dS,
            dB,
        )

        Omega = (
            Omega + Omega.conj().T
        ) / 2.0

        evals, evecs = np.linalg.eigh(
            Omega
        )

        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        largest = max(
            np.max(np.abs(evals)),
            1e-30,
        )

        keep = (
            evals
            > support_tol * largest
        )

        basis = evecs[:, keep]

        results.append(
            {
                "syndrome_index":
                    syn["index"],
                "alpha_eigenvalue":
                    syn["alpha_eigenvalue"],
                "Omega":
                    Omega,
                "bath_eigenvalues":
                    evals,
                "bath_basis":
                    basis,
                "bath_rank":
                    basis.shape[1],
            }
        )

    return results


