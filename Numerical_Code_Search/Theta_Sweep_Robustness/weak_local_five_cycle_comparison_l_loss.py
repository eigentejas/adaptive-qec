import argparse

from pathlib import Path



import numpy as np

import pandas as pd



from adaptive_qec_l_loss import (

    effective_errors,

    raw_kl_loss,

    search_exact_codes_raw,

    canonical_syndromes,

    conditional_bath_supports,

)



from weak_local_qec_setup_l_loss import (

    canonical_recovery,

    transpose_recovery,

    five_qubit_code_and_recovery,

    encode_joint_probes,

    apply_noise,

    apply_one_system_kraus,

    apply_recovery,

    decode_joint_probes,

    max_event_probability,

    worst_case_fidelity,

)





def projector(B):

    return B @ B.conj().T





class CodeCache:

    def __init__(self, tol=1e-7):

        self.tol = tol

        self.items = []



    def find(self, B):

        P = projector(B)



        for i, x in enumerate(self.items):

            if x["B"].shape[1] != B.shape[1]:

                continue



            if (

                np.linalg.norm(

                    P - x["P"],

                    "fro",

                )

                < self.tol

            ):

                return i



        return None



    def add(

        self,

        B,

        V,

        source,

        kl_loss=np.nan,

    ):

        i = self.find(B)



        if i is not None:

            return i



        self.items.append({

            "B": np.asarray(B, dtype=complex),

            "P": projector(B),

            "V": np.asarray(V, dtype=complex),

            "source": source,

            "kl_loss": float(kl_loss),

        })



        return len(self.items) - 1





def search_exact(
    errors,
    rng,
    exact_loss_tol,
    stages,
):
    """Direct exact-code search using only the unnormalized KL loss L."""
    best_loss = np.inf

    for j, (starts, steps) in enumerate(stages, start=1):
        max_nfev = max(160, steps // 5)
        result = search_exact_codes_raw(
            errors=errors,
            n_codes=1,
            rng=rng,
            loss_exact_tol=exact_loss_tol,
            max_attempts=starts,
            max_nfev=max_nfev,
            uniqueness_tol=1e-7,
        )

        if result.candidates:
            best_loss = min(best_loss, result.candidates[0].kl_loss)

        if result.exact:
            cand = result.exact[0]
            print(
                f"        exact-search stage {j}: starts<={starts}, "
                f"max_nfev={max_nfev}, L={cand.kl_loss:.3e}"
            )
            return cand.V, float(cand.kl_loss), j

        print(
            f"        exact-search stage {j}: starts<={starts}, "
            f"max_nfev={max_nfev}, no exact code; best L={best_loss:.3e}"
        )

    raise RuntimeError(
        f"No exact code found for a reachable bath support; best L={best_loss:.3e}."
    )


def get_child_code(

    U,

    B,

    cache,

    rng,

    exact_loss_tol,

    stages,

):

    i = cache.find(B)



    if i is not None:

        return i, False



    errors, _ = effective_errors(

        U,

        B,

        32,

        3,

    )



    V, kl_loss, stage = (

        search_exact(

            errors,

            rng,

            exact_loss_tol,

            stages,

        )

    )



    i = cache.add(

        B,

        V,

        "searched",

        kl_loss,

    )



    print(

        f"        new support #{i}: "

        f"rank={B.shape[1]}, "

        f"L={kl_loss:.3e}, "

        f"stage={stage}"

    )



    return i, True





def static_history(

    U,

    rhoB0,

    V,

    recovery,

    n_cycles,

):

    state = encode_joint_probes(

        V,

        rhoB0,

    )



    Fs = []

    Rs = []



    for n in range(

        1,

        n_cycles + 1,

    ):



        state = apply_noise(

            state,

            U,

        )



        state = apply_recovery(

            state,

            recovery,

        )



        logical = decode_joint_probes(

            state,

            V,

        )



        f, r = worst_case_fidelity(

            logical

        )



        Fs.append(f)

        Rs.append(r)



        print(

            f"      cycle {n}: "

            f"F_wc={f:.12f}, "

            f"1-F_wc={1-f:.6e}"

        )



    return (

        np.asarray(Fs),

        np.asarray(Rs),

    )





def adaptive_history(

    U,

    rhoB0,

    B0,

    V0,

    first_supports,

    first_Vs,

    n_cycles,

    exact_loss_tol,

    support_tol,

    search_seed,

    stages,

    failure_tol,

):

    rng = np.random.default_rng(

        search_seed

    )



    cache = CodeCache(

        support_tol

    )



    root_errors, _ = effective_errors(

        U,

        B0,

        32,

        3,

    )



    root_kl_loss = raw_kl_loss(V0, root_errors)



    if root_kl_loss >= exact_loss_tol:

        raise RuntimeError(

            "Root code is not exact: "

            f"{root_kl_loss:.3e}"

        )



    root = cache.add(

        B0,

        V0,

        "root",

        root_kl_loss,

    )



    if (

        len(first_supports)

        != len(first_Vs)

    ):

        raise RuntimeError(

            "Saved cycle-2 codes do not "

            "match cycle-1 supports."

        )



    # Preload the cycle-2 codes already found

    # in the previous production run.

    for m, (support, V) in enumerate(

        zip(first_supports, first_Vs)

    ):



        errors, _ = effective_errors(

            U,

            support["bath_basis"],

            32,

            3,

        )



        kl_loss = raw_kl_loss(V, errors)



        if kl_loss >= exact_loss_tol:

            raise RuntimeError(

                f"Saved branch code {m} "

                f"is not exact: "

                f"{kl_loss:.3e}"

            )



        cache.add(

            support["bath_basis"],

            V,

            "saved-cycle2",

            kl_loss,

        )



    # Each node contains the unnormalized

    # branch action on the four logical probes.

    nodes = {

        root:

            encode_joint_probes(

                V0,

                rhoB0,

            )

    }



    Fs = []

    Rs = []



    node_counts = []

    cache_sizes = []

    worst_kl_losses = []

    fail_probs = []

    # Directed support graph.  An edge recorded at cycle n connects
    # an incoming support/code used in cycle n to the conditional
    # support/code that would enter cycle n+1.
    edge_parent = []
    edge_child = []
    edge_cycle = []
    edge_syndrome = []

    # Save the incoming branch states at the start of the final cycle.
    # A later continuation run can resume from this frontier, re-process
    # only that final cycle, and then continue to cycles n+1, n+2, ... .
    frontier_cycle = int(n_cycles)
    frontier_node_ids = None
    frontier_states = None



    for cycle in range(

        1,

        n_cycles + 1,

    ):



        print(

            "\n"

            + "-" * 72

        )



        print(

            f"ADAPTIVE CYCLE {cycle}: "

            f"{len(nodes)} incoming "

            f"distinct supports"

        )



        print(

            "-" * 72

        )



        # Verify every incoming code on its

        # corresponding physical bath support.

        worst_kl_loss = 0.0



        for i in nodes:



            item = cache.items[i]



            errors, _ = effective_errors(

                U,

                item["B"],

                32,

                3,

            )



            kl_loss = raw_kl_loss(item["V"], errors)



            worst_kl_loss = max(

                worst_kl_loss,

                kl_loss,

            )

            if kl_loss >= exact_loss_tol:

                raise RuntimeError(

                    f"Incoming code on support {i} "

                    f"has KL loss {kl_loss:.3e}"

                )



        logical = np.zeros(

            (4, 2, 2),

            dtype=complex,

        )



        max_fail = 0.0



        # On the last cycle we do not need a code for cycle n+1 in this
        # run.  Preserve the complete incoming frontier first so a later
        # continuation can resume here without replaying cycles 1..n-1.
        if cycle == n_cycles:

            frontier_node_ids = np.asarray(
                list(nodes.keys()),
                dtype=int,
            )

            frontier_states = np.stack(
                [nodes[i] for i in frontier_node_ids]
            )



            for i, state in (

                nodes.items()

            ):



                item = cache.items[i]



                errors, _ = (

                    effective_errors(

                        U,

                        item["B"],

                        32,

                        3,

                    )

                )



                success, failure, _ = (

                    canonical_recovery(

                        item["V"],

                        errors,

                    )

                )



                noisy = apply_noise(

                    state,

                    U,

                )



                fail = np.zeros_like(

                    noisy

                )



                for R in failure:

                    fail += (

                        apply_one_system_kraus(

                            noisy,

                            R,

                        )

                    )



                max_fail = max(

                    max_fail,

                    max_event_probability(

                        fail

                    ),

                )



                out = apply_recovery(

                    noisy,

                    success + failure,

                )



                logical += (

                    decode_joint_probes(

                        out,

                        item["V"],

                    )

                )



            f, r = worst_case_fidelity(

                logical

            )



            Fs.append(f)

            Rs.append(r)



            node_counts.append(

                len(nodes)

            )



            cache_sizes.append(

                len(cache.items)

            )



            worst_kl_losses.append(

                worst_kl_loss

            )



            fail_probs.append(

                max_fail

            )



            print(

                "    worst KL loss = "

                f"{worst_kl_loss:.3e}"

            )



            print(

                "    max completion prob <= "

                f"{max_fail:.3e}"

            )



            print(

                f"F_wc={f:.12f}, "

                f"1-F_wc={1-f:.6e}"

            )



            break



        new_nodes = {}

        new_searches = 0



        for i, state in (

            nodes.items()

        ):



            item = cache.items[i]



            B = item["B"]

            V = item["V"]
            errors, _ = effective_errors(

                U,

                B,

                32,

                3,

            )



            syndromes, _ = (

                canonical_syndromes(

                    V,

                    errors,

                )

            )



            supports = (

                conditional_bath_supports(

                    U=U,

                    V=V,

                    bath_basis=B,

                    syndromes=syndromes,

                    dS=32,

                    dB=3,

                    support_tol=1e-9,

                )

            )



            child_ids = []

            child_Vs = []



            for support in supports:



                child, was_new = (

                    get_child_code(

                        U,

                        support[

                            "bath_basis"

                        ],

                        cache,

                        rng,

                        exact_loss_tol,

                        stages,

                    )

                )



                edge_parent.append(i)
                edge_child.append(child)
                edge_cycle.append(cycle)
                edge_syndrome.append(
                    int(support["syndrome_index"])
                )

                child_ids.append(

                    child

                )



                child_Vs.append(

                    cache.items[

                        child

                    ]["V"]

                )



                new_searches += int(

                    was_new

                )



            success, failure, _ = (

                canonical_recovery(

                    V,

                    errors,

                    V_outs=child_Vs,

                )

            )



            noisy = apply_noise(

                state,

                U,

            )



            fail = np.zeros_like(

                noisy

            )



            for R in failure:

                fail += (

                    apply_one_system_kraus(

                        noisy,

                        R,

                    )

                )



            max_fail = max(

                max_fail,

                max_event_probability(

                    fail

                ),

            )



            for R, child in zip(

                success,

                child_ids,

            ):



                child_state = (

                    apply_one_system_kraus(

                        noisy,

                        R,

                    )

                )



                child_V = (

                    cache.items[

                        child

                    ]["V"]

                )



                logical += (

                    decode_joint_probes(

                        child_state,

                        child_V,

                    )

                )



                if child in new_nodes:

                    new_nodes[

                        child

                    ] += child_state

                else:

                    new_nodes[

                        child

                    ] = child_state



        if max_fail > failure_tol:

            raise RuntimeError(

                f"Cycle {cycle}: "

                "completion branch too large: "

                f"{max_fail:.3e}"

            )



        f, r = worst_case_fidelity(

            logical

        )



        Fs.append(f)

        Rs.append(r)



        node_counts.append(

            len(nodes)

        )



        cache_sizes.append(

            len(cache.items)

        )



        worst_kl_losses.append(

            worst_kl_loss

        )



        fail_probs.append(

            max_fail

        )



        print(

            "    newly searched supports = "

            f"{new_searches}"

        )



        print(

            "    outgoing supports       = "

            f"{len(new_nodes)}"

        )



        print(

            "    total cached supports   = "

            f"{len(cache.items)}"

        )



        print(

            "    worst KL loss         = "

            f"{worst_kl_loss:.3e}"

        )



        print(

            "    max completion prob <= "

            f"{max_fail:.3e}"

        )



        print(

            f"F_wc={f:.12f}, "

            f"1-F_wc={1-f:.6e}"

        )



        nodes = new_nodes



    # Export the complete cache in fixed-shape arrays.  The actual
    # bath basis used for each cached support is retained because the
    # KL loss is evaluated on that actual physical error set.
    cache_ranks = np.asarray(
        [item["B"].shape[1] for item in cache.items],
        dtype=int,
    )

    cache_B_padded = np.zeros(
        (len(cache.items), 3, 3),
        dtype=complex,
    )

    for j, item in enumerate(cache.items):
        r = item["B"].shape[1]
        cache_B_padded[j, :, :r] = item["B"]

    cache_projectors = np.stack(
        [item["P"] for item in cache.items]
    )
    cache_Vs = np.stack(
        [item["V"] for item in cache.items]
    )
    cache_kl_loss = np.asarray(
        [item["kl_loss"] for item in cache.items],
        dtype=float,
    )
    cache_sources = np.asarray(
        [item["source"] for item in cache.items],
        dtype="U32",
    )

    return {
        "F": np.asarray(Fs),
        "R": np.asarray(Rs),
        "node_counts": np.asarray(node_counts),
        "cache_sizes": np.asarray(cache_sizes),
        "worst_kl_loss": np.asarray(worst_kl_losses),
        "fail_prob": np.asarray(fail_probs),

        "cache_B_padded": cache_B_padded,
        "cache_projectors": cache_projectors,
        "cache_Vs": cache_Vs,
        "cache_kl_loss": cache_kl_loss,
        "cache_ranks": cache_ranks,
        "cache_sources": cache_sources,

        "edge_parent": np.asarray(edge_parent, dtype=int),
        "edge_child": np.asarray(edge_child, dtype=int),
        "edge_cycle": np.asarray(edge_cycle, dtype=int),
        "edge_syndrome": np.asarray(edge_syndrome, dtype=int),

        "frontier_cycle": frontier_cycle,
        "frontier_node_ids": frontier_node_ids,
        "frontier_states": frontier_states,
    }



def main(args):



    path = Path(

        args.data

    )



    dat = np.load(

        path,

        allow_pickle=True,

    )



    U = dat["U"]

    B0 = dat["B0"]

    rhoB0 = dat["rhoB0"]



    V0 = dat[

        "selected_cycle1_V"

    ]



    first_Vs = dat[

        "adapted_branch_Vs"

    ]



    V_full = dat[

        "full_bath_V"

    ]



    print(

        "=" * 78

    )



    print(

        "FIVE-CYCLE FOUR-STRATEGY "

        "PRODUCTION RUN"

    )



    print(

        "=" * 78

    )



    print(

        f"data  = {path}"

    )



    print(

        f"seed  = "

        f"{int(dat['h_seed'])}"

    )



    print(

        f"theta = "

        f"{float(dat['theta'])}"

    )



    errors0, _ = effective_errors(
        U,
        B0,
        32,
        3,
    )

    # Use the exact conditional bath bases saved by the two-cycle
    # production run whenever available.  This preserves the actual
    # physical bath supports saved by the two-cycle search.
    if (
        "adapted_branch_bath_bases" in dat.files
        and "adapted_branch_ranks" in dat.files
    ):
        padded = dat["adapted_branch_bath_bases"]
        ranks = dat["adapted_branch_ranks"].astype(int)

        first_supports = [
            {
                "bath_basis": padded[m, :, :r],
                "bath_rank": int(r),
                "syndrome_index": (
                    int(dat["adapted_branch_syndrome_indices"][m])
                    if "adapted_branch_syndrome_indices" in dat.files
                    else m
                ),
            }
            for m, r in enumerate(ranks)
        ]
    else:
        syndromes0, _ = canonical_syndromes(
            V0,
            errors0,
        )
        first_supports = conditional_bath_supports(
            U=U,
            V=V0,
            bath_basis=B0,
            syndromes=syndromes0,
            dS=32,
            dB=3,
            support_tol=1e-9,
        )

    # Static exact-cycle-1 strategy:

    # same code AND same recovery forever.

    success, failure, _ = (

        canonical_recovery(

            V0,

            errors0,

        )

    )



    R_static = (

        success + failure

    )



    # Full-bath optimized:

    # same Petz recovery forever.

    full_errors, _ = (

        effective_errors(

            U,

            np.eye(

                3,

                dtype=complex,

            ),

            32,

            3,

        )

    )



    R_full = transpose_recovery(

        V_full,

        [

            E / np.sqrt(3)

            for E in full_errors

        ],

    )



    # Standard five-qubit code.

    V_stab, R_stab = (

        five_qubit_code_and_recovery()

    )



    print(

        "\nSTATIC REUSE"

    )



    F_static, R_static_wc = (

        static_history(

            U,

            rhoB0,

            V0,

            R_static,

            args.cycles,

        )

    )



    print(

        "\nFULL-BATH OPTIMIZED"

    )



    F_full, R_full_wc = (

        static_history(

            U,

            rhoB0,

            V_full,

            R_full,

            args.cycles,

        )

    )



    print(

        "\n[[5,1,3]] STABILIZER"

    )



    F_stab, R_stab_wc = (

        static_history(

            U,

            rhoB0,

            V_stab,

            R_stab,

            args.cycles,

        )

    )



    stages = [

        (args.stage1_starts, args.stage1_steps),

        (args.stage2_starts, args.stage2_steps),

        (args.stage3_starts, args.stage3_steps),

    ]



    adapt = adaptive_history(

        U,

        rhoB0,

        B0,

        V0,

        first_supports,

        first_Vs,

        args.cycles,

        args.exact_loss_tol,

        args.support_projector_tol,

        args.adaptive_search_seed,

        stages,

        args.failure_tol,

    )



    datasets = [

        (

            "syndrome_adapted",

            "Syndrome-adapted",

            adapt["F"],

            adapt["R"],

        ),

        (

            "static_reuse",

            "Exact cycle-1 "

            "(static reuse)",

            F_static,

            R_static_wc,

        ),

        (

            "full_bath_optimized",

            "Full-bath optimized",

            F_full,

            R_full_wc,

        ),

        (

            "five_qubit_stabilizer",

            "[[5,1,3]] stabilizer",

            F_stab,

            R_stab_wc,

        ),

    ]



    rows = []



    adaptive_exact = (

        adapt["worst_kl_loss"]

        < args.exact_loss_tol

    )



    for key, label, F, R in datasets:



        for j in range(

            args.cycles

        ):



            inf = max(

                0.0,

                1.0

                - float(F[j]),

            )



            exact = (

                (

                    key

                    == "syndrome_adapted"

                    and adaptive_exact[j]

                )

                or (

                    key

                    == "static_reuse"

                    and j == 0

                )

            )



            rows.append(

                {

                    "strategy":

                        key,



                    "label":

                        label,



                    "cycle":

                        j + 1,



                    "F_wc":

                        float(F[j]),



                    "infidelity":

                        inf,



                    "fidelity_loss":

                        inf,



                    "exact":

                        bool(exact),



                    "worst_rx":

                        float(

                            R[j, 0]

                        ),



                    "worst_ry":
                        float(

                            R[j, 1]

                        ),



                    "worst_rz":

                        float(

                            R[j, 2]

                        ),

                }

            )



    df = pd.DataFrame(

        rows

    )



    prefix = (

        path.stem.replace(

            "_data",

            "",

        )

    )



    csv = (

        path.parent

        / (

            f"{prefix}"

            "_five_cycle_four_strategy.csv"

        )

    )





    tree = (

        path.parent

        / (

            f"{prefix}"

            "_five_cycle_adaptive_tree.npz"

        )

    )



    df.to_csv(

        csv,

        index=False,

    )



    np.savez_compressed(
        tree,

        # Physical instance / provenance
        U=U,
        B0=B0,
        rhoB0=rhoB0,
        theta=float(dat["theta"]),
        h_seed=int(dat["h_seed"]),
        U_sha256=(
            dat["U_sha256"]
            if "U_sha256" in dat.files
            else ""
        ),
        input_two_cycle_file=str(path),

        # Codes needed to reconstruct all four protocols
        selected_cycle1_V=V0,
        adapted_cycle2_Vs=first_Vs,
        full_bath_V=V_full,
        stabilizer_V=V_stab,

        # Five-cycle logical performance
        F_adapt=adapt["F"],
        F_static=F_static,
        F_full=F_full,
        F_stab=F_stab,
        worst_r_adapt=adapt["R"],
        worst_r_static=R_static_wc,
        worst_r_full=R_full_wc,
        worst_r_stab=R_stab_wc,

        # Adaptive diagnostics by cycle
        adaptive_node_counts=adapt["node_counts"],
        adaptive_cache_sizes=adapt["cache_sizes"],
        adaptive_worst_kl_loss=adapt["worst_kl_loss"],
        adaptive_failure_prob=adapt["fail_prob"],
        adaptive_exact=(
            adapt["worst_kl_loss"] < args.exact_loss_tol
        ),

        # Complete cached support/code set
        cache_B_padded=adapt["cache_B_padded"],
        cache_projectors=adapt["cache_projectors"],
        cache_Vs=adapt["cache_Vs"],
        cache_kl_loss=adapt["cache_kl_loss"],
        cache_ranks=adapt["cache_ranks"],
        cache_sources=adapt["cache_sources"],

        # Directed adaptive support graph
        edge_parent=adapt["edge_parent"],
        edge_child=adapt["edge_child"],
        edge_cycle=adapt["edge_cycle"],
        edge_syndrome=adapt["edge_syndrome"],

        # Continuation frontier: incoming branch states at the start of
        # cycle n_cycles.  To extend to later cycles, re-process this cycle
        # using the cached code/support set and then continue forward.
        frontier_cycle=adapt["frontier_cycle"],
        frontier_node_ids=adapt["frontier_node_ids"],
        frontier_states=adapt["frontier_states"],

        # Run settings / reproducibility
        n_cycles=args.cycles,
        exact_loss_tol=args.exact_loss_tol,
        support_projector_tol=args.support_projector_tol,
        failure_tol=args.failure_tol,
        adaptive_search_seed=args.adaptive_search_seed,
        search_stages=np.asarray(stages, dtype=int),
    )


    print(

        "\n"

        + "=" * 78

    )



    print(

        "FINAL RESULTS"

    )



    print(

        "=" * 78

    )



    print(

        df[

            [

                "strategy",

                "cycle",

                "F_wc",

                "infidelity",

                "exact",

            ]

        ].to_string(

            index=False

        )

    )



    print(

        "Adaptive incoming supports:",

        adapt["node_counts"],

    )



    print(

        "Adaptive cache sizes:",

        adapt["cache_sizes"],

    )



    print(

        "Adaptive worst KL loss:",

        adapt["worst_kl_loss"],

    )



    print(

        f"saved: {csv}"

    )



    print(

        f"saved: {tree}"

    )





if __name__ == "__main__":



    p = argparse.ArgumentParser()



    p.add_argument(

        "--data",

        default=(

            "weak_local_rank2_seed0_theta0p2"

            "_data.npz"

        ),

    )



    p.add_argument(

        "--cycles",

        type=int,

        default=5,

    )



    p.add_argument(

        "--exact-loss-tol",

        type=float,

        default=1e-24,

        help="Numerical exactness criterion: unnormalized KL loss L below this value.",

    )



    p.add_argument(

        "--support-projector-tol",

        type=float,

        default=1e-7,

    )



    p.add_argument(

        "--failure-tol",

        type=float,

        default=1e-8,

    )



    p.add_argument(

        "--adaptive-search-seed",

        type=int,

        default=314159,

    )





    # Progressive positive-search budgets.

    p.add_argument(

        "--stage1-starts",

        type=int,

        default=8,

    )



    p.add_argument(

        "--stage1-steps",

        type=int,

        default=900,

    )





    p.add_argument(

        "--stage2-starts",

        type=int,

        default=24,

    )



    p.add_argument(

        "--stage2-steps",

        type=int,

        default=1400,

    )





    p.add_argument(

        "--stage3-starts",

        type=int,

        default=64,

    )



    p.add_argument(

        "--stage3-steps",

        type=int,

        default=2000,

    )





    main(

        p.parse_args()

    )
