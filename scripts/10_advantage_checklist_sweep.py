"""Script 10 - The screening checklist evaluated on the catalog of 109 problems and code families.

Section (A) prints the checklist: gates C0-C8, each with the kind of evidence that settles it.
Section (B) evaluates the gates on 109 named problems, application areas, code families and DQI
variants.  Numeric gates (C1 density window, C2 distance bound, C5 Prange screen, C6 regime,
C7 list-recovery vacuity) are computed from each row's parameters (alpha, lambda, rho, q, gamma);
the other gates carry a short justification (an arXiv identifier or an analytic argument).
Rows that declare an average-case reach also print a D4' token: the margin under an in-measure
licence (Jordan et al., Sec. 10.3-10.4), shown beside the worst-case margin.

Labels.  The paper's rules are (R1)-(R4).  This script prints them under earlier labels:
(D0)-(D3) are the parts of (R1), (D4) is (R2), (D5) is (R3), (D6) is (R4), and (D7) is the
objective-alphabet form of (R1) and (R2).

The rows are representatives of classes: every objective is in exactly one C0 class, and inside
the linear-agreement class the outcome of C2/C3 is decided by the bound class of the decoder code
(Singleton, TVZ, Zyablov, GV, capacity, vanishing).  The checks are necessary conditions; passing
them is not a claim of quantum advantage.
In the row notes, the "starting list" is the list of previously studied code families that
the search started from ("registered" rows are entries of it); "cat" marks an entry of that
list and "none-reg" means that no attack is recorded there.  Scores quoted in the notes of groups
A and B without a script number come from earlier studies of those families; those runs
are not part of this repository.

Run from the repository root:  python scripts/10_advantage_checklist_sweep.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isqrt, log, sqrt

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from dqi_explorer.discovery import (
    IN_MEASURE_MARGIN_CAP,
    alternant_half_designed_reach,
    column_inserted_reach,
    log_cauchy_reach,
    sparse_mixed_reach,
    construction_d_restricted_reach_ceiling,
    in_measure_margin_cap_at,
    licensed_radius_is_realizable,
    prange_quality,
    qary_entropy,
)
from dqi_explorer.spectral import canonical_asymptotic_quality

NF = "not found"


# --------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------
def inverse_qary_entropy(value: float, q: int) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1 - 1 / q
    lower, upper = 0.0, 1 - 1 / q
    for _ in range(200):
        middle = (lower + upper) / 2
        if qary_entropy(middle, q) < value:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def kl_nats(a: float, b: float) -> float:
    if a <= 0:
        return -log(1 - b)
    if a >= 1:
        return -log(b)
    return a * log(a / b) + (1 - a) * log((1 - a) / (1 - b))


def existential_quality(alpha: float, rho: float, q: float) -> float:
    budget = alpha * log(q)
    if budget >= -log(rho):
        return 1.0
    lower, upper = rho, 1.0
    for _ in range(200):
        middle = (lower + upper) / 2
        if kl_nats(middle, rho) < budget:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def density_window_ok(alpha: float, lam: float, rho: float) -> bool:
    """General C1 inequality: Q_DQI(lambda, rho) > Q_Prange(alpha, rho)."""
    return canonical_asymptotic_quality(lam, rho) > prange_quality(alpha, rho) + 1e-12


def distance_bound_ok(alpha: float, lam: float) -> bool:
    """C2 with worst-case radius lambda = delta/2: lambda(1-lambda) > alpha^2/4."""
    return lam * (1 - lam) > alpha * alpha / 4


def realizability_token(r: Row) -> str:
    """C2R: is the declared (alpha, lambda) pair realized by any code at all?

    Sphere packing caps a worst-case unique radius at H_q^-1(alpha), so a pair with
    H_q(lambda) > alpha is realized by no code, efficiently or otherwise.  Judged only
    for worst-case and designed-distance guarantees: a channel-average or probabilistic
    radius may legitimately exceed the ceiling, and a capacity radius sits on the
    boundary by construction.  This token reads the row's declared alphabet.
    """
    if r.lam is None or r.alpha is None or r.q is None or r.lam <= 0:
        return "-"
    order = int(round(r.q))
    if abs(r.q - order) > 1e-12 or order < 2:
        return "-"
    if r.lam > 1 - 1 / order:
        return "FAIL(past max entropy)"
    floor = qary_entropy(r.lam, order)
    token = r.c3.lower()
    worst_case = any(mark in token for mark in ("/wc", "/dd", "worst", "designed"))
    if not worst_case:
        if abs(r.alpha - floor) <= 5e-4:
            return f"boundary({floor:.4f}; average-case only)"
        return f"unjudged({floor:.4f}; not worst-case)"
    if licensed_radius_is_realizable(r.alpha, r.lam, order):
        return f"ok({floor:.4f}<={r.alpha:.4f})"
    return f"FAIL(needs {floor:.4f}, has {r.alpha:.4f})"


# --------------------------------------------------------------------------
# (A) the checklist
# --------------------------------------------------------------------------
CHECKLIST = (
    ("C0", "objective form",
     "maximize #{i:(Bx)_i in F_i} over a finite field / abelian group with an exact or approximation-preserving reduction witness",
     "reduction_witness(exact|approx); structural_refusal(quadratic|nonlinear|real-inequality|non-group image)",
     "not_linear_agreement; variant_needed(weighted|Hamiltonian|grouped-density)"),
    ("C1", "density window",
     "rho constant in n and Q_DQI(lambda,rho) > Q_Prange(alpha,rho); MDS form rho(1-rho) > alpha/(8(1-alpha/2))",
     "closed_form_inequality(alpha,lambda,rho) evaluated live",
     "exact_value_objective(rho=1/q); vanishing_density"),
    ("C2", "decoder-code distance",
     "delta bounded away from 0 with delta > 1-sqrt(1-alpha^2) (worst-case lambda=delta/2)",
     "code_bound(Singleton|TVZ|designed|exact_finite_distance)",
     "vanishing_distance(girth|gadget|BCH-designed|Zyablov|product)"),
    ("C2R", "radius realizability",
     "alpha >= H_q(lambda) for a worst-case unique guarantee, evaluated in the metric and alphabet the radius uses; equivalently lambda <= delta_GV(1-alpha)",
     "closed_form_inequality(alpha,lambda,q) evaluated live",
     "unrealizable_radius; indeterminate(block width undeclared)"),
    ("C3", "decoder",
     "efficient decoder to radius lambda; guarantee domain in {worst_case_unique, designed_distance, channel_average, probabilistic}",
     "literature_theorem(arXiv, thm); probabilistic => imperfect-decoder theorem required",
     "no_efficient_decoder; unlicensed_probabilistic"),
    ("C4", "encoding fidelity",
     "field semantics faithful: no-wrap quantization, alphabet >= min-alphabet table, evaluation points designer-chosen for AG codes",
     "analytic_argument",
     "density_O(1/n); data_dictated_points; boolean_gadget_blowup"),
    ("C5", "Prange screen",
     "Q_DQI > Q_Prange at the declared (alpha, lambda, rho)",
     "live numeric",
     "no_prange_screen_gap"),
    ("C6", "regime",
     "q^alpha*rho vs 1 and existential line Q*: 'sat' (competitor = satisfiable-CSP search) or 'unsat' (genuine optimization)",
     "live numeric",
     "informational, never a rejection"),
    ("C7", "structural classical threats",
     "none of: local-dual/direct-sum subcode, constituent-product, RD-encodable optimizer, non-vacuous list recovery, hook-fiber decomposition, known polynomial algorithm",
     "analytic_argument; list-recovery vacuity (alpha+gamma) rho q > 1 live",
     "dominated_by_structured_attack; rd_encodable_optimizer; known_polynomial_algorithm"),
    ("C8", "literature status",
     "published reproduction known, or none found",
     "cat(entry of the starting list); lit(arXiv); not found",
     "informational"),
)


# --------------------------------------------------------------------------
# (B) the sweep
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Row:
    group: str
    name: str
    encoding: str
    c0: str                       # lin | quad | nonlin | real | exact | nongroup | variant
    alpha: float | None = None
    lam: float | None = None
    rho: float | None = 0.5
    q: float | None = None
    gamma: float = 0.0
    c2: str = ""                  # bound class token when numeric, else fail label
    c3: str = ""
    c4: str = ""
    c7: str = ""
    c8: str = ""
    bucket: str = ""
    reason: str = ""
    # Gate D4' (licensed in measure): the reach of a decoder that succeeds on all but a
    # vanishing fraction of *uniformly random* weight-<= ell patterns, which is all canonical
    # DQI needs (Jordan et al. 2408.08292, Sec. 10.3-10.4, Thm 10.1).  None means no such
    # decoder is declared beyond the worst-case radius ``lam``.  The in-measure Singleton
    # bound caps it at alpha; the in-measure cap on the margin is (sqrt5-1)/4 = 0.309.
    lam_avg: float | None = None
    # Upper bound on the decoder's failure fraction at lam_avg (Jordan et al. Thm 10.1 charges
    # eps (m+1)/m against the quality); None when no bound is declared.
    eps_avg: float | None = None


G = 1 / 15  # GS tower genus ratio at s = 16
S2 = sqrt(2)

ROWS: tuple[Row, ...] = (
    # ---------------- A. canonical / known DQI problems ----------------
    Row("A", "sparse MAX-XOR-SAT / LDPC syndrome", "B sparse over F_2, singleton sets", "lin", 0.5, 0.11, 0.5, 2, 0,
        "GV-capacity(avg)", "BP/avg", "ok", "RD-enc+OGP/AMP", "cat:dqi-paper 2408.08292; 2509.14509; 2607.28120",
        "known_reproduction", "showcase family; RD line 0.89 > semicircle 0.81 at alpha=1/2; AMP/MCMC match"),
    Row("A", "dense syndrome decoding / LPN", "random dense B over F_2", "lin", 0.5, 0.0, 0.5, 2, 0,
        "GV(existential only)", "none (exhaustive only)", "ok", "n/a", "cat:dqi-paper",
        "rejected", "C3: absence of a decoder is the hardness assumption"),
    Row("A", "generic MAX-LINSAT, structured B", "field F_q, equal-size sets", "lin", None, None, None, None, 0,
        "depends on ker(B^T)", "depends on the code", "ok", "depends", "cat:dqi-paper",
        "known_reproduction", "class row: outcome decided by the decoder-code bound class of ker(B^T)"),
    Row("A", "OPI / GRS (noisy polynomial reconstruction)", "Vandermonde B, balanced sets", "lin", 0.2, 0.1, 0.5, 101, 0,
        "Singleton", "BM/wc", "ok", "none-reg; MCMC ~1.1^n", "cat",
        "known_reproduction", "the headline family; passes every gate; no lower bound",
        lam_avg=1 - sqrt(1 - 0.2)),  # D4': power decoding to the Johnson radius 1-sqrt(1-alpha) on random errors (Rosenkilde, AMC 2018)
    Row("A", "multivariate OPI / Reed-Muller", "RM evaluation B (starting-list point alpha=3/32)", "lin", 3 / 32, (1 - sqrt(1 - 1 / 64)) / 2, 0.5, 2, 0,
        "RM designed", "majority-logic/wc", "ok", "none-reg", "cat",
        "known_reproduction", "starting-list control 0.5625 vs 0.546875 reproduced from its (alpha, lambda)"),
    Row("A", "Hermitian OPI", "Hermitian curve AG code", "lin", 0.25, 0.0667, 0.5, 256, 0.0625,
        "AG designed", "Farran/dd", "designer-pts", "none-reg", "lit:2510.06603 (Gu-Jordan)",
        "known_reproduction", "starting-list control 0.8133 vs 0.625; alphabet n^(2/3)"),
    Row("A", "weighted OPI / weighted regression", "weights per constraint", "variant", None, None, None, None, 0,
        "Singleton", "BM/wc + imperfect-decoder thm", "ok", "none-reg", "lit:2605.10666",
        "variant_needed", "published theory (incl. imperfect decoders)"),
    Row("A", "rank-metric / Gabidulin MaxAgreement", "rank-metric shells", "lin", None, None, None, None, 0,
        "Singleton-like", "Gabidulin/wc", "ok", "none-reg", "lit:2606.04843",
        "known_reproduction", "published; no advantage claim in source"),
    Row("A", "folded / block MAX-LINSAT (F_q^b blocks)", "block predicates", "lin", None, None, None, None, 0,
        "block-Hamming", "block decoder", "ok", "whole-block IS attacks", "cat:dqi-paper s14",
        "known_reproduction", "block-alphabet quality law extends the scalar proof"),
    Row("A", "MaxCut (unweighted)", "incidence B over F_2", "lin", 0.5, 0.0, 0.5, 2, 0,
        "girth (vanishing)", "cycle-space exhaustive", "ok", "poly-alg (Parekh)", "cat; 2509.19966",
        "rejected", "d = girth; nontrivial regime is polynomial"),
    # ---------------- B. starting-list families ----------------
    Row("B", "star-twisted RS intersection", "low-star MDS optimizer, high-star dual", "lin", 0.25, 0.125, 0.5, 401, 0,
        "Singleton", "BW/wc (unscale+star check)", "ok", "hook-fiber (5/8 thm); CP-SAT, portfolio negative", "cat",
        "registered_candidate", "the only row that instantiates a published theorem"),
    Row("B", "self-dual GS MaxAgreement (s=16)", "C = C^perp multipoint AG", "lin", 0.5, 13 / 60, 0.5, 256, G,
        "TVZ", "Lee/wc (level 2 only)", "designer-pts", "none-reg; finite controls negative", "cat",
        "registered_candidate", "exact identity lambda(1-lambda)-1/16>0"),
    Row("B", "decreasing norm-trace intersection", "norm-trace AG, Q_r = 2 sqrt(n)", "lin", 7 / 6 - 1 / S2, (1 - 1 / S2) / 2, 0.5, 4096, 1 / 6,
        "AG designed (72>49)", "Farran/dd", "growing alphabet", "one-grid DP exp(n log n); two-grid open", "cat",
        "registered_candidate", "passes the screen; finite tests inconclusive"),
    Row("B", "GS tower function intersection (F_256)", "one-point AG on GS tower", "lin", 0.25, (0.25 - G) / 2, 0.5, 256, G,
        "TVZ", "Farran/dd (not instantiated)", "designer-pts", "none-reg", "cat",
        "registered_candidate", "screened only"),
    Row("B", "Kopparty-Tamo expander dual", "Tanner code with RS local codes", "lin", 0.6, 0.25, 0.5, 729, 0,
        "global degree bound", "ambient RS/wc", "growing degree", "local-dual (subexp) dominates iid controls", "cat",
        "registered_candidate", "arbitrary-set hypothesis only"),
    Row("B", "AEL near-MDS expander dual", "additive expander code", "lin", 0.5, 0.225, 0.5, 11, 0,
        "near-MDS (unlicensed)", "GI decoder (not instantiated)", "ok", "right-local dual union 10/10", "cat",
        "registered_candidate", "exploratory; small finite test negative"),
    Row("B", "Hermite-jet order-4 projection", "jet blocks", "lin", 0.25, 0.125, 0.5, 23, 0,
        "block (unlicensed)", "multiplicity WB", "ok", "best-of-256 IS beats DQI", "cat",
        "registered_candidate", "exploratory; order-2 control negative"),
    Row("B", "multivariate multiplicity dual", "Hasse-derivative blocks", "lin", 0.3, 0.0482, 0.5, 7, 0,
        "block (unlicensed)", "multivariate half-distance", "ok", "compatibility-aware interpolation 9/9", "cat",
        "registered_candidate", "exploratory; not dual-closed (interesting), controls negative"),
    Row("B", "binary 2x2 sum-rank dual", "sum-rank blocks", "lin", 0.2288, 0.0225, 0.5, 16, 0,
        "block (unlicensed)", "two-step Wu/wc", "ok", "product (constituent) attack; gap 0.0035", "cat",
        "registered_candidate", "new construction; margin within the noise of the finite test"),
    Row("B", "dual-polar max agreement", "polar SC decoder code", "lin", 0.2, 0.0249, 0.5, 2, 0,
        "capacity(avg)", "SC/avg", "ok", "RD-enc (polar source coding 0.737 > 0.656)", "cat",
        "registered_candidate", "the starting-list instance of the RD principle"),
    Row("B", "anchored-band grouped-density q=64", "grouped densities", "variant", None, None, None, None, 0,
        "exact finite", "syndrome/wc", "no finance semantics", "anchor-aware IS 48-52 > 43.2", "cat",
        "rejected", "synthetic test case for a rejection"),
    Row("B", "Treasury CMT curve fitting", "RS over quantized yields", "lin", 0.25, 0.125, None, None, 0,
        "Singleton", "BM/wc", "O(1/n) density", "n/a", "cat:finance (rejected in the starting list)",
        "rejected", "C1/C4: (2W+1)/q = O(1/n) since n <= q+1"),
    Row("B", "RTGS liquidity-saving subset", "modular equality", "real", None, None, None, None, 0,
        "n/a", "n/a", "real-ineq", "n/a", "cat",
        "rejected", "C0: inequalities, indivisibility, timing not code-linear"),
    Row("B", "BCH bounded-distance", "designed distance", "lin", 0.25, 0.0, 0.5, 2, 0,
        "designed (vanishing)", "BM/wc", "ok", "n/a", "cat",
        "rejected", "C2: relative radius -> 0"),
    Row("B", "scenario-agreement chance-constraint surrogate", "return rows b_t, bands S_t", "lin", None, None, None, None, 0,
        "unknown (dense real B)", "none for ker(B^T)", "quantized holdings", "n/a", "cat:finance",
        "variant_needed", "C3: empirical return matrices have no decoder; C0 drops variance/cardinality"),
    Row("B", "price-and-Greeks Hermite-jet consensus", "value+derivative jets", "lin", 0.25, 0.125, None, None, 0,
        "block", "multiplicity WB", "joint density rho ~ w^4 needs w > 0.4389", "n/a", "cat:finance",
        "variant_needed", "C1: joint tolerance density must exceed 0.0371 at alpha=1/4"),
    # ---------------- C. code families ----------------
    Row("C", "Roth-Lempel / multi-twist TGRS", "non-GRS MDS", "lin", 0.25, 0.125, 0.5, 401, 0,
        "Singleton", "Zhu-Jin/wc (2512.24217v2)", "ok", "hook-fiber generalization open", NF,
        "new_candidate", "does the 5/8 trap survive l hooks?"),
    Row("C", "interleaved RS, s=4 (collaborative decoding)", "block predicates over F_q^4", "lin", 0.25, 0.2, 0.5, 401, 0,
        "Singleton (block)", "collab/prob (Schmidt-Sidorenko-Bossert)", "ok", "none known; block IS attacks open", NF,
        "new_candidate", "largest projected gap; decoder probabilistic => unlicensed worst-case, licensed under D4' at ell=79 (SSB failure bound 2.4e-16)",
        # D4': Schmidt-Sidorenko-Bossert collaborative decoding, t_max = (s/(s+1))(N-K) = 80 block errors;
        # at ell = 79 the failure bound (cs/0610074 Thm 6) is ((q^s-1/q)/(q^s-1))^79 q^-5/(q-1) = 2.4e-16.
        lam_avg=79 / 400, eps_avg=((401**4 - 1 / 401) / (401**4 - 1)) ** 79 * 401 ** (-5) / 400),
    Row("C", "elliptic (genus-1) codes", "n <= q+1+2 sqrt q", "lin", 0.25, (0.25 - 1 / 289) / 2, 0.5, 256, 1 / 289,
        "near-MDS", "AG/wc", "designer-pts", "none-reg", NF,
        "new_candidate", "length gain 1.125 at q=256; marginal"),
    Row("C", "Suzuki (Castle) codes q=32", "n=q^2+1, g=q0(q-1)", "lin", 0.5, (0.5 - 124 / 1025) / 2, 0.5, 32, 124 / 1025,
        "AG designed", "Feng-Rao/dd", "designer-pts; alphabet ~ sqrt n", "none-reg", NF,
        "new_candidate", "better alphabet scaling than Hermitian"),
    Row("C", "GS tower F_256, alpha=0.10 (unsat regime)", "one-point AG, low rate", "lin", 0.10, (0.10 - G) / 2, 0.5, 256, G,
        "TVZ", "Farran/dd", "designer-pts", "OGP analysis open", NF,
        "new_candidate", "only regime with Q* < 1 and a passing screen"),
    Row("C", "self-dual / LCD AG sequences q>=8 (Marques-Quoos 6.4)", "TVZ-attaining self-dual", "lin", 0.5, (0.5 - 1 / 7) / 2, 0.5, 64, 1 / 7,
        "TVZ", "Lee/wc (preprocessing open)", "designer-pts", "none-reg", "cat",
        "registered_candidate", "the growing construction is left open"),
    Row("C", "product / tensor code RS x RS, R=0.5", "grid evaluations", "lin", 0.75, 0.125, 0.5, 256, 0,
        "product delta1*delta2", "row-column/wc", "ok", "n/a", NF,
        "predicted_rejection", "C2/C5: 0.8307 < 0.875"),
    Row("C", "concatenated / Justesen (Zyablov), q=256", "outer RS, inner random", "lin", 0.25, 0.0056, 0.5, 256, 0,
        "Zyablov (fails all q)", "GMD/wc", "ok", "n/a", NF,
        "predicted_rejection", "C2: Zyablov delta below 1-sqrt(1-alpha^2) at every q"),
    Row("C", "binary Goppa n=2^20", "McEliece-type", "lin", 0.25, 0.0125, 0.5, 2, 0,
        "designed 2t+1, lambda=alpha/log n", "Patterson/wc", "ok", "n/a", NF,
        "predicted_rejection", "C2 window closes at alpha < 4/log2 n (passes only n <= 2^14)"),
    Row("C", "Z4-linear Kerdock / Preparata", "Gray-map nonlinear binary", "nongroup", None, None, None, None, 0,
        "Preparata d=6 fixed; Kerdock rate->0", "Hammons et al.", "ring, not field", "n/a", NF,
        "predicted_rejection", "C0 (ring) and C2 (dual distance fixed)"),
    Row("C", "Chinese-remainder (CRT) codes", "residues of an integer interval", "nongroup", None, None, None, None, 0,
        "n/a", "GRS'00 unique/list", "n/a", "n/a", NF,
        "predicted_rejection", "C0: image of an interval is not a subgroup; no character dual"),
    Row("C", "Tamo-Barg locally recoverable codes", "RS subcode with local groups", "lin", 0.25, 0.125, 0.5, 256, 0,
        "Singleton-like", "RS local/wc", "ok", "local-dual direct-sum subcode (KT mechanism)", NF,
        "predicted_rejection", "C7: orbit-local attack applies verbatim"),
    Row("C", "finite-geometry (EG/PG) LDPC", "majority-logic decodable", "lin", 0.5, 0.0, 0.5, 4, 0,
        "delta ~ 1/q (vanishing)", "majority-logic/wc", "ok", "n/a", NF,
        "predicted_rejection", "C2"),
    Row("C", "convolutional codes", "sliding window", "lin", 0.5, 0.0, 0.5, 2, 0,
        "free distance fixed (vanishing)", "Viterbi/ML", "ok", "n/a", NF,
        "predicted_rejection", "C2"),
    Row("C", "Sipser-Spielman expander codes", "bounded-degree Tanner, small local codes", "lin", 0.5, 0.01, 0.5, 2, 0,
        "constant delta, small decodable fraction", "flip/wc (small radius)", "ok", "bounded-degree cap 2606.13570; RD-enc (LDGM)", "lit:2606.13570",
        "predicted_rejection", "C5 at the flip radius; C7 bounded-degree limit"),
    Row("C", "quantum-Tanner classical components (C_X, C_Z)", "good qLDPC classical codes", "lin", 0.5, 0.02, 0.5, 2, 0,
        "constant delta", "linear-time/wc (small radius)", "ok", "bounded-degree cap; local-dual", NF,
        "predicted_rejection", "same mechanism as expander codes"),
    Row("C", "Hadamard / simplex / first-order RM", "rate -> 0", "lin", 0.99, 0.25, 0.5, 2, 0,
        "delta 1/2, rate -> 0", "FFT ML/wc", "ok", "Prange -> 1", NF,
        "predicted_rejection", "C5: alpha -> 1 makes the Prange screen ~1"),
    Row("C", "lattice / Construction-A over Z_q, interval sets", "q-ary lattice code", "lin", None, None, None, None, 0,
        "depends on base code", "base-code decoder", "interval sets = MSB/HNP shape", "lattice reduction attacks", NF,
        "known_reproduction", "reduces to the base code's class; interval sets are ordinary sets to the Jacobi law"),
    Row("C", "cyclic / quasi-cyclic RS subcodes", "automorphism-invariant", "lin", 0.25, 0.125, 0.5, 401, 0,
        "Singleton", "BM/wc", "ok", "reduces to OPI", "cat:dqi-paper",
        "known_reproduction", "difference / symmetry constraints reduce to plain OPI (reduction witness)"),
    # ---------------- D. application areas ----------------
    Row("D", "ML: partial-label learning, polynomial hypotheses over F_p", "candidate label set = F_i", "lin", 0.2, 0.1, 0.5, 101, 0,
        "Singleton", "BM/wc", "ok (finite-field labels)", "none-reg", NF,
        "new_framing", "exactly OPI with rho = candidate-set fraction"),
    Row("D", "ML: tolerance-band polynomial regression (quantized)", "interval sets width 2W+1", "lin", 0.25, 0.125, 0.5, 4096, 0,
        "Singleton", "BM/wc", "no-wrap; growing alphabet or AG", "none-reg", NF,
        "new_framing", "density (2W+1)/q constant only if q fixed => AG towers (C4)"),
    Row("D", "ML: heteroscedastic tolerance regression", "unequal |F_i|", "variant", None, None, None, None, 0,
        "Singleton", "BM/wc", "ok", "none-reg", NF,
        "variant_needed", "grouped variational identity exists; circuit theorem missing"),
    Row("D", "ML: agnostic learning of parities", "random B over F_2", "lin", 0.5, 0.0, 0.5, 2, 0,
        "GV(existential)", "none", "ok", "n/a", "cat:dqi-paper (LPN)",
        "rejected", "= LPN; C3"),
    Row("D", "ML: real halfspaces / linear classifiers", "sign of real inner product", "real", None, None, None, None, 0,
        "n/a", "n/a", "real-ineq (mod-p sign is not a halfspace)", "n/a", NF,
        "rejected", "C0/C4"),
    Row("D", "ML: clustering / community detection", "quadratic objective", "quad", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", "cat:graph rejections",
        "variant_needed", "C0: quadratic"),
    Row("D", "ML: matrix completion / low rank", "bilinear", "nonlin", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", NF,
        "rejected", "C0"),
    Row("D", "crypto: noisy polynomial reconstruction / OPE assumption", "OPI", "lin", 0.2, 0.1, 0.5, 101, 0,
        "Singleton", "BM/wc", "ok", "lattice attack on OPE (Sun-Wootters note)", "cat; 2604.09533",
        "known_reproduction", "DQI returns quality-Q samples, not the polynomial"),
    Row("D", "crypto: hidden-number / MSB-of-polynomial predicates", "half-interval sets", "lin", 0.25, 0.125, 0.5, 4096, 0,
        "Singleton", "BM/wc", "ok", "lattice (BV) when almost all samples correct", NF,
        "new_framing", "interval sets are ordinary sets to the Jacobi law; explicit non-claim on crypto"),
    Row("D", "crypto: LWE / LWR", "random lattice B", "lin", 0.5, 0.0, 0.5, 4096, 0,
        "unknown", "none", "ok", "n/a", NF,
        "rejected", "C3: no decoder for random q-ary lattices"),
    Row("D", "crypto: McEliece / syndrome decoding", "Goppa exact syndrome", "exact", 0.25, 0.0125, 1 / 2**20, 2, 0,
        "designed", "Patterson/wc", "ok", "n/a", NF,
        "rejected", "C1: exact-value objective (rho = 1/q) and C2 window 4/log n"),
    Row("D", "crypto: hash preimage / AES", "nonlinear", "nonlin", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", NF, "rejected", "C0"),
    Row("D", "crypto: lattice CVP / SVP", "real lattice", "real", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", NF, "rejected", "C0/C4"),
    Row("D", "coded computing: AG-coded workers with tolerance verification", "designer-chosen curve points", "lin", 0.5, 13 / 60, 0.5, 256, G,
        "TVZ", "Lee/Farran/dd", "designer-pts (ok)", "none-reg", NF,
        "new_framing", "useful only beyond the unique-decoding fraction of faulty workers"),
    Row("D", "coded computing: exact Lagrange coded computing", "singleton returned values", "exact", 0.5, 0.25, 1 / 4096, 4096, 0,
        "Singleton", "BM/wc", "ok", "n/a", "lit:2101.11653 (classical)",
        "rejected", "C1: exact value; plain decoding already optimal"),
    Row("D", "storage: repair / regenerating codes", "exact recovery", "exact", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", NF, "rejected", "C1"),
    Row("D", "networks: network coding (rank metric)", "rank-metric agreement", "lin", None, None, None, None, 0,
        "Singleton-like", "Gabidulin/wc", "ok", "none-reg", "lit:2606.04843",
        "known_reproduction", "see row A rank-metric"),
    Row("D", "DNA storage constraints (homopolymer/GC)", "local nonlinear patterns", "nonlin", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", NF, "rejected", "C0"),
    Row("D", "PIR / batch codes", "exact retrieval", "exact", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", NF, "rejected", "C1"),
    Row("D", "verification: FRI / RS proximity testing", "closest codeword", "exact", 0.25, 0.125, 1 / 4096, 4096, 0,
        "Singleton", "BM/wc", "ok", "n/a", NF,
        "rejected", "C1: rho = 1/q"),
    Row("D", "verification: MPC / Shamir cheater identification", "singleton shares", "exact", 0.5, 0.25, 1 / 256, 256, 0,
        "Singleton", "BM/wc", "ok", "n/a", NF, "rejected", "C1"),
    Row("D", "finance: portfolio optimization", "quadratic", "quad", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", "cat:finance", "variant_needed", "C0"),
    Row("D", "OR: ILP / knapsack / scheduling", "DQI-Kit gadgets", "lin", 0.5, 0.0, 0.5, 2, 0,
        "gadget d <= 2 (vanishing)", "n/a", "boolean_gadget_blowup", "n/a", "lit:2605.16955",
        "rejected", "C2/C4"),
    Row("D", "OR: periodic event scheduling (PESP)", "difference constraints mod T", "lin", 0.5, 0.0, 0.5, 60, 0,
        "girth (vanishing)", "cycle-space", "ok", "n/a", NF,
        "rejected", "C2: incidence matrix => cycle code => girth"),
    Row("D", "OR: grid expansion planning (DC congestion)", "incidence B over F_2; build layer inexpressible", "real", 0.65, 0.0, 0.5, 2, 0,
        "girth=3 (vanishing, certified)", "cycle-space", "data_dictated_points", "OGP/AMP on tuned-hard instances", NF,
        "rejected", "C0/C2/C4: measured on IEEE 14/30-bus test cases"),
    Row("D", "OR: assignment / flow / matching", "polynomial problems", "real", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "poly-alg", "cat:graph search", "rejected", "C7: already polynomial"),
    Row("D", "OR: TSP / set cover / MaxSAT", "nonlinear or clause logic", "nonlin", None, None, None, None, 0,
        "n/a", "n/a", "XOR-gadget blowup", "n/a", "lit:2605.16955", "rejected", "C0/C4"),
    Row("D", "graphs: correlation clustering / signed Max-2-LIN", "signed incidence", "lin", 0.5, 0.0, 0.5, 2, 0,
        "girth", "cycle-space", "ok", "n/a", "analytic", "rejected", "C2"),
    Row("D", "graphs: Max-q-Cut / coloring", "folded incidence", "lin", 0.5, 0.0, 0.5, 4, 0,
        "girth (block)", "cycle-space", "ok", "n/a", "analytic", "rejected", "C2"),
    Row("D", "graphs: random Max-k-XOR, bounded degree", "sparse hypergraph", "lin", 0.5, 0.11, 0.5, 2, 0,
        "GV-capacity(avg)", "BP/avg", "ok", "OGP/AMP; r/q + O(1/sqrt D) cap", "lit:2509.14509; 2606.13570",
        "rejected", "C7"),
    Row("D", "physics: dense p-spin / QUBO", "quadratic Hamiltonian", "quad", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", "lit:2510.07913 (Hamiltonian DQI)",
        "variant_needed", "C0: Hamiltonian-DQI, separate certificates"),
    Row("D", "physics: sparse spin glass (diluted)", "sparse XOR", "lin", 0.5, 0.11, 0.5, 2, 0,
        "GV-capacity(avg)", "BP/avg", "ok", "OGP/AMP", "lit:2509.14509", "rejected", "C7"),
    Row("D", "signal processing: sparse recovery / Prony", "exact exponential fit", "exact", 0.5, 0.25, 1 / 4096, 4096, 0,
        "Singleton", "BM/wc", "real semantics", "n/a", NF, "rejected", "C1/C4"),
    Row("D", "signal processing: filter / sequence design", "real-valued spectra", "real", None, None, None, None, 0,
        "n/a", "n/a", "real-ineq", "n/a", NF, "rejected", "C0/C4"),
    Row("D", "bio: sequence alignment / phylogeny", "edit distance / trees", "nonlin", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", NF, "rejected", "C0"),
    Row("D", "statistics: robust mean / median", "real", "real", None, None, None, None, 0,
        "n/a", "n/a", "n/a", "n/a", NF, "rejected", "C0"),
    Row("D", "statistics: DQI as a quantum lossy encoder", "singleton distortion", "exact", 0.5, 0.25, 1 / 256, 256, 0,
        "Singleton", "BM/wc", "ok", "RD line is the classical side", NF,
        "rejected", "C1: singleton distortion is an exact-value objective"),
    # ---------------- C (continued). the alternant trapdoor family, rows 83-85 ----------------
    # B = H_pub^T for a scrambled alternant parity check over F_p with fixed extension degree s;
    # generic dimension gives delta = alpha/s and worst-case reach alpha/(2s).  The classical
    # solver sees only B (a Niederreiter public key); the key holder sees trace-OPI over F_(p^s).
    # C7 prints the list-recovery token at q = p, which is the wrong alphabet for the key holder's
    # view (lists live in F_(p^s), ratio alpha*rho*p^s/s >> 1, vacuous); without the key there is no
    # structure to list-recover.
    Row("C", "alternant trapdoor s=4, p=11 (Niederreiter public matrix)", "B = H_pub^T, dense over F_11, m <= 11^4", "lin", 0.12, alternant_half_designed_reach(0.12, 4), 0.5, 11, 0,
        "alternant designed (delta=alpha/s)", "BM-on-hidden-GRS/wc (key holder only)", "ok; sets random", "none known without key; hardness = key indistinguishability + Prange-optimality on random B", NF,
        "new_candidate", "not OPI to the solver; UNSAT regime at p=11; mu=+0.0616 (finite +0.050 at m=6000); syzygy distinguisher caps (A) at superpolynomial"),
    Row("C", "alternant trapdoor s=3, p=11", "B = H_pub^T, dense over F_11, m <= 11^3", "lin", 0.2, alternant_half_designed_reach(0.2, 3), 0.5, 11, 0,
        "alternant designed (delta=alpha/s)", "BM-on-hidden-GRS/wc (key holder only)", "ok; sets random", "none known without key; same hardness route", NF,
        "new_candidate", "mu=+0.0795 at alpha=0.2; window alpha < 0.6; finite restart threshold ~3000 exceeds 11^3 and 13^3, so p >= 17 is needed"),
    Row("C", "alternant trapdoor s=2, p=13 (attack risk)", "B = H_pub^T, dense over F_13, m <= 169", "lin", 0.25, alternant_half_designed_reach(0.25, 2), 0.5, 13, 0,
        "alternant designed (delta=alpha/s)", "BM-on-hidden-GRS/wc (key holder only)", "ok; sets random", "ATTACK RISK: Couvreur-Otmani-Tillich 1402.3264 recovers wild Goppa keys at s=2", NF,
        "new_candidate", "largest alternant margin (mu=+0.117) but the quadratic extension is where the known key recovery lives"),
    # ---------------- C (continued). the mechanism search (script 31), rows 86-92 ----------------
    # Problem statements as different from OPI as "beat a public key" is; each row records the mechanism it
    # uses for the decoder (secret structure, quantum decoder, non-algebraic public structure) and what closes it.
    Row("C", "ISIS-infinity on a McEliece key (interval sets, s=4, p=11)", "B = H_pub^T, sets {a_i..a_i+4} mod 11", "lin", 0.12, alternant_half_designed_reach(0.12, 4), 0.5, 11, 0,
        "alternant designed (delta=alpha/s)", "BM-on-hidden-GRS/wc (key holder only)", "ok; sets are intervals", "lattice view: ideal CVP would reach 0.79 of rows (GH), LLL/BKZ cannot at m=6000; hardness becomes LWE-type", NF,
        "new_candidate", "same class as row 83 with an LWE-type hardness assumption; not a new problem class"),
    Row("C", "hidden-LDPC key (sparse H, dense B = P H^T S, BP with the key)", "planted weight-w parity checks", "lin", 0.5, 0.084, 0.5, 2, 0,
        "BP threshold (in measure, D4')", "BP/avg (key holder only)", "ok", "one information set exposes a planted weight-w row with prob. w alpha (1-alpha)^(w-1): (3,6) in 10 sets; reach clears D2 only for w <= 16", NF,
        "rejected", "C7: decodable ensembles are recoverable; hidden ensembles (w >= 100) have reach 0.0004 < 0.016"),
    Row("C", "lattice key (LWE/GPV trapdoor, Gaussian objective over Z_q)", "trapdoor basis", "lin", 0.25, 0.125, 0.5, 3329, 0,
        "n/a", "Gaussian sampler / BDD with the trapdoor", "ok", "the trapdoor solves the primal BDD directly: a classical key holder beats DQI", NF,
        "rejected", "C7: no asymmetry; decoding the dual lattice and quantizing the primal are the same trapdoor operation"),
    Row("C", "hidden multi-twist TGRS key (MDS, reach alpha/2)", "Zhu-Jin decodable regime k' < (n+k)^2/4n", "lin", 0.25, 0.125, 0.5, 401, 0,
        "Singleton", "Zhu-Jin/wc (2512.24217) with the key", "ok", "shorten 2k'-n+1 positions: Schur square is a proper subspace (CPTZ 2412.15160 Thm 1), then Wieschebrink / Sidelnikov-Shestakov", NF,
        "rejected", "C7: decodable regime is distinguishable; structure public => twisted OPI (row 27)"),
    Row("C", "random dense B with a quantum decoder (USD / PGM)", "no structure, no key", "lin", 0.25, 0.125, 0.5, 11, 0,
        "none (random code)", "USD per coordinate (Chailloux-Tillich 2310.20651) / pretty good measurement", "ok", "USD inside Regev's reduction returns exactly Prange's bound (CT Thm 8); set-membership noise spans r < q dimensions so USD is impossible; PGM not efficient", NF,
        "rejected", "C3: the only efficient quantum decoder equals the classical baseline"),
    Row("C", "multilevel concatenated (Blokh-Zyablov) code, q=2^16", "GCC, RS outer, short inner, multistage GMD", "lin", 0.25, 0.028, 0.5, 65536, 0,
        "Blokh-Zyablov (passes D2 at q >= 2^16)", "multistage GMD/wc", "ok", "inner-block locality (right-local dual attack, as for AEL); outer codes RS-derived", NF,
        "new_candidate", "the only non-algebraic bound class that clears D2: delta_BZ=0.056 > 0.032, mu=+0.040; not a new problem class"),
    Row("C", "SSAG (Hermitian, F_11^4) / wild Goppa keys", "alternant on a curve; Gamma(L, g^(p-1))", "lin", 0.12, alternant_half_designed_reach(0.12, 4) - 0.002, 0.5, 11, 0.004,
        "alternant designed minus genus", "AG key-equation / Goppa decoder (key holder only)", "ok", "same class and attack surface as row 83; tangent-space attack outside the regime", NF,
        "new_candidate", "members of the McEliece class, not new problems; 121x the length at p=11 for a reach loss of 0.002"),
    # ---------------- C (continued). metric and alphabet relaxations, rows 93-97 ----------------
    # A smooth objective f_i(v) = cos(2 pi v/q) has Fourier support {+-1}, so DQI's error register holds
    # {0,+-1}-patterns and the decoder is a restricted-error decoder (script 33).
    # For a q-ary or Construction-D lattice the restricted reach is capped by delta_H(level-0 code)/4
    # (Euclidean BDD), so the metric inherits gate D1 from the level-0 code.  The cosine rows use the
    # cosine law sqrt(2 lam (1-lam)) against the baseline alpha; the live C1/C5 tokens below are the
    # set-membership gates at the same (alpha, lambda) and are printed for comparability only.
    Row("C", "nearest lattice point with per-coordinate tolerance: Barnes-Wall BW_1024 (public BDD decoder)", "Construction D on the RM chain, modulus 2^6; cosine or interval objective", "lin", 0.25, construction_d_restricted_reach_ceiling(2**5 / 2**10), 0.5, 64, 0,
        "restricted reach <= delta_H(RM)/4 = 0.0078", "Micciancio-Nicolosi BDD/wc (Euclidean radius d_E/2)", "ring Z_64; ceiling is metric-independent", "D1 in the restricted metric: 0.0078 < 0.0312 needed; ceiling -> 1/(4 sqrt N)", NF,
        "rejected", "C2: the restricted metric inherits D1 from the level-0 Reed-Muller code; reopened only by a non-BDD restricted decoder"),
    Row("C", "nearest lattice point with per-coordinate tolerance: polar lattice", "Construction D on polar codes", "lin", 0.25, 0.0, 0.5, 65536, 0,
        "vanishing (polar distance)", "SC/avg (D4')", "ring", "D5: polar lattices are quantisation-good (2405.04051), the solver has the existential line", NF,
        "rejected", "C7: efficient quantiser; C2: vanishing level-0 distance"),
    Row("C", "OPI with a smooth objective: cos(2 pi (Bx)_i/q) on Construction-A Reed-Solomon", "restricted {+-1} errors; KV soft decoding", "lin", 0.25, 0.125, 0.5, 401, 0,
        "Singleton (restricted reach >= alpha/4 by BDD, alpha/2 Hamming, more with KV)", "Koetter-Vardy/soft (2411.12553)", "ok", "cosine law sqrt(2 lam(1-lam)) vs baseline alpha: mu=+0.218 at lam=alpha/2, threshold 972 (range 2); LR vacuous", "lit:2411.12553; 2511.22691",
        "known_reproduction", "class 1 with a smooth objective; the metric change widens D0 (floor 0.164 at q=11, lam=0.1) but the decoder source is still evaluation"),
    Row("C", "Lee-metric BCH / negacyclic agreement over F_p", "Lee errors; Berlekamp negacyclic, Roth-Siegel", "lin", 0.25, 0.0, 0.5, 11, 0,
        "vanishing (redundancy Theta(t log_p n))", "Lee BCH/wc", "ok", "t/n = O(alpha/log_p n): 0.031-0.062 at buildable lengths with c = 1/2..1, below the 0.031 the cosine D2 needs; Lee-metric ISD (1903.07692) is the baseline", NF,
        "rejected", "C2: vanishing Lee reach at constant rate, the row-24 argument in another metric"),
    Row("C", "beat a factoring key: Reed-Solomon over Z_N, N = pq secret", "B over Z_N; Welch-Berlekamp needs only invertible pivots", "lin", 0.25, 0.125, 0.5, 10403, 0,
        "Singleton over Z_N", "Welch-Berlekamp/wc (no factorisation needed)", "ring Z_N", "no asymmetry: elimination decodes (182/200) or returns a factor of N (18/200) at N = 101*103; Lemma 9.2 over a ring unproved", NF,
        "rejected", "C7: the secret is not needed to decode; OPI over a ring to everyone"),
    # ---------------- rows 98-103: other secrets and other settings (scripts 34, 36, 38) ----------------
    Row("C", "sparse-mixed GRS key w=3, p=4001 (BBCRS-type block mixing)", "B = (S H T)^T, T of exact column weight 3, dense over F_p, m <= p-1", "lin", 0.07, sparse_mixed_reach(0.07, 3), 0.5, 4001, 0,
        "GRS designed through T (delta = alpha/w)", "BM-on-hidden-GRS via T^-1/wc (key holder only)", "ok; sets random", "none known at exact column weight >= 2 (Couvreur et al. 1501.03736 needs average density < 1+R); square battery full at m=600, 2000 (script 34)", NF,
        "new_candidate", "not OPI to the solver; prime field so D6 void; UNSAT needs alpha < log_p 2; mu=+0.072 (finite +0.064 at m=4000); the key holder's problem is polynomial intersection with pairwise-mixed evaluations"),
    Row("C", "sparse-mixed GRS key w=2 (square-distinguishable)", "B = (S H T)^T, T of exact column weight 2", "lin", 0.07, sparse_mixed_reach(0.07, 2), 0.5, 6007, 0,
        "GRS designed through T (delta = alpha/w)", "BM-on-hidden-GRS via T^-1/wc (key holder only)", "ok; sets random", "square of im(B) has dimension <= 2r-1+m/2 < m for alpha < 1/4 (measured 539 = 239 + 300 at m=600, r=120; script 34)", NF,
        "rejected", "C7: Schur-square distinguishable throughout the unsatisfiable regime (alpha < 0.08 < 1/4)"),
    Row("C", "column-inserted GRS key u=r, p=4001 (RLCE-type)", "GRS [m0,k] + u dense-functional coordinates mixed pairwise; B = H_pub^T, m = m0+u", "lin", 0.07, column_inserted_reach(0.07, 1.0), 0.5, 4001, 0,
        "GRS designed over m0+u (delta = alpha/(1+u/r))", "unmix, BM on the GRS part, recompute/wc (key holder only)", "ok; sets random", "Couvreur-Lequesne-Tillich 1805.11489 recovers short keys (u < r); u = r is the boundary (unverified); random-shortening square battery full at every u/r (script 34), which is not evidence against the targeted attack", NF,
        "new_candidate", "not OPI to the solver; prime field; mu=+0.096 (finite +0.088 at m=4000); attack-regime boundary flagged"),
    Row("C", "expanded GRS key (secret bases per position)", "GRS over F_(p^s) written over F_p in a secret basis per position", "lin", 0.12, alternant_half_designed_reach(0.12, 4), 0.5, 11, 0,
        "GRS designed (delta = alpha/s)", "regroup + BM/wc (key holder only)", "ok; sets random", "Couvreur-Lequesne 2009.05826: twisted-product distinguisher for subspace subcodes of RS with subspace dimension > s/2; the full expansion is inside", NF,
        "rejected", "C7: polynomial-time distinguisher; subspace dimension <= s/2 is the subfield-subcode mechanism again"),
    Row("C", "hash-and-sign signature with a quantum signer (DQI sample on a class-2 key)", "sets F_i = Hash(msg, i); signature x; accept at tau m with portfolio < tau < Q_DQI", "lin", 0.12, alternant_half_designed_reach(0.12, 4), 0.5, 11, 0,
        "alternant designed (delta=alpha/s)", "BM-on-hidden-GRS/wc (key holder only, quantum)", "ok; sets hashed", "forgery = key-less max-agreement at tau; signature law P(f)^2/Z(sets) is key-independent; classical key holder below the key-less portfolio (script 36)", NF,
        "new_candidate", "a scheme on the class-2 mechanism, not a new decoder source; the first hash-and-sign code signature with no rejection sampling"),
    Row("C", "classical key holder on the alternant key (trace-OPI baseline)", "the solver knows (a_i, v_i): F_q-lines in g-space pin r-1 rows against n-1 for F_p-lines", "lin", 0.12, alternant_half_designed_reach(0.12, 4), 0.5, 11, 0,
        "alternant designed (delta=alpha/s)", "the key as a classical tool", "ok; sets random", "F_q Prange and F_q line search score below the key-less portfolio (script 36)", NF,
        "known_reproduction", "baseline row: the OPI-hard core of class 2, measured"),
    # ---------------- rows 104-106: the objective selects the error alphabet (scripts 40-41) ----------------
    Row("C", "multiplicative polynomial intersection: log-Cauchy code B_ij = log(t_i + alpha_j) over F_8191, cosine objective", "public; restricted {+-1} errors exponentiate to a rational function; rational reconstruction", "lin", 0.07, log_cauchy_reach(0.07), 0.5, 8191, 0,
        "rational reconstruction: n >= 2 ell + 1 (reach alpha/2 for {0,+-1} errors)", "exponentiate + rational reconstruction/wc (public)", "smooth objective (cosine); no sets", "over F_M the code is not an evaluation code (entries are logs, square full); the primal is a subset-product knapsack with set-valued targets; classical hardness lattice-type (max-cos on a q-ary lattice); portfolio 0.149-0.152 vs DQI 0.243 at m=2000 (script 41); no known quantizer in either domain", NF,
        "new_candidate", "class 3 under the smooth-objective rule: an evaluation-type decoder in a domain the exponential map disconnects from the objective; not in the starting list (its law is set-membership)"),
    Row("C", "log-Cauchy code, in-measure column (random-sign patterns beyond ell)", "same code; D4' reach from the probe of script 40", "lin", 0.07, 0.75 * 0.07, 0.5, 8191, 0,
        "in-measure: random signs recovered to about 0.75 n (probe: 1.00 at 0.74 n, 0.08 at 0.99 n)", "same decoder, typical patterns", "smooth objective (cosine)", "the D4' Singleton ceiling is w < n; measured reach about 0.75 alpha", NF,
        "new_candidate", "in-measure margin about +0.29 at alpha = 0.07; worst-case column is row 104"),
    Row("C", "keyed Chor-Rivest (g, alpha, t secret): beat a discrete-log key", "the same matrix with its construction hidden", "lin", 0.07, log_cauchy_reach(0.07), 0.5, 8191, 0,
        "as row 104", "exponentiate + rational reconstruction/wc (key holder only)", "smooth objective (cosine)", "Vaudenay 1998 recovers the Chor-Rivest secret when h has small factors; the public form (row 104) needs no secret at all, so the key adds an attack surface and no reach", NF,
        "rejected", "C7: the secret is unnecessary; the public code already has the decoder"),
    # ---------------- rows 107-109: other one-way maps from the exponents (script 42) ----------------
    Row("C", "multiplicative polynomial intersection, safe-prime form: B_ij = log((t_i + alpha_j)^2) over F_M, M = (q-1)/2", "public; squares of a rational function; reach alpha/4; odd characteristic, alphabet at any size", "lin", 0.12, 0.03, 0.5, 6053, 0,
        "rational reconstruction of R^2 (degree <= 2 ell) + polynomial square roots: n >= 4 ell + 1", "exponentiate + reconstruct + sqrt/wc (public)", "smooth objective (cosine)", "the same mechanism as row 104 without the Mersenne restriction; square full; decoder exact through ell, all-plus at ell+1 rejected (script 42)", NF,
        "new_candidate", "class 3 under the smooth-objective rule; a parameter family of row 104 (half the reach, any alphabet size)"),
    Row("C", "beat a hidden exponent-knapsack key: the log-Cauchy matrix under a secret dense change of basis T", "B_pub = B T; im(B_pub) = im(B); the key holder undoes T^T on the syndrome", "lin", 0.07, 0.035, 0.5, 8191, 0,
        "as row 104 (key holder)", "exponentiate + rational reconstruction/wc (key holder only)", "smooth objective (cosine)", "key recovery = find two code vectors whose exponentials differ by a constant: M^(2n) candidates against m-1 field equations, no linear handle; no attack known; the key-less optimization is row 104 verbatim (script 42 C)", NF,
        "new_candidate", "a key on the class-3 mechanism: the secret is the basis in which the exponential of the matrix is Cauchy; not a subfield, not a mixing of positions"),
    Row("C", "keyless verifiable challenge on multiplicative polynomial intersection", "sample (t, alpha, centres) publicly; accept x at tau m cosine score", "lin", 0.07, 0.035, 0.5, 8191, 0,
        "as row 104", "exponentiate + rational reconstruction/wc (public)", "smooth objective (cosine)", "soundness = one assumption (no classical algorithm reaches DQI's 0.252 on the instance); portfolio 0.115; forgery bound 2^40 exp(-2 (tau-alpha)^2 m/((1-alpha) 4)) = 2e-12 at tau = 0.20 (script 42 D)", NF,
        "new_candidate", "a use, not a mechanism: the first publicly sampled, publicly verifiable DQI challenge with no key-indistinguishability assumption"),
)


def gate_string(r: Row) -> str:
    parts = [f"C0:{r.c0}"]
    numeric = r.alpha is not None and r.lam is not None and r.rho is not None
    if numeric:
        c1 = "ok" if density_window_ok(r.alpha, r.lam, r.rho) else "FAIL"
        parts.append(f"C1:{c1}(rho={r.rho:.4g})")
        c2 = "ok" if (r.rho == 0.5 and distance_bound_ok(r.alpha, r.lam)) or (r.rho != 0.5 and r.lam > 0) else "FAIL"
        parts.append(f"C2:{c2}[{r.c2}]")
        parts.append(f"C2R:{realizability_token(r)}")
    else:
        parts.append("C1:-")
        parts.append(f"C2:[{r.c2}]")
        parts.append("C2R:-")
    parts.append(f"C3:{r.c3}")
    parts.append(f"C4:{r.c4}")
    if numeric:
        qd, qp = canonical_asymptotic_quality(r.lam, r.rho), prange_quality(r.alpha, r.rho)
        parts.append(f"C5:{qd:.4f}{'>' if qd > qp + 1e-12 else '<='}{qp:.4f}")
        if r.q:
            ratio = r.q**r.alpha * r.rho
            qe = existential_quality(r.alpha, r.rho, r.q)
            parts.append(f"C6:{'sat' if ratio > 1 else 'UNSAT'}({ratio:.3g};Q*={qe:.3f})")
            lr = (r.alpha + r.gamma) * r.rho * r.q
            parts.append(f"C7:{r.c7}; LR {'vacuous' if lr > 1 else 'LIVE'}({lr:.3g})")
        else:
            parts.append("C6:-")
            parts.append(f"C7:{r.c7}")
    else:
        parts.append("C5:-")
        parts.append("C6:-")
        parts.append(f"C7:{r.c7}")
    parts.append(f"C8:{r.c8}")
    if r.lam_avg is not None and r.alpha is not None and r.rho is not None:
        parts.append(f"D4':{in_measure_token(r)}")
    return " ".join(parts)


def in_measure_token(r: Row) -> str:
    """Second column: the margin under an in-measure licence at the declared average-case reach.

    Reported beside the worst-case margin, never in place of it.  The reach is checked against
    the in-measure Singleton ceiling ``lam_avg < alpha``; a declared reach at or above it is
    impossible for every decoder and is flagged.
    """

    if r.lam_avg >= r.alpha:
        return f"IMPOSSIBLE(lam_avg={r.lam_avg:.4g}>=alpha)"
    qd = canonical_asymptotic_quality(r.lam_avg, r.rho)
    qp = prange_quality(r.alpha, r.rho)
    eps = "" if r.eps_avg is None else f"; eps<={r.eps_avg:.1e}"
    return f"mu'={qd - qp:+.4f}(lam_avg={r.lam_avg:.4g}{eps}; cap {in_measure_margin_cap_at(r.alpha, r.rho):.4f})"


if __name__ == "__main__":
    print("=" * 78)
    print("(A) The checklist: gates and admissible certificates")
    print("=" * 78)
    print("| gate | name | statement | certificates that discharge it | failure labels |")
    print("|---|---|---|---|---|")
    for gate in CHECKLIST:
        print("| " + " | ".join(gate) + " |")

    print()
    print("=" * 78)
    print("(B) The sweep")
    print("=" * 78)
    print("| # | group | problem / area | encoding | gates (live certificates) | bucket | reason |")
    print("|---|---|---|---|---|---|---|")
    for i, r in enumerate(ROWS, 1):
        print(f"| {i} | {r.group} | {r.name} | {r.encoding} | {gate_string(r)} | {r.bucket} | {r.reason} |")

    print()
    counts = Counter(r.bucket for r in ROWS)
    print("bucket counts:", dict(sorted(counts.items())))
    print("rows:", len(ROWS), "| groups:", dict(sorted(Counter(r.group for r in ROWS).items())))
    print("C0 classes:", dict(sorted(Counter(r.c0 for r in ROWS).items())))
    print(
        "margin caps at rho=1/2: worst-case licence (sqrt2-1)/2 = %.4f | in-measure licence D4' (sqrt5-1)/4 = %.4f"
        % ((S2 - 1) / 2, IN_MEASURE_MARGIN_CAP)
    )
    print("rows carrying a D4' token:", sum(1 for r in ROWS if r.lam_avg is not None))
