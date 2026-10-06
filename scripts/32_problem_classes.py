"""Script 32 - Every catalog row has a recorded disposition; the window a new mechanism would occupy.

DQI's decoder for ker(B^T) is either a polynomial-time algorithm of B alone (public) or needs
advice not computable from B (secret).  Public decoders that clear the gates come from evaluation
codes (class 1, polynomial intersection); secret decoders that survive structure recovery are
hidden subfield subcodes of evaluation codes (class 2, beat a public key).  Everything else that
passes a gate is closed by a named result: locality (AMP/OGP, gate D5), quantum decoders on random
codes (Chailloux-Tillich), preprocessing advice (vanishing reach), or a change of rule (metric,
alphabet, linearity).

This script (A) classifies every row of 10_advantage_checklist_sweep.py by name and checks that
every row whose Prange screen passes belongs to class 1, class 2, a hybrid, or a closed category
with a stated reason; (B) prints the feasible window H_q(lambda) <= alpha < 2 sqrt(lambda(1-lambda))
at several alphabets; (C) lists asymptotically good, efficiently decodable code families with the
category each falls in.  A row of script 10 that is not mapped makes the script fail.

Run from the repository root:  python scripts/32_problem_classes.py
"""

from __future__ import annotations

import os
import sys
from math import sqrt

EXAMPLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(EXAMPLES), "src"))
sys.path.insert(0, EXAMPLES)

from dqi_explorer.discovery import prange_quality, qary_entropy  # noqa: E402
from dqi_explorer.spectral import canonical_asymptotic_quality  # noqa: E402


def import_sweep():
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        return __import__("10_advantage_checklist_sweep")
    finally:
        sys.argv = saved


# --------------------------------------------------------------------------
# (A) the classification, keyed by sweep row name
# --------------------------------------------------------------------------
CATEGORIES = {
    "class1": "public algebraic structure: polynomial intersection (on a line, a curve, a twist, a block, several variables)",
    "class2": "secret structure: beat a public key (hidden subfield subcode of an evaluation code)",
    "class3": "public restricted-error decoder in a domain the exponential map disconnects from the objective (smooth-objective rule; log-Cauchy)",
    "hybrid": "algebraic outer code with local inner structure; hardness, if any, is class 1 at the outer level",
    "local": "sparse or local structure: closed by AMP/OGP (2509.14509), block-Gibbs (2607.28120), Parekh (MaxCut), the local-dual attacks, or gate D1 (girth)",
    "d5": "efficient rate-distortion encoder for the optimizer code: gate D5",
    "quantum": "quantum decoder on a random code: USD reproduces Prange, PGM is not efficient (2310.20651, 2509.24796)",
    "secret-closed": "secret structure that is recovered or that solves its own primal (hidden LDPC, lattice trapdoor, hidden twisted RS)",
    "none": "no decoder for ker(B^T) at all: the absence is the hardness assumption",
    "rulebreak": "outside the Hamming max-LINSAT rules (rank metric, quadratic objective, restricted/Lee metric, ring alphabet); script 33",
    "meta": "a class row, not an instance",
    "variant": "needs a DQI variant (weighted, grouped-density, Hamiltonian)",
    "notlin": "not a linear-agreement objective (exact value, real, nonlinear, non-group)",
}

CLASS_OF: dict[str, str] = {
    "sparse MAX-XOR-SAT / LDPC syndrome": "local",
    "dense syndrome decoding / LPN": "none",
    "generic MAX-LINSAT, structured B": "meta",
    "OPI / GRS (noisy polynomial reconstruction)": "class1",
    "multivariate OPI / Reed-Muller": "class1",
    "Hermitian OPI": "class1",
    "weighted OPI / weighted regression": "variant",
    "rank-metric / Gabidulin MaxAgreement": "rulebreak",
    "folded / block MAX-LINSAT (F_q^b blocks)": "class1",
    "MaxCut (unweighted)": "local",
    "star-twisted RS intersection": "class1",
    "self-dual GS MaxAgreement (s=16)": "class1",
    "decreasing norm-trace intersection": "class1",
    "GS tower function intersection (F_256)": "class1",
    "Kopparty-Tamo expander dual": "local",
    "AEL near-MDS expander dual": "local",
    "Hermite-jet order-4 projection": "class1",
    "multivariate multiplicity dual": "class1",
    "binary 2x2 sum-rank dual": "class1",
    "dual-polar max agreement": "d5",
    "anchored-band grouped-density q=64": "variant",
    "Treasury CMT curve fitting": "class1",
    "RTGS liquidity-saving subset": "notlin",
    "BCH bounded-distance": "class1",
    "scenario-agreement chance-constraint surrogate": "none",
    "price-and-Greeks Hermite-jet consensus": "class1",
    "Roth-Lempel / multi-twist TGRS": "class1",
    "interleaved RS, s=4 (collaborative decoding)": "class1",
    "elliptic (genus-1) codes": "class1",
    "Suzuki (Castle) codes q=32": "class1",
    "GS tower F_256, alpha=0.10 (unsat regime)": "class1",
    "self-dual / LCD AG sequences q>=8 (Marques-Quoos 6.4)": "class1",
    "product / tensor code RS x RS, R=0.5": "class1",
    "concatenated / Justesen (Zyablov), q=256": "hybrid",
    "binary Goppa n=2^20": "class1",
    "Z4-linear Kerdock / Preparata": "notlin",
    "Chinese-remainder (CRT) codes": "notlin",
    "Tamo-Barg locally recoverable codes": "local",
    "finite-geometry (EG/PG) LDPC": "local",
    "convolutional codes": "local",
    "Sipser-Spielman expander codes": "local",
    "quantum-Tanner classical components (C_X, C_Z)": "local",
    "Hadamard / simplex / first-order RM": "class1",
    "lattice / Construction-A over Z_q, interval sets": "class1",
    "cyclic / quasi-cyclic RS subcodes": "class1",
    "ML: partial-label learning, polynomial hypotheses over F_p": "class1",
    "ML: tolerance-band polynomial regression (quantized)": "class1",
    "ML: heteroscedastic tolerance regression": "variant",
    "ML: agnostic learning of parities": "none",
    "ML: real halfspaces / linear classifiers": "notlin",
    "ML: clustering / community detection": "variant",
    "ML: matrix completion / low rank": "notlin",
    "crypto: noisy polynomial reconstruction / OPE assumption": "class1",
    "crypto: hidden-number / MSB-of-polynomial predicates": "class1",
    "crypto: LWE / LWR": "none",
    "crypto: McEliece / syndrome decoding": "notlin",
    "crypto: hash preimage / AES": "notlin",
    "crypto: lattice CVP / SVP": "notlin",
    "coded computing: AG-coded workers with tolerance verification": "class1",
    "coded computing: exact Lagrange coded computing": "notlin",
    "storage: repair / regenerating codes": "notlin",
    "networks: network coding (rank metric)": "rulebreak",
    "DNA storage constraints (homopolymer/GC)": "notlin",
    "PIR / batch codes": "notlin",
    "verification: FRI / RS proximity testing": "notlin",
    "verification: MPC / Shamir cheater identification": "notlin",
    "finance: portfolio optimization": "variant",
    "OR: ILP / knapsack / scheduling": "local",
    "OR: periodic event scheduling (PESP)": "local",
    "OR: grid expansion planning (DC congestion)": "notlin",
    "OR: assignment / flow / matching": "notlin",
    "OR: TSP / set cover / MaxSAT": "notlin",
    "graphs: correlation clustering / signed Max-2-LIN": "local",
    "graphs: Max-q-Cut / coloring": "local",
    "graphs: random Max-k-XOR, bounded degree": "local",
    "physics: dense p-spin / QUBO": "variant",
    "physics: sparse spin glass (diluted)": "local",
    "signal processing: sparse recovery / Prony": "notlin",
    "signal processing: filter / sequence design": "notlin",
    "bio: sequence alignment / phylogeny": "notlin",
    "statistics: robust mean / median": "notlin",
    "statistics: DQI as a quantum lossy encoder": "notlin",
    "alternant trapdoor s=4, p=11 (Niederreiter public matrix)": "class2",
    "alternant trapdoor s=3, p=11": "class2",
    "alternant trapdoor s=2, p=13 (attack risk)": "class2",
    "ISIS-infinity on a McEliece key (interval sets, s=4, p=11)": "class2",
    "hidden-LDPC key (sparse H, dense B = P H^T S, BP with the key)": "secret-closed",
    "lattice key (LWE/GPV trapdoor, Gaussian objective over Z_q)": "secret-closed",
    "hidden multi-twist TGRS key (MDS, reach alpha/2)": "secret-closed",
    "random dense B with a quantum decoder (USD / PGM)": "quantum",
    "multilevel concatenated (Blokh-Zyablov) code, q=2^16": "hybrid",
    "SSAG (Hermitian, F_11^4) / wild Goppa keys": "class2",
    # rows 93-97: metric and alphabet relaxations (script 33)
    "nearest lattice point with per-coordinate tolerance: Barnes-Wall BW_1024 (public BDD decoder)": "rulebreak",
    "nearest lattice point with per-coordinate tolerance: polar lattice": "d5",
    "OPI with a smooth objective: cos(2 pi (Bx)_i/q) on Construction-A Reed-Solomon": "class1",
    "Lee-metric BCH / negacyclic agreement over F_p": "rulebreak",
    "beat a factoring key: Reed-Solomon over Z_N, N = pq secret": "rulebreak",
    # rows 98-103: other secrets and other settings (scripts 34, 36, 38)
    "sparse-mixed GRS key w=3, p=4001 (BBCRS-type block mixing)": "class2",
    "sparse-mixed GRS key w=2 (square-distinguishable)": "secret-closed",
    "column-inserted GRS key u=r, p=4001 (RLCE-type)": "class2",
    "expanded GRS key (secret bases per position)": "secret-closed",
    "hash-and-sign signature with a quantum signer (DQI sample on a class-2 key)": "class2",
    "classical key holder on the alternant key (trace-OPI baseline)": "class2",
    # rows 104-106: the objective selects the error alphabet (scripts 40-41)
    "multiplicative polynomial intersection: log-Cauchy code B_ij = log(t_i + alpha_j) over F_8191, cosine objective": "class3",
    "log-Cauchy code, in-measure column (random-sign patterns beyond ell)": "class3",
    "keyed Chor-Rivest (g, alpha, t secret): beat a discrete-log key": "secret-closed",
    # rows 107-109: other one-way maps from the exponents (script 42)
    "multiplicative polynomial intersection, safe-prime form: B_ij = log((t_i + alpha_j)^2) over F_M, M = (q-1)/2": "class3",
    "beat a hidden exponent-knapsack key: the log-Cauchy matrix under a secret dense change of basis T": "class3",
    "keyless verifiable challenge on multiplicative polynomial intersection": "class3",
}

OPEN_CATEGORIES = {"class1", "class2", "class3", "hybrid"}


def live_screen_passes(sweep, row) -> bool | None:
    """The row's live C5 screen at its declared reach (None when the row is not numeric)."""

    if row.alpha is None or row.lam is None or row.rho is None:
        return None
    return sweep.density_window_ok(row.alpha, row.lam, row.rho)


def section_a(sweep) -> None:
    rows = sweep.ROWS
    names = [r.name for r in rows]
    unmapped = [n for n in names if n not in CLASS_OF]
    stale = [n for n in CLASS_OF if n not in names]
    if unmapped or stale:
        raise SystemExit(f"classification out of date: unmapped {unmapped}; stale {stale}")
    print("=" * 100)
    print("(A) every sweep row classified; invariant: a passing row is class 1, class 2, a hybrid, or closed with a reason")
    print("=" * 100)
    violations = []
    by_category: dict[str, list[tuple[int, object, bool | None]]] = {}
    for index, row in enumerate(rows, 1):
        category = CLASS_OF[row.name]
        passes = live_screen_passes(sweep, row)
        by_category.setdefault(category, []).append((index, row, passes))
        if passes and category in {"meta", "variant", "notlin"}:
            violations.append((index, row.name, category))
    for category, description in CATEGORIES.items():
        members = by_category.get(category, [])
        if not members:
            continue
        passing = sum(1 for _, _, p in members if p)
        print(f"\n  [{category}] {description}")
        print(f"    {len(members)} rows, {passing} with a passing live screen")
        for index, row, passes in members:
            flag = "pass" if passes else ("fail" if passes is False else "n/a ")
            print(f"    {index:2d} {flag} {row.bucket:20s} {row.name}")
    print()
    open_rows = [(i, r) for i, r, p in sum((by_category.get(c, []) for c in OPEN_CATEGORIES), []) if p]
    closed_passing = [(i, r) for c in ("local", "d5", "quantum", "secret-closed") for i, r, p in by_category.get(c, []) if p]
    print(f"  passing rows in an open class (1, 2, hybrid): {len(open_rows)}")
    print(f"  passing rows closed by a named result:       {len(closed_passing)}")
    if violations:
        raise SystemExit(f"invariant violated: passing rows outside every class: {violations}")
    print("  invariant holds: no passing row is unclassified.")


# --------------------------------------------------------------------------
# (B) the window a third class would occupy
# --------------------------------------------------------------------------
def section_b() -> None:
    print()
    print("=" * 100)
    print("(B) the third-class window: H_q(lambda) <= alpha < 2 sqrt(lambda(1-lambda)), rho = 1/2")
    print("=" * 100)
    print("  A dense, non-evaluation, non-local code family would have to put its (rate loss alpha, worst-case reach")
    print("  lambda) inside this window, with a polynomial-time decoder, and have no efficient set-membership quantizer")
    print("  for its dual.  No such family is known (section C).")
    print()
    print("  %-8s %-8s %-10s %-10s %-8s %-12s" % ("q", "lambda", "alpha_min", "alpha_max", "width", "mu at centre"))
    for q in (2, 16, 256, 2**16):
        for lam in (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3):
            if lam >= 1 - 1 / q:
                continue
            lo = qary_entropy(lam, q)
            hi = 2 * sqrt(lam * (1 - lam))
            if lo < hi:
                centre = (lo + hi) / 2
                mu = canonical_asymptotic_quality(lam, 0.5) - prange_quality(centre, 0.5)
                print("  %-8d %-8.3f %-10.4f %-10.4f %-8.4f %+.4f" % (q, lam, lo, hi, hi - lo, mu))
            else:
                print("  %-8d %-8.3f %-10.4f %-10.4f %-8s empty" % (q, lam, lo, hi, "-"))


# --------------------------------------------------------------------------
# (C) the catalog of asymptotically good, efficiently decodable families
# --------------------------------------------------------------------------
CATALOG = (
    ("Reed-Solomon / GRS, doubly extended", "algebraic", "class 1; rows 4, 45"),
    ("algebraic-geometry codes (elliptic, Hermitian, Suzuki, GS tower)", "algebraic", "class 1; rows 6, 12-14, 29-32"),
    ("twisted / Roth-Lempel GRS", "algebraic", "class 1; rows 11, 27"),
    ("folded, interleaved, block RS", "algebraic", "class 1; rows 9, 28"),
    ("multivariate: Reed-Muller, multiplicity, lifted, tensor RM (2601.16164), BiD (2601.09390)", "algebraic", "class 1; rows 5, 18; the 2026 families are evaluation codes"),
    ("subfield subcodes: BCH, Goppa, alternant, SSAG", "algebraic (trace)", "class 1 when public (rows 24, 35); class 2 when hidden (rows 83-86, 92)"),
    ("sum-rank / linearized RS, Gabidulin", "algebraic, other metric", "rows 8, 19, 62: rule break or embedded"),
    ("polar codes", "local (capacity)", "gate D5; row 20"),
    ("LDPC, expander, Tanner, quantum-Tanner components", "local", "rows 1, 15, 16, 39, 41, 42"),
    ("AEL near-MDS expander codes", "local + algebraic", "row 16: licence needs an exponential alphabet"),
    ("Tamo-Barg locally recoverable codes", "local + algebraic", "row 38: orbit-local attack"),
    ("concatenated (single level, Zyablov)", "hybrid", "row 34: fails D2 at every q"),
    ("multilevel concatenated (Blokh-Zyablov)", "hybrid", "row 91: passes D0-D3 at q >= 2^16; C7 not examined"),
    ("product / tensor codes", "algebraic, product", "row 33: fails D2 (delta = delta1 delta2)"),
    ("convolutional / tail-biting", "local", "row 40: fixed free distance"),
    ("simplicial anticodes (2608.29631)", "algebraic-combinatorial, binary", "not swept: binary with vanishing rate-distance trade-off at these gates"),
    ("random codes with polynomial advice (2510.14347)", "none", "reach O((log n)^2 / n): gate D1"),
    ("code-chain lattices: Barnes-Wall, Construction A/D, polar lattices", "algebraic or local at level 0", "rows 93, 94: restricted-error reach <= delta_H(level 0)/4 (script 33 A2)"),
    ("Lee-metric BCH / negacyclic codes", "algebraic (BCH-type)", "row 96: vanishing Lee reach at constant rate"),
    ("Reed-Solomon over Z_N, Galois rings", "algebraic (evaluation over a ring)", "row 97: Welch-Berlekamp needs no factorisation"),
)


def section_c() -> None:
    print()
    print("=" * 100)
    print("(C) catalog: asymptotically good, efficiently decodable families, and the bucket each falls in")
    print("=" * 100)
    for family, bucket, where in CATALOG:
        print("  %-84s %-26s %s" % (family, bucket, where))
    print("  Every entry is algebraic, local, a hybrid of the two, or has vanishing reach; none is a dense non-evaluation")
    print("  family with a polynomial-time decoder to constant relative radius.  That is the catalog fact behind the")
    print("  two-class statement, and it is where a third class would have to come from.")
    print("  Under the smooth-objective rule the log-Cauchy code (rows 104-105) is a dense non-evaluation")
    print("  code over F_M with a public restricted-error decoder (class 3); under the set objective")
    print("  the question above is unchanged.")


if __name__ == "__main__":
    sweep = import_sweep()
    section_a(sweep)
    section_b()
    section_c()
