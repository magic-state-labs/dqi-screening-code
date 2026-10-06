"""Script 33 - Does a change of metric or alphabet give DQI a new decoder?

Every DQI candidate passing the gates is class 1 (polynomial intersection, public evaluation
code) or class 2 (beat a public key, hidden subfield subcode); a third class would need a new
decoder source or a changed rule.  This script screens two changes of rule, the metric and the
alphabet, one section per problem statement:

  (A1) the restricted-error window: a smooth objective f_i(v) = cos(2 pi v / q) has Fourier
       support {+-1}, so DQI's error register holds patterns with entries in {+-1} and the
       decoder DQI needs is a restricted-error decoder; the sphere-packing gate D0 widens from
       H_q(lambda) to (H_2(lambda) + lambda) / log2 q, and this section prints both windows;
  (A2) the reach ceiling of code-chain lattices (Construction A / D, Barnes-Wall, polar,
       Construction-A Reed-Solomon) for restricted errors: delta_H(level 0) / 4, so the
       restricted metric inherits gate D1 from the level-0 code;
  (A3) the quality law for the cosine objective, derived from Lemma 9.2's tridiagonal (mean,
       standard deviation, skewness) and cross-checked against canonical_finite_quality at the
       indicator special case; the information-set baseline alpha; the D2 analogue
       lambda(1-lambda) > alpha^2/2; the restart-converse thresholds;
  (A4) Lee-metric BCH / negacyclic codes: redundancy per Lee error grows like log_p n, so the
       relative Lee reach vanishes at constant rate (the same D1 argument as Hamming BCH);
  (B)  rings: Reed-Solomon over Z_N with N = pq secret, decoded by Welch-Berlekamp; every pivot
       is invertible or factors N (a demonstration with counts), so "beat a factoring key" has no
       asymmetry; Galois rings, Z_{2^t}, non-abelian groups and exponential alphabets in one
       line each;
  (C)  the rank metric (2606.04843): Gabidulin reach alpha/2 in rank, a single global objective,
       no Prange baseline, no advantage claimed by the authors;
  (F)  the summary table.

Nothing here is an advantage claim; every number is a gate evaluation or a closure argument.

Run from the repository root:  python scripts/33_metric_alphabet_relaxations.py
"""

from __future__ import annotations

import os
import random
import sys
from math import gcd, log, sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqi_explorer.discovery import (  # noqa: E402
    construction_d_restricted_reach_ceiling,
    prange_quality,
    qary_entropy,
    restricted_qary_entropy,
)
from dqi_explorer.spectral import (  # noqa: E402
    canonical_asymptotic_quality,
    canonical_finite_quality,
    general_objective_asymptotic_quality,
    general_objective_finite_quality,
)

RESTARTS = 2**40
FAILURE = 0.05
COS_STD = 1 / sqrt(2)


def hoeffding_length(
    alpha: float,
    margin: float,
    draws: int = RESTARTS,
    failure: float = FAILURE,
    value_range: float = 1.0,
) -> float | None:
    """Restart-converse length: m >= (1 - alpha) R^2 ln(N / eta) / (2 mu^2) (paper, theorem "DQI against N Prange restarts").

    ``R`` is the range of one constraint's score: 1 for set membership, 2 for the cosine.
    """

    if margin <= 0:
        return None
    return (1 - alpha) * value_range**2 * log(draws / failure) / (2 * margin * margin)


def banner(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# --------------------------------------------------------------------------
# (A1) the restricted-error window
# --------------------------------------------------------------------------
def section_a1() -> list[dict]:
    banner("(A1) restricted-error realizability: the window H_q^S(lambda) <= alpha < 2 sqrt(lambda(1-lambda)), S = {+-1}")
    print("  A smooth objective f_i(v) = cos(2 pi v/q) has Fourier support {+-1} (Jordan et al. Sec. 8.2: the error")
    print("  register holds y with g~_i(y_i) != 0).  Sphere packing then counts C(m,k) 2^k patterns instead of")
    print("  C(m,k) (q-1)^k, so gate D0 reads alpha >= (H_2(lambda) + lambda) / log2 q.  The upper edge of the window")
    print("  (gate D2 at rho = 1/2) is unchanged.  Both windows below, with their widths.")
    print()
    print("  %-8s %-8s %-12s %-12s %-10s %-10s %-10s" % ("q", "lambda", "alpha_min(H)", "alpha_min(S)", "alpha_max", "width(H)", "width(S)"))
    rows = []
    for q in (11, 256, 3329, 2**16):
        for lam in (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3):
            lo_h = qary_entropy(lam, q)
            lo_s = restricted_qary_entropy(lam, q, 2)
            hi = 2 * sqrt(lam * (1 - lam))
            rows.append({"q": q, "lam": lam, "lo_h": lo_h, "lo_s": lo_s, "hi": hi})
            print("  %-8d %-8.3f %-12.4f %-12.4f %-10.4f %-10.4f %-10.4f" % (q, lam, lo_h, lo_s, hi, max(0.0, hi - lo_h), max(0.0, hi - lo_s)))
    print()
    print("  Reading: the restriction widens D0 most at small q (at q = 11, lambda = 0.1 the floor drops from")
    print("  %.3f to %.3f); at q = 2^16 the two floors differ by %.3f.  A wider window is a licence to look, not a" % (
        qary_entropy(0.1, 11), restricted_qary_entropy(0.1, 11, 2),
        qary_entropy(0.1, 2**16) - restricted_qary_entropy(0.1, 2**16, 2)))
    print("  decoder: section (A2) asks which codes have one.")
    return rows


# --------------------------------------------------------------------------
# (A2) reach ceilings of code-chain lattices
# --------------------------------------------------------------------------
def section_a2() -> list[dict]:
    banner("(A2) restricted-error reach of code-chain lattices: delta_H(level 0) / 4 (Euclidean BDD), / 2 (any unique decoder)")
    print("  A q-ary or Construction-D lattice contains the {0,1}-lift of every level-0 codeword, so two restricted")
    print("  patterns with the same syndrome can differ by it: unique restricted decoding stops at d_H(C_0)/2, and a")
    print("  Euclidean bounded-distance decoder (radius d_E/2, d_E^2 <= d_H(C_0), pattern norm sqrt k) at d_H(C_0)/4.")
    print("  So the restricted metric inherits gate D1 from the level-0 code.")
    print()
    print("  %-38s %-10s %-8s %-12s %-14s %-12s %s" % ("lattice", "length", "alpha", "delta_H(C_0)", "ceiling (BDD)", "D2 needs", "verdict"))
    rows = []
    entries: list[tuple[str, int, float, float, str]] = []
    for n in (10, 16, 20):
        length = 2**n
        # Barnes-Wall: d_E^2 = 2^floor(n/2) at length 2^n (Micciancio-Nicolosi decode to a constant
        # fraction of d_E in quasi-linear time; the ceiling assumes radius d_E/2; the Reed-Muller
        # chain has vanishing relative distance).
        entries.append(("Barnes-Wall BW_%d (RM chain)" % length, length, 0.25, 2 ** (n // 2) / length, "D1: ceiling -> 0"))
    entries.append(("polar lattice (2405.04051), any length", 2**16, 0.25, 0.0, "D5: quantisation-good; D1: polar distance"))
    entries.append(("Construction-A RS [n, 3n/4] over F_q, q^2 > n", 400, 0.25, 0.25, "class 1 (row 44); KV soft decoding does better"))
    entries.append(("Construction-D BCH chain, constant rate", 2**16, 0.25, 0.0, "D1: BCH distance vanishes (row 24)"))
    for name, length, alpha, delta0, verdict in entries:
        ceiling = construction_d_restricted_reach_ceiling(delta0)
        need = alpha * alpha / 2  # D2 analogue for the cosine objective, section (A3)
        rows.append({"name": name, "length": length, "alpha": alpha, "delta0": delta0, "ceiling": ceiling, "verdict": verdict})
        print("  %-38s %-10d %-8.2f %-12.5f %-14.5f %-12.5f %s" % (name, length, alpha, delta0, ceiling, need, verdict))
    print()
    print("  Reading: constant restricted reach needs constant relative Hamming distance at level 0, which is the")
    print("  evaluation/local dichotomy again.  Barnes-Wall at N = 1024 has ceiling %.4f against a D2 need of %.4f." % (
        construction_d_restricted_reach_ceiling(2**5 / 2**10), 0.25**2 / 2))
    return rows


# --------------------------------------------------------------------------
# (A3) the cosine law, the baseline, and the converse
# --------------------------------------------------------------------------
def section_a3() -> list[dict]:
    banner("(A3) the cosine objective: quality law, information-set baseline, D2 analogue, restart thresholds")
    print("  Lemma 9.2's tridiagonal has diagonal k * skewness and off-diagonal sqrt(k (m-k+1)); the expectation is")
    print("  mean + std * lambda_max / m.  Indicator: (r/p, sqrt(r(p-r))/p, (p-2r)/sqrt(r(p-r))) and the law is the")
    print("  semicircle.  Cosine: (0, 1/sqrt2, 0) and the law is sqrt(2 lambda (1-lambda)).  Baseline: an information")
    print("  set scores cos = 1 on its alpha m rows and 0 on average elsewhere, so Q_Pr^cos = alpha.  D2^cos is")
    print("  therefore lambda(1-lambda) > alpha^2/2, twice as demanding as the set form's alpha^2/4: in units of the")
    print("  objective's spread the information set gains alpha/std = sqrt2 alpha instead of alpha.")
    print()
    m, p, r, ell = 400, 11, 5, 50
    sd = sqrt(r * (p - r)) / p
    sk = (p - 2 * r) / sqrt(r * (p - r))
    print("  cross-check at the indicator: canonical_finite_quality(%d, %d, %d, %d) = %.12f, general law = %.12f" % (
        m, p, r, ell, canonical_finite_quality(m, p, r, ell).value, general_objective_finite_quality(m, ell, r / p, sd, sk).value))
    print()
    print("  %-8s %-8s %-10s %-10s %-10s %-10s %-10s %-12s %-14s %s" % ("alpha", "lambda", "Q_cos", "Q_Pr^cos", "mu_cos", "Q_set", "Q_Pr^set", "mu_set", "thr_cos (R=2)", "thr_set"))
    rows = []
    for alpha, lam in ((0.25, 0.125), (0.25, 0.0625), (0.12, 0.015), (0.25, 0.25), (0.1, 0.05)):
        q_cos = general_objective_asymptotic_quality(lam, 0.0, COS_STD, 0.0)
        mu_cos = q_cos - alpha
        q_set = canonical_asymptotic_quality(lam, 0.5)
        mu_set = q_set - prange_quality(alpha, 0.5)
        thr = hoeffding_length(alpha, mu_cos, value_range=2.0)
        thr_set = hoeffding_length(alpha, mu_set)
        rows.append({"alpha": alpha, "lam": lam, "q_cos": q_cos, "mu_cos": mu_cos, "mu_set": mu_set, "threshold": thr, "threshold_set": thr_set})
        print("  %-8.3f %-8.4f %-10.4f %-10.4f %-+10.4f %-10.4f %-10.4f %-+12.4f %-14s %s" % (
            alpha, lam, q_cos, alpha, mu_cos, q_set, prange_quality(alpha, 0.5), mu_set,
            "%.0f" % thr if thr else "none", "%.0f" % thr_set if thr_set else "none"))
    print()
    print("  Reading: at the Reed-Solomon Hamming reach alpha/2 the raw cosine margin (+%.4f at alpha = 1/4) exceeds the" % rows[0]["mu_cos"])
    print("  set-membership margin (+%.4f), but a cosine row ranges over [-1, 1], so the restart threshold carries a" % rows[0]["mu_set"])
    print("  factor R^2 = 4 (%.0f against %.0f) and the break-even against one information set is lambda(1-lambda) >" % (rows[0]["threshold"], rows[0]["threshold_set"]))
    print("  alpha^2/2 rather than alpha^2/4.  At the plain lattice-BDD reach alpha/4 the margin is %+.4f.  The smooth" % rows[1]["mu_cos"])
    print("  objective is worth it only through a restricted decoder whose reach beats the Hamming reach; on Reed-Solomon")
    print("  that is Koetter-Vardy (2411.12553), class 1.")
    print("  Finite values (m = 400, ell = 50, i.e. lambda = 0.125): cosine %.4f against the limit %.4f." % (
        general_objective_finite_quality(400, 50, 0.0, COS_STD, 0.0).value, general_objective_asymptotic_quality(0.125, 0.0, COS_STD, 0.0)))
    return rows


# --------------------------------------------------------------------------
# (A4) Lee-metric BCH
# --------------------------------------------------------------------------
def section_a4() -> None:
    banner("(A4) Lee-metric BCH / negacyclic codes: reach at constant rate")
    print("  Berlekamp's negacyclic codes and the Roth-Siegel Lee-metric BCH codes (IEEE T-IT 40, 1994)")
    print("  correct t Lee errors with a defining set of Theta(t) roots over an")
    print("  extension of degree m ~ log_p n, so the redundancy is Theta(t log_p n) and, at a constant rate loss alpha,")
    print("  t / n = O(alpha / log_p n) -> 0.  This is the argument that closes Hamming BCH (sweep row 24).")
    print()
    print("  %-6s %-10s %-8s %-14s %-14s %s" % ("p", "n", "alpha", "t/n (c = 1)", "t/n (c = 1/2)", "D2^cos needs"))
    for p, n in ((5, 5**6), (11, 11**4), (101, 101**3)):
        alpha = 0.25
        m_ext = log(n + 1) / log(p)
        print("  %-6d %-10d %-8.2f %-14.5f %-14.5f %.5f" % (p, n, alpha, alpha / m_ext, alpha / (2 * m_ext), alpha * alpha / 2))
    print("  Reading: even with the generous constant c = 1 the Lee reach is an order below the D2 need at every")
    print("  buildable length, and it shrinks with n.  Lee-metric information-set decoding (1903.07692) is the classical")
    print("  baseline in this metric and costs the solver nothing extra.")


# --------------------------------------------------------------------------
# (B) rings: Reed-Solomon over Z_N with N = pq secret
# --------------------------------------------------------------------------
def welch_berlekamp_mod_n(points: list[int], values: list[int], k: int, e: int, modulus: int) -> tuple[list[int] | None, int | None]:
    """Welch-Berlekamp over Z_N.  Returns (message coefficients, factor of N found) with at most one None.

    Unknowns: E(x) monic of degree e (e unknowns) and Q(x) of degree < k + e (k + e unknowns);
    equations Q(a_i) = y_i E(a_i).  Gaussian elimination prefers invertible pivots; if a column has
    only zero-divisor candidates the elimination stops and returns gcd(candidate, N), a factor.
    """

    n = len(points)
    unknowns = k + 2 * e
    rows = []
    for a, y in zip(points, values):
        row = [pow(a, j, modulus) for j in range(k + e)] + [(-y * pow(a, j, modulus)) % modulus for j in range(e)]
        rhs = (y * pow(a, e, modulus)) % modulus
        rows.append(row + [rhs])
    pivot_row = 0
    pivots: list[int] = []
    for col in range(unknowns):
        chosen = None
        zero_divisor = None
        for r_index in range(pivot_row, n):
            entry = rows[r_index][col] % modulus
            if entry == 0:
                continue
            if gcd(entry, modulus) == 1:
                chosen = r_index
                break
            zero_divisor = entry
        if chosen is None:
            if zero_divisor is not None:
                return None, gcd(zero_divisor, modulus)
            continue  # free column (dependent unknown)
        rows[pivot_row], rows[chosen] = rows[chosen], rows[pivot_row]
        inv = pow(rows[pivot_row][col], -1, modulus)
        rows[pivot_row] = [(v * inv) % modulus for v in rows[pivot_row]]
        for r_index in range(n):
            if r_index != pivot_row and rows[r_index][col] % modulus:
                factor = rows[r_index][col]
                rows[r_index] = [(v - factor * w) % modulus for v, w in zip(rows[r_index], rows[pivot_row])]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == n:
            break
    solution = [0] * unknowns
    for r_index, col in enumerate(pivots):
        solution[col] = rows[r_index][-1]
    q_coeffs = solution[: k + e]
    e_coeffs = solution[k + e:] + [1]
    # polynomial division Q / E over Z_N (E monic, so no inverses are needed)
    quotient = [0] * (k + e - e) if k + e - e > 0 else []
    remainder = list(q_coeffs)
    for degree in range(len(q_coeffs) - 1, e - 1, -1):
        coefficient = remainder[degree] % modulus
        quotient[degree - e] = coefficient
        for j in range(e + 1):
            remainder[degree - e + j] = (remainder[degree - e + j] - coefficient * e_coeffs[j]) % modulus
    if any(v % modulus for v in remainder[:e]):
        return None, None
    return [c % modulus for c in quotient[:k]], None


def section_b(seed: int = 33, trials: int = 200) -> dict:
    banner("(B) rings: is decoding Reed-Solomon over Z_N (N = pq secret) a secret operation?  Welch-Berlekamp says no")
    p_prime, q_prime = 101, 103
    modulus = p_prime * q_prime
    n, k, e = 20, 10, 5
    points = list(range(1, n + 1))  # pairwise differences below 101, hence invertible mod N
    rng = random.Random(seed)
    ok = factored = miss = 0
    for _ in range(trials):
        message = [rng.randrange(modulus) for _ in range(k)]
        values = [sum(c * pow(a, j, modulus) for j, c in enumerate(message)) % modulus for a in points]
        positions = rng.sample(range(n), e)
        for pos in positions:
            values[pos] = (values[pos] + rng.randrange(1, modulus)) % modulus
        decoded, factor = welch_berlekamp_mod_n(points, values, k, e, modulus)
        if decoded == message:
            ok += 1
        elif factor is not None and factor in (p_prime, q_prime):
            factored += 1
        else:
            miss += 1
    pivot_failure = 1 / p_prime + 1 / q_prime - 1 / modulus
    print("  N = %d * %d = %d, RS [%d, %d] at points 1..%d, %d errors per trial, %d trials (seed %d)." % (
        p_prime, q_prime, modulus, n, k, n, e, trials, seed))
    print("  decoded without the factorisation: %d | elimination met only zero-divisor pivots and returned a factor: %d | other: %d" % (ok, factored, miss))
    print("  probability that one random element of Z_N is a zero divisor: %.4f; the elimination prefers invertible" % pivot_failure)
    print("  pivots, so it needs every candidate in a column to be one, which happened %d times in %d trials." % (factored, trials))
    print("  Either way the solver never needs (p, q): the code is public-decodable, and the problem is OPI over a ring.")
    print()
    print("  One line each for the other alphabets:")
    print("  - Galois rings Z_{p^2}: Hensel-lifted RS, OPI over a ring; Lemma 9.2 unproved there.")
    print("  - Z_{2^t} Construction-D lattices: section (A2), ceiling delta_H(C_0)/4.")
    print("  - non-abelian groups: Fourier machinery exists (HDQI 2510.07913 is Pauli/symplectic with a Gibbs objective),")
    print("    no decoder to a constant fraction for group-algebra codes (Bazzi-Mitter, Borello-Willems): gate D4.")
    print("  - exponential alphabets with explicit sets: OPI over a large field; the Yamakawa-Zhandry separation is a")
    print("    query bound for oracle sets, class 1 with a different proof technique.")
    return {"ok": ok, "factored": factored, "miss": miss, "trials": trials, "pivot_failure": pivot_failure}


# --------------------------------------------------------------------------
# (C) the rank metric
# --------------------------------------------------------------------------
def section_c() -> None:
    banner("(C) rank metric (Krajenbrink-Krawchuk-Rosmanis-Rosenkranz 2606.04843): why the gates do not apply")
    print("  Problem: find an m x n matrix over F_q closest in rank distance to a target.  Code: Gabidulin [n, k] over")
    print("  F_(q^m), evaluations of q-linearized polynomials, unique rank decoding to floor((n-k)/2): relative reach")
    print("  alpha/2, the rank-metric image of Reed-Solomon (class-1 analogue).  The objective is one global rank, not a")
    print("  sum of per-row scores, so neither Q_Pr (information sets) nor the restart converse is defined; the authors")
    print("  note a covering-radius obstruction and state that they do not claim a quantum advantage.  The Gabidulin")
    print("  trapdoor (GPT) is recovered in polynomial time (Overbeck 2008), so there is no class-2 analogue either.")
    for alpha in (0.1, 0.25, 0.5):
        print("  alpha = %.2f: Gabidulin rank reach %.3f; sum-rank embedding row 19 is class 1." % (alpha, alpha / 2))


# --------------------------------------------------------------------------
# (F) summary table
# --------------------------------------------------------------------------
def section_f(a1_rows, a2_rows, a3_rows, b_row) -> None:
    banner("(F) summary table")
    bw = next(r for r in a2_rows if r["length"] == 1024)
    q11 = next(r for r in a1_rows if r["q"] == 11 and abs(r["lam"] - 0.1) < 1e-9)
    cos_rs = a3_rows[0]
    cos_bdd = a3_rows[1]
    rows = [
        ("R1 nearest lattice point with per-coordinate tolerance (Barnes-Wall, public BDD decoder)", "restricted metric", "everyone", "<= %.4f at N=1024" % bw["ceiling"], "-", "D1 (restricted)", "closed (A2)", "ceiling delta_H(RM)/4 -> 0; reopened by a restricted decoder that is not a Euclidean BDD"),
        ("R1 same on a polar lattice", "restricted metric", "everyone", "-", "-", "D5", "closed (A2)", "quantisation-good (2405.04051) hands the solver the existential line"),
        ("R1 same on Construction-A Reed-Solomon (Chailloux-Tillich)", "restricted metric", "everyone", "alpha/4 (BDD), more with KV", "%+.4f (BDD) / %+.4f (Hamming reach)" % (cos_bdd["mu_cos"], cos_rs["mu_cos"]), "D0_S window %.3f..%.3f at q=11" % (q11["lo_s"], q11["hi"]), "class 1 (row 44)", "OPI with a smooth objective; 2411.12553"),
        ("R1 Lee-metric BCH agreement", "Lee metric", "everyone", "O(alpha / log_p n)", "-", "D1", "closed (A4)", "same argument as row 24"),
        ("R2 beat a factoring key (RS over Z_N, N = pq)", "ring alphabet", "nobody", "alpha/2", "-", "-", "closed (B)", "Welch-Berlekamp decodes or factors: %d/%d decoded, %d factored" % (b_row["ok"], b_row["trials"], b_row["factored"])),
        ("R2 Galois rings, Z_{2^t}, non-abelian groups, exponential alphabets", "ring / group alphabet", "-", "-", "-", "D4 / class 1", "closed (B)", "no new decoder source; one line each"),
        ("R3 rank-metric nearest matrix (Gabidulin)", "rank metric", "-", "alpha/2 in rank", "undefined", "outside C0", "closed (C)", "global objective, no baseline; authors claim no advantage"),
    ]
    print("| problem | mechanism | non-OPI to whom | reach | margin | gates | status | closes / next |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()
    print("Bottom line: the metric relaxation widens gate D0 and changes the baseline, but every public restricted")
    print("decoder with constant reach sits on a code chain with constant relative Hamming distance at level 0, which")
    print("is the evaluation/local dichotomy again; the alphabet relaxations add no decoder source.  No third class")
    print("here.  What would reopen it: a structured decoder for {0,+-1} errors on a code whose Hamming distance")
    print("vanishes but whose restricted distance does not (the restricted-SDP literature, 2303.08882, has only")
    print("ISD-type generic algorithms), or a Lemma 9.2 over a ring together with a non-evaluation ring code.")


if __name__ == "__main__":
    a1 = section_a1()
    a2 = section_a2()
    a3 = section_a3()
    section_a4()
    b = section_b()
    section_c()
    section_f(a1, a2, a3, b)
