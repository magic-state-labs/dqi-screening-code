"""Script 31 - New-problem search by mechanism: what besides OPI can give DQI its decoder?

A DQI candidate needs an efficient decoder for ``ker(B^T)`` and no efficient quantizer for
``im(B)``.  The decoder can come from four places: public algebraic structure (OPI and its
curves), locality (sparse parity checks, closed by AMP/OGP and gate D5), secret structure known
to the operator (the McEliece mechanism), or a quantum decoder.  This script prints the numbers
for each problem statement examined, one section each:

  (A5) hidden-LDPC key: Gallager-B thresholds of regular ensembles against gate D2, and the
       information-set probability of recovering a planted weight-w parity check (Prange);
  (A7) hidden multi-twist TGRS key: the Zhu-Jin decodable regime (2512.24217, Thm 4) against the
       shorten-and-square distinguisher (Couvreur-Pratihar-Tanisali-Zappatore 2412.15160, Thm 1);
  (B1) random dense B with the unambiguous-state-discrimination quantum decoder
       (Chailloux-Tillich 2310.20651): the reach it needs and why set-membership noise defeats it;
  (C1) Blokh-Zyablov multilevel concatenation against D2, the first non-algebraic bound class
       that passes;
  (A2) ISIS-infinity on a McEliece key: the Gaussian-heuristic size of the shortest lattice
       vector against the interval half-width, i.e. whether rounding attacks can beat rho;
  (A1) the tangent-space-attack regime number (Lemoine 2505.10184) at the design point;
  (F)  the summary table.

Nothing here is an advantage claim; every number is a gate evaluation or a closure argument.

Run from the repository root:  python scripts/31_new_problem_mechanisms.py
"""

from __future__ import annotations

import os
import sys
from math import comb, erf, exp, log, pi, sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqi_explorer.discovery import (  # noqa: E402
    alternant_half_designed_reach,
    prange_quality,
    qary_entropy,
)
from dqi_explorer.spectral import canonical_asymptotic_quality  # noqa: E402

RESTARTS = 2**40
FAILURE = 0.05


def d2_line(alpha: float) -> float:
    """Gate D2 at rho = 1/2 in terms of relative distance: delta > 1 - sqrt(1 - alpha^2)."""

    return 1 - sqrt(1 - alpha * alpha)


def margin(alpha: float, reach: float, rho: float = 0.5) -> float:
    return canonical_asymptotic_quality(reach, rho) - prange_quality(alpha, rho)


# --------------------------------------------------------------------------
# (A5) hidden LDPC keys: reach versus key recovery
# --------------------------------------------------------------------------
def gallager_b_threshold(dv: int, dc: int) -> float:
    """Largest BSC crossover p0 the Gallager-B decoder corrects for the (dv, dc) regular ensemble.

    Standard density-evolution recursion for Gallager B with the majority threshold b = dv - 1
    (flip when all dv - 1 incoming check messages disagree):
      p_{t+1} = p0 - p0 * ((1 + (1-2p_t)^(dc-1))/2)^(dv-1) + (1-p0) * ((1 - (1-2p_t)^(dc-1))/2)^(dv-1).
    The threshold is found by bisection on p0; it lower-bounds the belief-propagation threshold.
    """

    def converges(p0: float) -> bool:
        p = p0
        for _ in range(2000):
            s = (1 - 2 * p) ** (dc - 1)
            p_next = p0 - p0 * ((1 + s) / 2) ** (dv - 1) + (1 - p0) * ((1 - s) / 2) ** (dv - 1)
            if p_next < 1e-12:
                return True
            if p_next >= p - 1e-15:
                return False
            p = p_next
        return p < 1e-9

    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if converges(mid):
            lo = mid
        else:
            hi = mid
    return lo


def prange_planted_row_probability(alpha: float, weight: int) -> float:
    """Probability that one random information set of im(B) exposes a planted weight-w codeword.

    im(B) has rate alpha.  A weight-w codeword is returned by information-set decoding when
    exactly one of its w nonzero positions falls inside the information set (the rest outside),
    so the probability per information set is about w * alpha * (1 - alpha)^(w - 1).
    """

    return weight * alpha * (1 - alpha) ** (weight - 1)


def section_a5() -> list[dict]:
    print("=" * 100)
    print("(A5) hidden-LDPC key: BP reach must clear D2 while the planted rows stay hidden from ISD")
    print("=" * 100)
    print("  B = P H^T S with H a (dv, dc)-regular parity check; the key holder runs belief propagation on H")
    print("  (licensed only in measure, D4'); the key-less solver must find the planted weight-dc codewords of im(B).")
    print("  Reach is the Gallager-B threshold (a lower bound on the BP threshold); rate loss alpha = dv/dc.")
    print()
    # Belief-propagation thresholds on the BSC where tabulated (Richardson-Urbanke 2001, regular ensembles);
    # BP is the decoder the key holder would actually run, Gallager B is the computable lower bound.
    bp_known = {(3, 6): 0.084, (4, 8): 0.076, (5, 10): 0.068}
    print("  %-8s %-7s %-10s %-9s %-10s %-4s %-9s %-14s %-12s %s" % ("(dv,dc)", "alpha", "reach(GB)", "BP(RU)", "D2 needs", "D2", "mu", "P(ISD finds)", "trials", "verdict"))
    rows = []
    for dv, dc in ((3, 6), (4, 8), (5, 10), (3, 12), (4, 16), (6, 24), (10, 40), (25, 100), (50, 200)):
        alpha = dv / dc
        gb = gallager_b_threshold(dv, dc)
        bp = bp_known.get((dv, dc))
        reach = bp if bp is not None else gb
        needs = d2_line(alpha) / 2
        passes = reach > needs
        mu = margin(alpha, reach)
        prob = prange_planted_row_probability(alpha, dc)
        trials = 1 / prob
        if passes and trials < 2**40:
            verdict = "passes D2 but the key is recovered in %.0e information sets" % trials
        elif passes:
            verdict = "passes D2 and key hard (would be a candidate)"
        else:
            verdict = "fails D2 (reach %.4f < %.4f)" % (reach, needs)
        print("  (%2d,%3d)  %-7.3f %-10.4f %-9s %-10.4f %-4s %-+9.4f %-14.2e %-12.2e %s" % (dv, dc, alpha, gb, ("%.3f" % bp) if bp else "n/a", needs, "ok" if passes else "no", mu, prob, trials, verdict))
        rows.append({"dv": dv, "dc": dc, "alpha": alpha, "reach": reach, "passes": passes, "trials": trials})
    print("  (Where BP is tabulated it is used as the reach; elsewhere Gallager B, which is a lower bound.  Every ensemble")
    print("   that clears D2 has row weight dc <= 16, and a planted row of that weight is exposed by a handful of")
    print("   information sets; at dc >= 100 the reach is far below the line whichever decoder is used.)")
    return rows


# --------------------------------------------------------------------------
# (A7) hidden multi-twist TGRS keys: decodable regime is distinguishable
# --------------------------------------------------------------------------
def section_a7() -> None:
    print()
    print("=" * 100)
    print("(A7) hidden multi-twist TGRS key: Zhu-Jin's decodable regime meets shorten-and-square")
    print("=" * 100)
    print("  Zhu-Jin (2512.24217, Thm 4): unique decoding to (n-k)/2 needs pseudo-dimension k' < (n+k)^2/(4n).")
    print("  CPTZ (2412.15160, Thm 1, third bound): after shortening a positions, dim C_I^2 <= 2(k'-1-a)+1 for any")
    print("  number of twists.  Choose a = 2k'-n+1: the square of the shortened code lies in a proper subspace, so the")
    print("  code is distinguishable from random (then Wieschebrink / Sidelnikov-Shestakov on that GRS square).")
    print()
    print("  %-6s %-5s %-6s %-8s %-8s %-14s %-10s %-14s %s" % ("n", "R", "k", "k'_max", "a*", "[n-a, k-a]", "sq bound", "random square", "verdict"))
    for n in (400, 6000):
        for R in (0.3, 0.5, 0.7, 0.9):
            k = int(round(R * n))
            kprime = (n + k) ** 2 // (4 * n)
            a = 2 * kprime - n + 1
            if a < 0:
                a = 0
            length, dim = n - a, k - a
            bound = 2 * (kprime - 1 - a) + 1
            random_square = min(length, comb(dim + 1, 2)) if dim > 0 else 0
            verdict = "distinguishable" if (dim > 0 and bound < random_square) else "not by this bound"
            print("  %-6d %-5.1f %-6d %-8d %-8d [%4d, %4d]   %-10d %-14d %s" % (n, R, k, kprime, a, length, dim, bound, random_square, verdict))
    n, k, kprime = 400, 300, 306
    a = 2 * kprime - n + 1
    print("  example point: n=400, k=300, k'<=306: a*=%d, shortened [%d, %d], square bound %d < %d: distinguishable"
          % (a, n - a, k - a, 2 * (kprime - 1 - a) + 1, min(n - a, comb(k - a + 1, 2))))


# --------------------------------------------------------------------------
# (B1) random B with the USD quantum decoder
# --------------------------------------------------------------------------
def perp(x: float, q: int) -> float:
    """Chailloux-Tillich's x^perp = (sqrt((1-x)(q-1)) - sqrt(x))^2 / q."""

    return (sqrt((1 - x) * (q - 1)) - sqrt(x)) ** 2 / q


def section_b1() -> None:
    print()
    print("=" * 100)
    print("(B1) random dense B with the unambiguous-state-discrimination (USD) quantum decoder")
    print("=" * 100)
    print("  Chailloux-Tillich (2310.20651, Thm 6): QDP with Bernoulli noise omega on a random code of rate R is solved")
    print("  in polynomial time by per-coordinate USD when q*omega^perp/(q-1) > R.  Thm 8: used inside Regev's reduction")
    print("  the USD decoder returns dual codewords down to exactly Prange's bound (q-1)(1-R)/q, no further.")
    print()
    print("  %-6s %-7s %-10s %-12s %s" % ("q", "alpha", "R=1-alpha", "omega_max", "(largest Bernoulli rate the USD decoder tolerates for ker(B^T))"))
    for q in (2, 11, 401):
        for alpha in (0.12, 0.25, 0.5):
            R = 1 - alpha
            target = (q - 1) * R / q
            # omega_max solves q*omega^perp/(q-1) = R, i.e. omega^perp = target: bisection
            lo, hi = 0.0, (q - 1) / q
            for _ in range(80):
                mid = (lo + hi) / 2
                if q * perp(mid, q) / (q - 1) > R:
                    lo = mid
                else:
                    hi = mid
            print("  %-6d %-7.2f %-10.2f %-12.4f" % (q, alpha, R, lo))
    print()
    print("  Why this does not apply to DQI's error superposition: for an accepted set of size r the per-coordinate")
    print("  noise states sum_u g^(u)|a+u>, a in F_q, have Fourier transforms supported on the r accepted values, so the q")
    print("  states span an r-dimensional space and cannot be discriminated unambiguously (USD success probability 0).")
    for q, r in ((11, 5), (401, 200), (2, 1)):
        print("    q=%d, r=%d: rank of the q shifted states = %d < %d" % (q, r, r, q))
    print("  The optimal decoder (pretty good measurement) reaches the Holevo limit H_q(|g^|^2) (2509.24796) but no")
    print("  efficient implementation is known, and at that limit Regev's reduction outputs minimum-weight dual codewords,")
    print("  i.e. it would solve the short-codeword problem of random codes.  Mechanism closed on random B.")


# --------------------------------------------------------------------------
# (C1) Blokh-Zyablov multilevel concatenation against D2
# --------------------------------------------------------------------------
def _inverse_entropy_table(q: int, points: int = 4000) -> list[tuple[float, float]]:
    top = 1 - 1 / q
    table = []
    for i in range(1, points + 1):
        x = top * i / points
        table.append((qary_entropy(x, q), x))
    return table


def _inverse_entropy(y: float, table: list[tuple[float, float]]) -> float:
    if y <= 0:
        return 0.0
    lo, hi = 0, len(table) - 1
    if y >= table[-1][0]:
        return table[-1][1]
    while lo < hi:
        mid = (lo + hi) // 2
        if table[mid][0] < y:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return table[0][1]
    (y0, x0), (y1, x1) = table[lo - 1], table[lo]
    return x0 + (x1 - x0) * (y - y0) / (y1 - y0)


def blokh_zyablov_rate(delta: float, q: int, table, steps: int = 2000) -> float:
    """R_BZ(delta) = 1 - H_q(delta) - delta * integral_0^{1-H_q(delta)} dx / H_q^{-1}(1-x)."""

    top = 1 - qary_entropy(delta, q)
    if top <= 0:
        return 0.0
    total = 0.0
    for i in range(steps):
        x = (i + 0.5) / steps * top
        total += 1 / _inverse_entropy(1 - x, table)
    return top - delta * total * top / steps


def blokh_zyablov_distance(alpha: float, q: int, table) -> float:
    lo, hi = 1e-6, 1 - 1 / q
    for _ in range(50):
        mid = (lo + hi) / 2
        if blokh_zyablov_rate(mid, q, table) > 1 - alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def zyablov_distance(alpha: float, q: int, table, steps: int = 2000) -> float:
    rate = 1 - alpha
    best = 0.0
    for i in range(1, steps):
        r = rate + (1 - rate) * i / steps
        best = max(best, (1 - rate / r) * _inverse_entropy(1 - r, table))
    return best


def section_c1() -> list[dict]:
    print()
    print("=" * 100)
    print("(C1) Blokh-Zyablov multilevel concatenation: the one non-algebraic bound class that clears D2")
    print("=" * 100)
    print("  Single-level (Zyablov) concatenation fails D2 at every q (sweep row 34).  With the number of levels growing,")
    print("  generalized concatenated codes reach the Blokh-Zyablov bound and multistage GMD decodes to half the designed")
    print("  distance (a worst-case licence).  Threat: the inner blocks are local (C7), and the outer codes are RS-derived.")
    print()
    print("  %-8s %-7s %-9s %-9s %-10s %-7s %-9s %s" % ("q", "alpha", "delta_Z", "delta_BZ", "D2 needs", "D2", "lambda", "mu at rho=1/2"))
    rows = []
    for q in (256, 2**16, 2**32):
        table = _inverse_entropy_table(q)
        for alpha in (0.15, 0.25, 0.4, 0.5, 0.6, 0.7):
            dz = zyablov_distance(alpha, q, table)
            db = blokh_zyablov_distance(alpha, q, table)
            needs = d2_line(alpha)
            passes = db > needs
            lam = db / 2
            mu = margin(alpha, lam) if passes else float("nan")
            print("  %-8d %-7.2f %-9.4f %-9.4f %-10.4f %-7s %-9.4f %s" % (q, alpha, dz, db, needs, "ok" if passes else "no", lam, ("%+.4f" % mu) if passes else "-"))
            rows.append({"q": q, "alpha": alpha, "delta_bz": db, "passes": passes, "mu": mu})
    return rows


# --------------------------------------------------------------------------
# (A2) ISIS-infinity on a McEliece key: can rounding beat rho?
# --------------------------------------------------------------------------
def section_a2() -> None:
    print()
    print("=" * 100)
    print("(A2) ISIS-infinity on a McEliece key: interval accepted sets and the lattice attack surface")
    print("=" * 100)
    print("  Rows accept (Bx)_i mod p in an interval of length r around a centre.  The key-less solver may treat")
    print("  {Bx + p k} as a q-ary lattice of determinant p^(m-n) and round toward the centres.  Gaussian heuristic:")
    print("  shortest vector ~ sqrt(m/(2 pi e)) * p^(1-alpha); rounding beats the random density rho per coordinate")
    print("  only if that length is below the interval half-width r/2 in most coordinates.")
    print()
    print("  %-6s %-7s %-6s %-11s %-9s %-10s %-11s %s" % ("p", "alpha", "m", "GH length", "per-coord", "half-width", "ideal CVP", "reading"))
    for p, alpha, m, r in ((11, 0.12, 2000, 5), (11, 0.12, 6000, 5), (11, 0.30, 6000, 5), (401, 0.25, 400, 200), (401, 0.10, 4000, 200)):
        gh = sqrt(m / (2 * pi * exp(1))) * p ** (1 - alpha)
        per_coordinate = gh / sqrt(m)
        # Gaussian model: an ideal closest-vector oracle leaves a per-coordinate error of scale per_coordinate;
        # the fraction landing within the half-width is erf(half/(sigma sqrt 2)).
        ideal = erf((r / 2) / (per_coordinate * sqrt(2)))
        reading = (
            "an ideal CVP oracle would beat rho; polynomial-time reduction (LLL/BKZ) has a 2^Theta(m) approximation gap"
            if ideal > r / p
            else "even an ideal CVP oracle stays near rho"
        )
        print("  %-6d %-7.2f %-6d %-11.1f %-9.2f %-10.1f %-11.2f %s" % (p, alpha, m, gh, per_coordinate, r / 2, ideal, reading))
    print("  So the key-less hardness of the interval form is an LWE-type assumption (dimension n, modulus p, error")
    print("  bounded by the half-width), not the random-set assumption (B).  The finite control with --sets interval")
    print("  (script 30, attack F_babai: least-squares rounding, no lattice reduction) measures what rounding alone gives.")


# --------------------------------------------------------------------------
# (A1) the tangent-space regime at the design point
# --------------------------------------------------------------------------
def section_a1() -> None:
    print()
    print("=" * 100)
    print("(A1) alternant key: the square-distinguishable regime of Lemoine's tangent-space attack (2505.10184)")
    print("=" * 100)
    print("  The attack recovers the key when the dual alternant code is square-distinguishable, i.e. when")
    print("  C(rs+1, 2) - s*C(r-1, 2) < m (r <= q).  Goppa codes with r >= q-1 have a trivial quadratic hull and are")
    print("  outside it entirely; at constant rate over F_p the Goppa degree r = alpha*m/s exceeds p by far.")
    print()
    for s, m, alpha in ((4, 6000, 0.12), (4, 14640, 0.12), (3, 4913, 0.16), (4, 2000, 0.12)):
        r = int(round(alpha * m / s))
        count = comb(r * s + 1, 2) - s * comb(r - 1, 2)
        print("  s=%d m=%5d alpha=%.2f r=%3d: C(rs+1,2) - s C(r-1,2) = %9d %s m  -> %s; r >= p-1: %s"
              % (s, m, alpha, r, count, "<" if count < m else ">=", "square-distinguishable" if count < m else "not distinguishable by squares", "yes" if r >= 10 else "no"))
    print("  reach at the design point: alpha/(2s) = %.4f" % alternant_half_designed_reach(0.12, 4))


# --------------------------------------------------------------------------
# (F) the summary table
# --------------------------------------------------------------------------
def section_f(a5_rows, c1_rows) -> None:
    print()
    print("=" * 100)
    print("(F) summary table")
    print("=" * 100)
    bz = next(r for r in c1_rows if r["q"] == 2**16 and abs(r["alpha"] - 0.25) < 1e-9)
    ldpc_best = max((r for r in a5_rows if r["passes"]), key=lambda r: r["trials"], default=None)
    rows = [
        ("A1 max-agreement on a McEliece key (alternant, s=4, p=11)", "secret structure", "key-less solver", "alpha/8", "+0.050 finite", "D0-D6 pass", "script 30", "assumption (A) superpolynomial (syzygy)"),
        ("A2 ISIS-infinity on a McEliece key (interval sets)", "secret structure", "key-less solver", "alpha/8", "same", "D0-D6 pass", "see script 30 --sets interval", "lattice rounding cannot beat rho at p=11 (GH); measured by F_babai"),
        ("A3 low-weight parity checks of a public McEliece code", "secret structure", "-", "-", "-", "D3 fails", "closed", "accepted set {0} has density 1/p"),
        ("A4 max-agreement on a lattice (LWE/GPV) key", "secret structure", "-", "-", "-", "-", "closed", "the trapdoor solves the primal BDD directly; classical key holder beats DQI"),
        ("A5 max-agreement on a hidden-LDPC key", "secret structure (in measure)", "-", "GB threshold", "-", "D2 vs ISD", "closed", "passing ensembles expose a planted row in <= %.0e information sets" % (ldpc_best["trials"] if ldpc_best else float("nan"))),
        ("A6 max-agreement on a QC-MDPC (BIKE) key", "secret structure", "-", "0.005", "-", "D2 fails", "closed", "reach too small at alpha=1/2"),
        ("A7 max-agreement on a hidden multi-twist TGRS key", "secret structure", "-", "alpha/2", "OPI-level", "D0-D6 pass", "closed", "decodable regime is distinguishable by shorten-and-square"),
        ("A8 hidden GRS / AG / RM / concatenated / polar / Gabidulin keys", "secret structure", "-", "-", "-", "-", "closed", "structure recovered by published attacks"),
        ("A9 SSAG / wild Goppa keys", "secret structure", "key-less solver", "~alpha/(2s)", "~+0.05", "D0-D6 pass", "on paper", "variants of A1"),
        ("B1 random B with the USD quantum decoder", "quantum decoder", "-", "Prange", "0", "-", "closed", "USD = Prange (CT Thm 8); set-membership noise defeats USD"),
        ("B2 random B with the PGM quantum decoder", "quantum decoder", "-", "Holevo limit", "-", "-", "closed", "no efficient PGM; would break code-based cryptography"),
        ("C1 multilevel concatenated (Blokh-Zyablov) code, q=2^16, alpha=0.25", "public non-algebraic", "everyone", "%.4f" % (bz["delta_bz"] / 2), "%+.4f" % bz["mu"], "D0-D3 pass", "C7 not examined", "inner-block locality threat; outer codes RS-derived"),
    ]
    print("| problem | mechanism | non-OPI to whom | reach | margin | gates | status | closes / next |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    a5 = section_a5()
    section_a7()
    section_b1()
    c1 = section_c1()
    section_a2()
    section_a1()
    section_f(a5, c1)
