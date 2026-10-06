"""Script 34 - Hidden keys whose secret is not a subfield subcode: the screen.

The class-2
problem, "beat a public key", had one member: max-agreement on an alternant (subfield
subcode) McEliece key.  Code-based cryptography knows two other ways to hide a
Reed--Solomon code that are not subfield tricks, and this script screens both as DQI
families (``dqi_explorer/hidden_keys.py``):

  K1  sparse-mixed key (BBCRS idea): ``H_pub = S H T``, ``T`` of exact column weight ``w``;
      the key holder decodes to ``floor((r-1)/2) // w`` errors, reach ``alpha/(2w)``.
  K2  column-inserted key (RLCE idea): ``u`` dense-functional coordinates mixed pairwise
      into a GRS code; the key holder decodes to ``floor((m0-k-1)/2)`` errors, reach
      ``(m0-k)/(2m) = alpha/(2(1+u/r))``.

Both live on a prime field with ``m <= p - 1``, so gate D6 is void and the field must be
as long as the instance (``p = 6007`` at ``m = 6000``).  That puts the unsatisfiable regime
at ``alpha < log_p 2``, which is where the screen looks.

(A) the analytic gates D0, D2, D3 (= D2 at rho = 1/2), the margin, the regime factor, ``Q*``
    and the restart-converse threshold, with the attack-literature regime flags; the finite
    Lemma 9.2 values at the design points.
(B) the numeric square-code battery on built instances (``--quick``: m = 600 only;
    default adds m = 2000): the ``w = 1`` and ``u = 0`` controls must collapse to
    ``2 dim - 1`` on the image side, a random matrix must be full, and the candidates are
    read off against both.  Decoders are verified on random patterns.
(C) what the battery proves and what it does not, the ``w = 2`` count, and the closed
    expanded-RS key (K3).

Nothing here is an advantage claim.  Run from the repository root:
  python scripts/34_hidden_key_screen.py [--quick]
"""

from __future__ import annotations

import os
import sys
import time
from math import comb, log, sqrt

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqi_explorer.discovery import (  # noqa: E402
    column_inserted_reach,
    licensed_radius_is_realizable,
    prange_quality,
    qary_entropy,
    sparse_mixed_reach,
    sparse_mixed_reach_window,
)
from dqi_explorer.hidden_keys import (  # noqa: E402
    build_column_inserted_key_instance,
    build_sparse_mixed_key_instance,
    random_matrix_battery,
    square_battery,
    verify_decoder,
)
from dqi_explorer.spectral import canonical_asymptotic_quality, canonical_finite_quality  # noqa: E402

QUICK = "--quick" in sys.argv
RESTARTS = 2**40
FAILURE = 0.05


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


def hoeffding_length(alpha: float, margin: float, draws: int, failure: float) -> float | None:
    if margin <= 0:
        return None
    return (1 - alpha) * log(draws / failure) / (2 * margin**2)


def margin(alpha: float, reach: float, rho: float = 0.5) -> float:
    return canonical_asymptotic_quality(reach, rho) - prange_quality(alpha, rho)


# --------------------------------------------------------------------------
# (A) analytic screen
# --------------------------------------------------------------------------
PRIMES = (1009, 2003, 4001, 6007)
ALPHAS = (0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25, 0.30)
WEIGHTS = (2, 3, 4)
U_RATIOS = (0.5, 1.0, 1.5)


def screen_row(kind: str, parameter: float, p: int, alpha: float, rho: float = 0.5) -> dict:
    if kind == "sparse_mixed":
        reach = sparse_mixed_reach(alpha, int(parameter))
        # the square of the public code at w = 2 has dimension <= 2r - 1 + m/2 (section C)
        flag = "square-distinguishable (alpha < 1/4)" if parameter == 2 and alpha < 0.25 else ""
    else:
        reach = column_inserted_reach(alpha, parameter)
        flag = "CLT-2019 regime (u < r)" if parameter < 1 else ("boundary u = r (unverified)" if parameter == 1 else "")
    mu = margin(alpha, reach, rho)
    regime = p**alpha * rho
    threshold = hoeffding_length(alpha, mu, RESTARTS, FAILURE)
    return {
        "kind": kind,
        "parameter": parameter,
        "p": p,
        "alpha": alpha,
        "lambda": reach,
        "H_p(lambda)": qary_entropy(reach, p),
        "D0": licensed_radius_is_realizable(alpha, reach, p),
        "D2": reach * (1 - reach) > alpha * alpha / 4,
        "mu": mu,
        "regime": regime,
        "unsat": regime < 1,
        "Q*": existential_quality(alpha, rho, p),
        "threshold": threshold,
        "buildable": threshold is not None and threshold <= p - 1,
        "flag": flag,
    }


def print_rows(rows: list[dict]) -> None:
    header = "  %-13s %-5s %-5s %-6s %-8s %-8s %-3s %-3s %-8s %-7s %-6s %-6s %-9s %s"
    print(header % ("kind", "par", "p", "alpha", "lambda", "H_p(lam)", "D0", "D2", "mu", "p^a/2", "unsat", "Q*", "threshold", "verdict"))
    print("  " + "-" * 120)
    for row in rows:
        verdict = []
        if not row["D0"]:
            verdict.append("D0 FAIL")
        if not row["D2"]:
            verdict.append("D2 FAIL")
        if row["threshold"] is None:
            verdict.append("no margin")
        elif not row["buildable"]:
            verdict.append("threshold > p-1")
        if not row["unsat"]:
            verdict.append("SAT regime")
        if row["flag"]:
            verdict.append(row["flag"])
        if not verdict:
            verdict.append("passes on paper")
        print(
            "  %-13s %-5s %-5d %-6.3f %-8.4f %-8.4f %-3s %-3s %+-8.4f %-7.3f %-6s %-6.3f %-9s %s"
            % (
                row["kind"],
                ("w=%d" % row["parameter"]) if row["kind"] == "sparse_mixed" else ("u/r=%.1f" % row["parameter"]),
                row["p"],
                row["alpha"],
                row["lambda"],
                row["H_p(lambda)"],
                "ok" if row["D0"] else "no",
                "ok" if row["D2"] else "no",
                row["mu"],
                row["regime"],
                "yes" if row["unsat"] else "no",
                row["Q*"],
                "%.0f" % row["threshold"] if row["threshold"] else "-",
                "; ".join(verdict),
            )
        )


def section_a() -> None:
    print("=" * 110)
    print("(A) analytic screen at rho = 1/2: D0 (sphere packing), D2 (reach vs rate), margin, regime, Q*, restart threshold")
    print("=" * 110)
    print("  reach laws: sparse-mixed alpha/(2w); column-inserted alpha/(2(1+u/r)).  D3 = D2 at rho = 1/2; D4 worst-case BM; D6 void.")
    print("  D2 window: sparse-mixed alpha < %s" % ", ".join("w=%d: %.3f" % (w, sparse_mixed_reach_window(w)) for w in WEIGHTS))
    print("  the prime field must have p > m, so the regime factor p^alpha/2 is below 1 only for alpha < log_p 2 (0.080 at p = 6007).")
    print()
    print("  p = 6007 (m = 6000), all alphas:")
    rows = [screen_row("sparse_mixed", w, 6007, a) for w in WEIGHTS for a in ALPHAS]
    rows += [screen_row("column_inserted", u, 6007, a) for u in U_RATIOS for a in ALPHAS]
    print_rows(rows)
    print()
    print("  other primes at alpha = 0.07 (the design point) and 0.12:")
    rows = [screen_row("sparse_mixed", 3, p, a) for p in PRIMES for a in (0.07, 0.12)]
    rows += [screen_row("column_inserted", 1.0, p, a) for p in PRIMES for a in (0.07, 0.12)]
    print_rows(rows)
    print()
    print("  finite Lemma 9.2 values at the design points (m = 6000, p = 6007, accepted sets of size 3003):")
    print("  %-22s %-6s %-5s %-5s %-8s %-8s %-8s %-10s %s" % ("family", "alpha", "n", "ell", "Q_fin", "Q_Pr", "mu_fin", "threshold", "above?"))
    for label, alpha, ell_of in (
        ("sparse-mixed w=2", 0.07, lambda r: ((r - 1) // 2) // 2),
        ("sparse-mixed w=3", 0.07, lambda r: ((r - 1) // 2) // 3),
        ("sparse-mixed w=4", 0.07, lambda r: ((r - 1) // 2) // 4),
        ("column-inserted u=r", 0.07, lambda n: (n // 2 - 1) // 2),
        ("sparse-mixed w=3", 0.05, lambda r: ((r - 1) // 2) // 3),
    ):
        m, p = 6000, 6007
        n = int(round(alpha * m))
        ell = ell_of(n)
        q = canonical_finite_quality(m, p, (p - 1) // 2, ell)
        rho = ((p - 1) // 2) / p
        q_pr = prange_quality(n / m, rho)
        mu_fin = q.value - q_pr
        thr = hoeffding_length(n / m, mu_fin, RESTARTS, FAILURE)
        print("  %-22s %-6.3f %-5d %-5d %-8.4f %-8.4f %+-8.4f %-10s %s" % (label, alpha, n, ell, q.value, q_pr, mu_fin, "%.0f" % thr if thr else "-", "yes" if thr and m >= thr else "no"))


# --------------------------------------------------------------------------
# (B) numeric battery on built instances
# --------------------------------------------------------------------------
def battery_line(name: str, battery: dict) -> str:
    parts = []
    for key, value in battery.items():
        if not isinstance(value, dict):
            continue
        mark = "" if value["full"] else " <-- deficit"
        parts.append("%s: dim %d len %d rank %d/%d%s" % (key.replace("_shorten_", "@"), value["dimension"], value["length"], value["square_rank"], value["random_expectation"], mark))
    return "  %-26s %s" % (name, " | ".join(parts))


def section_b() -> None:
    print()
    print("=" * 110)
    print("(B) square-code battery on built instances (random codewords, plain and auto-shortened, both sides)")
    print("=" * 110)
    print("  a Schur-product distinguisher fires when a rank falls short of min(length, C(dim+1,2)); the controls calibrate it.")
    sizes = [(600, 1009, 0.2)] if QUICK else [(600, 1009, 0.2), (2000, 2003, 0.07)]
    for m, p, alpha in sizes:
        print()
        print("  m = %d, p = %d, alpha = %.2f" % (m, p, alpha))
        rng = np.random.default_rng(34)
        t0 = time.time()
        for w in (1, 2, 3):
            inst = build_sparse_mixed_key_instance(m=m, seed=34, p=p, alpha=alpha, w=w)
            dec = verify_decoder(inst, 40, 1, beyond_trials=6)
            name = "sparse-mixed w=%d%s" % (w, " (control)" if w == 1 else "")
            print(battery_line(name, square_battery(inst, rng)))
            print("  %-26s decoder %d/%d exact, ell %d, beyond-radius wrong %d" % ("", dec["exact_recoveries"], dec["trials"], inst.ell, dec["beyond_radius"]["wrong"]))
        for u in (0.0, 0.5, 1.0):
            inst = build_column_inserted_key_instance(m=m, seed=34, p=p, alpha=alpha, u_ratio=u)
            dec = verify_decoder(inst, 40, 1, beyond_trials=6)
            name = "column-inserted u/r=%.1f%s" % (u, " (control)" if u == 0 else "")
            print(battery_line(name, square_battery(inst, rng)))
            print("  %-26s decoder %d/%d exact, ell %d, u %d r %d" % ("", dec["exact_recoveries"], dec["trials"], inst.ell, inst.u, inst.r))
        n = int(round(alpha * m))
        print(battery_line("random matrix (control)", random_matrix_battery(m, n, p, rng)))
        print("  (%.0f s)" % (time.time() - t0))


# --------------------------------------------------------------------------
# (C) reading
# --------------------------------------------------------------------------
def section_c() -> None:
    print()
    print("=" * 110)
    print("(C) what the battery says")
    print("=" * 110)
    print("  w = 2: a public coordinate mixes two GRS coordinates a, b, so a square codeword restricted to the block is a")
    print("  combination of g_a g'_a, g_b g'_b (coordinates of the GRS-dual square, dimension 2r-1) and g_a g'_b + g_b g'_a")
    print("  (one cross term per block, m/2 of them): dim(square) <= 2r - 1 + m/2, below m exactly when alpha < 1/4.")
    for m, r in ((600, 120), (2000, 140), (6000, 420)):
        print("    m = %d, r = %d: bound %d against length %d (%s)" % (m, r, 2 * r - 1 + m // 2, m, "distinguishable" if 2 * r - 1 + m // 2 < m else "full"))
    print("  The unsatisfiable regime at p ~ m needs alpha < 0.08, so w = 2 is closed there; w >= 3 has 3 cross terms per")
    print("  block (m per type) and the battery finds no deficit, plain or shortened.")
    print("  Column-inserted keys: no deficit at u/r = 0.5, 1.0 or 1.5 with random shortening positions.  The published")
    print("  attack (Couvreur-Lequesne-Tillich 2019, u < r) chooses its shortening positions; the battery does not reproduce")
    print("  it and is not evidence against it.  u/r = 1 is screened as the boundary of that attack's stated regime.")
    print("  K3, expanded RS with secret bases per position: Couvreur-Lequesne (IEEE T-IT 2022, arXiv 2009.05826) distinguish")
    print("  every subspace subcode of RS whose subspaces have dimension above s/2 in polynomial time; the full expansion")
    print("  (dimension s) is inside that regime.  Closed; no instance generator was written.")
    print("  Not shown by any of this: hardness.")


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
