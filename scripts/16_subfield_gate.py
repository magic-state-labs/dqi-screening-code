"""Gate (D6): subfield structure of the accepted sets.

Over F_q with q = p^b, an accepted set that contains a coset of an F_p-subspace of codimension c
turns each constraint into c F_p-linear equations in the b*n base-field coordinates of x.
Information-set decoding over F_p then reaches Q_Pr(alpha_eff, rho) with
alpha_eff = min(1, b*alpha/c), provided the equations are independent.

This script computes, for random accepted sets of density rho:
  * the expected-count estimate of the largest contained affine subspace, via Gaussian binomials
    and exact hypergeometric containment probabilities;
  * a Monte Carlo check: exhaustive over all affine subspaces for F_32 (b=5), randomized greedy
    growth for F_256 (b=8);
and the recomputed margins for several code families over extension fields.
"""
import math
import random
from itertools import combinations


def gauss_binom(b, k, p):
    num = den = 1
    for i in range(k):
        num *= p ** (b - i) - 1
        den *= p ** (i + 1) - 1
    return num // den


def contain_prob(q, size, points):
    """Probability that a uniformly random `size`-subset of F_q contains `points` given points."""
    if points > size:
        return 0.0
    logp = 0.0
    for i in range(points):
        logp += math.log(size - i) - math.log(q - i)
    return math.exp(logp)


def expected_counts(p, b, rho):
    q = p ** b
    size = round(rho * q)
    return {k: gauss_binom(b, k, p) * p ** (b - k) * contain_prob(q, size, p ** k) for k in range(b + 1)}


def span(vectors):
    s = {0}
    for v in vectors:
        s |= {x ^ v for x in s}
    return frozenset(s)


def all_linear_subspaces(b):
    subs = {frozenset({0})}
    frontier = {frozenset({0})}
    while frontier:
        nxt = set()
        for sub in frontier:
            for v in range(1, 2 ** b):
                if v not in sub:
                    nxt.add(frozenset(sub | {x ^ v for x in sub}))
        subs |= nxt
        frontier = nxt
    return subs


def largest_affine_exhaustive(accept, b, subspaces_by_dim):
    for k in range(b, -1, -1):
        for sub in subspaces_by_dim.get(k, []):
            for a in range(2 ** b):
                if all(accept[a ^ x] for x in sub):
                    return k
    return -1


def largest_affine_greedy(accept, b, rng, tries):
    members = [v for v in range(2 ** b) if accept[v]]
    best = 0
    for _ in range(tries):
        a0 = rng.choice(members)
        cur, dim = {0}, 0
        dirs = list(range(1, 2 ** b))
        rng.shuffle(dirs)
        for d in dirs:
            if d in cur:
                continue
            new = {x ^ d for x in cur}
            if all(accept[a0 ^ x] for x in new):
                cur |= new
                dim += 1
        best = max(best, dim)
    return best


def semicircle(lam, rho):
    return (math.sqrt(lam * (1 - rho)) + math.sqrt(rho * (1 - lam))) ** 2


def qpr(alpha, rho):
    return rho + (1 - rho) * alpha


def main():
    rho = 0.5
    rng = random.Random(4242)
    print("Expected number of affine subspaces of each dimension inside a random density-1/2 set")
    for p, b in ((2, 5), (2, 8)):
        ec = expected_counts(p, b, rho)
        kstar = max(k for k, v in ec.items() if v >= 1)
        print(f"  F_{p**b}: " + ", ".join(f"dim {k}: {v:.3g}" for k, v in ec.items())
              + f"  -> estimate k*={kstar}, c*={b - kstar}")

    # Monte Carlo, F_32 exhaustive
    subs = all_linear_subspaces(5)
    by_dim = {}
    for s in subs:
        by_dim.setdefault(int(math.log2(len(s))), []).append(s)
    hist32 = {}
    for _ in range(300):
        members = set(rng.sample(range(32), 16))
        accept = [v in members for v in range(32)]
        k = largest_affine_exhaustive(accept, 5, by_dim)
        hist32[k] = hist32.get(k, 0) + 1
    print("  F_32 Monte Carlo (300 random sets, exhaustive): largest affine dim histogram", dict(sorted(hist32.items())))

    # Monte Carlo, F_256 greedy
    hist256 = {}
    for _ in range(40):
        members = set(rng.sample(range(256), 128))
        accept = [v in members for v in range(256)]
        k = largest_affine_greedy(accept, 8, rng, 1500)
        hist256[k] = hist256.get(k, 0) + 1
    print("  F_256 Monte Carlo (40 random sets, greedy lower bound): largest affine dim histogram", dict(sorted(hist256.items())))

    # Recomputed margins for several code families
    families = [
        # name, b, alpha, Q_DQI at instance, Q_DQI limit
        ("twisted RS over F_401", 1, 0.25, None, None),
        ("interleaved RS over F_401^4", 4, 0.25, 0.8279, semicircle(0.125, rho)),
        ("elliptic over F_256", 8, 0.25, 0.8267, semicircle((0.25 - 1 / 288) / 2, rho)),
        ("Suzuki over F_32", 5, 0.50, 0.8911, semicircle((0.5 - 124 / 1024) / 2, rho)),
        ("GS tower over F_256", 8, 0.10, None, semicircle((0.10 - 1 / 15) / 2, rho)),
    ]
    print("\nGate (D6) on several code families, random accepted sets, rank hypothesis assumed")
    for name, b, alpha, q_inst, q_lim in families:
        if b == 1:
            print(f"  {name}: prime base field, b=1, D6 void")
            continue
        if name.startswith("interleaved"):
            # block alphabet F_401^4 is a 4-dim F_401-space; a random half set contains no affine
            # F_401-line: expected count ~ 401^6 * 2^-401, so c = 4 and alpha_eff = alpha
            lines = gauss_binom(4, 1, 401) * 401 ** 3
            print(f"  {name}: expected affine F_401-lines inside a random half set ~ {lines:.3g} * 2^-401 ~ 0;"
                  f" c=4, alpha_eff={alpha:.4f}, margin unchanged")
            continue
        for c in ((3, 2) if b == 5 else (4,)):
            a_eff = min(1.0, b * alpha / c)
            line = f"  {name}: c={c}, alpha_eff={a_eff:.4f}, Q_Pr(alpha_eff)={qpr(a_eff, rho):.4f}"
            if q_inst is not None:
                line += f", instance margin {q_inst - qpr(a_eff, rho):+.4f}"
            line += f", limit margin {q_lim - qpr(a_eff, rho):+.4f}"
            print(line)

    # the finite-control check: stuck cells, nibble mapping, b=8, c=1, alpha=28/240
    a = 28 / 240
    print("\nStuck-cell control (nibble mapping, c=1): alpha_eff=%.4f, Q_Pr(alpha_eff)=%.4f; "
          "measured best 237/240=%.4f with full-rank bit equations" % (min(1, 8 * a), qpr(min(1, 8 * a), rho), 237 / 240))


if __name__ == "__main__":
    main()
