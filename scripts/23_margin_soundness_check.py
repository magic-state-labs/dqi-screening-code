"""Soundness of the rules with respect to the margin.

The paper fixes the pipeline B -> (alpha, delta, licensed lambda, rho) -> Q_DQI(lambda, rho),
Q_Pr(alpha, rho) -> mu = Q_DQI - Q_Pr, with Q_DQI the semicircle law saturated at 1 for
lambda > 1 - rho.  The advantage margin over the best applicable baseline is
mu+ = Q_DQI - max(Q_Pr(alpha), Q_Pr(alpha_sub) [D6], Q* [D5]).

For each rule this sweeps a grid and checks the claim of the soundness theorem:
  D1       delta -> 0 forces mu < 0                                   (sound)
  D2       at rho = 1/2, lambda = delta/2: fires <=> mu <= 0           (exact)
  D3       fires (with alpha/2 <= 1 - rho) => mu <= 0 for all lambda <= alpha/2   (sound)
           and the same claim WITHOUT the hypothesis, to exhibit the defect
  window   the general density window with lambda <= 1 - rho: fires <=> mu <= 0 (exact)
  D4       lambda = 0 => mu = -(1 - rho) alpha <= 0                    (sound)
  D5/exist realisable lambda (<= alpha/2 and <= H_q^-1(alpha)) => Q_DQI <= Q*   (bound)
  cap      rho = 1/2, lambda <= alpha/2 => mu <= kappa(alpha)          (bound)
  D6       alpha_sub >= alpha => mu+ <= mu, and fires <=> mu+ <= 0     (sound for mu+)
Then it checks, for every family in 20_passing_families_screen.py, that each rule that fires is
matched by mu <= 0 or mu+ <= 0 as its claim says.

Usage: python scripts/23_margin_soundness_check.py
"""
import importlib.util
import math

import numpy as np

EPS = 1e-12


def load_screen():
    spec = importlib.util.spec_from_file_location("screen20", "scripts/20_passing_families_screen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def semicircle(lam, rho):
    return (np.sqrt(lam * (1 - rho)) + np.sqrt(rho * (1 - lam))) ** 2


def qdqi(lam, rho):
    return np.where(lam > 1 - rho, 1.0, semicircle(lam, rho))


def qpr(alpha, rho):
    return rho + (1 - rho) * alpha


def hq(x, q):
    x = np.clip(x, 1e-300, 1 - 1e-15)
    return x * math.log(q - 1, q) - x * np.log(x) / math.log(q) - (1 - x) * np.log(1 - x) / math.log(q)


def hq_inv(a, q):
    lo, hi = np.zeros_like(a), np.full_like(a, 1 - 1 / q)
    for _ in range(80):
        mid = (lo + hi) / 2
        big = hq(mid, q) > a
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    return lo


def kl(a, b):
    a = np.clip(a, 1e-15, 1 - 1e-15)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def qstar(alpha, q, rho):
    sat = q ** alpha * rho >= 1
    lo, hi = np.array(rho, dtype=float) + 0 * alpha, np.full_like(alpha, 1 - 1e-15)
    for _ in range(80):
        mid = (lo + hi) / 2
        below = kl(mid, rho) < alpha * math.log(q)
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.where(sat, 1.0, lo)


def flat_second_moment(b, k, s):
    """Exact E[X], E[X^2] and the Paley-Zygmund bound for X = number of affine k-flats of F_2^b
    inside a uniform s-subset."""
    from fractions import Fraction
    q, size = 2 ** b, 2 ** k
    subs = {frozenset({0})}
    for _ in range(k):
        subs = {frozenset(t | {x ^ v for x in t}) for t in subs for v in range(1, q) if v not in t}
    flats = list({frozenset(a ^ v for v in t) for t in subs for a in range(q)})

    def contain(t):
        return Fraction(math.comb(q - t, s - t), math.comb(q, s)) if t <= s else Fraction(0)

    ex = len(flats) * contain(size)
    prof = {}
    for other in flats:
        i = len(flats[0] & other)
        prof[i] = prof.get(i, 0) + 1
    ex2 = sum(len(flats) * c * contain(2 * size - i) for i, c in prof.items())
    return float(ex), float(ex2), float(ex * ex / ex2)


def report(name, violations, total, note=""):
    status = "PASS" if violations == 0 else f"FAIL ({violations} of {total})"
    print(f"  {name:62s} {status} {note}")
    return violations


def main():
    alphas = np.linspace(0.01, 0.99, 99)
    rhos = np.array(sorted(set([0.01, 0.02, 0.05] + list(np.round(np.linspace(0.1, 0.9, 17), 3)) + [0.95, 0.98])))
    fr = np.linspace(0.0, 1.0, 41)  # lambda as a fraction of alpha/2
    A, R, F = np.meshgrid(alphas, rhos, fr, indexing="ij")
    LAM = F * A / 2
    MU = qdqi(LAM, R) - qpr(A, R)
    total = MU.size
    bad = 0
    print("Soundness sweep over", total, "grid points (alpha, rho, lambda <= alpha/2)")

    # D1: delta -> 0 (lambda -> 0) forces mu -> -(1 - rho) alpha < 0
    tiny = qdqi(1e-9 * A / 2, R) - qpr(A, R)
    bad += report("D1: lambda -> 0 gives mu < 0", int(np.sum(tiny >= 0)), total)

    # D2 at rho = 1/2 with lambda = delta/2, delta in (0, alpha]: fires <=> mu <= 0
    a = alphas[:, None]
    delta = np.linspace(0.001, 1, 400)[None, :] * a
    mu2 = qdqi(delta / 2, 0.5) - qpr(a, 0.5)
    fires2 = delta <= 1 - np.sqrt(1 - a ** 2)
    viol = np.sum(fires2 & (mu2 > EPS)) + np.sum(~fires2 & (mu2 <= -EPS))
    bad += report("D2 (rho = 1/2): fires <=> mu <= 0", int(viol), mu2.size)

    # D3 (MDS form): fires => mu <= 0 for every lambda <= alpha/2
    fires3 = R * (1 - R) <= A / (8 * (1 - A / 2))
    hyp3 = A / 2 <= 1 - R
    report("D3 without the hypothesis alpha/2 <= 1 - rho (expected to fail)",
           int(np.sum(fires3 & (MU > EPS))), total)
    ex = np.argwhere(fires3 & (MU > EPS))
    if len(ex):
        i, j, k = ex[np.argmax([R[tuple(e)] for e in ex])]
        print(f"      e.g. alpha={A[i, j, k]:.2f}, rho={R[i, j, k]:.2f}, lambda={LAM[i, j, k]:.3f}: "
              f"Q_DQI={qdqi(LAM[i, j, k], R[i, j, k]):.4f} > Q_Pr={qpr(A[i, j, k], R[i, j, k]):.4f}")
    bad += report("D3 with the hypothesis: fires => mu <= 0", int(np.sum(fires3 & hyp3 & (MU > EPS))), total)

    # general density window (lambda <= 1 - rho): fires <=> mu <= 0
    ok = LAM <= 1 - R
    lhs = LAM - 2 * LAM * R + 2 * np.sqrt(LAM * R * (1 - LAM) * (1 - R))
    firesw = lhs <= A * (1 - R)
    viol = np.sum(ok & firesw & (MU > EPS)) + np.sum(ok & ~firesw & (MU <= -EPS))
    bad += report("general window (lambda <= 1 - rho): fires <=> mu <= 0", int(viol), int(ok.sum()))

    # D4: lambda = 0
    mu0 = qdqi(0 * A, R) - qpr(A, R)
    bad += report("D4: lambda = 0 gives mu = -(1 - rho) alpha <= 0", int(np.sum(mu0 > EPS)), total)

    # cap at rho = 1/2: mu <= kappa(alpha) for lambda <= alpha/2
    u = alphas[:, None] / 2
    lamc = np.linspace(0, 1, 201)[None, :] * u
    muc = qdqi(lamc, 0.5) - qpr(alphas[:, None], 0.5)
    kappa = np.sqrt(u * (1 - u)) - u
    bad += report("cap (rho = 1/2): mu <= kappa(alpha)", int(np.sum(muc > kappa + EPS)), muc.size)

    # D5 / existential: realisable lambda <= min(alpha/2, H_q^-1(alpha)) gives Q_DQI <= Q*
    for q in (2, 4, 8, 32, 256, 401, 4001):
        rset = sorted(set([1 / q] + [x for x in rhos if x * q >= 1]))
        Aq, Rq, Fq = np.meshgrid(alphas, np.array(rset), fr, indexing="ij")
        cap_l = np.minimum(Aq / 2, hq_inv(Aq, q))
        L = Fq * cap_l
        v = qdqi(L, Rq) > qstar(Aq, q, Rq) + 1e-9
        bad += report(f"existential (q = {q}): Q_DQI <= Q* on realisable lambda", int(v.sum()), v.size)

    # D6: mu+ <= mu whenever alpha_sub >= alpha; fires <=> mu+ <= 0 by definition
    asub = np.minimum(1.0, A * np.array([1, 1.5, 2, 3])[None, None, None, :].max())
    muplus = qdqi(LAM, R) - np.maximum(qpr(A, R), qpr(asub, R))
    bad += report("D6: mu+ <= mu", int(np.sum(muplus > MU + EPS)), total)

    print("\nFamilies of 20_passing_families_screen.py: does every firing rule match its claim?")
    screen = load_screen()
    fams = []
    for p in (401, 1009, 4001):
        for al in (0.1, 0.25):
            fams.append((f"RS / OPI alpha={al}", f"F_{p}", p, 1, p - 1, round((p - 1) * al), 0))
    for q in (401, 1009):
        M = (q - 1) // 2
        fams.append(("star-twisted RS", f"F_{q}", q, 1, M, M // 4, 0))
    for p in (409, 2003):
        fams.append(("elliptic", f"F_{p}", p, 1, p, round(p / 4), 1))
    for p in (7, 11, 13):
        fams.append(("Hermitian", f"F_{p}^2", p, 2, p ** 3, round(p ** 3 / 4), p * (p - 1) // 2))
    fams += [("Hermitian OPI", "F_256", 2, 8, 4096, 1024, 120),
             ("elliptic max length", "F_256", 2, 8, 288, 72, 1),
             ("GS tower alpha=0.1", "F_256", 2, 8, 4096, 410, round(4096 / 15)),
             ("Suzuki", "F_32", 2, 5, 1024, 512, 124),
             ("norm-trace anchor", "F_64", 2, 6, 1024, 471, 150)]
    claims = {"D0": None, "D1": "mu", "D2": "mu", "D3": "mu", "window": "mu", "D6": "mu_plus"}
    fam_bad = 0
    for f in fams:
        e = screen.evaluate(*f)
        fired = [k for k, v in e["fires"].items() if v]
        wrong = [k for k in fired if claims[k] and e[claims[k]] > EPS]
        fam_bad += len(wrong)
        print(f"  {e['name']:22s} {e['field']:8s} mu={e['mu']:+.4f} mu+={e['mu_plus']:+.4f} "
              f"fires={fired or '-'} {'CONSISTENT' if not wrong else 'INCONSISTENT: ' + str(wrong)}")

    print("\nRigorous D6 inputs for the families it filters (the screen's coset shares are estimates):")
    suz = screen.evaluate("Suzuki", "F_32", 2, 5, 1024, 512, 124)
    print("  Suzuki, F_32, 16-sets: a set with no affine 2-flat is a Sidon set; the largest in F_2^5 has 7")
    print("    elements (exhaustive, script 17), so every row has c <= 3 and alpha_sub >= 5*0.5/3 = 0.833:")
    print(f"    Q_Pr(alpha_sub) >= 0.9167 > Q_DQI = {suz['q_dqi']:.4f}, so mu+ <= {suz['q_dqi'] - 0.9167:+.4f}")
    ex, ex2, bound = flat_second_moment(6, 3, 32)
    nt = screen.evaluate("norm-trace anchor", "F_64", 2, 6, 1024, 471, 150)
    need = 2 * (nt["q_dqi"] - 0.5)  # alpha_sub with Q_Pr(alpha_sub) = Q_DQI at rho = 1/2
    base = 6 * nt["alpha"] / 4
    frac_needed = max(0.0, 4 * (need - base))  # alpha_sub = 6a/4 + f/4 while f <= 6a/3
    guaranteed = min(bound, 6 * nt["alpha"] / 3)
    print(f"  norm-trace, F_64, 32-sets: E[X]={ex:.2f}, E[X^2]={ex2:.2f} for affine 3-flats, so a row contains")
    print(f"    one with probability >= {bound:.4f} (Paley-Zygmund), and every row has c <= 4 (Sidon bound 11).")
    print(f"    Q_DQI = {nt['q_dqi']:.4f} needs alpha_sub >= {need:.4f}, i.e. a share >= {frac_needed:.3f} of rows with")
    print(f"    c = 3; the guaranteed share {bound:.3f} gives alpha_sub >= {base + guaranteed / 4:.4f} and "
          f"mu+ <= {nt['q_dqi'] - (0.5 + 0.5 * (base + guaranteed / 4)):+.4f} (rows independent, so it concentrates)")
    print(f"\nsweep violations (excluding the deliberate D3 check without hypothesis): {bad}; "
          f"family inconsistencies: {fam_bad}")


if __name__ == "__main__":
    main()
