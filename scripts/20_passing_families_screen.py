"""Screen: which decoder-code families pass every check, with finite-size numbers.

For each family and instance size this computes
  * gates D0 (H_q(lambda) <= alpha), D2 (delta > 1 - sqrt(1 - alpha^2)), D3 (the general density
    window, equivalent to an asymptotic margin), D5 (recorded: no efficient rate-distortion encoder
    is known for any family here), and D6 (the rate seen over the prime subfield, from expected
    affine-subspace counts in a random half-set);
  * DQI's finite expected score from Lemma 9.2 of Jordan et al. at the licensed radius
    ell = floor((d* - 2)/2), with d* = n - g + 1 for a one-point AG optimizer code of dimension n
    and genus g (g = 0: Reed-Solomon);
  * the finite margin against the information-set baseline at the subfield rate, and the restart
    threshold of the converse with the finite value, m >= (1 - a) ln(N/eta) / (2 mu^2),
    N = 2^40, eta = 0.05.
A family passes when every gate passes, the finite margin is positive, and m clears the threshold.

Usage: python scripts/20_passing_families_screen.py
"""
import math

import numpy as np

LOGN = math.log(2 ** 40 / 0.05)


def dqi_finite(m, ell, q, r):
    a = np.zeros((ell + 1, ell + 1))
    for k in range(1, ell + 1):
        a[k, k - 1] = a[k - 1, k] = math.sqrt(k * (m - k + 1))
    d = (q - 2 * r) / math.sqrt(r * (q - r))
    for k in range(ell + 1):
        a[k, k] = k * d
    return (m * r / q + math.sqrt(r * (q - r)) / q * float(np.linalg.eigvalsh(a).max())) / m


def hq(x, q):
    if x <= 0:
        return 0.0
    return x * math.log(q - 1, q) - x * math.log(x, q) - (1 - x) * math.log(1 - x, q)


def gauss_binom(b, k, p):
    num = den = 1
    for i in range(k):
        num *= p ** (b - i) - 1
        den *= p ** (i + 1) - 1
    return num // den


def contain_prob(q, size, points):
    if points > size:
        return 0.0
    return math.exp(sum(math.log(size - i) - math.log(q - i) for i in range(points)))


def subfield_rate(p, b, alpha, q, r):
    """Rate seen over F_p for random sets of size r.  Rows are sorted by codimension; each row of
    codimension c costs c equations out of b*alpha*m.  The fraction of rows whose set contains an
    affine subspace of dimension k is estimated by min(1, expected count)."""
    if b == 1:
        return alpha, "void"
    frac = []
    for k in range(b, -1, -1):
        e = gauss_binom(b, k, p) * p ** (b - k) * contain_prob(q, r, p ** k)
        frac.append((k, min(1.0, e)))
    # fraction of rows with largest dimension exactly k (greedy from the top)
    remaining, dist = 1.0, []
    for k, f in frac:
        take = min(remaining, f)
        if take > 0:
            dist.append((b - k, take))
            remaining -= take
        if remaining <= 1e-12:
            break
    budget, fixed = b * alpha, 0.0
    for c, share in sorted(dist):
        if c == 0:
            return 1.0, "c=0"
        use = min(share, budget / c)
        fixed += use
        budget -= use * c
        if budget <= 1e-12:
            break
    desc = ", ".join(f"c={c}:{share:.3g}" for c, share in sorted(dist))
    return min(1.0, fixed), desc


def semicircle(lam, rho):
    return (math.sqrt(lam * (1 - rho)) + math.sqrt(rho * (1 - lam))) ** 2


def qdqi_sat(lam, rho):
    """The semicircle quality law, saturated at 1 for lam > 1 - rho."""
    return 1.0 if lam > 1 - rho else semicircle(lam, rho)


def evaluate(name, field, p, b, m, n, g):
    """All screening quantities, rule outcomes, and the finite converse for one instance."""
    q = p ** b
    r = (q - 1) // 2 if q % 2 else q // 2
    rho = r / q
    alpha = n / m
    dstar = n - g + 1
    ell = (dstar - 2) // 2
    lam, delta = ell / m, dstar / m
    a_sub, d6desc = subfield_rate(p, b, alpha, q, r)
    q_dqi = qdqi_sat(lam, rho)
    q_pr = rho + (1 - rho) * alpha
    q_sub = rho + (1 - rho) * a_sub
    mu = q_dqi - q_pr
    mu_plus = q_dqi - max(q_pr, q_sub)
    fires = {
        "D0": hq(lam, q) > alpha,
        "D1": False,  # a limit statement; every family here has delta bounded away from 0
        "D2": abs(rho - 0.5) < 0.01 and delta <= 1 - math.sqrt(1 - alpha ** 2),
        "D3": alpha / 2 <= 1 - rho and rho * (1 - rho) <= alpha / (8 * (1 - alpha / 2)),
        "window": lam <= 1 - rho and (lam - 2 * lam * rho + 2 * math.sqrt(lam * rho * (1 - lam) * (1 - rho))
                                      <= alpha * (1 - rho)),
        "D6": b > 1 and q_dqi <= q_sub,
    }
    qf = dqi_finite(m, ell, q, r)
    mu_fin = qf - max(q_pr, q_sub)
    thr = (1 - a_sub) * LOGN / (2 * mu_fin * mu_fin) if mu_fin > 0 else float("inf")
    return dict(name=name, field=field, q=q, m=m, n=n, g=g, ell=ell, alpha=alpha, delta=delta, lam=lam,
                rho=rho, a_sub=a_sub, d6desc=d6desc, q_dqi=q_dqi, q_pr=q_pr, q_sub=q_sub, mu=mu,
                mu_plus=mu_plus, fires=fires, dqi_finite=qf, mu_finite=mu_fin, threshold=thr)


def screen(name, field, p, b, m, n, g, alpha_note=""):
    e = evaluate(name, field, p, b, m, n, g)
    fired = [k for k, v in e["fires"].items() if v]
    if fired:
        verdict = "filtered by " + ", ".join(fired)
    elif e["mu_plus"] <= 0:
        verdict = "filtered: mu+ <= 0"
    elif e["mu_finite"] <= 0:
        verdict = "mu+ > 0 asymptotically, but not at this length"
    elif e["m"] < e["threshold"]:
        verdict = f"PASS (restart guarantee not yet reached: m={e['m']} < {e['threshold']:.0f})"
    else:
        verdict = "PASS (restart guarantee holds)"
    thr = e["threshold"]
    print(f"{name:34s} {field:9s} m={e['m']:5d} n={e['n']:5d} g={g:4d} ell={e['ell']:4d} "
          f"a={e['alpha']:.3f} a_sub={e['a_sub']:.3f} Q_DQI={e['q_dqi']:.4f} mu={e['mu']:+.4f} "
          f"mu+={e['mu_plus']:+.4f} finite mu+={e['mu_finite']:+.4f} "
          f"thr={thr if thr < 1e9 else float('inf'):.0f} qubits={math.ceil(math.log2(e['q']))} "
          f"{verdict}   [{e['d6desc']}]")
    return verdict


def main():
    print("Prime fields")
    for p in (401, 1009, 4001):
        for a in (0.1, 0.25):
            screen(f"RS / OPI (alpha={a})", f"F_{p}", p, 1, p - 1, round((p - 1) * a), 0)
    for p in (401, 1009):
        screen("doubly extended RS (alpha=1/4)", f"F_{p}", p, 1, p + 1, round((p + 1) / 4), 0)
    for q in (401, 433, 521, 601, 1009):
        M = (q - 1) // 2
        screen("star-twisted RS [M,3M/4,M/4+1]", f"F_{q}", q, 1, M, M // 4, 0)
    for p in (409, 1009, 2003):
        screen("elliptic, genus 1 (alpha=1/4)", f"F_{p}", p, 1, p, round(p / 4), 1)
    for p in (409, 1009, 2003):
        m = p + 1 + int(4 * math.sqrt(p))  # Hasse-Weil maximum for genus 2
        screen("genus 2, max length (alpha=1/4)", f"F_{p}", p, 1, m, round(m / 4), 2)

    print("\nPrime-square fields, odd characteristic")
    for p in (7, 11, 13, 17, 19):
        m = p ** 3
        screen("Hermitian (alpha=1/4)", f"F_{p}^2", p, 2, m, round(m / 4), p * (p - 1) // 2)
    for p in (7, 11, 13):
        # GS tower over F_{p^2}: genus ratio approaches 1/(p-1); use a length-10^4 illustrative level
        m = 10000
        g = round(m / (p - 1))
        screen("GS tower limit gamma=1/(p-1), a=1/4", f"F_{p}^2", p, 2, m, round(m / 4), g)

    print("\nBinary extension fields")
    screen("Hermitian OPI (alpha=1/4)", "F_256", 2, 8, 4096, 1024, 120)
    screen("elliptic, max length (alpha=1/4)", "F_256", 2, 8, 288, 72, 1)
    screen("GS tower F_256 (alpha=0.1), m=4096", "F_256", 2, 8, 4096, 410, round(4096 / 15))
    screen("Suzuki (alpha=1/2)", "F_32", 2, 5, 1024, 512, 124)
    # norm-trace curve x^21 = y^16 + y^4 + y has genus (21-1)(16-1)/2 = 150, so d* = 471 - 150 + 1 = 322
    screen("norm-trace anchor (alpha=0.46)", "F_64", 2, 6, 1024, 471, 150)
    print("\nNot screenable: interleaved RS (block-constraint quality law unproved); multi-twist TGRS")
    print("(as RS if an MDS instance is certified).")
    print("Hypothesis for every extension-field row: the semicircle law over F_(p^b), asserted by")
    print("Gu-Jordan (2510.06603) but proved by Jordan et al. (2408.08292, Thm 4.1, Lemma 9.2) for prime p.")


if __name__ == "__main__":
    main()
