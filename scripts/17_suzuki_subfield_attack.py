"""Gate (D6) at finite size: a Suzuki-curve code under a bit-level attack.

Instance: q = 32 = 2*q0^2 with q0 = 4, Suzuki curve
y^q + y = x^q0 (x^q + x), genus 124, D = the 1024 affine points (all of F_32^2), G = 635 Q.
B in F_32^{1024 x 512} evaluates one monomial in x, y, z, w (pole orders 32, 36, 40, 41) per
pole order up to 635.  Accepted sets: 16 of the 32 values, drawn uniformly at random per row.

Attack: each accepted set contains a coset a0 + V of an F_2-subspace V of F_32 = F_2^5.  The row
constraint then implies c = 5 - dim V linear equations over F_2 in the 2560 bits of x.  Rows are
added (smallest c first, random order within) to an F_2 system with rollback on inconsistency;
the solution is completed at random and re-scored directly over F_32 against the original sets.

Checks: rank of B over F_32 is 512; the pole orders of z and w are confirmed by a dimension count
(all monomials of weight <= A span exactly A - g + 1 dimensions); the curve equation holds at all
1024 points; re-scoring is independent of the F_2 system.

Also: the largest subset of F_2^5 with no affine 2-flat (so every accepted set of size 16 contains
a 2-flat, whatever its structure), and the finite DQI value from Lemma 9.2 of Jordan et al.

Usage: python scripts/17_suzuki_subfield_attack.py [--quick]
"""
import itertools
import math
import random
import sys
import time

import numpy as np

QUICK = "--quick" in sys.argv
Q, Q0, G_GENUS, DEG_G = 32, 4, 124, 635
SEEDS = [1801, 1802, 1803]
RESTARTS = 2 if QUICK else 6

# ---------------------------------------------------------------------------------------------
# GF(32), polynomial x^5 + x^2 + 1, generator 2
EXP = [0] * 62
LOG = [0] * 32
_x = 1
for _i in range(31):
    EXP[_i] = _x
    LOG[_x] = _i
    _x <<= 1
    if _x & 0x20:
        _x ^= 0x25
for _i in range(31, 62):
    EXP[_i] = EXP[_i - 31]
assert len(set(EXP[:31])) == 31, "2 is not primitive"
MUL = [[0 if (a == 0 or b == 0) else EXP[LOG[a] + LOG[b]] for b in range(32)] for a in range(32)]
MT = np.array(MUL, dtype=np.uint8)
INV = [0] + [EXP[(31 - LOG[a]) % 31] for a in range(1, 32)]


def gpow(a, e):
    if e == 0:
        return 1
    return 0 if a == 0 else EXP[(LOG[a] * e) % 31]


def gf_rank(mat):
    m = np.array(mat, dtype=np.uint8)
    rows, cols = m.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if m[i, c]), None)
        if piv is None:
            continue
        m[[r, piv]] = m[[piv, r]]
        m[r] = MT[INV[int(m[r, c])], m[r]]
        f = m[:, c].copy()
        f[r] = 0
        m ^= MT[f[:, None], m[r][None, :]]
        r += 1
        if r == rows:
            break
    return r


# ---------------------------------------------------------------------------------------------
# the Suzuki curve and the evaluation matrix
POINTS = [(x, y) for x in range(32) for y in range(32)]
for x, y in POINTS:
    lhs = gpow(y, Q) ^ y
    rhs = MUL[gpow(x, Q0)][gpow(x, Q) ^ x]
    assert lhs == rhs


def zw(x, y):
    z = gpow(x, 2 * Q0 + 1) ^ gpow(y, 2 * Q0)
    w = MUL[x][gpow(y, 2 * Q0)] ^ gpow(z, 2 * Q0)
    return z, w


FUNCS = [(x, y) + zw(x, y) for x, y in POINTS]
WEIGHTS = (32, 36, 40, 41)


def monomial_column(e):
    return [MUL[MUL[gpow(fx, e[0])][gpow(fy, e[1])]][MUL[gpow(fz, e[2])][gpow(fw, e[3])]]
            for fx, fy, fz, fw in FUNCS]


def exponents_up_to(a):
    out = []
    for i in range(a // 32 + 1):
        for j in range((a - 32 * i) // 36 + 1):
            for k in range((a - 32 * i - 36 * j) // 40 + 1):
                for l in range((a - 32 * i - 36 * j - 40 * k) // 41 + 1):
                    out.append((i, j, k, l))
    return out


def one_per_order(a):
    chosen = {}
    for e in exponents_up_to(a):
        v = sum(w * t for w, t in zip(WEIGHTS, e))
        if v not in chosen or sum(e) < sum(chosen[v]):
            chosen[v] = e
    return [chosen[v] for v in sorted(chosen)]


def pole_order_check(a):
    cols = [monomial_column(e) for e in exponents_up_to(a)]
    return gf_rank(np.array(cols, dtype=np.uint8).T), a - G_GENUS + 1, len(cols)


# ---------------------------------------------------------------------------------------------
# F_2 structure of F_32
def parity(v):
    return bin(v).count("1") & 1


# T[a][phi] = 5-bit mask s -> phi(a * 2^s)
T = [[sum(parity(phi & MUL[a][1 << s]) << s for s in range(5)) for phi in range(32)] for a in range(32)]


def span(vectors):
    s = {0}
    for v in vectors:
        s |= {x ^ v for x in s}
    return frozenset(s)


def linear_subspaces():
    by_dim = {0: {frozenset({0})}}
    for k in range(1, 6):
        by_dim[k] = {frozenset(s | {x ^ v for x in s}) for s in by_dim[k - 1] for v in range(1, 32)
                     if v not in s}
    return {k: sorted(map(sorted, v)) for k, v in by_dim.items()}


SUBS = linear_subspaces()


def annihilator(sub):
    ann = [phi for phi in range(1, 32) if all(parity(phi & v) == 0 for v in sub)]
    basis = []
    for phi in ann:
        if phi not in span(basis):
            basis.append(phi)
    return basis


ANN = {k: [annihilator(s) for s in SUBS[k]] for k in SUBS}


def best_cosets(accept):
    """All cosets a0 + V inside the set, V of the largest possible dimension."""
    for k in range(5, -1, -1):
        found = [(a0, idx) for idx, sub in enumerate(SUBS[k]) for a0 in range(32)
                 if all(accept[a0 ^ v] for v in sub)]
        if found:
            return k, found
    raise AssertionError


def largest_flat_free_set():
    """Largest subset of F_2^5 containing no affine 2-flat {a, b, c, a^b^c}."""
    best = []

    def extend(cur, start):
        nonlocal best
        if len(cur) > len(best):
            best = list(cur)
        if len(cur) + (32 - start) <= len(best):
            return
        for v in range(start, 32):
            ok = all((a ^ b ^ v) not in cur_set for a, b in itertools.combinations(cur, 2))
            if ok:
                cur.append(v)
                cur_set.add(v)
                extend(cur, v + 1)
                cur.pop()
                cur_set.discard(v)

    cur_set = set()
    extend([], 0)
    return best


# ---------------------------------------------------------------------------------------------
# Lemma 9.2 finite DQI value, r = q/2 (zero diagonal)
def dqi_lemma92(m, ell, q=Q, r=Q // 2):
    off = [math.sqrt(k * (m - k + 1)) for k in range(1, ell + 1)]

    def n_greater(x):
        cnt, d = 0, -x
        cnt += d > 0
        for a in off:
            d = -x - a * a / (d if d != 0 else 1e-300)
            cnt += d > 0
        return cnt

    lo, hi = 0.0, 2 * max(off) + 1
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if n_greater(mid) >= 1 else (lo, mid)
    return m * r / q + math.sqrt(r * (q - r)) / q * lo


def semicircle(lam, rho):
    return (math.sqrt(lam * (1 - rho)) + math.sqrt(rho * (1 - lam))) ** 2


# ---------------------------------------------------------------------------------------------
def attack(B, sets, rng, force_dim=None):
    m, n = B.shape
    nbits = 5 * n
    rows = []
    for i in range(m):
        accept = [v in sets[i] for v in range(32)]
        k, cosets = best_cosets(accept)
        if force_dim is not None:
            k = force_dim
            cosets = [(a0, idx) for idx, sub in enumerate(SUBS[k]) for a0 in range(32)
                      if all(accept[a0 ^ v] for v in sub)]
        rows.append((5 - k, cosets))
    Bl = B.tolist()
    order = list(range(m))
    rng.shuffle(order)
    order.sort(key=lambda i: rows[i][0])
    basis = {}
    fixed = 0
    for i in order:
        c, cosets = rows[i]
        a0, idx = rng.choice(cosets)
        k = 5 - c
        eqs = []
        for phi in ANN[k][idx]:
            vec = 0
            for j in range(n):
                vec |= T[Bl[i][j]][phi] << (5 * j)
            eqs.append((vec, parity(phi & a0)))
        added = []
        ok = True
        for vec, rhs in eqs:
            while vec:
                p = vec.bit_length() - 1
                if p in basis:
                    bv, br = basis[p]
                    vec ^= bv
                    rhs ^= br
                else:
                    break
            if vec:
                basis[vec.bit_length() - 1] = (vec, rhs)
                added.append(vec.bit_length() - 1)
            elif rhs:
                ok = False
                break
        if ok:
            fixed += 1
        else:
            for p in added:
                del basis[p]
        if len(basis) == nbits:
            break
    xbits = [rng.randrange(2) for _ in range(nbits)]
    for p in sorted(basis):
        vec, rhs = basis[p]
        low = vec ^ (1 << p)
        acc = rhs
        while low:
            b = low.bit_length() - 1
            acc ^= xbits[b]
            low ^= 1 << b
        xbits[p] = acc
    x = [sum(xbits[5 * j + s] << s for s in range(5)) for j in range(n)]
    return x, len(basis), fixed, sum(c for c, _ in rows) / m, sum(1 for c, _ in rows if c == 2)


def score(B, x, sets):
    xs = np.array(x, dtype=np.uint8)
    y = np.bitwise_xor.reduce(MT[B, xs[None, :]], axis=1)
    return sum(int(y[i]) in sets[i] for i in range(len(sets)))


def score_independent(Bl, x, sets):
    total = 0
    for i, row in enumerate(Bl):
        acc = 0
        for bij, xj in zip(row, x):
            acc ^= MUL[bij][xj]
        total += acc in sets[i]
    return total


def main():
    t0 = time.time()
    flat_free = largest_flat_free_set()
    print(f"largest 2-flat-free subset of F_2^5: size {len(flat_free)} {flat_free}")
    print("  so every accepted set of size 16 contains an affine 2-flat: c <= 3 for any sets")

    for a in (300, 400):
        got, want, ncols = pole_order_check(a)
        print(f"pole-order check at A={a}: rank of all {ncols} monomials of weight <= A is {got},"
              f" A-g+1 = {want}")
        assert got == want
    basis_exps = one_per_order(DEG_G)
    assert len(basis_exps) == DEG_G - G_GENUS + 1 == 512
    B = np.array([monomial_column(e) for e in basis_exps], dtype=np.uint8).T
    rk = gf_rank(B)
    print(f"B: {B.shape}, rank over F_32 = {rk}")
    assert rk == 512

    m, ell = 1024, 193
    dqi = dqi_lemma92(m, ell)
    print(f"DQI (Lemma 9.2, m={m}, ell={ell}): {dqi:.1f} = {dqi / m:.4f}; semicircle "
          f"{semicircle(ell / m, 0.5):.4f}; Q_Pr(1/2) = 768 = 0.7500")

    Bl = B.tolist()
    for seed in SEEDS:
        rng = random.Random(seed)
        sets = [frozenset(rng.sample(range(32), 16)) for _ in range(m)]
        for label, force in (("largest cosets", None), ("2-flats only (any sets)", 2)):
            best = None
            for r in range(RESTARTS):
                x, rank2, fixed, mean_c, n_c2 = attack(B, sets, rng, force)
                s = score(B, x, sets)
                if best is None or s > best[0]:
                    best = (s, x, rank2, fixed, mean_c, n_c2)
                if s == m:
                    break
            s, x, rank2, fixed, mean_c, n_c2 = best
            s_ind = score_independent(Bl, x, sets)
            assert s == s_ind
            print(f"seed {seed}, {label}: rows with c=2: {n_c2}, mean c {mean_c:.3f};"
                  f" F_2 rank {rank2}/2560; rows fixed {fixed}; satisfied {s}/1024 = {s / m:.4f}"
                  f" (independent re-score {s_ind}); DQI {dqi / m:.4f}")
    print(f"time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
