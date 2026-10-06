"""Script 49 - Concrete cost of the 2026 GIJS distinguisher (ePrint 2026/1630) on our alternant key.

The attack.  Ghoshal-Ishai-Jain-Sun hold out one public column, compute the homogeneous
degree-d polynomials whose order-<s Hasse jets vanish at every other column, and accept
when they all vanish at the held-out column.  For a code whose F_q^m-embedding lies in an
ambient GRS code of dimension D+1, Hermite interpolation forces acceptance when
    s (n' - h) > d D                                  (reconstruction, their (6))
and a random matrix is rejected (advantage >= 1 - d/q, or 1-(1-1/q)^d for reduced
polynomials) when the kernel is nonzero:
    (n' - k' - h) * c(k', s) < dim P(k', d)            (kernel, their (7))

Which matrix.  The attack needs columns on a hidden curve, i.e. the code itself must embed
in a GRS code.  Our alternant code A = GRS_{n-r}(alpha, v)|F_p is that code, so the attack
runs on the generator matrix of A (computed from Hpub by a kernel over F_11).
Hpub spans the trace code of the dual GRS, which is not a GRS subcode.

Mapping t -> r.  For a square-free binary Goppa code of degree t the paper uses
D = n - 2t - 1 (Goppa(g) = Goppa(g^2), ambient GRS codimension 2t) and codimension m t.
For a generic alternant code with r parity rows over F_{p^s} the ambient GRS code is
GRS_{n-r}, so D = n - r - 1 and codimension s r.  So their "2t" is our r and their "m t" is
our s r; the ratio (F_q-codim)/(ambient codim) is m/2 = 6 for mceliece348864 and s for us (s = 6 in the paper).
Shortening keeps n' - k' = codim fixed, and reconstruction gives n' < (d(R+1) - s h)/(d - s)
with R the ambient codimension, so the attack needs s_att >~ codim/R.  That ratio (not n)
is what drives the cost.

Port to F_q (q = 11).  Variants evaluated, attacker takes the minimum:
  ML-sys    multilinear homogeneous degree-d polynomials over F_11 (the paper's F_2 space,
            used verbatim over F_q; Vedenev 2026/1747 Sec. 3 states the soundness bound for
            reduced polynomials over any F_q).  Hasse derivatives with some a_i >= 2 vanish,
            and the systematic-form trick of Sec. 5.1 carries over unchanged (d > s), so
            the kernel condition is exactly their (7).  Pr[H(y)=0] <= 1-(10/11)^d.
  e-sys     homogeneous, individual degree <= e, 2 <= e <= d - s (our generalisation; the
            systematic columns still impose nothing because a nonzero jet at u_i needs
            A_i >= d - s + 1).  Dim and per-point constraint counts by inclusion-exclusion.
  full      the paper's q-ary space (Thm 3.2; Vedenev (7)): all homogeneous degree-d
            polynomials, d < q, constraints (n'-h) C(k'+s-1, s-1), no systematic trick.
  HF-heur   the full space with the kernel condition replaced by the paper's random-nullity
            prediction (App. C.2, eq. (34)): HF(d) = [z^d] (1 - z^{d-s+1})^{n'-h} / (1-z)^{k'} > 0.
            Heuristic (the paper derives it in char 2; we test it over F_11 below).  This is
            in the spirit of Remark 5.1's second optimisation, not part of Table 2.

Cost model: Sec. 5.3 verbatim (Eberly block Lanczos, eps0 = 572, reliability params 16, 18,
K_lin = 5N' + 3bR + 4bL, T_lin = K_lin Tmv + (72bL+8)N'^2 + (112bL^2+5184)N',
Tmv = 2 (nnz(M) + E[Z]), nnz(M) <= (n'-k'-h) Dim sum_{r<s} C(d,r)).  These are F_q
operations.  Conversion to binary operations: each F_q operation costs ceil(log2 q)^2 bit
operations, the paper's own convention in (13) (m^2 per F_{2^m} op).  For F_11 that is 16
(+4 bits); for F_2 it is 1.  log2 of the field-op count is reported too.
Remark 5.1 (Weis): two further optimisations reduce the time exponent by "roughly 20 bits"
at Classic McEliece sizes; we report baseline - 20 as an indicative optimistic line.

Run: python scripts/49_gijs_cost_estimate.py            (estimates + small experiment)
     python scripts/49_gijs_cost_estimate.py --no-exp   (estimates only)
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from functools import lru_cache
from math import comb, log2

import numpy as np

EPS0, ALPHA_R, BETA_R = 572, 16, 18


# ---------------------------------------------------------------- combinatorics
@lru_cache(maxsize=None)
def bounded(k: int, d: int, e: int) -> int:
    """# monomials of degree d in k variables with every exponent <= e."""
    if d < 0:
        return 0
    if e >= d:
        return comb(k + d - 1, d) if k > 0 else int(d == 0)
    if e == 1:
        return comb(k, d)
    tot = 0
    for j in range(0, min(k, d // (e + 1)) + 1):
        tot += (-1) ** j * comb(k, j) * comb(k - 1 + d - j * (e + 1), k - 1)
    return tot


def cons_per_point(k: int, s: int, e: int) -> int:
    """# derivative multi-indices a with |a| < s and a_i <= e."""
    return sum(bounded(k, r, e) for r in range(s))


def nnz_row_bound(d: int, s: int) -> int:
    """Max # nonzero order-<s Hasse derivatives of one degree-d monomial (paper's bound)."""
    return sum(comb(d, r) for r in range(s))


def hf_random(k: int, d: int, s: int, npts: int) -> int:
    """Random-nullity Hilbert function (34) for npts points of multiplicity s, truncated (+)."""
    e = d - s + 1
    coeffs = []
    for i in range(d + 1):
        c = 0
        for lam in range(0, i // e + 1):
            if lam > npts:
                break
            c += (-1) ** lam * comb(npts, lam) * comb(k - 1 + i - lam * e, k - 1)
        coeffs.append(c)
    if any(c <= 0 for c in coeffs):
        return 0
    return coeffs[d]


def lg(x) -> float:
    return log2(x) if x > 0 else float("-inf")


# ---------------------------------------------------------------- Sec. 5.3 cost model
def t_lin(N: int, J: int, nnz: int) -> dict:
    """Block Lanczos cost of Sec. 5.3 in field operations.  N rows (monomials), J columns."""
    lN = math.ceil(lg(N))
    Np = J + 2 * lN
    lNp = math.ceil(lg(Np))
    bR = EPS0 + ALPHA_R + 1
    bL = bR + 2 * (lNp + BETA_R)
    c = 3 * lNp + 2 * lN
    zsum = sum(min(1.0, c / 2 ** i) for i in range(1, 64))
    EZ = (Np + J) * zsum
    Tmv = 2 * (nnz + EZ)
    K = 5 * Np + 3 * bR + 4 * bL
    T = K * Tmv + (72 * bL + 8) * Np ** 2 + (112 * bL ** 2 + 5184) * Np
    S = (24 * bL + 4) * Np + 8 * bL ** 2 + 24 * bL * lNp + 2 * lNp * EZ + nnz * lg(N)
    return {"T_fieldops": T, "Np": Np, "bL": bL, "S_bits_approx": S}


def bitcost(q: int) -> int:
    return 1 if q == 2 else math.ceil(log2(q)) ** 2


# ---------------------------------------------------------------- distinguisher search
def evaluate(variant, q, n, k, R, kp, s, d, e, h):
    """Return dict for one shortened instance (k' = kp) or None if conditions fail."""
    c = n - k
    npr = kp + c
    D = npr - R - 1
    if not s * (npr - h) > d * D:
        return None
    if variant in ("ML-sys", "e-sys"):
        dim = bounded(kp, d, e)
        per = cons_per_point(kp, s, e)
        pts = npr - kp - h
        J = pts * per
        if not J < dim:
            return None
        nnz = pts * dim * nnz_row_bound(d, s)
    elif variant == "full":
        dim = bounded(kp, d, d)
        per = comb(kp + s - 1, s - 1)
        pts = npr - h
        J = pts * per
        if not J < dim:
            return None
        nnz = pts * dim * nnz_row_bound(d, s)
    elif variant == "HF-heur":
        dim = bounded(kp, d, d)
        pts = npr - h
        hf = hf_random(kp, d, s, pts)
        if hf <= 0:
            return None
        J = dim - hf  # predicted rank
        nnz = pts * dim * nnz_row_bound(d, s)
    else:
        raise ValueError(variant)
    cost = t_lin(dim, J, nnz)
    bc = bitcost(q)
    if variant in ("ML-sys",):
        acc = 1 - ((q - 1) / q) ** d
    else:
        acc = min(1.0, d / q) if d < q else 1.0
    adv = 0.99 - acc ** h
    return {
        "variant": variant, "shorten": k - kp, "n'": npr, "k'": kp, "D": D, "s": s, "d": d,
        "e": e, "holdouts": h, "log2_Dim": round(lg(dim), 2), "log2_J": round(lg(J), 2),
        "log2_T_fieldops": round(lg(cost["T_fieldops"]), 2),
        "log2_T_bits": round(lg(cost["T_fieldops"] * bc), 2),
        "log2_S_bits": round(lg(cost["S_bits_approx"]), 2),
        "adv_lower": round(adv, 3),
    }


def search(q, n, k, R, variants=("ML-sys", "e-sys", "full", "HF-heur"), smax=40, dextra=6,
           objective="T"):
    """Min-cost (or min-Dim) feasible choice per variant.  Cost is monotone in k', so for each
    (s, d, e) the smallest k' satisfying the kernel condition is optimal."""
    c = n - k
    best = {}
    for variant in variants:
        for h in (1, 2, 3):
            for s in range(1, smax + 1):
                for d in range(s + 1, s + dextra + 1):
                    if variant in ("full", "HF-heur") and q > 2 and d >= q:
                        continue
                    if variant in ("full", "HF-heur") and q == 2:
                        continue
                    es = [1] if variant == "ML-sys" else (
                        range(2, min(d - s, q - 1) + 1) if variant == "e-sys" else [d])
                    for e in es:
                        num = d * (R + 1) - s * h
                        npmax = math.ceil(num / (d - s)) - 1
                        kpmax = min(npmax - c, k)
                        if kpmax < 1:
                            continue
                        # smallest k' with kernel condition: binary search (monotone)
                        lo, hi = 1, kpmax
                        if evaluate(variant, q, n, k, R, hi, s, d, e, h) is None:
                            continue
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if evaluate(variant, q, n, k, R, mid, s, d, e, h) is None:
                                lo = mid + 1
                            else:
                                hi = mid
                        r = evaluate(variant, q, n, k, R, lo, s, d, e, h)
                        if r is None or (q > 2 and r["adv_lower"] < 0.01):  # F_2: paper GHW count gives >0.42
                            continue
                        key = r["log2_T_bits"] if objective == "T" else r["log2_Dim"]
                        cur = best.get(variant)
                        ck = None if cur is None else (
                            cur["log2_T_bits"] if objective == "T" else cur["log2_Dim"])
                        if cur is None or key < ck:
                            best[variant] = r
    return best


# ---------------------------------------------------------------- GIJS Sec. 7 key recovery
def keyrec_eval(q, mbr, n, k, R, kp, d, sbar, nslack_minus, goppa_t=0):
    """Baseline cost (13) for one shortening.  mbr = # Frobenius branches (m)."""
    npr = kp + (n - k)
    D = npr - R - 1
    S_tot = d * D + 1 - nslack_minus + sbar
    rest = S_tot - 4 * sbar
    if npr <= 4 or rest <= 0:
        return None
    a = rest // (npr - 4)
    hi_cnt = rest - a * (npr - 4)
    if a < sbar or a + (1 if hi_cnt else 0) >= d:
        return None
    prof = {sbar: 4}
    prof[a] = prof.get(a, 0) + (npr - 4 - hi_cnt)
    if hi_cnt:
        prof[a + 1] = prof.get(a + 1, 0) + hi_cnt
    Nd = comb(kp + d - 1, d)
    M = sum(cnt * comb(kp + sj - 1, sj - 1) for sj, cnt in prof.items())

    def HF(pr):
        coeffs = []
        # product over groups of (1 - z^{d-sj+1})^cnt divided by (1-z)^kp, coefficient at d
        poly = [1] + [0] * d
        for sj, cnt in pr.items():
            ex = d - sj + 1
            newp = [0] * (d + 1)
            for i, pc in enumerate(poly):
                if pc == 0:
                    continue
                for lam in range(0, (d - i) // ex + 1):
                    if lam > cnt:
                        break
                    newp[i + lam * ex] += pc * (-1) ** lam * comb(cnt, lam)
            poly = newp
        for i in range(d + 1):
            coeffs.append(sum(poly[j] * comb(kp - 1 + i - j, kp - 1) for j in range(i + 1)))
        if any(cc <= 0 for cc in coeffs):
            return 0
        return coeffs[d]

    hf = HF(prof)
    excess = mbr * sbar + (mbr - 1 if goppa_t else 0)
    if hf <= 0:
        return None
    margins = []
    Bmax = 0
    for sj in prof:
        pr2 = dict(prof)
        pr2[sj] -= 1
        if pr2[sj] == 0:
            del pr2[sj]
        sl = sj - sbar + 1
        pr2[sl] = pr2.get(sl, 0) + 1
        Bj = sum(comb(kp + u - 1, u) for u in range(sl, sj))
        Bmax = max(Bmax, Bj)
        margins.append(HF(pr2) - hf - excess)
    if min(margins) <= 0:
        return None
    Ls = comb(kp + sbar - 1, sbar - 1)
    unknowns = 2 * M
    eqs = 2 * Nd + 4 * (Ls - 1) + 1 + Bmax
    nnz = 2 * Nd * sum(cnt * nnz_row_bound(d, sj) for sj, cnt in prof.items()) + 2 * (
        4 * (Ls - 1) + 1 + Bmax)
    cst = t_lin(unknowns, eqs, nnz)  # dimensions interchanged as in Sec. 7.4
    Rr = math.ceil((n - 4) / (npr - 4))
    mult = 1.0101 * bitcost(q) * Rr * (npr + 1) * (q + 1)
    return {
        "shorten": k - kp, "n'": npr, "k'": kp, "d": d, "sbar": sbar,
        "profile": {int(a_): int(b_) for a_, b_ in sorted(prof.items())},
        "log2_system": f"{lg(eqs):.2f} x {lg(unknowns):.2f}",
        "log2_margin_HF": round(lg(hf + excess), 2), "log2_min_locator_margin": round(lg(min(margins)), 2),
        "R_runs": Rr, "log2_T_loc_fieldops": round(lg(cst["T_fieldops"]), 2),
        "log2_T_rec_bits": round(lg(cst["T_fieldops"] * mult), 2),
    }


def keyrec_search(q, mbr, n, k, R, nslack_minus=0, goppa=False, drange=range(3, 13),
                  sbars=(3, 5, 9), step=1):
    best = None
    c = n - k
    for d in drange:
        for sbar in sbars:
            if sbar >= d:
                continue
            for npr in range(c + 2, min(n, c + k) + 1, step):
                kp = npr - c
                r = keyrec_eval(q, mbr, n, k, R, kp, d, sbar, nslack_minus, goppa)
                if r and (best is None or r["log2_T_rec_bits"] < best["log2_T_rec_bits"]):
                    best = r
    return best


# ---------------------------------------------------------------- small experiment over F_11
P = 11


def rank_mod(A: np.ndarray, p: int = P) -> int:
    A = A.copy() % p
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        piv = np.nonzero(A[r:, c])[0]
        if piv.size == 0:
            continue
        i = r + piv[0]
        if i != r:
            A[[r, i]] = A[[i, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        nz = np.nonzero(A[:, c])[0]
        nz = nz[nz != r]
        if nz.size:
            A[nz] = (A[nz] - np.outer(A[nz, c], A[r])) % p
        r += 1
    return r


def rref_mod(A: np.ndarray, p: int = P):
    A = A.copy() % p
    rows, cols = A.shape
    r = 0
    pivs = []
    for c in range(cols):
        if r == rows:
            break
        piv = np.nonzero(A[r:, c])[0]
        if piv.size == 0:
            continue
        i = r + piv[0]
        if i != r:
            A[[r, i]] = A[[i, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        nz = np.nonzero(A[:, c])[0]
        nz = nz[nz != r]
        if nz.size:
            A[nz] = (A[nz] - np.outer(A[nz, c], A[r])) % p
        pivs.append(c)
        r += 1
    return A[:r], pivs


def nullspace_mod(A: np.ndarray, p: int = P) -> np.ndarray:
    R_, pivs = rref_mod(A, p)
    n = A.shape[1]
    free = [j for j in range(n) if j not in pivs]
    B = np.zeros((len(free), n), dtype=np.int64)
    for t, f in enumerate(free):
        B[t, f] = 1
        for i, pc in enumerate(pivs):
            B[t, pc] = (-R_[i, f]) % p
    return B


class Ext:
    """F_{11^m} by a monic irreducible of degree m; elements are ints in base 11."""

    def __init__(self, m: int, rng):
        self.m, self.q = m, P ** m
        while True:
            f = [rng.randrange(P) for _ in range(m)] + [1]
            if self._irreducible(f):
                break
        self.f = f
        q = self.q
        self.mul = np.zeros((q, q), dtype=np.int32) if q <= 1400 else None
        for a in range(q):
            for b in range(a, q):
                v = self._mul(a, b)
                self.mul[a, b] = self.mul[b, a] = v

    def vec(self, a):
        return [(a // P ** i) % P for i in range(self.m)]

    def unvec(self, v):
        return sum(int(x) * P ** i for i, x in enumerate(v))

    def _mul(self, a, b):
        A, B = self.vec(a), self.vec(b)
        prod = [0] * (2 * self.m - 1)
        for i, x in enumerate(A):
            for j, y in enumerate(B):
                prod[i + j] = (prod[i + j] + x * y) % P
        for i in range(len(prod) - 1, self.m - 1, -1):
            c = prod[i]
            if c:
                for j in range(self.m + 1):
                    prod[i - self.m + j] = (prod[i - self.m + j] - c * self.f[j]) % P
        return self.unvec(prod[: self.m])

    def _irreducible(self, f):
        m = len(f) - 1
        if m == 1:
            return True
        # no roots, and for m <= 3 that suffices
        for x in range(P):
            if sum(c * pow(x, i, P) for i, c in enumerate(f)) % P == 0:
                return False
        assert m <= 3
        return True


def alternant_generator(ext: Ext, n: int, r: int, rng) -> np.ndarray:
    q = ext.q
    supp = rng.sample(range(q), n)
    mults = [rng.randrange(1, q) for _ in range(n)]
    rows = []
    for j in range(n):
        col = []
        pw = mults[j]
        for i in range(r):
            col.extend(ext.vec(pw))
            pw = int(ext.mul[pw, supp[j]])
        rows.append(col)
    H = np.array(rows, dtype=np.int64).T  # (m r) x n over F_11
    return nullspace_mod(H)


def shorten(G: np.ndarray, ell: int, rng) -> np.ndarray:
    k, n = G.shape
    for _ in range(50):
        I = rng.sample(range(n), ell)
        if rank_mod(G[:, I]) == ell:
            break
    else:
        raise RuntimeError("no info subset")
    W = nullspace_mod(G[:, I].T)  # left kernel of G[:, I]
    keep = [j for j in range(n) if j not in set(I)]
    Gs = (W @ G[:, keep]) % P
    return Gs


def ml_monomials(k, d):
    from itertools import combinations
    return list(combinations(range(k), d))


def distinguisher_ml(Y: np.ndarray, s: int, d: int, hold: int) -> bool:
    """Multilinear + systematic-form hold-out test over F_11; True = accept (structured)."""
    from itertools import combinations
    k, n = Y.shape
    others = [j for j in range(n) if j != hold]
    Rr, pivs = rref_mod(Y[:, others])
    assert len(pivs) == k
    Yp = np.zeros_like(Y)
    # systematic transform: solve Y_P^{-1} Y
    YP = Y[:, [others[p_] for p_ in pivs]]
    aug = np.concatenate([YP, np.eye(k, dtype=np.int64)], axis=1)
    Rinv, _ = rref_mod(aug)
    inv = Rinv[:, k:]
    Yp = (inv @ Y) % P
    sysc = {others[p_] for p_ in pivs}
    pts = [j for j in others if j not in sysc]
    mons = ml_monomials(k, d)
    idx = {m: i for i, m in enumerate(mons)}
    ders = [a for r_ in range(s) for a in combinations(range(k), r_)]
    M = np.zeros((len(mons), len(pts) * len(ders) + 1), dtype=np.int64)
    col = 0
    for j in pts:
        y = Yp[:, j]
        for a in ders:
            aset = set(a)
            for mi, A in enumerate(mons):
                if aset.issubset(A):
                    v = 1
                    for i in A:
                        if i not in aset:
                            v = (v * int(y[i])) % P
                    M[mi, col] = v
            col += 1
    yh = Yp[:, hold]
    for mi, A in enumerate(mons):
        v = 1
        for i in A:
            v = (v * int(yh[i])) % P
        M[mi, col] = v
    # check that systematic columns impose nothing (d > s): no jet constraint is needed
    rM = rank_mod(M[:, :-1])
    rMb = rank_mod(M)
    return rM == rMb, len(mons), M.shape[1] - 1, rM


def hasse_mon(A, a):
    c = 1
    for Ai, ai in zip(A, a):
        if ai > Ai:
            return 0
        c = (c * comb(Ai, ai)) % P
    return c


def exps(k, d):
    if k == 1:
        yield (d,)
        return
    for i in range(d, -1, -1):
        for rest in exps(k - 1, d - i):
            yield (i,) + rest


def jet_rank_full(Ypts: np.ndarray, s: int, d: int) -> tuple[int, int, int]:
    """Rank of the full-homogeneous order-<s jet matrix at the given points."""
    k, npts = Ypts.shape
    mons = list(exps(k, d))
    ders = [a for r_ in range(s) for a in exps(k, r_)]
    M = np.zeros((len(mons), npts * len(ders)), dtype=np.int64)
    col = 0
    for j in range(npts):
        y = [int(v) for v in Ypts[:, j]]
        for a in ders:
            for mi, A in enumerate(mons):
                c = hasse_mon(A, a)
                if c:
                    v = c
                    for i in range(k):
                        e_ = A[i] - a[i]
                        if e_:
                            v = (v * pow(y[i], e_, P)) % P
                    M[mi, col] = v
            col += 1
    return rank_mod(M), len(mons), M.shape[1]


def experiment(seed=4901):
    rng = random.Random(seed)
    out = {}
    t0 = time.time()
    # ---- (1) F_121 alternant, multilinear + systematic distinguisher (paper's port)
    ext = Ext(2, rng)
    n, r = 100, 20
    G = alternant_generator(ext, n, r, rng)
    k = G.shape[0]
    s, d = 2, 3
    R_amb = r
    c = n - k
    # smallest k' meeting both conditions
    kp = None
    for cand in range(1, k + 1):
        npr = cand + c
        if s * (npr - 1) > d * (npr - R_amb - 1) and (npr - cand - 1) * (1 + cand) < comb(cand, d):
            kp = cand
            break
    ell = k - kp
    trials = []
    for t in range(6):
        Gs = shorten(G, ell, rng)
        hold = rng.randrange(Gs.shape[1])
        acc, N, J, rk = distinguisher_ml(Gs, s, d, hold)
        Yr = np.array([[rng.randrange(P) for _ in range(Gs.shape[1])] for _ in range(Gs.shape[0])],
                      dtype=np.int64)
        accr, *_ = distinguisher_ml(Yr, s, d, hold)
        trials.append({"structured_accept": bool(acc), "random_accept": bool(accr)})
    out["F121_alternant_ML"] = {
        "field": "F_11 base, F_121 ambient", "n": n, "r": r, "k": int(k), "shorten": ell,
        "n'": kp + c, "k'": kp, "s": s, "d": d, "Dim": N, "constraints": J,
        "recon": f"{s}*({kp + c}-1)={s * (kp + c - 1)} > {d}*{kp + c - R_amb - 1}={d * (kp + c - R_amb - 1)}",
        "trials": trials,
    }
    # ---- (2) F_1331 alternant (extension degree 3), same test
    ext3 = Ext(3, rng)
    n3, r3 = 200, 30
    G3 = alternant_generator(ext3, n3, r3, rng)
    k3 = G3.shape[0]
    c3 = n3 - k3
    s3, d3 = 3, 4
    kp3 = None
    for cand in range(1, k3 + 1):
        npr = cand + c3
        if s3 * (npr - 1) > d3 * (npr - r3 - 1) and (npr - cand - 1) * (1 + cand + comb(cand, 2)) < comb(cand, d3):
            kp3 = cand
            break
    res3 = {"n": n3, "r": r3, "k": int(k3), "s": s3, "d": d3, "k'_needed": kp3}
    if kp3 is not None and comb(kp3, d3) <= 6000:
        trials3 = []
        for t in range(3):
            Gs = shorten(G3, k3 - kp3, rng)
            hold = rng.randrange(Gs.shape[1])
            acc, N, J, rk = distinguisher_ml(Gs, s3, d3, hold)
            Yr = np.array([[rng.randrange(P) for _ in range(Gs.shape[1])] for _ in range(Gs.shape[0])], dtype=np.int64)
            accr, *_ = distinguisher_ml(Yr, s3, d3, hold)
            trials3.append({"structured_accept": bool(acc), "random_accept": bool(accr)})
        res3.update({"n'": kp3 + c3, "Dim": N, "constraints": J, "trials": trials3})
    else:
        res3["note"] = ("no k' meets both conditions at s=3,d=4 here (needs r >~ 36, then Dim = C(k',4) ~ 2^17-2^18); "
                       "beyond this dense O(Dim^3) prototype, skipped")
    out["F1331_alternant_ML"] = res3
    # ---- (3) odd-characteristic check of the random-nullity prediction (34)
    hf_checks = []
    for (kk, dd, ss, npts) in [(8, 5, 4, 6), (10, 4, 3, 12), (12, 3, 2, 10), (7, 6, 4, 9)]:
        Yp = np.array([[rng.randrange(P) for _ in range(npts)] for _ in range(kk)], dtype=np.int64)
        rk, Nd, cols = jet_rank_full(Yp, ss, dd)
        pred = Nd - hf_random(kk, dd, ss, npts)
        hf_checks.append({"k": kk, "d": dd, "s": ss, "points": npts, "N_d": Nd, "jet_cols": cols,
                          "rank_observed": rk, "rank_predicted_(34)": pred})
    out["HF_prediction_check_F11"] = hf_checks
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- main
def main():
    res = {"conventions": {
        "bit_ops_per_Fq_op": "ceil(log2 q)^2 (paper's (13) convention): F_2 -> 1, F_11 -> 16, F_11^4 -> 196",
        "remark_5_1": "paper: Weis optimisations lower the time exponent by ~20 bits; not modelled",
        "t_to_r": "binary Goppa: ambient codim 2t, codim m t; alternant: ambient codim r, codim s r",
    }}
    # ---- calibration
    cal = {}
    for name, m, n, t in [("mceliece348864", 12, 3488, 64), ("mceliece460896", 13, 4608, 96),
                          ("mceliece6688128", 13, 6688, 128)]:
        k = n - m * t
        bestDim = search(2, n, k, 2 * t, variants=("ML-sys",), objective="Dim")["ML-sys"]
        bestT = search(2, n, k, 2 * t, variants=("ML-sys",), objective="T")["ML-sys"]
        cal[name] = {"min_Dim_choice": bestDim, "min_T_choice": bestT}
        print(f"[cal] {name}: minDim {bestDim}\n       minT  {bestT}")
    res["calibration_table2"] = cal
    print("paper Table 2: 348864 l=2503 d=8 s=7 Dim 2^46.60 log2T 114.16; 460896 120.29; 6688128 123.95")

    # ---- our key and alternatives
    sets = [
        ("ours s=4 n=6000 r=180", 4, 6000, 180),
        ("s=4 n=14000 r=420 (alpha .12)", 4, 14000, 420),
        ("s=4 n=14641 r=180 (alpha .049)", 4, 14641, 180),
        ("s=4 n=6000 r=90 (alpha .06)", 4, 6000, 90),
        ("s=5 n=6000 r=144 (alpha .12)", 5, 6000, 144),
        ("s=5 n=20000 r=480 (alpha .12)", 5, 20000, 480),
        ("s=5 n=50000 r=1200 (alpha .12)", 5, 50000, 1200),
        ("s=6 n=6000 r=120 (alpha .12)", 6, 6000, 120),
        ("s=6 n=20000 r=400 (alpha .12)", 6, 20000, 400),
        ("s=6 n=50000 r=1000 (alpha .12)", 6, 50000, 1000),
        ("s=7 n=6000 r=103 (alpha .12)", 7, 6000, 103),
        ("s=8 n=6000 r=90 (alpha .12)", 8, 6000, 90),
        ("ours s=6 n=14000 r=187", 6, 14000, 187),
        ("s=6 n=20000 r=267", 6, 20000, 267),
        ("s=6 n=40000 r=533", 6, 40000, 533),
    ]
    alt = {}
    for label, se, n, r in sets:
        k = n - se * r
        b = search(11, n, k, r)
        mn = min(b.values(), key=lambda x: x["log2_T_bits"]) if b else None
        mn_rig = min((v for kk, v in b.items() if kk != "HF-heur"), key=lambda x: x["log2_T_bits"], default=None)
        alt[label] = {"p": 11, "s_ext": se, "n": n, "r": r, "k": k, "per_variant": b,
                      "min_rigorous": mn_rig, "min_incl_heuristic": mn}
        print(f"\n[{label}] k={k}")
        for v, x in b.items():
            print(f"   {v:8s} l={x['shorten']} n'={x[chr(110)+chr(39)]} k'={x[chr(107)+chr(39)]} s={x['s']} d={x['d']} e={x['e']} "
                  f"Dim=2^{x['log2_Dim']} J=2^{x['log2_J']} T=2^{x['log2_T_bits']} (Fq ops 2^{x['log2_T_fieldops']}) adv>={x['adv_lower']}")
    res["alternant_sets"] = alt

    # ---- key recovery (GIJS Sec. 7), calibration + ours
    print("\n[keyrec] calibrating against Table 3 (348864: l=2569 d=7 log2T 130.38)")
    cm = keyrec_eval(2 ** 12, 12, 3488, 2720, 128, 2720 - 2569, 7, 5, 64, goppa_t=64)
    print("   at paper's choice:", cm)
    kr = {"calibration_348864_at_paper_choice": cm}
    keyrec_labels = ("ours s=4 n=6000 r=180", "s=5 n=6000 r=144 (alpha .12)",
                     "s=6 n=6000 r=120 (alpha .12)", "ours s=6 n=14000 r=187", "s=6 n=40000 r=533")
    for label, se, n, r in [x for x in sets if x[0] in keyrec_labels]:
        k = n - se * r
        b = keyrec_search(11 ** se, se, n, k, r)
        kr[label] = b
        print(f"   {label}: {b}")
    res["key_recovery_gijs_sec7"] = kr

    if "--no-exp" not in sys.argv:
        print("\n[experiment] small instances over F_11")
        ex = experiment()
        res["small_experiment"] = ex
        print(json.dumps(ex, indent=1))

    outp = os.path.join(os.path.dirname(__file__), "..", "results", "gijs-cost-estimate.json")
    with open(os.path.normpath(outp), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("wrote", os.path.relpath(outp))


if __name__ == "__main__":
    main()
