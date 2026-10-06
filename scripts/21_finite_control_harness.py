"""Generic finite-control harness: DQI's finite value against a classical attack battery.

Families (all pass the screen of 20_passing_families_screen.py):
  rs         Reed-Solomon over F_p, evaluation points F_p^* (calibration against script 18)
  elliptic   one-point codes on y^2 = x^3 + a x + b over prime F_p, G = n O, basis x^i, x^i y
  hermitian  one-point codes on y^p + y = x^(p+1) over F_(p^2), p odd, basis x^i y^j (j < p)
Accepted sets are uniformly random of size floor(q/2) (q odd: (q-1)/2).

DQI's value is the exact finite formula of Jordan et al. Lemma 9.2 at the licensed radius
ell = floor((d* - 2)/2), d* = n - g + 1.

Attacks, every witness checked to be a codeword and re-scored independently:
  A  Prange restarts over F_q (systematic form L = B B_S^-1, accepted targets on S)
  B  pivoting line search: the line y + t L[:, k] keeps the other n-1 rows of the information set
     S fixed; all q points are scored with one histogram; S is then pivoted to a satisfied row
  C  heat-bath version of B (t drawn proportional to exp(beta * count), beta annealed)
  D  (F_(p^2) only) subfield information sets over F_p: rows whose set contains an affine F_p-line
     cost one F_p-equation instead of two (gate D6)

Usage:
  python scripts/21_finite_control_harness.py --family elliptic --sizes 409,1009 [--quick]
  python scripts/21_finite_control_harness.py --merge
Writes results/finite-control-<family>-run-<tag>.json.
"""
import glob
import json
import math
import sys
import time

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


QUICK = "--quick" in sys.argv
FAMILY = arg("--family", "elliptic")
SIZES = [int(s) for s in arg("--sizes", "409").split(",")]
TAG = arg("--tag", "quick" if QUICK else "all")
ALPHA = float(arg("--alpha", "0.25"))
SEEDS = [3101, 3102, 3103] if not QUICK else [3101]
T_LINE = 5 if QUICK else 40
T_HEAT = 5 if QUICK else 40
SUB_RESTARTS = 3 if QUICK else 20
OUT = "results/finite-control-%s-run-%s.json"


# ---------------------------------------------------------------------------------------------
# fields: elements are integers 0..q-1; for F_(p^2) the integer u + p*v encodes u + v*w, w^2 = nu
def primitive_root(p):
    x, fac, d = p - 1, set(), 2
    while d * d <= x:
        while x % d == 0:
            fac.add(d)
            x //= d
        d += 1
    if x > 1:
        fac.add(x)
    return next(g for g in range(2, p) if all(pow(g, (p - 1) // f, p) != 1 for f in fac))


class PrimeField:
    def __init__(self, p):
        self.p = self.q = p
        self.b = 1
        self.INV = np.array([0] + [pow(a, p - 2, p) for a in range(1, p)], dtype=np.int64)

    def add(self, a, b):
        return (a + b) % self.p

    def sub(self, a, b):
        return (a - b) % self.p

    def mul(self, a, b):
        return (a * b) % self.p

    def inv(self, a):
        return self.INV[a]

    def matmul(self, a, x):
        return (np.asarray(a, dtype=np.float64) @ np.asarray(x, dtype=np.float64) % self.p).astype(np.int64)


class PrimeSquareField:
    def __init__(self, p):
        self.p, self.q, self.b = p, p * p, 2
        self.nu = next(a for a in range(2, p) if pow(a, (p - 1) // 2, p) == p - 1)
        e = np.arange(self.q, dtype=np.int64)
        u, v = e % p, e // p
        norm = (u * u - self.nu * v * v) % p
        ninv = np.array([0] + [pow(int(a), p - 2, p) for a in range(1, p)], dtype=np.int64)[norm]
        self.INV = ((u * ninv) % p + p * ((-v * ninv) % p)) % self.q
        self.INV[0] = 0

    def split(self, a):
        a = np.asarray(a, dtype=np.int64)
        return a % self.p, a // self.p

    def join(self, u, v):
        return (u % self.p) + self.p * (v % self.p)

    def add(self, a, b):
        (u1, v1), (u2, v2) = self.split(a), self.split(b)
        return self.join(u1 + u2, v1 + v2)

    def sub(self, a, b):
        (u1, v1), (u2, v2) = self.split(a), self.split(b)
        return self.join(u1 - u2, v1 - v2)

    def mul(self, a, b):
        (u1, v1), (u2, v2) = self.split(a), self.split(b)
        return self.join(u1 * u2 + self.nu * v1 * v2, u1 * v2 + v1 * u2)

    def inv(self, a):
        return self.INV[np.asarray(a, dtype=np.int64)]

    def matmul(self, a, x):
        (a0, a1), (x0, x1) = self.split(a), self.split(x)
        f = lambda s, t: np.asarray(s, dtype=np.float64) @ np.asarray(t, dtype=np.float64)
        u = f(a0, x0) + self.nu * f(a1, x1)
        v = f(a0, x1) + f(a1, x0)
        return self.join((u % self.p).astype(np.int64), (v % self.p).astype(np.int64))


def fpow(field, a, k):
    a = np.asarray(a, dtype=np.int64)
    result = np.ones_like(a)
    base = a.copy()
    while k:
        if k & 1:
            result = field.mul(result, base)
        base = field.mul(base, base)
        k >>= 1
    return result


def gauss_inverse(field, a):
    """Inverse of a square matrix over the field, or None if singular."""
    n = a.shape[0]
    dt = _elim_dtype(field)
    aug = np.concatenate([np.asarray(a, dtype=dt), np.eye(n, dtype=dt)], axis=1)
    for c in range(n):
        nz = np.nonzero(aug[c:, c])[0]
        if len(nz) == 0:
            return None
        piv = c + nz[0]
        if piv != c:
            aug[[c, piv]] = aug[[piv, c]]
        aug[c] = field.mul(aug[c], int(field.inv(aug[c, c])))
        f = aug[:, c].copy()
        f[c] = 0
        rows = np.nonzero(f)[0]
        if len(rows):
            aug[rows] = field.sub(aug[rows], field.mul(f[rows, None], aug[c][None, :]))
    return aug[:, n:].astype(np.int64)


def _elim_dtype(field):
    """int16 is exact for elimination over a small prime field (entries < p, products < p^2 < 2^15)."""
    return np.int16 if type(field).__name__ == "PrimeField" and field.p * field.p < 2**15 else np.int64


def rank_mod(field, a):
    a = np.asarray(a, dtype=_elim_dtype(field)).copy()
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        nz = np.nonzero(a[r:, c])[0]
        if len(nz) == 0:
            continue
        piv = r + nz[0]
        a[[r, piv]] = a[[piv, r]]
        a[r] = field.mul(a[r], int(field.inv(a[r, c])))
        f = a[:, c].copy()
        f[r] = 0
        idx = np.nonzero(f)[0]
        if len(idx):
            a[idx] = field.sub(a[idx], field.mul(f[idx, None], a[r][None, :]))
        r += 1
        if r == rows:
            break
    return r


# ---------------------------------------------------------------------------------------------
# DQI model
def dqi_expected(m, ell, q, r):
    a = np.zeros((ell + 1, ell + 1))
    for k in range(1, ell + 1):
        a[k, k - 1] = a[k - 1, k] = math.sqrt(k * (m - k + 1))
    d = (q - 2 * r) / math.sqrt(r * (q - r))
    for k in range(ell + 1):
        a[k, k] = k * d
    return m * r / q + math.sqrt(r * (q - r)) / q * float(np.linalg.eigvalsh(a).max())


def semicircle(lam, rho):
    return 1.0 if lam > 1 - rho else (math.sqrt(lam * (1 - rho)) + math.sqrt(rho * (1 - lam))) ** 2


# ---------------------------------------------------------------------------------------------
# code builders: return field, B (m x n), genus, checks dict
def build_rs(p, alpha):
    field = PrimeField(p)
    m = p - 1
    n = round(m * alpha)
    pts = np.arange(1, p, dtype=np.int64)
    B = np.ones((m, n), dtype=np.int64)
    for j in range(1, n):
        B[:, j] = B[:, j - 1] * pts % p
    return field, B, 0, {}


def build_elliptic(p, alpha, seed):
    field = PrimeField(p)
    rng = np.random.default_rng(seed + 17)
    sq = {}
    for y in range(p):
        sq.setdefault(y * y % p, []).append(y)
    while True:
        a, b = (int(v) for v in rng.integers(0, p, 2))
        if (4 * a ** 3 + 27 * b ** 2) % p:
            break
    pts = [(x, y) for x in range(p) for y in sq.get((x ** 3 + a * x + b) % p, [])]
    X = np.array([t[0] for t in pts], dtype=np.int64)
    Y = np.array([t[1] for t in pts], dtype=np.int64)
    m = len(pts)
    n = round(m * alpha)
    orders = [0] + list(range(2, n + 1))  # the n pole orders <= n (1 is a gap)
    cols = []
    for k in orders:
        if k % 2 == 0:
            cols.append(fpow(field, X, k // 2))
        else:
            cols.append(field.mul(fpow(field, X, (k - 3) // 2), Y))
    B = np.stack(cols, axis=1)
    # dimension check: every monomial x^i y^j with 2i + 3j <= n spans exactly n dimensions
    allcols = [field.mul(fpow(field, X, i), fpow(field, Y, j))
               for j in range(0, n // 3 + 1) for i in range(0, (n - 3 * j) // 2 + 1)]
    dim = rank_mod(field, np.stack(allcols, axis=1)) if n <= 300 else None
    curve_ok = bool(np.all((Y * Y - (X ** 3 + a * X + b)) % p == 0))
    return field, B, 1, {"curve": f"y^2 = x^3 + {a} x + {b}", "points": m, "curve_ok": curve_ok,
                         "dimension_count": dim, "expected_dimension": n}


def build_hermitian(p, alpha):
    field = PrimeSquareField(p)
    q = p * p
    e = np.arange(q, dtype=np.int64)
    xn = fpow(field, e, p + 1)
    tr = field.add(fpow(field, e, p), e)
    pts = [(int(x), int(y)) for x in range(q) for y in np.nonzero(tr == xn[x])[0]]
    X = np.array([t[0] for t in pts], dtype=np.int64)
    Y = np.array([t[1] for t in pts], dtype=np.int64)
    m = len(pts)
    assert m == p ** 3
    g = p * (p - 1) // 2
    n = round(m * alpha)
    a = n + g - 1
    monos = sorted([(i, j) for j in range(p) for i in range(a // p + 1) if i * p + j * (p + 1) <= a],
                   key=lambda t: t[0] * p + t[1] * (p + 1))
    assert len(monos) == n, (len(monos), n)
    xp = {i: fpow(field, X, i) for i in {t[0] for t in monos}}
    yp = {j: fpow(field, Y, j) for j in range(p)}
    B = np.stack([field.mul(xp[i], yp[j]) for i, j in monos], axis=1)
    lhs = field.add(fpow(field, Y, p), Y)
    curve_ok = bool(np.all(lhs == fpow(field, X, p + 1)))
    return field, B, g, {"points": m, "genus": g, "pole_bound": a, "curve_ok": curve_ok,
                         "monomials": len(monos)}


# ---------------------------------------------------------------------------------------------
class Control:
    def __init__(self, field, B, g, sets, rng):
        self.f, self.B, self.g, self.rng = field, B, g, rng
        self.m, self.n = B.shape
        self.q = field.q
        self.sets = [set(map(int, s)) for s in sets]
        self.r = len(sets[0])
        self.A = np.array([sorted(s) for s in sets], dtype=np.int64)
        self.M = np.zeros((self.m, self.q), dtype=bool)
        self.M[np.arange(self.m)[:, None], self.A] = True
        self.ar = np.arange(self.m)

    def score(self, y):
        return int(self.M[self.ar, y].sum())

    def verify(self, y):
        for _ in range(20):
            S = np.sort(self.rng.choice(self.m, self.n, replace=False))
            inv = gauss_inverse(self.f, self.B[S])
            if inv is not None:
                break
        else:
            raise RuntimeError("no invertible information set found")
        x = self.f.matmul(inv, y[S][:, None])[:, 0]
        assert np.array_equal(self.f.matmul(self.B, x[:, None])[:, 0], y), "witness not in the code"
        return sum(int(v) in s for v, s in zip(y.tolist(), self.sets))

    def systematic(self):
        while True:
            S = self.rng.choice(self.m, self.n, replace=False)
            inv = gauss_inverse(self.f, self.B[S])
            if inv is not None:
                return S, self.f.matmul(self.B, inv)

    # A: Prange
    def prange(self, restarts):
        best, best_y, scores = -1, None, []
        per = max(1, restarts // 4)
        for _ in range(4):
            S, L = self.systematic()
            for start in range(0, per, 64):
                k = min(64, per - start)
                T = self.A[S[:, None], self.rng.integers(0, self.r, (self.n, k))]
                Y = self.f.matmul(L, T)
                sc = self.M[self.ar[:, None], Y].sum(axis=0)
                scores.extend(sc.tolist())
                j = int(sc.argmax())
                if sc[j] > best:
                    best, best_y = int(sc[j]), Y[:, j].copy()
        return {"best": best, "restarts": len(scores), "mean": float(np.mean(scores)) / self.m}, best_y

    def _line_counts(self, y, u):
        free = np.nonzero(u)[0]
        uinv = self.f.inv(u[free])
        tv = self.f.mul(self.f.sub(self.A[free], y[free, None]), uinv[:, None])
        counts = np.bincount(tv.ravel(), minlength=self.q)
        fixed = np.setdiff1d(self.ar, free)
        return counts + int(self.M[fixed, y[fixed]].sum())

    def _pivot(self, S, L, k, sat_mask):
        cand = np.nonzero(sat_mask & (L[:, k] != 0))[0]
        cand = np.setdiff1d(cand, S)
        if len(cand) == 0:
            return S, L
        j = int(self.rng.choice(cand))
        piv_inv = self.f.inv(L[j, k])
        rowj = self.f.mul(L[j, :], piv_inv)  # L[j,l] / L[j,k]
        colk = L[:, k].copy()
        Lnew = self.f.sub(L, self.f.mul(colk[:, None], rowj[None, :]))
        Lnew[:, k] = self.f.mul(colk, piv_inv)
        S = S.copy()
        S[k] = j
        return S, Lnew

    def _start(self):
        S, L = self.systematic()
        t = self.A[S, self.rng.integers(0, self.r, self.n)]
        return S, L, self.f.matmul(L, t[:, None])[:, 0]

    # B: pivoting line search
    def line(self, seconds, patience=60):
        t_end = time.time() + seconds
        best, best_y, moves, restarts = -1, None, 0, 0
        while time.time() < t_end:
            S, L, y = self._start()
            cur, stale = self.score(y), 0
            restarts += 1
            while stale < patience and time.time() < t_end:
                k = int(self.rng.integers(self.n))
                counts = self._line_counts(y, L[:, k])
                assert counts[0] == cur
                top = int(counts.max())
                moves += 1
                if top > cur:
                    t = int(self.rng.choice(np.nonzero(counts == top)[0]))
                    y = self.f.add(y, self.f.mul(t, L[:, k]))
                    cur, stale = top, 0
                else:
                    stale += 1
                S, L = self._pivot(S, L, k, self.M[self.ar, y])
            if cur > best:
                best, best_y = cur, y.copy()
        return {"best": best, "moves": moves, "restarts": restarts}, best_y

    # C: heat bath
    def heat(self, seconds, beta0=0.05, beta1=4.0):
        t0 = time.time()
        S, L, y = self._start()
        cur = self.score(y)
        best, best_y, moves = cur, y.copy(), 0
        while True:
            frac = (time.time() - t0) / seconds
            if frac >= 1:
                break
            beta = beta0 * (beta1 / beta0) ** frac
            k = int(self.rng.integers(self.n))
            counts = self._line_counts(y, L[:, k])
            w = np.exp(beta * (counts - counts.max()))
            t = int(self.rng.choice(self.q, p=w / w.sum()))
            y = self.f.add(y, self.f.mul(t, L[:, k]))
            cur = int(counts[t])
            moves += 1
            if cur > best:
                best, best_y = cur, y.copy()
            S, L = self._pivot(S, L, k, self.M[self.ar, y])
        return {"best": best, "moves": moves}, best_y

    # D: subfield information sets over F_p (F_(p^2) only)
    def subfield(self, restarts):
        f, p = self.f, self.f.p
        dirs = [(1, 0)] + [(t, 1) for t in range(p)]
        lines, meta = [], []
        for d0, d1 in dirs:
            seen = set()
            for a in range(self.q):
                pts = frozenset(f.join((a % p) + d0 * t, (a // p) + d1 * t) for t in range(p))
                if pts in seen:
                    continue
                seen.add(pts)
                lines.append(sorted(pts))
                meta.append((a, d0, d1))
        lines = np.array(lines, dtype=np.int64)
        inside = [np.nonzero(self.M[i][lines].all(axis=1))[0] for i in range(self.m)]
        has_line = np.array([len(x) > 0 for x in inside])
        b0, b1 = f.split(self.B)
        nu = f.nu
        best, best_y = -1, None
        for _ in range(restarts):
            order = self.rng.permutation(self.m)
            order = np.concatenate([order[has_line[order]], order[~has_line[order]]])
            rows_eq, rhs = [], []
            for i in order:
                # coefficient rows over (x0, x1): u-part and v-part of (Bx)_i
                cu = np.concatenate([b0[i], nu * b1[i]]) % p
                cv = np.concatenate([b1[i], b0[i]]) % p
                if has_line[i]:
                    li = int(self.rng.choice(inside[i]))
                    a, d0, d1 = meta[li]
                    # phi(u, v) = d1*u - d0*v vanishes on the direction (d0, d1)
                    rows_eq.append((d1 * cu - d0 * cv) % p)
                    rhs.append((d1 * (a % p) - d0 * (a // p)) % p)
                else:
                    t = int(self.rng.choice(self.A[i]))
                    rows_eq.extend([cu, cv])
                    rhs.extend([t % p, t // p])
                if len(rows_eq) >= 2 * self.n + 8:
                    break
            E = np.array(rows_eq, dtype=np.int64)
            h = np.array(rhs, dtype=np.int64)
            x = self._solve_consistent(E, h, p)
            xx = f.join(x[:self.n], x[self.n:])
            y = f.matmul(self.B, xx[:, None])[:, 0]
            s = self.score(y)
            if s > best:
                best, best_y = s, y
        return {"best": best, "restarts": restarts, "rows_with_line": float(has_line.mean())}, best_y

    def _solve_consistent(self, E, h, p):
        """Greedy sequential elimination mod p: keep each equation that is independent or
        consistent, drop inconsistent ones; free variables random."""
        cols = E.shape[1]
        basis = np.zeros((0, cols + 1), dtype=np.int64)
        pivots = []
        inv = np.array([0] + [pow(a, p - 2, p) for a in range(1, p)], dtype=np.int64)
        for row, rh in zip(E, h):
            v = np.concatenate([row, [rh]]) % p
            if len(pivots):
                coef = v[pivots]
                v = (v - coef @ basis) % p
            nz = np.nonzero(v[:cols])[0]
            if len(nz) == 0:
                continue  # dependent: consistent or not, drop
            c = nz[0]
            v = v * inv[v[c]] % p
            if len(pivots):
                basis = (basis - np.outer(basis[:, c], v)) % p
            basis = np.vstack([basis, v])
            pivots.append(c)
        x = self.rng.integers(0, p, cols)
        free = np.setdiff1d(np.arange(cols), pivots)
        x[pivots] = (basis[:, cols] - basis[:, free] @ x[free]) % p
        return x


# ---------------------------------------------------------------------------------------------
def run():
    results = []
    for size in SIZES:
        for seed in SEEDS:
            t0 = time.time()
            rng = np.random.default_rng(seed)
            extra = {}
            if FAMILY == "rs":
                field, B, g, info = build_rs(size, ALPHA)
                sets = None
            elif FAMILY == "elliptic":
                field, B, g, info = build_elliptic(size, ALPHA, seed)
                sets = None
            elif FAMILY == "hermitian":
                field, B, g, info = build_hermitian(size, ALPHA)
                sets = None
            else:
                raise SystemExit("unknown family")
            m, n = B.shape
            q = field.q
            if sets is None:
                r = (q - 1) // 2 if q % 2 else q // 2
                sets = [rng.choice(q, r, replace=False) for _ in range(m)]
            ctl = Control(field, B, g, sets, rng)
            rk = rank_mod(field, B)
            dstar = n - g + 1
            ell = (dstar - 2) // 2
            dqi = dqi_expected(m, ell, q, ctl.r)
            rho = ctl.r / q
            rec = {"family": FAMILY, "size": size, "seed": seed, "q": q, "m": m, "n": n, "genus": g,
                   "rank": rk, "ell": ell, "rho": rho, "dqi_lemma92": dqi,
                   "semicircle": semicircle(ell / m, rho), "Q_Pr": rho + (1 - rho) * n / m,
                   "info": info, "attacks": {}}
            assert rk == n, f"rank {rk} != {n}"
            print(f"{FAMILY} size={size} seed={seed}: q={q} m={m} n={n} g={g} ell={ell} rank ok; "
                  f"DQI {dqi:.1f} ({dqi / m:.4f}); Q_Pr {rec['Q_Pr']:.4f}; info {info}", flush=True)
            budget = int(min(8000, max(1000, 4e9 / (m * n))))
            for key, call in (("A_prange", lambda: ctl.prange(budget)),
                              ("B_line", lambda: ctl.line(T_LINE)),
                              ("C_heat", lambda: ctl.heat(T_HEAT))):
                res, y = call()
                s_ind = ctl.verify(y)
                assert s_ind == res["best"], (key, s_ind, res["best"])
                res["fraction"] = res["best"] / m
                rec["attacks"][key] = res
            if isinstance(field, PrimeSquareField):
                res, y = ctl.subfield(SUB_RESTARTS)
                assert ctl.verify(y) == res["best"]
                res["fraction"] = res["best"] / m
                rec["attacks"]["D_subfield"] = res
            best = max(v["best"] for v in rec["attacks"].values())
            rec["best_classical"] = best
            rec["classical_minus_dqi"] = best - dqi
            rec["seconds"] = time.time() - t0
            print("   " + "; ".join(f"{k} {v['best']} ({v['fraction']:.4f})" for k, v in rec["attacks"].items())
                  + f" | Prange mean {rec['attacks']['A_prange']['mean']:.4f} vs {rec['Q_Pr']:.4f}"
                  + f" | best - DQI = {best - dqi:+.1f} | {rec['seconds']:.0f}s", flush=True)
            results.append(rec)
            with open(OUT % (FAMILY, TAG), "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"family": FAMILY, "alpha": ALPHA,
                           "budgets_seconds": {"line": T_LINE, "heat": T_HEAT}, "results": results},
                          fh, indent=2, default=str)
    print("wrote", OUT % (FAMILY, TAG))


def merge():
    rows = {}
    for path in sorted(p for fam in ("rs", "elliptic", "hermitian") for p in glob.glob("results/finite-control-%s-run-*.json" % fam)):
        if path.endswith("-quick.json"):
            continue
        with open(path, encoding="utf-8") as fh:
            for rec in json.load(fh)["results"]:
                rows.setdefault((rec["family"], rec["size"]), []).append(rec)
    print("family      size    q      m     DQI(L9.2)  Q_Pr    best classical (per attack)                      best-DQI")
    for (fam, size) in sorted(rows):
        recs = rows[(fam, size)]
        m = recs[0]["m"]
        keys = sorted({k for r in recs for k in r["attacks"]})
        per = "  ".join(f"{k.split('_')[0]}:{np.mean([r['attacks'][k]['fraction'] for r in recs if k in r['attacks']]):.4f}"
                        for k in keys)
        ms = sorted({r["m"] for r in recs})
        mlabel = f"{ms[0]}" if len(ms) == 1 else f"{ms[0]}-{ms[-1]}"
        print(f"{fam:10s} {size:5d} {recs[0]['q']:6d} {mlabel:>9s}  "
              f"{np.mean([r['dqi_lemma92'] / r['m'] for r in recs]):.4f}    "
              f"{np.mean([r['Q_Pr'] for r in recs]):.4f}  {per}   "
              f"{np.mean([r['classical_minus_dqi'] / r['m'] for r in recs]):+.4f}  (n={len(recs)})")


if __name__ == "__main__":
    merge() if "--merge" in sys.argv else run()
