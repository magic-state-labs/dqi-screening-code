"""Finite control and dequantization attempt: optimal polynomial intersection (OPI).

Instance (Jordan et al., arXiv 2408.08292, headline regime): prime p, evaluation points all of
F_p^* (m = p - 1), polynomials of degree < n with n = round(m/10), accepted sets F_i uniformly
random of size r = (p-1)/2.  B_ij = a_i^j, optimizer code = RS[m, n], decoder code ker(B^T) =
GRS[m, m-n, n+1], licensed radius ell = floor((d-2)/2) with d = n+1.

Checks: rank of B, random maximal minors, Berlekamp-Massey syndrome decoding
of random weight-ell patterns (the licence of gate D4), gates D0-D6, the margin cap, the restart
threshold, the regime factor and Q*, and DQI's finite value from Lemma 9.2 of Jordan et al.

Attacks, every witness checked to lie in the code and re-scored independently:
  A. Prange restarts over F_p (information set + accepted targets, Lagrange completion);
  B. pinned-line search: fix n-1 currently satisfied rows; the polynomials of degree < n agreeing
     with the current one there form a line f + t*g, g = prod (z - a_l); score all p points of the
     line exactly with one histogram and move to the best;
  C. heat-bath line search: the same lines, t drawn with probability proportional to
     exp(beta * count(t)), beta annealed (a block-Gibbs sampler);
  D. pinned-plane search (small p): n-2 pins, a p x p histogram over two directions, started from
     the best of B;
  E. CP-SAT on the exact model (small p), hinted with the best heuristic witness;
  F. Guruswami-Sudan list-recovery vacuity (computed).

Usage:
  python scripts/18_opi_finite_control.py [--quick] [--sizes 101,211] [--tag name]
  python scripts/18_opi_finite_control.py --merge
Writes results/opi-finite-control-run-<tag>.json.
"""
import glob
import json
import math
import sys
import time

import numpy as np

QUICK = "--quick" in sys.argv


def arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


SIZES = [int(s) for s in arg("--sizes", "101,211,401" if QUICK else "101,211,401,1009,2003,4001").split(",")]
TAG = arg("--tag", "quick" if QUICK else "all")
SEEDS = [2101, 2102, 2103]
T_LINE = 4 if QUICK else 40
T_HEAT = 4 if QUICK else 40
T_PLANE = 3 if QUICK else 30
T_CPSAT = 10 if QUICK else 90
PLANE_MAX_P = 211
CPSAT_MAX_P = 211
BM_TRIALS = 20 if QUICK else 100
OUT = "results/opi-finite-control-run-%s.json"


# ---------------------------------------------------------------------------------------------
# prime-field helpers
def primitive_root(p):
    fac, x, d = set(), p - 1, 2
    while d * d <= x:
        while x % d == 0:
            fac.add(d)
            x //= d
        d += 1
    if x > 1:
        fac.add(x)
    for g in range(2, p):
        if all(pow(g, (p - 1) // f, p) != 1 for f in fac):
            return g
    raise ValueError


def solve_mod(a, b, p):
    """Solve a x = b (mod p) for square nonsingular a; b may be a matrix. Returns None if singular."""
    a = np.array(a, dtype=np.int64) % p
    b = np.array(b, dtype=np.int64) % p
    vec = b.ndim == 1
    if vec:
        b = b[:, None]
    k = a.shape[0]
    aug = np.concatenate([a, b], axis=1)
    for c in range(k):
        nz = np.nonzero(aug[c:, c])[0]
        if len(nz) == 0:
            return None
        piv = c + nz[0]
        if piv != c:
            aug[[c, piv]] = aug[[piv, c]]
        aug[c] = aug[c] * pow(int(aug[c, c]), p - 2, p) % p
        f = aug[:, c].copy()
        f[c] = 0
        aug = (aug - f[:, None] * aug[c][None, :]) % p
    x = aug[:, k:]
    return x[:, 0] if vec else x


def rank_mod(a, p):
    a = np.array(a, dtype=np.int64) % p
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        nz = np.nonzero(a[r:, c])[0]
        if len(nz) == 0:
            continue
        piv = r + nz[0]
        a[[r, piv]] = a[[piv, r]]
        a[r] = a[r] * pow(int(a[r, c]), p - 2, p) % p
        f = a[:, c].copy()
        f[r] = 0
        a = (a - f[:, None] * a[r][None, :]) % p
        r += 1
        if r == rows:
            break
    return r


# ---------------------------------------------------------------------------------------------
# models (as in 15_partially_stuck_finite_control.py)
def dqi_expected(m, ell, q, rsize):
    a = np.zeros((ell + 1, ell + 1))
    for k in range(1, ell + 1):
        a[k, k - 1] = a[k - 1, k] = math.sqrt(k * (m - k + 1))
    d = (q - 2 * rsize) / math.sqrt(rsize * (q - rsize))
    for k in range(ell + 1):
        a[k, k] = k * d
    lam_max = float(np.linalg.eigvalsh(a).max())
    return m * rsize / q + math.sqrt(rsize * (q - rsize)) / q * lam_max


def semicircle(lam, rho):
    return 1.0 if lam > 1 - rho else (math.sqrt(lam * (1 - rho)) + math.sqrt(rho * (1 - lam))) ** 2


def kl(a, b):
    t = 0.0
    if a > 0:
        t += a * math.log(a / b)
    if a < 1:
        t += (1 - a) * math.log((1 - a) / (1 - b))
    return t


def qstar(alpha, q, rho):
    if q ** alpha * rho >= 1:
        return 1.0
    lo, hi = rho, 1 - 1e-15
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kl(mid, rho) < alpha * math.log(q) else (lo, mid)
    return lo


def hq(x, q):
    return x * math.log(q - 1, q) - x * math.log(x, q) - (1 - x) * math.log(1 - x, q)


# ---------------------------------------------------------------------------------------------
class Instance:
    def __init__(self, p, seed):
        self.p, self.seed = p, seed
        self.m, self.n, self.r = p - 1, round((p - 1) / 10), (p - 1) // 2
        self.rng = np.random.default_rng(seed)
        self.g = primitive_root(p)
        self.EXP = np.zeros(2 * (p - 1), dtype=np.int64)
        self.LOG = np.zeros(p, dtype=np.int64)
        x = 1
        for i in range(p - 1):
            self.EXP[i] = x
            self.LOG[x] = i
            x = x * self.g % p
        self.EXP[p - 1:] = self.EXP[:p - 1]
        self.pts = np.arange(1, p, dtype=np.int64)
        B = np.ones((self.m, self.n), dtype=np.int64)
        for j in range(1, self.n):
            B[:, j] = B[:, j - 1] * self.pts % p
        self.B = B
        self.A = np.stack([self.rng.choice(p, self.r, replace=False) for _ in range(self.m)]).astype(np.int64)
        self.M = np.zeros((self.m, p), dtype=bool)
        self.M[np.arange(self.m)[:, None], self.A] = True
        self.sets = [set(map(int, row)) for row in self.A]
        diff = (self.pts[:, None] - self.pts[None, :]) % p
        logs = self.LOG[diff]
        np.fill_diagonal(logs, 0)
        self.DLOG = logs.astype(np.int16 if p < 32768 else np.int32)
        self.ar = np.arange(self.m)

    def inv(self, x):
        return self.EXP[(self.p - 1 - self.LOG[x]) % (self.p - 1)]

    def score(self, y):
        return int(self.M[self.ar, y].sum())

    def score_independent(self, y):
        return sum(int(v) in s for v, s in zip(y.tolist(), self.sets))

    def coefficients(self, y):
        c = solve_mod(self.B[:self.n], y[:self.n], self.p)
        return c

    def in_code(self, y):
        c = self.coefficients(y)
        return c is not None and np.array_equal((self.B @ c) % self.p, y % self.p)

    def pin_direction(self, pins):
        """Values at all points of prod_{l in pins} (z - a_l); zero on the pins."""
        s = self.DLOG[:, pins].sum(axis=1, dtype=np.int64)
        u = self.EXP[s % (self.p - 1)]
        u[pins] = 0
        return u

    def line_counts(self, y, u):
        p = self.p
        free = np.nonzero(u)[0]
        uinv = self.inv(u[free])
        tv = ((self.A[free] - y[free, None]) * uinv[:, None]) % p
        counts = np.bincount(tv.ravel(), minlength=p)
        return counts + (self.m - len(free))  # pinned rows are satisfied for every t


# ---------------------------------------------------------------------------------------------
def berlekamp_massey(s, p):
    c, b = [1], [1]
    L, mshift, bb = 0, 1, 1
    for i in range(len(s)):
        d = s[i]
        for j in range(1, L + 1):
            d = (d + c[j] * s[i - j]) % p
        if d == 0:
            mshift += 1
            continue
        coef = d * pow(bb, p - 2, p) % p
        t = c[:]
        c = c + [0] * (len(b) + mshift - len(c))
        for j in range(len(b)):
            c[j + mshift] = (c[j + mshift] - coef * b[j]) % p
        if 2 * L <= i:
            L, b, bb, mshift = i + 1 - L, t, d, 1
        else:
            mshift += 1
    return c[:L + 1], L


def decoder_check(inst, ell, trials):
    p, m, n = inst.p, inst.m, inst.n
    ok = 0
    for _ in range(trials):
        pos = inst.rng.choice(m, ell, replace=False)
        e = np.zeros(m, dtype=np.int64)
        e[pos] = inst.rng.integers(1, p, ell)
        s = (inst.B.T @ e) % p
        conn, L = berlekamp_massey([int(v) for v in s], p)
        # roots of the connection polynomial are the inverses of the error points
        xinv = inst.inv(inst.pts)
        val = np.zeros(m, dtype=np.int64)
        for coef in reversed(conn):
            val = (val * xinv + coef) % p
        found = np.nonzero(val == 0)[0]
        if L != ell or len(found) != L:
            continue
        V = np.stack([inst.B[found, j] for j in range(L)])
        ev = solve_mod(V, s[:L], p)
        rec = np.zeros(m, dtype=np.int64)
        rec[found] = ev
        ok += int(np.array_equal(rec, e))
    return ok


# ---------------------------------------------------------------------------------------------
def attack_prange(inst):
    p, m, n, rng = inst.p, inst.m, inst.n, inst.rng
    total = int(min(20000, max(2000, 2e10 / (m * n))))
    best, best_y, scores = -1, None, []
    per_set = total // 4
    for _ in range(4):
        S = np.sort(rng.choice(m, n, replace=False))
        binv = solve_mod(inst.B[S], np.eye(n, dtype=np.int64), p)
        L = ((inst.B.astype(np.float64) @ binv.astype(np.float64)) % p).astype(np.float64)
        for start in range(0, per_set, 256):
            k = min(256, per_set - start)
            T = inst.A[S[:, None], rng.integers(0, inst.r, (n, k))].astype(np.float64)
            Y = (L @ T % p).astype(np.int64)
            sc = inst.M[inst.ar[:, None], Y].sum(axis=0)
            scores.extend(sc.tolist())
            j = int(sc.argmax())
            if sc[j] > best:
                best, best_y = int(sc[j]), Y[:, j].copy()
    theory = (n + (m - n) * inst.r / p) / m
    return {"restarts": len(scores), "best": best, "mean": float(np.mean(scores)) / m,
            "theory_mean": theory, "y": best_y}


def random_start(inst):
    p, n, rng = inst.p, inst.n, inst.rng
    S = np.sort(rng.choice(inst.m, n, replace=False))
    t = inst.A[S, rng.integers(0, inst.r, n)]
    c = solve_mod(inst.B[S], t, p)
    return (inst.B @ c) % p


def attack_line(inst, seconds, patience=40):
    p, n, rng = inst.p, inst.n, inst.rng
    t_end = time.time() + seconds
    best, best_y, moves, restarts, local_optima = -1, None, 0, 0, []
    while time.time() < t_end:
        y = random_start(inst)
        cur = inst.score(y)
        restarts += 1
        stale = 0
        while stale < patience and time.time() < t_end:
            sat = np.nonzero(inst.M[inst.ar, y])[0]
            pins = rng.choice(sat, n - 1, replace=False)
            u = inst.pin_direction(pins)
            counts = inst.line_counts(y, u)
            assert counts[0] == cur
            top = counts.max()
            moves += 1
            if top > cur:
                t = int(rng.choice(np.nonzero(counts == top)[0]))
                y = (y + t * u) % p
                cur = int(top)
                stale = 0
            else:
                stale += 1
        local_optima.append(cur)
        if cur > best:
            best, best_y = cur, y.copy()
    return {"best": best, "moves": moves, "restarts": restarts,
            "mean_local_optimum": float(np.mean(local_optima)) / inst.m, "y": best_y}


def attack_heat(inst, seconds, beta0=0.05, beta1=4.0):
    p, n, rng = inst.p, inst.n, inst.rng
    t0 = time.time()
    y = random_start(inst)
    cur = inst.score(y)
    best, best_y, moves = cur, y.copy(), 0
    while True:
        frac = (time.time() - t0) / seconds
        if frac >= 1:
            break
        beta = beta0 * (beta1 / beta0) ** frac
        sat = np.nonzero(inst.M[inst.ar, y])[0]
        pins = rng.choice(sat, n - 1, replace=False)
        u = inst.pin_direction(pins)
        counts = inst.line_counts(y, u)
        w = beta * (counts - counts.max())
        prob = np.exp(w)
        prob /= prob.sum()
        t = int(rng.choice(p, p=prob))
        y = (y + t * u) % p
        cur = int(counts[t])
        moves += 1
        if cur > best:
            best, best_y = cur, y.copy()
    return {"best": best, "moves": moves, "final": cur, "y": best_y}


def attack_plane(inst, y0, seconds, patience=30):
    p, n, rng, m = inst.p, inst.n, inst.rng, inst.m
    y, cur = y0.copy(), inst.score(y0)
    t_end, moves, stale = time.time() + seconds, 0, 0
    t2 = np.arange(p, dtype=np.int64)
    while stale < patience and time.time() < t_end:
        sat = np.nonzero(inst.M[inst.ar, y])[0]
        pins = rng.choice(sat, n - 2, replace=False)
        u = inst.pin_direction(pins)
        free = np.nonzero(u)[0]
        z = ((inst.A[free] - y[free, None]) * inst.inv(u[free])[:, None]) % p
        a = inst.pts[free]
        t1 = (z[:, :, None] - a[:, None, None] * t2[None, None, :]) % p
        idx = (t1 * p + t2[None, None, :]).ravel()
        counts = np.bincount(idx, minlength=p * p) + (m - len(free))
        assert counts[0] == cur
        top = counts.max()
        moves += 1
        if top > cur:
            k = int(rng.choice(np.nonzero(counts == top)[0]))
            a1, a2 = divmod(k, p)
            y = (y + a1 * u + a2 * u * inst.pts) % p
            cur = int(top)
            stale = 0
        else:
            stale += 1
    return {"best": cur, "moves": moves, "y": y}


def attack_cpsat(inst, hint_y, seconds):
    try:
        from ortools.sat.python import cp_model
    except Exception as exc:  # a broken OR-tools/protobuf install should not kill the control
        print(f"   CP-SAT unavailable: {type(exc).__name__}", flush=True)
        return None
    p, m, n = inst.p, inst.m, inst.n
    model = cp_model.CpModel()
    coef = [model.NewIntVar(0, p - 1, f"c{j}") for j in range(n)]
    hits = []
    for i in range(m):
        w = [int(v) for v in inst.B[i]]
        word = model.NewIntVar(0, p - 1, f"w{i}")
        quo = model.NewIntVar(0, sum(w) * (p - 1) // p, f"t{i}")
        model.Add(sum(wj * cj for wj, cj in zip(w, coef)) == word + p * quo)
        hit = model.NewBoolVar(f"b{i}")
        model.AddAllowedAssignments([word], [[int(v)] for v in sorted(inst.sets[i])]).OnlyEnforceIf(hit)
        hits.append(hit)
    model.Maximize(sum(hits))
    hc = inst.coefficients(hint_y)
    for v, h in zip(coef, hc.tolist()):
        model.AddHint(v, int(h))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(seconds)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 7
    status = solver.StatusName(solver.Solve(model))
    c = np.array([solver.Value(v) for v in coef], dtype=np.int64)
    y = (inst.B @ c) % p
    return {"status": status, "best": inst.score(y), "bound": int(solver.BestObjectiveBound()),
            "wall": solver.WallTime(), "y": y}


# ---------------------------------------------------------------------------------------------
def checks_and_model(inst):
    p, m, n, r = inst.p, inst.m, inst.n, inst.r
    d = n + 1
    ell = (d - 2) // 2
    alpha, lam, rho, delta = n / m, ell / m, r / p, d / m
    rk = rank_mod(inst.B, p)
    minors = sum(solve_mod(inst.B[np.sort(inst.rng.choice(m, n, replace=False))],
                           np.zeros(n, dtype=np.int64), p) is not None for _ in range(5))
    dec_ok = decoder_check(inst, ell, BM_TRIALS if p <= 401 else max(10, BM_TRIALS // 5))
    q_dqi = semicircle(lam, rho)
    q_pr = rho + (1 - rho) * alpha
    u = alpha / 2
    kappa = math.sqrt(u * (1 - u)) - u
    mu = q_dqi - q_pr
    return {
        "p": p, "m": m, "n": n, "r": r, "d": d, "ell": ell, "alpha": alpha, "lambda": lam,
        "rho": rho, "rank": rk, "minors_nonsingular": f"{minors}/5",
        "decoder_recovered": dec_ok,
        "D0": {"H_q(lambda)": hq(lam, p), "alpha": alpha, "pass": hq(lam, p) <= alpha},
        "D1": {"delta": delta, "pass": True},
        "D2": {"threshold": 1 - math.sqrt(1 - alpha ** 2), "pass": delta > 1 - math.sqrt(1 - alpha ** 2)},
        "D3": {"rho(1-rho)": rho * (1 - rho), "threshold": alpha / (8 * (1 - alpha / 2)),
               "pass": rho * (1 - rho) > alpha / (8 * (1 - alpha / 2))},
        "D4": {"pass": None},
        "D5": "no efficient rate-distortion encoder known for RS",
        "D6": "void (prime field)",
        "Q_DQI_semicircle": q_dqi, "Q_Pr": q_pr, "mu": mu, "kappa": kappa,
        "mu_over_kappa": mu / kappa,
        "restart_threshold_m": (1 - alpha) * math.log(2 ** 40 / 0.05) / (2 * mu * mu),
        "regime_factor": p ** alpha * rho, "Q_star": qstar(alpha, p, rho),
        "dqi_lemma92": dqi_expected(m, ell, p, r),
        "gs_list_recovery": {"sqrt((n-1) m r)": math.sqrt((n - 1) * m * r), "m": m,
                             "vacuous": math.sqrt((n - 1) * m * r) >= m},
    }


def finish(inst, res, name):
    y = res.pop("y")
    assert inst.in_code(y), f"{name}: witness not in the code"
    s_ind = inst.score_independent(y)
    assert s_ind == res["best"], f"{name}: re-score mismatch {s_ind} vs {res['best']}"
    res["fraction"] = res["best"] / inst.m
    return y


def run_size(p):
    out = []
    for seed in SEEDS:
        t0 = time.time()
        inst = Instance(p, seed)
        chk = checks_and_model(inst)
        ell = chk["ell"]
        chk["D4"]["pass"] = chk["decoder_recovered"] > 0 and chk["decoder_recovered"] == (
            BM_TRIALS if p <= 401 else max(10, BM_TRIALS // 5))
        rec = {"seed": seed, "checks": chk, "attacks": {}}
        dqi = chk["dqi_lemma92"]
        print(f"p={p} seed={seed}: m={inst.m} n={inst.n} ell={ell} rank={chk['rank']} "
              f"decoder {chk['decoder_recovered']} ok; DQI Lemma 9.2 {dqi:.1f} ({dqi / inst.m:.4f}), "
              f"semicircle {chk['Q_DQI_semicircle']:.4f}, Q_Pr {chk['Q_Pr']:.4f}, "
              f"regime {chk['regime_factor']:.3f}, Q* {chk['Q_star']:.3f}", flush=True)
        a = attack_prange(inst)
        finish(inst, a, "prange")
        rec["attacks"]["A_prange"] = a
        b = attack_line(inst, T_LINE)
        yb = finish(inst, b, "line")
        rec["attacks"]["B_line"] = b
        c = attack_heat(inst, T_HEAT)
        finish(inst, c, "heat")
        rec["attacks"]["C_heat"] = c
        best_y = yb
        if p <= PLANE_MAX_P:
            dres = attack_plane(inst, yb, T_PLANE)
            best_y = finish(inst, dres, "plane")
            rec["attacks"]["D_plane"] = dres
        if p <= CPSAT_MAX_P:
            e = attack_cpsat(inst, best_y, T_CPSAT)
            if e is not None:
                finish(inst, e, "cpsat")
                rec["attacks"]["E_cpsat"] = e
        best_classical = max(v["best"] for v in rec["attacks"].values())
        rec["best_classical"] = best_classical
        rec["classical_minus_dqi"] = best_classical - dqi
        rec["seconds"] = time.time() - t0
        print("   " + "; ".join(f"{k} {v['best']} ({v['fraction']:.4f})" for k, v in rec["attacks"].items())
              + f" | Prange mean {a['mean']:.4f} vs theory {a['theory_mean']:.4f}"
              + (f" | CP-SAT bound {rec['attacks']['E_cpsat']['bound']}" if "E_cpsat" in rec["attacks"] else "")
              + f" | best - DQI = {best_classical - dqi:+.1f} | {rec['seconds']:.0f}s", flush=True)
        out.append(rec)
    return out


def merge():
    rows = {}
    for path in sorted(glob.glob(OUT % "*")):
        if path.endswith("-quick.json"):
            continue
        with open(path, encoding="utf-8") as fh:
            for rec in json.load(fh)["results"]:
                rows.setdefault(rec["checks"]["p"], []).append(rec)
    print("p      m     DQI(L9.2)  semicirc  Prange    line     heat     plane    cpsat    best-DQI  regime")
    xs, ys = [], []
    for p in sorted(rows):
        recs = rows[p]
        m = recs[0]["checks"]["m"]

        def mean_frac(key):
            vals = [r["attacks"][key]["fraction"] for r in recs if key in r["attacks"]]
            return f"{np.mean(vals):.4f}" if vals else "  -   "
        dqi = np.mean([r["checks"]["dqi_lemma92"] for r in recs]) / m
        gap = np.mean([r["classical_minus_dqi"] for r in recs]) / m
        line = [r["attacks"]["B_line"]["fraction"] for r in recs]
        heat = [r["attacks"]["C_heat"]["fraction"] for r in recs]
        xs.append(math.sqrt(math.log(p) / m))
        ys.append(max(np.mean(line), np.mean(heat)))
        print(f"{p:5d} {m:5d}  {dqi:.4f}    {recs[0]['checks']['Q_DQI_semicircle']:.4f}   "
              f"{mean_frac('A_prange')}   {mean_frac('B_line')}   {mean_frac('C_heat')}   "
              f"{mean_frac('D_plane')}   {mean_frac('E_cpsat')}   {gap:+.4f}   "
              f"{recs[0]['checks']['regime_factor']:.3f}")
    if len(xs) >= 3:
        slope, icpt = np.polyfit(xs, ys, 1)
        print(f"fit best local-search fraction = {icpt:.4f} + {slope:.4f} * sqrt(ln p / m)"
              f"  (m -> infinity: {icpt:.4f}; Prange 0.55, DQI 0.7179)")


def main():
    if "--merge" in sys.argv:
        merge()
        return
    results = []
    for p in SIZES:
        results.extend(run_size(p))
        with open(OUT % TAG, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"sizes": SIZES, "seeds": SEEDS,
                       "budgets_seconds": {"line": T_LINE, "heat": T_HEAT, "plane": T_PLANE, "cpsat": T_CPSAT},
                       "results": results}, fh, indent=2)
    print("wrote", OUT % TAG)


if __name__ == "__main__":
    main()
