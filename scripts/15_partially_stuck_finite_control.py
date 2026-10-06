"""Finite control: maximum masking of partially stuck-at memory cells.

Problem: Wachter-Zeh & Yaakobi's masking construction (arXiv 1505.03281)
stores y = w + zH over GF(256); a symbol whose cell is partially stuck at level s is masked when
that cell's level in (w + zH)_i is >= s.  Choosing z to mask as many stuck symbols as possible is
max-LINSAT with B = H_u^T (parity-check columns at the stuck positions), |F_i| = 128, and decoder
code ker(H_u) = GRS[u, u-r, r+1].

Each symbol is two 16-level cells.  Two identifications of bytes with cell pairs are compared:
  (A) nibble: the stuck cell is the high nibble of the byte;
  (B) random bijection: a fixed random permutation of bytes, then the high nibble.

Compared against DQI's modelled finite-size score (Jordan et al., Lemma 9.2):
  a. GF(256) information-set decoding with restarts,
  b. GF(2) affine-subspace information-set decoding (uses subfield structure of the accepted sets),
  c. simulated annealing over z (from random starts and from the best solution of b),
  d. Guruswami-Sudan list-recovery vacuity (computed, not run).
Checks: MDS rank of H_u, a Reed-Solomon codeword check, fidelity by simulating stuck cells,
Prange mean against theory, Lemma 9.2 against the semicircle law, independent re-scoring.

Usage: python scripts/15_partially_stuck_finite_control.py [--quick]
Writes results/partially-stuck-finite-control-run.json.
"""
import json
import math
import sys
import time

import numpy as np

QUICK = "--quick" in sys.argv
Q, N_CODE = 256, 255
U, R, S_LEVEL = 240, 28, 8
SEEDS = [1701, 1702, 1703]
PERM_SEED = 90210
PRANGE_BATCHES, PRANGE_T = (4, 5000) if QUICK else (25, 20000)
AFFINE_RESTARTS = 40 if QUICK else 600
SA_STEPS = 5000 if QUICK else 150000
POOL_TRIES = 400 if QUICK else 4000

# ---------------------------------------------------------------------------------------------
# GF(256), AES polynomial 0x11B, generator 3
EXP = [0] * 510
LOG = [0] * 256
_x = 1
for _i in range(255):
    EXP[_i] = _x
    LOG[_x] = _i
    _x2 = (_x << 1) ^ _x
    if _x2 & 0x100:
        _x2 ^= 0x11B
    _x = _x2
for _i in range(255, 510):
    EXP[_i] = EXP[_i - 255]
assert len(set(EXP[:255])) == 255, "3 is not primitive"
MUL = [[0 if (a == 0 or b == 0) else EXP[LOG[a] + LOG[b]] for b in range(256)] for a in range(256)]
MT = np.array(MUL, dtype=np.uint8)
INV = [0] + [EXP[255 - LOG[a]] for a in range(1, 256)]


def gf_rank(rows):
    m = [list(r) for r in rows]
    rank, cols = 0, len(m[0])
    for c in range(cols):
        piv = next((i for i in range(rank, len(m)) if m[i][c]), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        inv = INV[m[rank][c]]
        m[rank] = [MUL[inv][v] for v in m[rank]]
        for i in range(len(m)):
            if i != rank and m[i][c]:
                f = m[i][c]
                m[i] = [a ^ MUL[f][b] for a, b in zip(m[i], m[rank])]
        rank += 1
    return rank


def gf_inverse(mat):
    n = len(mat)
    m = [list(mat[i]) + [1 if i == j else 0 for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = next(i for i in range(c, n) if m[i][c])
        m[c], m[piv] = m[piv], m[c]
        inv = INV[m[c][c]]
        m[c] = [MUL[inv][v] for v in m[c]]
        for i in range(n):
            if i != c and m[i][c]:
                f = m[i][c]
                m[i] = [a ^ MUL[f][b] for a, b in zip(m[i], m[c])]
    return [row[n:] for row in m]


def gf_matmul(a, b):
    """a: r x r (lists), b: r x u (numpy uint8) -> r x u numpy."""
    out = np.zeros((len(a), b.shape[1]), dtype=np.uint8)
    for i, row in enumerate(a):
        acc = np.zeros(b.shape[1], dtype=np.uint8)
        for k, coef in enumerate(row):
            if coef:
                acc ^= MT[coef, b[k]]
        out[i] = acc
    return out


def row_times(z, mat):
    """z (length r) times mat (r x u), vectorized."""
    acc = np.zeros(mat.shape[1], dtype=np.uint8)
    for j, zj in enumerate(z):
        if zj:
            acc ^= MT[zj, mat[j]]
    return acc


# Reed-Solomon parity-check matrix: H[i][j] = a_j^i with a_j = 3^j
H = np.array([[EXP[(i * j) % 255] for j in range(N_CODE)] for i in range(R)], dtype=np.uint8)


# ---------------------------------------------------------------------------------------------
# DQI model (Jordan et al. Lemma 9.2, balanced sets so the diagonal term d = 0)
def dqi_expected(m, ell, q=Q, rsize=Q // 2):
    a = np.zeros((ell + 1, ell + 1))
    for k in range(1, ell + 1):
        a[k, k - 1] = a[k - 1, k] = math.sqrt(k * (m - k + 1))
    d = (q - 2 * rsize) / math.sqrt(rsize * (q - rsize))
    for k in range(ell + 1):
        a[k, k] = k * d
    lam_max = float(np.linalg.eigvalsh(a).max())
    return m * rsize / q + math.sqrt(rsize * (q - rsize)) / q * lam_max


def dqi_fraction_large_m(m, ell):
    """Lemma 9.2 top eigenvalue (balanced case, zero diagonal) by Sturm-sequence bisection."""
    off = [math.sqrt(k * (m - k + 1)) for k in range(1, ell + 1)]

    def n_greater(x):
        neg, d = 0, -x
        neg += d < 0
        for a in off:
            d = -x - a * a / (d if d != 0 else 1e-300)
            neg += d < 0
        return (ell + 1) - neg

    lo, hi = 0.0, 2 * max(off) + 1
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if n_greater(mid) >= 1 else (lo, mid)
    return 0.5 + lo / (2 * m)


def semicircle(lam, rho):
    return (math.sqrt(lam * (1 - rho)) + math.sqrt(rho * (1 - lam))) ** 2


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


# ---------------------------------------------------------------------------------------------
# Identifications and accepted set A = {bytes whose stuck cell level is >= s}
def identification(kind):
    if kind == "A_nibble":
        fwd = np.arange(256)
    else:
        fwd = np.random.default_rng(PERM_SEED).permutation(256)
    inv = np.argsort(fwd)
    level = fwd >> 4
    return fwd, inv, level >= S_LEVEL


def affine_pool(accept, rng, tries):
    """Randomized greedy growth of affine subspaces a0 + span(basis) inside the accepted set."""
    members = [v for v in range(256) if accept[v]]
    found = {}
    for _ in range(tries):
        a0 = members[rng.integers(len(members))]
        span, basis = {0}, []
        for d in rng.permutation(np.arange(1, 256)):
            d = int(d)
            if d in span:
                continue
            new = {x ^ d for x in span}
            if all(accept[a0 ^ x] for x in new):
                span |= new
                basis.append(d)
        key = frozenset(a0 ^ x for x in span)
        found[key] = (a0, basis)
    best_dim = max(len(b) for _, b in found.values())
    pool = [(a0, b) for a0, b in found.values() if len(b) == best_dim]
    return best_dim, pool


def annihilator(basis):
    sols = [phi for phi in range(1, 256) if all(bin(phi & s).count("1") % 2 == 0 for s in basis)]
    xb = {}
    for phi in sols:
        v = phi
        while v:
            p = v.bit_length() - 1
            if p in xb:
                v ^= xb[p]
            else:
                xb[p] = v
                break
    return list(xb.values())


# ---------------------------------------------------------------------------------------------
def gf2_rank(rows):
    basis = {}
    for v in rows:
        while v:
            p = v.bit_length() - 1
            if p in basis:
                v ^= basis[p]
            else:
                basis[p] = v
                break
    return len(basis)


def run_instance(kind, seed):
    rng = np.random.default_rng(seed)
    fwd, inv_perm, accept = identification("A_nibble" if kind.startswith("C_") else kind)
    positions = np.sort(rng.choice(N_CODE, U, replace=False))
    hu = H[:, positions]
    if kind.startswith("C_"):
        # generalized Reed-Solomon: random nonzero column multipliers (same MDS code family)
        mult = rng.integers(1, 256, U)
        hu = np.array([[MUL[int(hu[i, k])][int(mult[k])] for k in range(U)] for i in range(R)], dtype=np.uint8)
    w = rng.integers(0, 256, U).astype(np.uint8)
    accept_np = accept.astype(bool)

    def score(z):
        return int(accept_np[w ^ row_times(z, hu)].sum())

    def score_independent(z):
        # pure-Python path: y_k = sum_j z_j * Hu[j,k]
        total = 0
        hul = hu.tolist()
        wl = w.tolist()
        for k in range(U):
            y = 0
            for j in range(R):
                y ^= MUL[int(z[j])][hul[j][k]]
            total += 1 if accept[wl[k] ^ y] else 0
        return total

    checks = {}
    # MDS: random r-subsets of columns of H_u have full rank
    ranks = [gf_rank(hu[:, np.sort(rng.choice(U, R, replace=False))].tolist()) for _ in range(20)]
    checks["mds_rank_20_subsets"] = all(x == R for x in ranks)

    # --- a. GF(256) information-set decoding
    t0 = time.time()
    accepted_vals = np.array([v for v in range(256) if accept[v]], dtype=np.uint8)
    best_a, best_a_z, total, count = -1, None, 0.0, 0
    for _ in range(PRANGE_BATCHES):
        info = rng.choice(U, R, replace=False)
        hi_inv = gf_inverse(hu[:, info].tolist())
        g = gf_matmul(hi_inv, hu)
        t = w[info][None, :] ^ accepted_vals[rng.integers(0, len(accepted_vals), (PRANGE_T, R))]
        y = np.zeros((PRANGE_T, U), dtype=np.uint8)
        for a in range(R):
            y ^= MT[t[:, a][:, None], g[a][None, :]]
        sc = accept_np[w[None, :] ^ y].sum(axis=1)
        total += float(sc.sum())
        count += PRANGE_T
        i = int(sc.argmax())
        if sc[i] > best_a:
            best_a = int(sc[i])
            # z = t * H_I^{-1}
            best_a_z = np.zeros(R, dtype=np.uint8)
            for a in range(R):
                if t[i, a]:
                    best_a_z ^= MT[t[i, a], np.array(hi_inv[a], dtype=np.uint8)]
    prange = {"samples": count, "mean_fraction": total / count / U, "best": best_a,
              "best_rescored": score(best_a_z), "seconds": round(time.time() - t0, 1)}

    # --- b. GF(2) affine-subspace information-set decoding
    t0 = time.time()
    dim, pool = affine_pool(accept, rng, POOL_TRIES)
    pool_eqs = [(a0, annihilator(basis)) for a0, basis in pool]
    rowmask = [[0] * 8 for _ in range(256)]
    for c in range(256):
        for b in range(8):
            rowmask[c][b] = sum(((MUL[1 << tt][c] >> b) & 1) << tt for tt in range(8))
    hul = hu.tolist()
    func = [[sum(rowmask[hul[j][i]][b] << (8 * j) for j in range(R)) for b in range(8)] for i in range(U)]
    wl = w.tolist()
    nbits = 8 * R
    best_b, best_b_z, sat_counts = -1, None, []
    for _ in range(AFFINE_RESTARTS):
        basis = {}
        n_forced = 0
        for i in rng.permutation(U):
            i = int(i)
            a0, phis = pool_eqs[rng.integers(len(pool_eqs))]
            target = wl[i] ^ a0
            added, ok = [], True
            for phi in phis:
                row = 0
                for b in range(8):
                    if (phi >> b) & 1:
                        row ^= func[i][b]
                rhs = bin(phi & target).count("1") % 2
                while row:
                    p = row.bit_length() - 1
                    if p in basis:
                        brow, brhs = basis[p]
                        row ^= brow
                        rhs ^= brhs
                    else:
                        basis[p] = (row, rhs)
                        added.append(p)
                        break
                if row == 0 and rhs == 1:
                    ok = False
                    break
            if ok:
                n_forced += 1
            else:
                for p in added:
                    del basis[p]
        x = int(rng.integers(0, 2 ** 62)) | (int(rng.integers(0, 2 ** 62)) << 62) \
            | (int(rng.integers(0, 2 ** 62)) << 124) | (int(rng.integers(0, 2 ** 62)) << 186)
        x &= (1 << nbits) - 1
        for p in sorted(basis):
            row, rhs = basis[p]
            rest = row & ~(1 << p)
            val = rhs ^ (bin(rest & x).count("1") % 2)
            x = (x | (1 << p)) if val else (x & ~(1 << p))
        z = np.array([(x >> (8 * j)) & 255 for j in range(R)], dtype=np.uint8)
        sc = score(z)
        sat_counts.append(n_forced)
        if sc > best_b:
            best_b, best_b_z = sc, z
    bit_ranks = [gf2_rank([func[i][b] for i in range(U)]) for b in range(8)]
    affine = {"max_subspace_dim": dim, "pool_size": len(pool), "restarts": AFFINE_RESTARTS,
              "gf2_rank_of_single_bit_functionals": bit_ranks,
              "mean_forced_constraints": float(np.mean(sat_counts)), "best": best_b,
              "best_rescored": score(best_b_z), "seconds": round(time.time() - t0, 1)}

    # --- c. simulated annealing
    def anneal(z0):
        z = z0.copy()
        y = row_times(z, hu)
        cur = int(accept_np[w ^ y].sum())
        best, best_z = cur, z.copy()
        for step in range(SA_STEPS):
            temp = 2.0 * (1 - step / SA_STEPS) + 0.05
            j = int(rng.integers(R))
            delta = int(rng.integers(1, 256))
            ny = y ^ MT[delta, hu[j]]
            new = int(accept_np[w ^ ny].sum())
            if new >= cur or rng.random() < math.exp((new - cur) / temp):
                y, cur = ny, new
                z[j] ^= delta
                if cur > best:
                    best, best_z = cur, z.copy()
        return best, best_z

    t0 = time.time()
    sa_rand, sa_rand_z = anneal(rng.integers(0, 256, R).astype(np.uint8))
    sa_seed, sa_seed_z = anneal(best_b_z)
    sa = {"steps": SA_STEPS, "best_random_start": sa_rand, "best_random_start_rescored": score(sa_rand_z),
          "best_from_affine": sa_seed, "best_from_affine_rescored": score(sa_seed_z),
          "seconds": round(time.time() - t0, 1)}

    # --- checks on the best solution overall
    cands = [(best_a, best_a_z), (best_b, best_b_z), (sa_rand, sa_rand_z), (sa_seed, sa_seed_z)]
    best_score, best_z = max(cands, key=lambda c: c[0])
    checks["independent_rescore_matches"] = score_independent(best_z) == best_score
    stored = w ^ row_times(best_z, hu)                      # symbol bytes at stuck positions
    code = fwd[stored]                                      # cell pair (c1 << 4) | c2
    c1, c2 = code >> 4, code & 15
    written = inv_perm[(np.maximum(c1, S_LEVEL) << 4) | c2]  # stuck cell cannot go below s
    corrupted = written != stored
    checks["fidelity_corrupted_equals_unmasked"] = bool(np.array_equal(corrupted, ~accept_np[stored]))
    return {"identification": kind, "seed": seed, "prange_gf256": prange, "affine_gf2": affine,
            "annealing": sa, "best_overall": best_score, "checks": checks}


def main():
    alpha, rho = R / U, 0.5
    ell = max(l for l in range(R + 1) if 2 * l + 1 < R + 1)
    dqi_count = dqi_expected(U, ell)
    lam = ell / U
    model = {
        "u_stuck_symbols": U, "r_redundancy": R, "alpha": alpha, "rho": rho, "ell": ell,
        "lambda": lam, "dqi_lemma92_count": dqi_count, "dqi_lemma92_fraction": dqi_count / U,
        "dqi_semicircle_fraction": semicircle(lam, rho), "prange_theory_fraction": rho + (1 - rho) * alpha,
        "regime_factor": Q ** alpha * rho, "qstar": qstar(alpha, Q, rho),
        "wzy_masking_bound_alpha": math.log(1 / rho) / math.log(Q),
        "gs_list_recovery_needed_agreement": math.sqrt(128 * U * (R - 1)),
        "bit_level_affine_hyperplane_estimate": rho + (1 - rho) * min(1.0, 8 * alpha),
    }
    gaps = []
    for big_m in (1000, 10000, 100000, 1000000):
        big_ell = int(round(lam * big_m))
        gaps.append(dqi_fraction_large_m(big_m, big_ell) - semicircle(big_ell / big_m, rho))
    model["lemma92_minus_semicircle_m1e3_to_1e6"] = gaps
    model_checks = {
        "lemma92_bisection_matches_numpy_at_m240": abs(dqi_fraction_large_m(U, ell) - dqi_count / U) < 1e-9,
        "lemma92_converges_to_semicircle": all(abs(a) > abs(b) for a, b in zip(gaps, gaps[1:]))
                                           and abs(gaps[-1]) < 5e-4,
        "licensing_2ell_plus_1_lt_d": 2 * ell + 1 < R + 1,
        "unsat_regime": Q ** alpha * rho < 1,
        "gs_list_recovery_vacuous": math.sqrt(128 * U * (R - 1)) > U,
    }
    mu = model["dqi_lemma92_fraction"] - model["prange_theory_fraction"]
    n_restarts = PRANGE_BATCHES * PRANGE_T
    model["restart_threshold_m"] = (1 - alpha) * math.log(n_restarts / 0.05) / (2 * mu * mu)
    model["restart_threshold_N"] = n_restarts

    results = []
    for kind in ("A_nibble", "B_random_bijection", "C_nibble_grs_multipliers"):
        for seed in SEEDS:
            res = run_instance(kind, seed)
            res["checks"]["prange_mean_near_theory"] = abs(res["prange_gf256"]["mean_fraction"]
                                                           - model["prange_theory_fraction"]) < 0.01
            results.append(res)
            print(f"{kind:20s} seed {seed}: prange best {res['prange_gf256']['best']}, "
                  f"affine(dim {res['affine_gf2']['max_subspace_dim']}) best {res['affine_gf2']['best']}, "
                  f"SA {res['annealing']['best_random_start']}/{res['annealing']['best_from_affine']}, "
                  f"checks {res['checks']}", flush=True)

    print("\nDQI model: %.2f of %d (%.4f); semicircle %.4f; Prange theory %.4f; Q* %.4f; regime %.3f"
          % (dqi_count, U, dqi_count / U, model["dqi_semicircle_fraction"], model["prange_theory_fraction"],
             model["qstar"], model["regime_factor"]))
    print("restart threshold at N=%d: m >= %.0f (u=%d)" % (n_restarts, model["restart_threshold_m"], U))
    print("model checks:", model_checks)
    out = {"quick": QUICK,
           "model": model, "model_checks": model_checks, "results": results}
    if not QUICK:
        with open("results/partially-stuck-finite-control-run.json", "w", encoding="utf-8",
                  newline="\n") as fh:
            json.dump(out, fh, indent=2)
        print("wrote results/partially-stuck-finite-control-run.json")


if __name__ == "__main__":
    main()
