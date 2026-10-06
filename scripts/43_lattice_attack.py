"""Script 43 - Row-subset lattice attack on multiplicative polynomial intersection (numpy + fpylll).

The exponent-domain problem "find x with (Bx)_i close to the centres c_i modulo M on many rows"
is a closest-vector problem on an M-ary lattice.  For a random subset R of m' rows and a block size
b, the script reduces the lattice L = {v : v = B_R x mod M} (systematic basis; LLL, then BKZ-b for
b > 2), runs Babai's nearest-plane algorithm towards the centres c_R, recovers x from an invertible
n-subset of R, and scores sum_i cos(2 pi ((Bx)_i - c_i)/M) on ALL m rows.  It repeats over random
subsets (at most 50) within the time budget and keeps the best assignment.

The script needs only numpy and fpylll.  It reads an instance exported by
``dqi_explorer.log_cauchy.export_instance_json`` and writes <instance>.lattice.json next to it.

Usage:
  python scripts/43_lattice_attack.py results/lattice/logcauchy-m600-s3101.json --subset-rows 100,150 --blocks 20 --seconds 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    from fpylll import BKZ, GSO, LLL, IntegerMatrix  # type: ignore
    from fpylll.algorithms.bkz2 import BKZReduction as BKZ2  # type: ignore

    FPYLLL = True
except Exception as exc:  # the library or one of its parts is missing
    FPYLLL = False
    FPYLLL_ERROR = exc


# ---------------------------------------------------------------------------
# arithmetic helpers
# ---------------------------------------------------------------------------
def inverse_mod_p(a: np.ndarray, p: int) -> np.ndarray | None:
    a = np.asarray(a, dtype=np.int64) % p
    n = a.shape[0]
    aug = np.concatenate([a, np.eye(n, dtype=np.int64)], axis=1)
    inv = [0] + [pow(v, p - 2, p) for v in range(1, p)]
    for c in range(n):
        nz = np.nonzero(aug[c:, c])[0]
        if len(nz) == 0:
            return None
        piv = c + int(nz[0])
        if piv != c:
            aug[[c, piv]] = aug[[piv, c]]
        aug[c] = (aug[c] * inv[int(aug[c, c])]) % p
        f = aug[:, c].copy()
        f[c] = 0
        idx = np.nonzero(f)[0]
        if len(idx):
            aug[idx] = (aug[idx] - f[idx, None] * aug[c][None, :]) % p
    return aug[:, n:]


def matmul_mod(a, b, p: int) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64) % p).astype(np.int64)


def cosine_score(y: np.ndarray, centres: np.ndarray, p: int) -> float:
    return float(np.cos(2 * np.pi * ((y - centres) % p) / p).sum())


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        inst = json.load(handle)
    inst["B"] = np.asarray(inst["B"], dtype=np.int64)
    inst["centres"] = np.asarray(inst["centres"], dtype=np.int64)
    return inst


# ---------------------------------------------------------------------------
# row-subset CVP
# ---------------------------------------------------------------------------
def qary_basis(BR: np.ndarray, p: int, rng=None):
    """The q-ary lattice {v : v = B_R x mod p} in systematic form, a basis with no dependent rows.

    With B_T invertible on n rows T, v[T] = u is free and v[rest] = A u mod p for A = B_rest B_T^{-1}.
    The basis is the n vectors (e_j on T, column j of A on rest) and p e_i for i outside T.  Building
    it directly avoids an LLL over n + rows generators, which is slow once n is in the hundreds.
    """

    rows, n = BR.shape
    rng = rng if rng is not None else np.random.default_rng(0)
    inv = None
    for _ in range(50):
        T = np.sort(rng.choice(rows, n, replace=False))
        inv = inverse_mod_p(BR[T], p)
        if inv is not None:
            break
    if inv is None:
        raise ValueError("no invertible n-row subset")
    rest = np.setdiff1d(np.arange(rows), T)
    A = matmul_mod(BR[rest], inv, p)  # (rows - n) x n
    basis = IntegerMatrix(rows, rows)
    for j in range(n):
        basis[j, int(T[j])] = 1
        for k, i in enumerate(rest):
            basis[j, int(i)] = int(A[k, j])
    for k, i in enumerate(rest):
        basis[n + k, int(i)] = int(p)
    return basis


def bkz(reduced, block: int, seconds: float) -> float:
    """LLL through fplll's adaptive-precision wrapper, then BKZ 2.0 for block sizes above two.
    Fixed double precision loops in size reduction once the dimension is in the hundreds."""
    t0 = time.time()
    LLL.reduction(reduced)
    if block > 2:
        # pip wheels of fpylll can ship without fplll's strategy file; plain BKZ (no pruning) then
        strategies = {"strategies": BKZ.DEFAULT_STRATEGY} if os.path.exists(str(BKZ.DEFAULT_STRATEGY)) else {}
        params = BKZ.Param(block_size=min(block, reduced.nrows), max_time=max(1.0, seconds - (time.time() - t0)),
                           flags=BKZ.AUTO_ABORT | BKZ.MAX_TIME, **strategies)
        try:
            M = GSO.Mat(reduced, float_type="dpe")
            M.update_gso()
            BKZ2(LLL.Reduction(M))(params)
        except Exception as exc:  # BKZ 2.0 in dpe can loop in size reduction; retry in mpfr
            # fplll's C++ BKZ.reduction throws an uncatchable C++ exception on the same failure and
            # kills the process, so the fallback stays in BKZ 2.0 with a 120-bit mpfr GSO.
            print("   BKZ2 (dpe) failed, retrying in mpfr:", type(exc).__name__, exc, file=sys.stderr, flush=True)
            try:
                from fpylll import FPLLL
                FPLLL.set_precision(120)
                params = BKZ.Param(block_size=min(block, reduced.nrows), max_time=max(1.0, seconds - (time.time() - t0)),
                                   flags=BKZ.AUTO_ABORT | BKZ.MAX_TIME, **strategies)
                M = GSO.Mat(reduced, float_type="mpfr")
                M.update_gso()
                BKZ2(LLL.Reduction(M))(params)
            except Exception as exc2:  # keep going with the LLL basis
                print("   BKZ2 (mpfr) failed:", type(exc2).__name__, exc2, file=sys.stderr, flush=True)
    return time.time() - t0


def babai_point(reduced, target: list[int]) -> np.ndarray:
    coeffs = None
    for float_type in ("d", "dpe", "mpfr"):
        try:
            M = GSO.Mat(reduced, float_type=float_type)
            M.update_gso()
            coeffs = M.babai(target)
            break
        except Exception:
            continue
    if coeffs is None:
        raise RuntimeError("Babai failed at every precision")
    v = np.zeros(reduced.ncols, dtype=np.int64)
    for r in range(reduced.nrows):
        if coeffs[r]:
            for c in range(reduced.ncols):
                v[c] += int(coeffs[r]) * int(reduced[r, c])
    return v


def attack_subsets(inst: dict, rows_list, blocks, seconds: float, rng) -> list[dict]:
    B, c, p, m, n = inst["B"], inst["centres"], inst["alphabet"], inst["m"], inst["n"]
    out = []
    for rows in rows_list:
        rows = min(rows, m)
        for block in blocks:
            t_end = time.time() + seconds
            best_all, best_sub, subsets, bkz_time = None, None, 0, 0.0
            while time.time() < t_end and subsets < 50:
                R = np.sort(rng.choice(m, rows, replace=False))
                BR = B[R]
                reduced = qary_basis(BR, p, rng)
                bkz_time += bkz(reduced, block, t_end - time.time())
                v = babai_point(reduced, [int(x) for x in c[R]]) % p
                x = None
                for _ in range(10):
                    T = np.sort(rng.choice(rows, n, replace=False))
                    inv = inverse_mod_p(BR[T], p)
                    if inv is not None:
                        x = matmul_mod(inv, v[T][:, None], p)[:, 0]
                        break
                if x is None:
                    continue
                y = matmul_mod(B, x[:, None], p)[:, 0]
                s_all = cosine_score(y, c, p) / m
                s_sub = cosine_score(y[R], c[R], p) / rows
                subsets += 1
                if best_all is None or s_all > best_all:
                    best_all, best_sub = s_all, s_sub
            rec = {"attack": "A_subset_cvp", "rows": rows, "block": block, "subsets": subsets, "bkz_seconds": bkz_time, "best_all_rows": best_all, "subset_score_at_best": best_sub}
            print("   A rows=%d block=%d: subsets %d, best mean cosine on all rows %s (on the subset %s), BKZ %.0fs" % (rows, block, subsets, "%.4f" % best_all if best_all is not None else "-", "%.4f" % best_sub if best_sub is not None else "-", bkz_time), flush=True)
            out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instance")
    parser.add_argument("--subset-rows", default="200,300,400,500,600")
    parser.add_argument("--blocks", default="20,30", help="block sizes; 2 means LLL alone")
    parser.add_argument("--seconds", type=float, default=120.0, help="time budget per (subset size, block size) pair")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if not FPYLLL:
        raise SystemExit(f"fpylll could not be imported ({FPYLLL_ERROR}); see README.md, Quick start")
    inst = load(args.instance)
    print("instance %s: m=%d n=%d alphabet=%d alpha=%.4f" % (inst["kind"], inst["m"], inst["n"], inst["alphabet"], inst["alpha"]))
    results = {"instance": os.path.basename(args.instance), "fpylll": FPYLLL, "records": []}
    rng = np.random.default_rng(args.seed)
    blocks = [int(b) for b in args.blocks.split(",")]
    results["records"] += attack_subsets(inst, [int(r) for r in args.subset_rows.split(",")], blocks, args.seconds, rng)
    out = os.path.splitext(args.instance)[0] + ".lattice.json"
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
