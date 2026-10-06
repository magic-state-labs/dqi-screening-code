"""Screening steps shared by scripts 41, 46, 48, 50 and 53: rule checks, decoder check, DQI value, classical solvers.

``screen_one`` takes an instance, or a callable ``build(m, seed)`` returning one: any object
exposing ``p`` (a prime), ``B_pub`` (``m x n`` int64 over
``F_p``), ``ell`` (the licensed radius), ``decode(word) -> pattern | None`` (public order), and
either ``accept_sets`` (set-membership objective) or ``centres`` (smooth cosine objective
``cos(2 pi (y_i - c_i)/p)``).  Adapters cover the alternant, hidden-key and log-Cauchy
instances of this package.

Steps (each returns a plain dict, all recorded in the output file):
  gates            D0 (sphere packing, restricted for the cosine), D2 (reach against rate),
                   D3 (density; sets only), D6 (void on a prime alphabet), the asymptotic law
  decoder_check    exact recovery of random patterns of weight <= ell (values on the objective's
                   error alphabet), with kernel codewords added when m <= 1200; beyond-radius
                   patterns classified
  distinguisher    Schur-square battery of hidden_keys plus a random-matrix control (m <= 2000)
  dqi_value        finite Lemma 9.2 value (canonical for sets, general-objective for the cosine),
                   the Prange baseline, the margin, and for sets the agreement law's spread
  classical_portfolio  A information-set restarts, B pivoting line search, C heat bath, on the
                   public matrix only, every witness re-scored independently (the harness's
                   arithmetic for sets, an FFT line for the cosine)
  converse         the restart-converse length threshold at N = 2^40, eta = 0.05
  verdict          a size survives iff the best classical score is below
                   the DQI value on every seed

The solver class is the one of scripts/21_finite_control_harness.py with scoring routed through
an objective object; for the set objective the order of random-number calls is the same.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
import os
import time
from typing import Callable

import numpy as np

from .alternant_trapdoor import kernel_basis_mod_p, matmul_mod_p
from .discovery import (
    canonical_beats_prange_balanced,
    cosine_baseline,
    cosine_beats_baseline,
    cosine_quality,
    existential_cosine,
    licensed_radius_is_realizable,
    prange_quality,
    qary_entropy,
    restricted_qary_entropy,
    restricted_radius_is_realizable,
)
from .hidden_keys import random_matrix_battery, square_battery
from .spectral import canonical_asymptotic_quality, canonical_finite_quality, general_objective_finite_quality

RESTARTS = 2.0**40
FAILURE = 0.05


# ---------------------------------------------------------------------------
# prime-field arithmetic (the harness's PrimeField)
# ---------------------------------------------------------------------------
class PrimeField:
    def __init__(self, p: int) -> None:
        self.p = self.q = int(p)
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


def gauss_inverse(field: PrimeField, a) -> np.ndarray | None:
    n = a.shape[0]
    aug = np.concatenate([np.asarray(a, dtype=np.int64), np.eye(n, dtype=np.int64)], axis=1)
    for c in range(n):
        nz = np.nonzero(aug[c:, c])[0]
        if len(nz) == 0:
            return None
        piv = c + nz[0]
        if piv != c:
            aug[[c, piv]] = aug[[piv, c]]
        aug[c] = field.mul(aug[c], field.inv(aug[c, c]))
        f = aug[:, c].copy()
        f[c] = 0
        rows = np.nonzero(f)[0]
        if len(rows):
            aug[rows] = field.sub(aug[rows], field.mul(f[rows, None], aug[c][None, :]))
    return aug[:, n:]


def rank_mod(field: PrimeField, a) -> int:
    a = np.asarray(a, dtype=np.int64).copy()
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        nz = np.nonzero(a[r:, c])[0]
        if len(nz) == 0:
            continue
        piv = r + nz[0]
        a[[r, piv]] = a[[piv, r]]
        a[r] = field.mul(a[r], field.inv(a[r, c]))
        f = a[:, c].copy()
        f[r] = 0
        idx = np.nonzero(f)[0]
        if len(idx):
            a[idx] = field.sub(a[idx], field.mul(f[idx, None], a[r][None, :]))
        r += 1
        if r == rows:
            break
    return r


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------
class SetObjective:
    """Per-row accepted sets; the harness's boolean-table arithmetic."""

    kind = "sets"
    is_integer = True
    value_range = 1.0

    def __init__(self, sets, p: int) -> None:
        self.p = p
        self.m = len(sets)
        self.sets = [set(map(int, s)) for s in sets]
        self.r = len(sets[0])
        self.A = np.array([sorted(s) for s in sets], dtype=np.int64)
        self.M = np.zeros((self.m, p), dtype=bool)
        self.M[np.arange(self.m)[:, None], self.A] = True
        self.ar = np.arange(self.m)

    @property
    def rho(self) -> float:
        return self.r / self.p

    def score(self, y):
        return int(self.M[self.ar, y].sum())

    def batch_scores(self, Y):
        return self.M[self.ar[:, None], Y].sum(axis=0)

    def satisfied(self, y):
        return self.M[self.ar, y]

    def prange_targets(self, S, rng, k):
        return self.A[S[:, None], rng.integers(0, self.r, (len(S), k))]

    def start_target(self, S, rng):
        return self.A[S, rng.integers(0, self.r, len(S))]

    def line_scores(self, y, u, field):
        free = np.nonzero(u)[0]
        uinv = field.inv(u[free])
        tv = field.mul(field.sub(self.A[free], y[free, None]), uinv[:, None])
        counts = np.bincount(tv.ravel(), minlength=self.p)
        fixed = np.setdiff1d(self.ar, free)
        return counts + int(self.M[fixed, y[fixed]].sum())

    def verify(self, y):
        return sum(int(v) in s for v, s in zip(y.tolist(), self.sets))

    def baseline(self, alpha):
        return prange_quality(alpha, self.rho)

    def better(self, a, b):
        return a > b

    def close(self, a, b):
        return a == b


class CosineObjective:
    """``sum_i cos(2 pi (y_i - c_i)/p)`` with per-row centres; the line search uses one FFT per move."""

    kind = "cos"
    is_integer = False
    value_range = 2.0

    def __init__(self, centres, p: int) -> None:
        self.p = p
        self.centres = np.asarray(centres, dtype=np.int64) % p
        self.m = len(self.centres)
        self.ar = np.arange(self.m)
        self.omega = np.exp(2j * np.pi * np.arange(p) / p)
        self.cos_table = np.cos(2 * np.pi * np.arange(p) / p)

    @property
    def rho(self):
        return None

    def score(self, y):
        return float(self.cos_table[(np.asarray(y, dtype=np.int64) - self.centres) % self.p].sum())

    def batch_scores(self, Y):
        return self.cos_table[(np.asarray(Y, dtype=np.int64) - self.centres[:, None]) % self.p].sum(axis=0)

    def satisfied(self, y):
        return self.cos_table[(np.asarray(y, dtype=np.int64) - self.centres) % self.p] >= 0.5

    def prange_targets(self, S, rng, k):
        return np.repeat(self.centres[S][:, None], k, axis=1)

    def start_target(self, S, rng):
        return self.centres[S]

    def line_scores(self, y, u, field):
        free = np.nonzero(u)[0]
        phase = np.exp(2j * np.pi * ((y[free] - self.centres[free]) % self.p) / self.p)
        a = np.bincount(u[free], weights=phase.real, minlength=self.p) + 1j * np.bincount(u[free], weights=phase.imag, minlength=self.p)
        line = np.real(self.p * np.fft.ifft(a))
        fixed = np.setdiff1d(self.ar, free)
        return line + float(self.cos_table[(y[fixed] - self.centres[fixed]) % self.p].sum())

    def verify(self, y):
        return float(sum(math.cos(2 * math.pi * ((int(v) - int(c)) % self.p) / self.p) for v, c in zip(y.tolist(), self.centres.tolist())))

    def baseline(self, alpha):
        return cosine_baseline(alpha)

    def better(self, a, b):
        return a > b + 1e-9

    def close(self, a, b):
        return abs(a - b) <= 1e-6 * (1 + abs(a) + abs(b))


# ---------------------------------------------------------------------------
# the attack battery (copy of the harness Control with objective-routed scoring)
# ---------------------------------------------------------------------------
class Control:
    def __init__(self, field: PrimeField, B, objective, rng) -> None:
        self.f, self.B, self.obj, self.rng = field, np.asarray(B, dtype=np.int64), objective, rng
        self.m, self.n = self.B.shape
        self.q = field.q
        self.ar = np.arange(self.m)

    def score(self, y):
        return self.obj.score(y)

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
        return self.obj.verify(y)

    def systematic(self):
        while True:
            S = self.rng.choice(self.m, self.n, replace=False)
            inv = gauss_inverse(self.f, self.B[S])
            if inv is not None:
                return S, self.f.matmul(self.B, inv)

    # A: information-set restarts
    def prange(self, restarts):
        best, best_y, scores = None, None, []
        if self.obj.kind == "cos":
            sets = max(4, restarts // 64)
            for _ in range(sets):
                S, L = self.systematic()
                y = self.f.matmul(L, self.obj.start_target(S, self.rng)[:, None])[:, 0]
                s = self.obj.score(y)
                scores.append(s)
                if best is None or self.obj.better(s, best):
                    best, best_y = s, y.copy()
            return {"best": best, "restarts": len(scores), "mean": float(np.mean(scores)) / self.m, "mode": "centre-solve"}, best_y
        per = max(1, restarts // 4)
        best = -1
        for _ in range(4):
            S, L = self.systematic()
            for start in range(0, per, 64):
                k = min(64, per - start)
                T = self.obj.prange_targets(S, self.rng, k)
                Y = self.f.matmul(L, T)
                sc = self.obj.batch_scores(Y)
                scores.extend(sc.tolist())
                j = int(sc.argmax())
                if sc[j] > best:
                    best, best_y = int(sc[j]), Y[:, j].copy()
        return {"best": best, "restarts": len(scores), "mean": float(np.mean(scores)) / self.m}, best_y

    def _pivot(self, S, L, k, sat_mask):
        cand = np.nonzero(sat_mask & (L[:, k] != 0))[0]
        cand = np.setdiff1d(cand, S)
        if len(cand) == 0:
            return S, L
        j = int(self.rng.choice(cand))
        piv_inv = self.f.inv(L[j, k])
        rowj = self.f.mul(L[j, :], piv_inv)
        colk = L[:, k].copy()
        Lnew = self.f.sub(L, self.f.mul(colk[:, None], rowj[None, :]))
        Lnew[:, k] = self.f.mul(colk, piv_inv)
        S = S.copy()
        S[k] = j
        return S, Lnew

    def _start(self):
        S, L = self.systematic()
        t = self.obj.start_target(S, self.rng)
        return S, L, self.f.matmul(L, t[:, None])[:, 0]

    # B: pivoting line search
    def line(self, seconds, patience=60):
        t_end = time.time() + seconds
        best, best_y, moves, restarts = None, None, 0, 0
        while time.time() < t_end:
            S, L, y = self._start()
            cur, stale = self.obj.score(y), 0
            restarts += 1
            while stale < patience and time.time() < t_end:
                k = int(self.rng.integers(self.n))
                counts = self.obj.line_scores(y, L[:, k], self.f)
                assert self.obj.close(counts[0], cur), (counts[0], cur)
                top = counts.max()
                moves += 1
                if self.obj.better(top, cur):
                    choices = np.nonzero(counts >= top - (0 if self.obj.is_integer else 1e-9))[0]
                    t = int(self.rng.choice(choices))
                    y = self.f.add(y, self.f.mul(t, L[:, k]))
                    cur, stale = (int(top) if self.obj.is_integer else float(counts[t])), 0
                else:
                    stale += 1
                S, L = self._pivot(S, L, k, self.obj.satisfied(y))
            if best is None or self.obj.better(cur, best):
                best, best_y = cur, y.copy()
        return {"best": best, "moves": moves, "restarts": restarts}, best_y

    # C: heat bath
    def heat(self, seconds, beta0=0.05, beta1=4.0):
        t0 = time.time()
        S, L, y = self._start()
        cur = self.obj.score(y)
        best, best_y, moves = cur, y.copy(), 0
        while True:
            frac = (time.time() - t0) / seconds
            if frac >= 1:
                break
            beta = beta0 * (beta1 / beta0) ** frac
            k = int(self.rng.integers(self.n))
            counts = self.obj.line_scores(y, L[:, k], self.f)
            w = np.exp(beta * (counts - counts.max()))
            t = int(self.rng.choice(self.q, p=w / w.sum()))
            y = self.f.add(y, self.f.mul(t, L[:, k]))
            cur = int(counts[t]) if self.obj.is_integer else float(counts[t])
            moves += 1
            if self.obj.better(cur, best):
                best, best_y = cur, y.copy()
            S, L = self._pivot(S, L, k, self.obj.satisfied(y))
        return {"best": best, "moves": moves}, best_y


# ---------------------------------------------------------------------------
# instance protocol
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Adapted:
    p: int
    m: int
    n: int
    ell: int
    B: np.ndarray
    objective_kind: str
    sets: list | None
    centres: np.ndarray | None
    decode: Callable
    kind: str
    reach_law: str
    designed_distance: int | None
    info: dict
    source: object

    @property
    def alpha(self) -> float:
        return self.n / self.m

    def make_objective(self):
        if self.objective_kind == "sets":
            return SetObjective(self.sets, self.p)
        return CosineObjective(self.centres, self.p)


def adapt(instance, *, objective: str = "auto", seed: int = 0) -> Adapted:
    """Normalize an instance of this package, or any duck-typed object, to the protocol."""

    from .alternant_trapdoor import AlternantTrapdoorInstance, decode as alternant_decode

    B = np.asarray(instance.B_pub, dtype=np.int64)
    m, n = B.shape
    p = int(instance.p)
    if isinstance(instance, AlternantTrapdoorInstance):
        decode = lambda w, _i=instance: (lambda r: None if r is None else np.asarray(r, dtype=np.int64))(alternant_decode(_i, w, public_order=True))
    else:
        decode = lambda w, _i=instance: (lambda r: None if r is None else np.asarray(r, dtype=np.int64))(_i.decode(w))
    has_sets = hasattr(instance, "accept_sets")
    has_centres = hasattr(instance, "centres")
    if objective == "auto":
        objective = "sets" if has_sets else "cos"
    sets = centres = None
    if objective == "sets":
        if not has_sets:
            raise ValueError("the instance has no accept_sets; use objective='cos'")
        sets = [np.array(sorted(s), dtype=np.int64) for s in instance.accept_sets]
    elif objective == "cos":
        centres = np.asarray(instance.centres, dtype=np.int64) if has_centres else np.random.default_rng(seed + 11).integers(0, p, m)
    else:
        raise ValueError("objective must be 'sets', 'cos' or 'auto'")
    return Adapted(
        p=p,
        m=m,
        n=n,
        ell=int(instance.ell),
        B=B,
        objective_kind=objective,
        sets=sets,
        centres=centres,
        decode=decode,
        kind=str(getattr(instance, "kind", type(instance).__name__)),
        reach_law=str(getattr(instance, "reach_law", "declared by the generator")),
        designed_distance=getattr(instance, "designed_distance", None),
        info={k: getattr(instance, k) for k in ("s", "r", "w", "u", "h") if hasattr(instance, k) and isinstance(getattr(instance, k), (int, float))},
        source=instance,
    )


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
def gates(alpha: float, lam: float, rho: float | None, p: int, *, objective: str = "sets") -> dict:
    out: dict = {"alpha": alpha, "lambda": lam, "p": p, "objective": objective}
    if objective == "sets":
        out["D0"] = {"pass": bool(licensed_radius_is_realizable(alpha, lam, p)), "H_q(lambda)": qary_entropy(lam, p)}
        out["D2"] = {"pass": bool(canonical_beats_prange_balanced(alpha, lam)), "rule": "lambda(1-lambda) > alpha^2/4"}
        q_dqi = canonical_asymptotic_quality(lam, rho)
        q_pr = prange_quality(alpha, rho)
        out["D3"] = {"pass": bool(q_dqi > q_pr), "Q_DQI": q_dqi, "Q_Pr": q_pr, "rho": rho}
        out["margin"] = q_dqi - q_pr
    else:
        out["D0"] = {"pass": bool(restricted_radius_is_realizable(alpha, lam, p)), "floor": restricted_qary_entropy(lam, p, 2), "support": 2}
        out["D2"] = {"pass": bool(cosine_beats_baseline(alpha, lam)), "rule": "lambda(1-lambda) > alpha^2/2"}
        out["D3"] = {"pass": None, "note": "no sets; the objective's moments enter the law"}
        out["margin"] = cosine_quality(lam) - cosine_baseline(alpha)
        out["Q_star"] = existential_cosine(alpha, p) if p <= 2**17 else None
    out["D1"] = {"pass": bool(lam > 0)}
    out["D6"] = {"pass": True, "note": "void on a prime alphabet"}
    out["all_pass"] = all(out[g]["pass"] for g in ("D0", "D1", "D2") if out[g]["pass"] is not None) and (out["D3"]["pass"] is not False)
    return out


def decoder_check(inst: Adapted, trials: int, seed: int, *, beyond_trials: int = 10) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    p, m = inst.p, inst.m
    values = np.array([1, p - 1], dtype=np.int64) if inst.objective_kind == "cos" else None
    kernel = kernel_basis_mod_p(inst.B.T, p) if m <= 1200 else None

    def pattern(weight):
        e = np.zeros(m, dtype=np.int64)
        if weight:
            support = rng.choice(m, weight, replace=False)
            e[support] = rng.choice(values, weight) if values is not None else rng.integers(1, p, weight)
        return e

    by_weight: dict[int, list[int]] = {}
    recovered = rejected = wrong = 0
    for trial in range(trials):
        weight = inst.ell if trial % 2 == 0 else int(rng.integers(0, inst.ell + 1))
        e = pattern(weight)
        word = e
        if kernel is not None and len(kernel):
            coeffs = rng.integers(0, p, len(kernel), dtype=np.int64)
            word = (e + matmul_mod_p(coeffs[None, :], kernel, p)[0]) % p
        result = inst.decode(word)
        entry = by_weight.setdefault(weight, [0, 0])
        entry[0] += 1
        if result is None:
            rejected += 1
        elif np.array_equal(result, e):
            recovered += 1
            entry[1] += 1
        else:
            wrong += 1
    beyond = {"weight": inst.ell + 1, "trials": beyond_trials, "rejected": 0, "recovered": 0, "wrong": 0}
    for _ in range(beyond_trials):
        e = pattern(inst.ell + 1)
        result = inst.decode(e)
        if result is None:
            beyond["rejected"] += 1
        elif np.array_equal(result, e):
            beyond["recovered"] += 1
        else:
            beyond["wrong"] += 1
    return {
        "trials": trials,
        "ell": inst.ell,
        "with_codewords": kernel is not None,
        "error_alphabet": "{+-1}" if values is not None else "F_p^*",
        "exact_recoveries": recovered,
        "all_recovered": recovered == trials,
        "by_weight": {str(k): v for k, v in sorted(by_weight.items())},
        "failures": {"rejected": rejected, "wrong": wrong},
        "beyond_radius": beyond,
        "seconds": time.time() - t0,
    }


def distinguisher(inst: Adapted, rng, *, max_m: int = 2000, force: bool = False) -> dict:
    if inst.m > max_m and not force:
        return {"skipped": f"m = {inst.m} > {max_m}; pass force=True"}

    class _Shell:
        pass

    shell = _Shell()
    shell.p, shell.m, shell.n, shell.B_pub = inst.p, inst.m, inst.n, inst.B
    battery = square_battery(shell, rng)
    control = random_matrix_battery(inst.m, inst.n, inst.p, rng)
    fires = any(isinstance(v, dict) and not v["full"] for v in battery.values())
    return {"square_battery": battery, "random_control": control, "fires": fires}


def dqi_value(inst: Adapted, *, law: bool = True, law_max_work: float = 5e7) -> dict:
    m, ell = inst.m, inst.ell
    if inst.objective_kind == "sets":
        r = len(inst.sets[0])
        rho = r / inst.p
        quality = canonical_finite_quality(m, inst.p, r, ell)
        q_pr = prange_quality(inst.alpha, rho)
        out = {"Q_fin": {"value": quality.value, "lower": quality.lower, "upper": quality.upper, "method": quality.method}, "dqi_count": quality.value * m, "asymptotic": canonical_asymptotic_quality(ell / m, rho), "Q_Pr": q_pr, "mu_fin": quality.value - q_pr, "rho": rho, "accepted_size": r, "value_range": 1.0}
        if law and m * ell * ell <= law_max_work:
            from .dqi_signature import agreement_distribution

            dist = agreement_distribution(m, inst.p, r, ell)
            probs = dist["probabilities"]
            cdf = np.cumsum(probs)
            out["agreement_law"] = {"std_fraction": dist["std"] / m, "mean_check_abs_diff": dist["abs_diff"], "tail_below_Q_Pr": float(probs[: int(q_pr * m) + 1].sum()), "q01": int(np.searchsorted(cdf, 0.01)), "q99": int(np.searchsorted(cdf, 0.99))}
    else:
        quality = general_objective_finite_quality(m, ell, 0.0, 1 / math.sqrt(2), 0.0)
        q_pr = cosine_baseline(inst.alpha)
        out = {"Q_fin": {"value": quality.value, "lower": quality.lower, "upper": quality.upper, "method": quality.method}, "dqi_count": quality.value * m, "asymptotic": cosine_quality(ell / m), "Q_Pr": q_pr, "mu_fin": quality.value - q_pr, "rho": None, "accepted_size": None, "value_range": 2.0}
    return out


def converse(alpha: float, mu: float, *, draws: float = RESTARTS, failure: float = FAILURE, value_range: float = 1.0) -> dict:
    threshold = None if mu <= 0 else (1 - alpha) * value_range**2 * math.log(draws / failure) / (2 * mu**2)
    return {"threshold": threshold, "draws": draws, "failure": failure, "value_range": value_range}


def classical_portfolio(ctl: Control, *, prange_restarts: int, line_seconds: float, heat_seconds: float, log=None) -> tuple[dict, np.ndarray]:
    attacks: dict = {}
    best_y = None
    for key, call in (("A_prange", lambda: ctl.prange(prange_restarts)), ("B_line", lambda: ctl.line(line_seconds)), ("C_heat", lambda: ctl.heat(heat_seconds))):
        res, y = call()
        independent = ctl.verify(y)
        assert ctl.obj.close(independent, res["best"]), (key, independent, res["best"])
        res["fraction"] = res["best"] / ctl.m
        res["witness_y"] = [int(v) for v in np.asarray(y).ravel()]  # the codeword Bx, re-scorable
        attacks[key] = res
        if best_y is None or ctl.obj.better(res["best"], ctl.score(best_y)):
            best_y = y.copy()
        if log:
            log(f"   {key}: {res['best']:.4f} ({res['fraction']:.4f})" if not ctl.obj.is_integer else f"   {key}: {res['best']} ({res['fraction']:.4f})")
    return attacks, best_y


@dataclasses.dataclass
class Budgets:
    prange_restarts: int | None = None
    line_seconds: float = 40.0
    heat_seconds: float = 40.0
    decoder_trials: int = 200
    square: bool | None = None
    law: bool = True

    @classmethod
    def quick(cls) -> "Budgets":
        return cls(prange_restarts=200, line_seconds=5.0, heat_seconds=5.0, decoder_trials=20, square=None, law=True)

    def restarts_for(self, m: int, n: int) -> int:
        return self.prange_restarts if self.prange_restarts is not None else int(min(8000, max(1000, 4e9 / (m * n))))


def screen_one(build_or_instance, m: int, seed: int, *, objective: str = "auto", budgets: Budgets | None = None, family: str = "generator", log=print, extra_attacks=()) -> dict:
    """Run every step on one instance; ``build_or_instance`` is a callable ``(m, seed)`` or an instance."""

    budgets = budgets or Budgets()
    t0 = time.time()
    instance = build_or_instance(m, seed) if callable(build_or_instance) else build_or_instance
    inst = adapt(instance, objective=objective, seed=seed)
    log(f"{family} size={inst.m} seed={seed}: p={inst.p} n={inst.n} alpha={inst.alpha:.4f} ell={inst.ell} objective={inst.objective_kind} kind={inst.kind}")
    field = PrimeField(inst.p)
    rank = int(rank_mod(field, inst.B))
    assert rank == inst.n, f"rank {rank} != {inst.n}"
    obj = inst.make_objective()
    g = gates(inst.alpha, inst.ell / inst.m, obj.rho, inst.p, objective=inst.objective_kind)
    log(f"   gates: D0 {g['D0']['pass']} D2 {g['D2']['pass']} D3 {g['D3']['pass']} margin {g['margin']:+.4f}")
    dec = decoder_check(inst, budgets.decoder_trials, seed + 1)
    log(f"   decoder: {dec['exact_recoveries']}/{dec['trials']} exact (failures {dec['failures']}); beyond radius {dec['beyond_radius']}; {dec['seconds']:.1f}s")
    rng = np.random.default_rng(seed)
    value = dqi_value(inst, law=budgets.law)
    conv = converse(inst.alpha, value["mu_fin"], value_range=value["value_range"])
    threshold = conv["threshold"]
    log(f"   model: Q_fin {value['Q_fin']['value']:.4f} ({value['dqi_count']:.1f}) Q_Pr {value['Q_Pr']:.4f} mu_fin {value['mu_fin']:+.4f} | threshold {threshold if threshold is None else round(threshold)} ({'above' if threshold and inst.m >= threshold else 'below'})")
    dist = distinguisher(inst, rng, force=True) if (budgets.square if budgets.square is not None else inst.m <= 600) else {"skipped": "disabled (auto for m <= 600; pass square=True)"}
    if "fires" in dist:
        log(f"   distinguisher: {'FIRES' if dist['fires'] else 'no deficit'}")
    ctl = Control(field, inst.B, obj, rng)
    attacks, best_y = classical_portfolio(ctl, prange_restarts=budgets.restarts_for(inst.m, inst.n), line_seconds=budgets.line_seconds, heat_seconds=budgets.heat_seconds, log=log)
    attacks["A_prange"]["mean_minus_Q_Pr"] = attacks["A_prange"]["mean"] - value["Q_Pr"]
    for name, call in extra_attacks:
        res, y = call(ctl, inst)
        if y is not None:
            independent = ctl.verify(y)
            assert ctl.obj.close(independent, res["best"]), (name, independent, res["best"])
            res["fraction"] = res["best"] / inst.m
            res["witness_y"] = [int(v) for v in np.asarray(y).ravel()]
            log(f"   {name}: {res['best']:.4f} ({res['fraction']:.4f})")
        else:
            log(f"   {name}: {res.get('skipped', 'no witness')}")
        attacks[name] = res
    attacks.setdefault("E_cpsat", {"skipped": "not part of the generic screen"})
    scored = {k: v for k, v in attacks.items() if "best" in v}
    best_key = max(scored, key=lambda k: scored[k]["best"])
    best = scored[best_key]["best"]
    dqi_count = value["dqi_count"]
    record = {
        "family": family,
        "kind": inst.kind,
        "objective": inst.objective_kind,
        "size": m,
        "seed": seed,
        "p": inst.p,
        "q": inst.p,
        "m": inst.m,
        "n": inst.n,
        "rank": rank,
        "alpha": inst.alpha,
        "ell": inst.ell,
        "designed_distance": inst.designed_distance,
        "reach_law": inst.reach_law,
        "accepted_size": value["accepted_size"],
        "sets": "random" if inst.objective_kind == "sets" else "cos",
        "rho": value["rho"],
        "genus": 0,
        "Q_fin": value["Q_fin"],
        "dqi_lemma92": dqi_count,
        "dqi_crosscheck_abs_diff": None,
        "semicircle": value["asymptotic"],
        "Q_Pr": value["Q_Pr"],
        "mu_fin": value["mu_fin"],
        "threshold": threshold,
        "above_threshold": bool(threshold is not None and inst.m >= threshold),
        "decoder": dec,
        "info": inst.info,
        "gates": g,
        "distinguisher": dist,
        "agreement_law": value.get("agreement_law"),
        "attacks": attacks,
        "best_attack": best_key,
        "best_classical": best,
        "classical_minus_dqi": best - dqi_count,
        "classical_minus_dqi_fraction": (best - dqi_count) / inst.m,
        "beats_dqi": bool(best >= dqi_count),
        "seconds": time.time() - t0,
        "budgets": {"prange_restarts": budgets.restarts_for(inst.m, inst.n), "line": budgets.line_seconds, "heat": budgets.heat_seconds, "decoder_trials": budgets.decoder_trials},
    }
    log(f"   best classical {best:.4f} ({best / inst.m:.4f}) by {best_key} | DQI {dqi_count:.1f} ({value['Q_fin']['value']:.4f}) | best - DQI = {best - dqi_count:+.1f} | {'BEATEN' if record['beats_dqi'] else 'below DQI'} | {record['seconds']:.0f}s")
    return record


def verdict(records: list[dict]) -> dict:
    groups: dict[tuple[str, int], list[dict]] = {}
    for rec in records:
        groups.setdefault((rec["family"], int(rec["m"])), []).append(rec)
    out = {}
    for (family, m), recs in sorted(groups.items()):
        survives = all(not r["beats_dqi"] for r in recs)
        above = all(r["above_threshold"] for r in recs)
        passes = survives and above
        out[f"{family}@{m}"] = {"family": family, "m": m, "seeds": [r["seed"] for r in recs], "survives": survives, "above_threshold": above, "reason": ("survives the generic portfolio above the restart threshold" if passes else ("beaten on the instance" if not survives else "survives, but below the restart threshold (no converse)")), "beaten_by": [(r["seed"], r["best_attack"]) for r in recs if r["beats_dqi"]]}
    return out


def write_draft(path: str, records: list[dict], *, family: str, objective: str, extra: dict | None = None) -> None:
    payload = {"family": family, "objective": objective, "generator": (extra or {}).get("generator"), "results": records}
    payload.update(extra or {})
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def draft_path(results_dir: str, family: str, m: int, seed: int, tag: str = "") -> str:
    if tag:
        return os.path.join(results_dir, f"finite-control-screen-{family}-run-{tag}-m{m}-s{seed}.json")
    return os.path.join(results_dir, f"finite-control-screen-{family}-run-m{m}-s{seed}.json")


def merge(results_dir: str, family: str | None = None) -> list[dict]:
    pattern = f"finite-control-screen-{family or '*'}-run-*.json"
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
        if "-quick" in os.path.basename(path):
            continue
        with open(path, encoding="utf-8") as handle:
            records.extend(json.load(handle)["results"])
    return records


__all__ = [
    "Adapted",
    "Budgets",
    "Control",
    "CosineObjective",
    "PrimeField",
    "SetObjective",
    "adapt",
    "classical_portfolio",
    "converse",
    "decoder_check",
    "distinguisher",
    "dqi_value",
    "draft_path",
    "gates",
    "gauss_inverse",
    "merge",
    "rank_mod",
    "screen_one",
    "verdict",
    "write_draft",
]
