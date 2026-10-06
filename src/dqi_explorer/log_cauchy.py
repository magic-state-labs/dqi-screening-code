"""The log-Cauchy code: a public restricted-error decoder from exponentiation.

The problem (multiplicative polynomial intersection).  Fix ``F = GF(2^h)`` with
``M = 2^h - 1`` a Mersenne prime, a generator ``g``, ``m`` points ``t_i`` and ``n`` points
``alpha_j`` with every ``t_i + alpha_j`` nonzero.  The public matrix is
``B[i][j] = log_g(t_i + alpha_j)``, an ``m x n`` matrix over the prime field ``F_M``.  For
exponents ``x in Z_M^n`` the row values are ``(Bx)_i = log_g prod_j (t_i + alpha_j)^{x_j}``,
and the smooth objective ``sum_i cos(2 pi ((Bx)_i - c_i) / M)`` asks that the discrete
logarithm of the product land near a prescribed centre ``c_i`` at as many points as possible.
Everything is public: there is no key.

The decoder.  DQI with a smooth objective needs to decode error patterns whose entries lie in
the Fourier support of the objective, ``{0, +1, -1}`` for the cosine (Jordan et al.
2408.08292, Sec. 8.2).  For such a pattern ``y`` of weight ``ell``
the syndrome ``s = B^T y`` satisfies ``g^{s_j} = prod_i (t_i + alpha_j)^{y_i} = R_y(alpha_j)``
with ``R_y = N/D`` a rational function whose numerator and denominator have degree at most
``ell`` and split over the ``t_i``.  Rational reconstruction from ``n >= 2 ell + 1`` values,
division by ``gcd(N, D)`` and a root scan over the ``t_i`` recover ``y``: a worst-case unique
decoder to ``ell = floor((n - 1)/2)``, reach ``alpha/2`` at ``alpha = n/m``.  For patterns with
mixed signs the degrees are about ``ell/2`` each and the decoder succeeds well beyond ``ell``;
``in_measure_probe`` measures that.

Why it is not an evaluation code over ``F_M``.  The entries of ``B`` are discrete
logarithms, so over the DQI alphabet ``F_M`` the code has no polynomial structure (its Schur
square is full); the algebra appears only after exponentiation, which is not linear.  The
construction is the Chor--Rivest knapsack (1988) with its trapdoor made public.  Nothing here
claims hardness; the finite control is ``scripts/41_log_cauchy_finite_control.py``.

Conventions.  Field elements are integers ``0 .. 2^h - 1`` in the polynomial basis
(``alternant_trapdoor.PrimeExtensionField``); ``g = x`` (the class's ``EXP``/``LOG`` tables).
Patterns are int64 vectors over ``F_M`` with ``+1 -> 1`` and ``-1 -> M - 1``.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from math import cos, pi

import numpy as np

from .alternant_trapdoor import PrimeExtensionField

# primitive trinomials/pentanomials over GF(2), coefficients low -> high
MERSENNE_MODULI: dict[int, tuple[int, ...]] = {
    13: (1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),  # x^13 + x^4 + x^3 + x + 1
    17: (1, 0, 0, 1) + (0,) * 13 + (1,),  # x^17 + x^3 + 1
}
MERSENNE_EXPONENTS = (13, 17, 19, 31, 61, 89, 107, 127)


def mersenne_prime(h: int) -> int:
    if h not in MERSENNE_EXPONENTS:
        raise ValueError(f"2^{h} - 1 is not a Mersenne prime in the supported list")
    return 2**h - 1


@functools.lru_cache(maxsize=None)
def mersenne_field(h: int) -> PrimeExtensionField:
    if h not in MERSENNE_MODULI:
        raise ValueError(f"no stored primitive modulus for h = {h}; tables exist for {sorted(MERSENNE_MODULI)}")
    return PrimeExtensionField(2, h, modulus=MERSENNE_MODULI[h])


# ---------------------------------------------------------------------------
# polynomials over GF(2^h): int64 arrays low -> high
# ---------------------------------------------------------------------------
def poly_degree(P) -> int:
    nz = np.nonzero(np.asarray(P))[0]
    return int(nz[-1]) if len(nz) else -1


def poly_mod(field: PrimeExtensionField, a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64).copy()
    b = np.asarray(b, dtype=np.int64)
    db = poly_degree(b)
    if db < 0:
        raise ZeroDivisionError("polynomial modulus by zero")
    inv = field.inv(b[db])
    da = poly_degree(a)
    while da >= db:
        c = field.mul(a[da], inv)
        a[da - db : da + 1] = field.sub(a[da - db : da + 1], field.mul(c, b[: db + 1]))
        da = poly_degree(a)
    return a


def poly_div_exact(field: PrimeExtensionField, a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64).copy()
    b = np.asarray(b, dtype=np.int64)
    db = poly_degree(b)
    inv = field.inv(b[db])
    q = np.zeros(len(a), dtype=np.int64)
    da = poly_degree(a)
    while da >= db:
        c = field.mul(a[da], inv)
        q[da - db] = c
        a[da - db : da + 1] = field.sub(a[da - db : da + 1], field.mul(c, b[: db + 1]))
        da = poly_degree(a)
    if da >= 0:
        raise ArithmeticError("inexact polynomial division")
    return q


def poly_gcd(field: PrimeExtensionField, a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    while poly_degree(b) >= 0:
        a, b = b, poly_mod(field, a, b)
    return a


def poly_eval(field: PrimeExtensionField, P, points) -> np.ndarray:
    """Horner evaluation of ``P`` at an array of points."""

    points = np.asarray(points, dtype=np.int64)
    value = np.zeros_like(points)
    for c in reversed(np.asarray(P, dtype=np.int64).tolist()):
        value = field.add(field.mul(value, points), c)
    return value


def nullspace_vector_over_field(field: PrimeExtensionField, matrix) -> np.ndarray | None:
    """One nonzero vector of the right kernel of ``matrix`` over ``GF(p^s)``, or ``None``."""

    a = np.asarray(matrix, dtype=np.int64).copy()
    rows, cols = a.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(a[r:, c])[0]
        if len(nz) == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        a[r] = field.mul(a[r], field.inv(a[r, c]))
        factors = a[:, c].copy()
        factors[r] = 0
        idx = np.nonzero(factors)[0]
        if len(idx):
            a[idx] = field.sub(a[idx], field.mul(factors[idx, None], a[r][None, :]))
        pivots.append(c)
        r += 1
    free = [c for c in range(cols) if c not in set(pivots)]
    if not free:
        return None
    f = free[0]
    v = np.zeros(cols, dtype=np.int64)
    v[f] = 1
    for row, c in enumerate(pivots):
        v[c] = field.neg(a[row, f])
    return v


# ---------------------------------------------------------------------------
# the instance
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LogCauchyInstance:
    kind: str
    h: int
    m: int
    n: int
    seed: int
    t_points: np.ndarray
    alpha_points: np.ndarray
    B_pub: np.ndarray
    centres: np.ndarray

    @property
    def p(self) -> int:
        return 2**self.h - 1

    @property
    def M(self) -> int:
        return self.p

    @property
    def field(self) -> PrimeExtensionField:
        return mersenne_field(self.h)

    @property
    def alpha(self) -> float:
        return self.n / self.m

    @property
    def ell(self) -> int:
        return (self.n - 1) // 2

    @property
    def designed_distance(self) -> int:
        return self.n + 1

    @property
    def reach_law(self) -> str:
        return "alpha/2 for {0,+-1} errors (worst case); about alpha in measure"

    @property
    def objective(self) -> str:
        return "cos"

    # -- syndromes and decoding ---------------------------------------------
    def syndrome(self, word) -> np.ndarray:
        w = np.asarray(word, dtype=np.int64) % self.p
        return (np.asarray(self.B_pub, dtype=np.float64).T @ w.astype(np.float64) % self.p).astype(np.int64)

    def decode_syndrome(self, syndrome, *, enforce_radius: bool = True) -> np.ndarray | None:
        """Pattern ``y`` in ``{0, 1, M-1}^m`` with ``B^T y = syndrome``, by rational reconstruction."""

        field = self.field
        s = np.asarray(syndrome, dtype=np.int64) % self.p
        if not np.any(s):
            return np.zeros(self.m, dtype=np.int64)
        R = field.EXP[s]  # R_y(alpha_j)
        ell = self.ell
        a = self.alpha_points
        cols = []
        power = np.ones(self.n, dtype=np.int64)
        for _ in range(ell + 1):
            cols.append(power.copy())
            power = field.mul(power, a)
        power = np.ones(self.n, dtype=np.int64)
        for _ in range(ell + 1):
            cols.append(field.neg(field.mul(R, power)))
            power = field.mul(power, a)
        system = np.stack(cols, axis=1)  # n x (2 ell + 2): N(a_j) - R_j D(a_j) = 0
        v = nullspace_vector_over_field(field, system)
        if v is None:
            return None
        N, D = v[: ell + 1], v[ell + 1 :]
        if poly_degree(N) < 0 or poly_degree(D) < 0:
            return None
        g = poly_gcd(field, N, D)
        if poly_degree(g) > 0:
            N, D = poly_div_exact(field, N, g), poly_div_exact(field, D, g)
        minus_t = field.neg(self.t_points)
        plus = np.nonzero(poly_eval(field, N, minus_t) == 0)[0]
        minus = np.nonzero(poly_eval(field, D, minus_t) == 0)[0]
        if len(plus) != poly_degree(N) or len(minus) != poly_degree(D):
            return None  # a factor did not split over the points: not a restricted pattern
        y = np.zeros(self.m, dtype=np.int64)
        y[plus] = 1
        y[minus] = self.p - 1
        if enforce_radius and len(plus) + len(minus) > ell:
            return None
        if not np.array_equal(self.syndrome(y), s):
            return None
        return y

    def decode(self, word, *, enforce_radius: bool = True) -> np.ndarray | None:
        """Unique restricted pattern with the syndrome of ``word`` (public order), else ``None``."""

        return self.decode_syndrome(self.syndrome(word), enforce_radius=enforce_radius)

    # -- objective ------------------------------------------------------------
    def cosine_score(self, y) -> float:
        y = np.asarray(y, dtype=np.int64) % self.p
        return float(np.cos(2 * pi * ((y - self.centres) % self.p) / self.p).sum())


def build_log_cauchy_instance(*, m: int, seed: int, h: int = 13, alpha: float | None = 0.07, n: int | None = None) -> LogCauchyInstance:
    """``m`` points ``t_i`` and ``n`` points ``alpha_j``, all distinct, no ``t_i + alpha_j = 0``."""

    field = mersenne_field(h)
    q = field.order
    if n is None:
        if alpha is None:
            raise ValueError("give alpha or n")
        n = int(round(alpha * m))
    if n < 3:
        raise ValueError("n must be at least 3 for a positive radius")
    if m + n > q:
        raise ValueError(f"m + n must be at most 2^h = {q} (distinct points; t_i + alpha_j = 0 iff t_i = alpha_j in characteristic 2)")
    rng = np.random.default_rng(seed)
    points = rng.choice(q, m + n, replace=False).astype(np.int64)
    alpha_points = points[:n]
    t_points = points[n:]
    sums = field.add(t_points[:, None], alpha_points[None, :])
    assert np.all(sums != 0)
    B = field.LOG[sums] % (q - 1)
    centres = rng.integers(0, q - 1, m, dtype=np.int64)
    for array in (t_points, alpha_points, B, centres):
        array.setflags(write=False)
    return LogCauchyInstance(kind="log_cauchy", h=h, m=m, n=n, seed=seed, t_points=t_points, alpha_points=alpha_points, B_pub=B, centres=centres)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def random_restricted_pattern(instance: LogCauchyInstance, weight: int, rng, *, signs: str = "random") -> np.ndarray:
    y = np.zeros(instance.m, dtype=np.int64)
    if weight:
        support = rng.choice(instance.m, weight, replace=False)
        if signs == "plus":
            y[support] = 1
        elif signs == "minus":
            y[support] = instance.p - 1
        else:
            y[support] = rng.choice([1, instance.p - 1], weight)
    return y


def verify_decoder(instance: LogCauchyInstance, trials: int, seed: int, *, beyond_trials: int = 10) -> dict:
    """Exact recovery of restricted patterns of weight <= ell, including the all-plus and all-minus worst cases."""

    rng = np.random.default_rng(seed)
    t0 = time.time()
    recovered = rejected = wrong = 0
    by_kind: dict[str, list[int]] = {}
    schedule = [("plus", instance.ell), ("minus", instance.ell)]
    for trial in range(trials):
        schedule.append(("random", instance.ell if trial % 2 == 0 else int(rng.integers(0, instance.ell + 1))))
    for signs, weight in schedule:
        y = random_restricted_pattern(instance, weight, rng, signs=signs)
        result = instance.decode(y)
        entry = by_kind.setdefault(signs, [0, 0])
        entry[0] += 1
        if result is None:
            rejected += 1
        elif np.array_equal(result, y):
            recovered += 1
            entry[1] += 1
        else:
            wrong += 1
    total = len(schedule)
    beyond = {"weight": instance.ell + 1, "trials": beyond_trials, "rejected": 0, "recovered": 0, "wrong": 0}
    for _ in range(beyond_trials):
        y = random_restricted_pattern(instance, instance.ell + 1, rng, signs="plus")
        result = instance.decode(y)
        if result is None:
            beyond["rejected"] += 1
        elif np.array_equal(result, y):
            beyond["recovered"] += 1
        else:
            beyond["wrong"] += 1
    return {
        "trials": total,
        "ell": instance.ell,
        "exact_recoveries": recovered,
        "all_recovered": recovered == total,
        "by_kind": {k: v for k, v in by_kind.items()},
        "failures": {"rejected": rejected, "wrong": wrong},
        "beyond_radius": beyond,
        "seconds": time.time() - t0,
    }


def in_measure_probe(instance: LogCauchyInstance, weights, trials: int, seed: int) -> dict:
    """Recovery fraction of *random-sign* patterns at weights beyond ell (the D4' reach)."""

    rng = np.random.default_rng(seed)
    out = {}
    for w in weights:
        ok = 0
        for _ in range(trials):
            y = random_restricted_pattern(instance, int(w), rng, signs="random")
            result = instance.decode(y, enforce_radius=False)
            ok += result is not None and np.array_equal(result, y)
        out[int(w)] = ok / trials
    return {"ell": instance.ell, "n": instance.n, "recovery_fraction": out}


def as_dense_arrays(instance: LogCauchyInstance) -> tuple[np.ndarray, np.ndarray]:
    return np.array(instance.B_pub, dtype=np.int64), np.array(instance.centres, dtype=np.int64)


def restricted_decodability_probe(B: np.ndarray, p: int, ell: int, trials: int, rng) -> dict:
    """Syndrome-collision test for ``{0, +-1}`` patterns of weight <= ell on any matrix ``B`` (m x n).

    Information-theoretic only: distinct restricted patterns with equal syndromes would defeat
    every decoder.  Reports collisions among ``trials`` random pairs and the pigeonhole count
    ``sum_k C(m,k) 2^k`` against ``p^n``.  A search tool for further domain-switch codes.
    """

    from math import comb

    B = np.asarray(B, dtype=np.int64)
    m, n = B.shape
    seen: dict[tuple, np.ndarray] = {}
    collisions = 0
    for _ in range(trials):
        w = int(rng.integers(1, ell + 1))
        y = np.zeros(m, dtype=np.int64)
        support = rng.choice(m, w, replace=False)
        y[support] = rng.choice([1, p - 1], w)
        s = tuple(((B.T.astype(np.float64) @ y.astype(np.float64)) % p).astype(np.int64).tolist())
        if s in seen and not np.array_equal(seen[s], y):
            collisions += 1
        seen[s] = y
    patterns = sum(comb(m, k) * 2**k for k in range(ell + 1))
    return {"trials": trials, "collisions": collisions, "restricted_patterns": patterns, "syndromes": p**n, "pigeonhole_ok": patterns <= p**n}


# ---------------------------------------------------------------------------
# odd characteristic: the safe-prime family (alphabet M = (p - 1)/2 prime, images are squares)
# ---------------------------------------------------------------------------
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def is_safe_prime(p: int) -> bool:
    """``p = 2M + 1`` with both ``p`` and ``M`` prime."""

    return _is_prime(p) and _is_prime((p - 1) // 2)


def next_safe_prime(start: int) -> int:
    p = max(5, start | 1)
    while not is_safe_prime(p):
        p += 2
    return p


@functools.lru_cache(maxsize=None)
def safe_prime_field(p: int) -> PrimeExtensionField:
    if not is_safe_prime(p):
        raise ValueError(f"{p} is not a safe prime")
    return PrimeExtensionField(p, 1)


def poly_derivative(field: PrimeExtensionField, P) -> np.ndarray:
    P = np.asarray(P, dtype=np.int64)
    out = np.zeros(len(P), dtype=np.int64)
    if len(P) > 1:
        k = np.arange(1, len(P), dtype=np.int64) % field.p
        out[:-1] = field.mul(P[1:], k)
    return out


def poly_sqrt_squarefree(field: PrimeExtensionField, Q) -> np.ndarray | None:
    """``N`` with ``N^2 = Q`` up to a scalar when ``N`` is squarefree (odd characteristic): ``gcd(Q, Q')``."""

    Q = np.asarray(Q, dtype=np.int64)
    if poly_degree(Q) < 0:
        return None
    if poly_degree(Q) == 0:
        return Q[:1].copy()
    g = poly_gcd(field, Q, poly_derivative(field, Q))
    if 2 * poly_degree(g) != poly_degree(Q):
        return None
    return g


@dataclasses.dataclass(frozen=True)
class SafePrimeLogCauchyInstance:
    """``B[i][j] = log_{g^2}((t_i + alpha_j)^2)`` over ``F_M``, ``M = (q - 1)/2``, ``q`` a safe prime.

    The index-two subgroup of ``F_q^*`` has prime order ``M``, so the alphabet is a prime field
    of any size (no Mersenne restriction, no subfield).  A ``{0, +-1}`` pattern ``y`` exponentiates
    to ``R_y(alpha_j)^2``; rational reconstruction with degree bound ``2 ell`` from
    ``n >= 4 ell + 1`` points, polynomial square roots and a root scan recover ``y``: reach
    ``alpha/4`` (half the Mersenne form's), still a worst-case licence.
    """

    kind: str
    q: int
    m: int
    n: int
    seed: int
    t_points: np.ndarray
    alpha_points: np.ndarray
    B_pub: np.ndarray
    centres: np.ndarray

    @property
    def p(self) -> int:
        return (self.q - 1) // 2

    @property
    def M(self) -> int:
        return self.p

    @property
    def field(self) -> PrimeExtensionField:
        return safe_prime_field(self.q)

    @property
    def alpha(self) -> float:
        return self.n / self.m

    @property
    def ell(self) -> int:
        return (self.n - 1) // 4

    @property
    def designed_distance(self) -> int:
        return self.n + 1

    @property
    def reach_law(self) -> str:
        return "alpha/4 for {0,+-1} errors (worst case; squares of a rational function)"

    @property
    def objective(self) -> str:
        return "cos"

    def syndrome(self, word) -> np.ndarray:
        w = np.asarray(word, dtype=np.int64) % self.p
        return (np.asarray(self.B_pub, dtype=np.float64).T @ w.astype(np.float64) % self.p).astype(np.int64)

    def decode_syndrome(self, syndrome, *, enforce_radius: bool = True) -> np.ndarray | None:
        field = self.field
        s = np.asarray(syndrome, dtype=np.int64) % self.p
        if not np.any(s):
            return np.zeros(self.m, dtype=np.int64)
        Q = field.EXP[(2 * s) % (self.q - 1)]  # R_y(alpha_j)^2
        bound = 2 * self.ell
        a = self.alpha_points
        cols = []
        power = np.ones(self.n, dtype=np.int64)
        for _ in range(bound + 1):
            cols.append(power.copy())
            power = field.mul(power, a)
        power = np.ones(self.n, dtype=np.int64)
        for _ in range(bound + 1):
            cols.append(field.neg(field.mul(Q, power)))
            power = field.mul(power, a)
        v = nullspace_vector_over_field(field, np.stack(cols, axis=1))
        if v is None:
            return None
        N2, D2 = v[: bound + 1], v[bound + 1 :]
        if poly_degree(N2) < 0 or poly_degree(D2) < 0:
            return None
        g = poly_gcd(field, N2, D2)
        if poly_degree(g) > 0:
            N2, D2 = poly_div_exact(field, N2, g), poly_div_exact(field, D2, g)
        N = poly_sqrt_squarefree(field, N2)
        D = poly_sqrt_squarefree(field, D2)
        if N is None or D is None:
            return None
        minus_t = field.neg(self.t_points)
        plus = np.nonzero(poly_eval(field, N, minus_t) == 0)[0]
        minus = np.nonzero(poly_eval(field, D, minus_t) == 0)[0]
        if len(plus) != poly_degree(N) or len(minus) != poly_degree(D):
            return None
        y = np.zeros(self.m, dtype=np.int64)
        y[plus] = 1
        y[minus] = self.p - 1
        if enforce_radius and len(plus) + len(minus) > self.ell:
            return None
        if not np.array_equal(self.syndrome(y), s):
            return None
        return y

    def decode(self, word, *, enforce_radius: bool = True) -> np.ndarray | None:
        return self.decode_syndrome(self.syndrome(word), enforce_radius=enforce_radius)

    def cosine_score(self, y) -> float:
        y = np.asarray(y, dtype=np.int64) % self.p
        return float(np.cos(2 * pi * ((y - self.centres) % self.p) / self.p).sum())


def build_safe_prime_instance(*, m: int, seed: int, q: int | None = None, alpha: float | None = 0.07, n: int | None = None) -> SafePrimeLogCauchyInstance:
    """``q`` a safe prime with ``q >= m + n + 1`` (default: the next safe prime above ``2m``)."""

    if n is None:
        if alpha is None:
            raise ValueError("give alpha or n")
        n = int(round(alpha * m))
    if n < 5:
        raise ValueError("n must be at least 5 for a positive radius")
    if q is None:
        q = next_safe_prime(2 * m + 1)
    field = safe_prime_field(q)
    if m + n > q - 1:
        raise ValueError("m + n must be at most q - 1")
    M = (q - 1) // 2
    rng = np.random.default_rng(seed)
    alpha_points = rng.choice(q - 1, n, replace=False).astype(np.int64) + 1  # nonzero, distinct
    forbidden = np.unique(np.concatenate([alpha_points, (-alpha_points) % q, np.zeros(1, dtype=np.int64)]))
    allowed = np.setdiff1d(np.arange(q, dtype=np.int64), forbidden)
    if len(allowed) < m:
        raise ValueError("q too small for m points avoiding the alpha_j and their negatives")
    t_points = rng.choice(allowed, m, replace=False).astype(np.int64)
    sums = (t_points[:, None] + alpha_points[None, :]) % q
    assert not np.any(sums == 0)
    logs = field.LOG[sums]  # log_g of the elements; the squares have even logs
    B = ((2 * logs) % (q - 1)) // 2 % M  # log_{g^2}((t+a)^2) = log_g(t+a) mod M
    centres = rng.integers(0, M, m, dtype=np.int64)
    for array in (t_points, alpha_points, B, centres):
        array.setflags(write=False)
    return SafePrimeLogCauchyInstance(kind="safe_prime_log_cauchy", q=q, m=m, n=n, seed=seed, t_points=t_points, alpha_points=alpha_points, B_pub=B, centres=centres)


# ---------------------------------------------------------------------------
# a key on the exponent knapsack: a secret dense change of basis
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MixedLogCauchyInstance:
    """``B_pub = B T`` for a secret dense invertible ``T`` over ``F_M`` (``n x n``).

    The code ``im(B)`` and the decoder code ``ker(B^T)`` are unchanged, so the optimization
    problem is the same as the public form up to the reparametrization ``x -> T x``; what the
    secret hides is the basis in which the entrywise exponential of the matrix is the Cauchy
    matrix ``t_i + alpha_j``.  The key holder undoes ``T^T`` on the syndrome and decodes as
    before.  Key recovery is the search for two code vectors whose exponentials differ by a
    constant vector (``exp_affine_pair_test``), a problem with no linear handle.
    """

    kind: str
    base: LogCauchyInstance
    T: np.ndarray
    T_inv: np.ndarray
    B_pub: np.ndarray

    @property
    def p(self) -> int:
        return self.base.p

    @property
    def m(self) -> int:
        return self.base.m

    @property
    def n(self) -> int:
        return self.base.n

    @property
    def h(self) -> int:
        return self.base.h

    @property
    def field(self) -> PrimeExtensionField:
        return self.base.field

    @property
    def alpha(self) -> float:
        return self.base.alpha

    @property
    def ell(self) -> int:
        return self.base.ell

    @property
    def centres(self) -> np.ndarray:
        return self.base.centres

    @property
    def designed_distance(self) -> int:
        return self.base.designed_distance

    @property
    def reach_law(self) -> str:
        return self.base.reach_law + " (key holder only)"

    @property
    def objective(self) -> str:
        return "cos"

    def syndrome(self, word) -> np.ndarray:
        w = np.asarray(word, dtype=np.int64) % self.p
        return (np.asarray(self.B_pub, dtype=np.float64).T @ w.astype(np.float64) % self.p).astype(np.int64)

    def decode(self, word, *, enforce_radius: bool = True) -> np.ndarray | None:
        s_pub = self.syndrome(word)
        s = (np.asarray(self.T_inv, dtype=np.float64).T @ s_pub.astype(np.float64) % self.p).astype(np.int64)
        return self.base.decode_syndrome(s, enforce_radius=enforce_radius)

    def cosine_score(self, y) -> float:
        return self.base.cosine_score(y)


def build_mixed_log_cauchy_instance(*, m: int, seed: int, h: int = 13, alpha: float | None = 0.07, n: int | None = None) -> MixedLogCauchyInstance:
    from .alternant_trapdoor import inverse_mod_p

    base = build_log_cauchy_instance(m=m, seed=seed, h=h, alpha=alpha, n=n)
    rng = np.random.default_rng(seed + 101)
    M = base.p
    while True:
        T = rng.integers(0, M, (base.n, base.n), dtype=np.int64)
        T_inv = inverse_mod_p(T, M)
        if T_inv is not None:
            break
    B_pub = (np.asarray(base.B_pub, dtype=np.float64) @ T.astype(np.float64) % M).astype(np.int64)
    for array in (T, T_inv, B_pub):
        array.setflags(write=False)
    return MixedLogCauchyInstance(kind="mixed_log_cauchy", base=base, T=T, T_inv=T_inv, B_pub=B_pub)


def exp_affine_pair_test(instance: MixedLogCauchyInstance, x, x_prime) -> dict:
    """Is ``exp(B_pub x) - exp(B_pub x')`` a constant vector?  True for ``x = T^-1 e_j``, ``x' = T^-1 e_k``.

    The key-recovery problem for the mixed key is to find such a pair without ``T``: the number of
    candidate pairs is ``M^(2n)`` and the condition is ``m - 1`` equations over the field, so the
    expected number of spurious solutions is about ``M^(2n) / q^(m-1)``, printed by the sweep.
    """

    field = instance.field
    M = instance.p
    v = (np.asarray(instance.B_pub, dtype=np.float64) @ np.asarray(x, dtype=np.float64) % M).astype(np.int64)
    w = (np.asarray(instance.B_pub, dtype=np.float64) @ np.asarray(x_prime, dtype=np.float64) % M).astype(np.int64)
    diff = field.sub(field.EXP[v], field.EXP[w])
    return {"constant": bool(np.all(diff == diff[0])), "distinct_values": int(len(np.unique(diff)))}


def switch_closures() -> list[dict]:
    """The domain-switch sweep: candidate isomorphisms and why each opens or closes."""

    from math import log

    return [
        {"switch": "exp/log into F_(2^h)^*, M = 2^h - 1 Mersenne", "identification": "rational reconstruction (degree <= ell)", "reach": "alpha/2", "status": "open: the log-Cauchy family"},
        {"switch": "exp/log into the squares of F_q^*, q = 2M + 1 safe prime", "identification": "reconstruct R_y^2 (degree <= 2 ell), polynomial square roots", "reach": "alpha/4", "status": "open: odd characteristic, alphabet at any size"},
        {"switch": "elliptic-curve discrete log, |E(F_q)| = M prime", "identification": "a sum of ell points is one point; the summands are not recoverable (no degree); a Jacobian of genus g keeps them only for ell <= g, and g ~ n needs a group of size q^g", "reach": "-", "status": "closed by argument"},
        {"switch": "pairing into mu_M subset F_(q^k)^*", "identification": "Miller functions of degree about M ell; nothing of low degree to reconstruct", "reach": "-", "status": "closed by argument"},
        {"switch": "integers modulo a prime P, generators = small primes p_i", "identification": "rational reconstruction of a rational number of height p_max^ell needs P > p_max^(2 ell): ell <= log P / (2 log p_max), a constant", "reach": "vanishing (D1); CRT across columns needs one modulus per column (rule break)", "status": "closed with the count: ell <= %.1f at P = 2^61, p_max = 100" % (61 * log(2) / (2 * log(100)))},
        {"switch": "super-increasing (Merkle-Hellman) knapsack", "identification": "greedy decoding needs M >= 2^m", "reach": "-", "status": "closed (exponential alphabet, rule break)"},
        {"switch": "the exponent knapsack under a secret dense change of basis T", "identification": "the same decoder with T^-T on the syndrome (key holder); key recovery = an exp-affine pair search over M^(2n) candidates", "reach": "alpha/2 (key holder)", "status": "open: a key on a class-3 code; the key-less problem is the public one"},
        {"switch": "the exponent knapsack on an elliptic curve (functions in a Riemann-Roch space)", "identification": "reconstruction in the function field at a genus penalty", "reach": "about alpha/2 - genus terms", "status": "variant: more points per field size, the same mechanism"},
    ]


# ---------------------------------------------------------------------------
# export for the lattice attack (scripts/43_lattice_attack.py)
# ---------------------------------------------------------------------------
def export_instance_json(instance, path: str) -> dict:
    """Write everything the standalone lattice script needs: the public matrix, centres, alphabet, points.

    Works for the characteristic-2 form, the safe-prime form and the mixed key (which exports its
    public matrix; the lattice attack does not use the secret).
    """

    import json

    payload = {
        "kind": instance.kind,
        "m": int(instance.m),
        "n": int(instance.n),
        "alphabet": int(instance.p),
        "ell": int(instance.ell),
        "alpha": float(instance.alpha),
        "objective": "cos",
        "B": np.asarray(instance.B_pub, dtype=np.int64).tolist(),
        "centres": np.asarray(instance.centres, dtype=np.int64).tolist(),
    }
    if hasattr(instance, "h"):
        payload["h"] = int(instance.h)
    if hasattr(instance, "q"):
        payload["q"] = int(instance.q)
    base = getattr(instance, "base", instance)
    if hasattr(base, "t_points"):
        payload["t_points"] = np.asarray(base.t_points, dtype=np.int64).tolist()
        payload["alpha_points"] = np.asarray(base.alpha_points, dtype=np.int64).tolist()
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.write("\n")
    return {k: v for k, v in payload.items() if k not in ("B", "centres", "t_points", "alpha_points")}


def in_measure_failure_probability(n: int, w: int) -> float:
    """``Pr[max(#plus, #minus) > ell]`` for a uniform random sign pattern of weight ``w``, ``ell = floor((n-1)/2)``.

    The rational reconstruction with degree bound ``ell`` succeeds iff both the numerator and the
    denominator degree are at most ``ell``; with ``P`` plus signs out of ``w`` (binomial), that is
    ``w - ell <= P <= ell``.  This is the in-measure licence of the log-Cauchy decoder (paper
    Theorem, part (ii)): exponentially small in ``n`` for ``w = (1 - eps) 2 ell``.
    """

    from math import comb

    ell = (n - 1) // 2
    if w <= ell:
        return 0.0
    ok = sum(comb(w, k) for k in range(max(0, w - ell), min(w, ell) + 1))
    return 1.0 - ok / 2**w


__all__ = [
    "LogCauchyInstance",
    "MERSENNE_EXPONENTS",
    "MERSENNE_MODULI",
    "as_dense_arrays",
    "build_log_cauchy_instance",
    "in_measure_probe",
    "mersenne_field",
    "mersenne_prime",
    "random_restricted_pattern",
    "restricted_decodability_probe",
    "verify_decoder",
    "SafePrimeLogCauchyInstance",
    "MixedLogCauchyInstance",
    "build_safe_prime_instance",
    "build_mixed_log_cauchy_instance",
    "exp_affine_pair_test",
    "is_safe_prime",
    "next_safe_prime",
    "poly_sqrt_squarefree",
    "safe_prime_field",
    "switch_closures",
    "export_instance_json",
    "in_measure_failure_probability",
]
