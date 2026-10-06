"""Alternant trapdoor instances: a public max-LINSAT matrix with a hidden decoder.

The family.  Fix a prime ``p`` and an extension degree ``s``, and write ``F`` for
``GF(p^s)``.  Choose ``m`` distinct nonzero evaluation points ``a_i`` and nonzero
multipliers ``v_i`` in ``F``.  The ``r x m`` matrix ``H[j][i] = v_i a_i^j`` is a
generalized Reed--Solomon parity check over ``F``.  Expanding every entry into its
``s`` coordinates over ``F_p`` gives ``H_p``, an ``(s r) x m`` matrix over ``F_p``,
and the alternant code is ``ker(H_p)``: designed distance ``r + 1``, dimension at
least ``m - s r``.  The public matrix is ``B = P H_p^T S`` for a random row
permutation ``P`` and a random invertible ``S`` over ``F_p`` (the Niederreiter
public key with the roles of rows and columns as DQI needs them).  The optimizer
code ``im(B)`` is the dual of the alternant code; the decoder code ``ker(B^T)`` is
the alternant code itself.

The decoder.  Anyone holding ``(a, v)`` computes the ``r`` syndromes of a pattern
over ``F`` and runs Berlekamp--Massey, root finding and an ``L x L`` solve: a
worst-case unique decoder to ``floor(r/2)`` errors, licensed at
``ell = floor((r - 1)/2)`` by the strict condition ``2 ell + 1 < r + 1``.  Anyone
without ``(a, v)`` sees a dense ``p``-ary matrix.

What this module does not claim.  It constructs instances, decodes with the key,
and exports the public matrix for the classical controls of
``scripts/30_alternant_trapdoor_finite_control.py``.  The generic
dimension ``m - s r`` is *certified* per instance by a rank computation.  No
statement about classical hardness, indistinguishability or quantum advantage is
made here; those are the assumptions stated in the paper.

Conventions.  Field elements are integers ``0 .. p^s - 1`` whose base-``p`` digits
are the coordinates in the polynomial basis ``1, x, ..., x^(s-1)``; the prime
subfield is therefore ``0 .. p - 1`` under the identity map.  Multiplication and
inversion go through log/antilog tables; addition is digit-wise modulo ``p``.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import time
from math import comb

import numpy as np


# ---------------------------------------------------------------------------
# polynomials over F_p (low -> high coefficient lists)
# ---------------------------------------------------------------------------
def _poly_mod(dividend: list[int], divisor: list[int], p: int) -> list[int]:
    """Remainder of ``dividend`` modulo the monic ``divisor`` over F_p."""

    rem = [c % p for c in dividend]
    d = len(divisor) - 1
    while len(rem) > d:
        lead = rem[-1]
        if lead:
            shift = len(rem) - 1 - d
            for k in range(d + 1):
                rem[shift + k] = (rem[shift + k] - lead * divisor[k]) % p
        rem.pop()
    while len(rem) > 1 and rem[-1] == 0:
        rem.pop()
    return rem


def polynomial_is_irreducible(coefficients: tuple[int, ...], p: int) -> bool:
    """Whether the monic polynomial (low -> high, leading 1) is irreducible over F_p.

    Brute force over monic divisors of degree ``1 .. deg/2``; supports degree at most 4,
    which is all the field constructions here need.
    """

    coeffs = [int(c) % p for c in coefficients]
    if len(coeffs) < 2 or coeffs[-1] != 1:
        raise ValueError("expected a monic polynomial given low -> high")
    degree = len(coeffs) - 1
    if degree > 4:
        raise ValueError("irreducibility test supports degree at most 4")
    if degree == 1:
        return True
    for d in range(1, degree // 2 + 1):
        for tail in itertools.product(range(p), repeat=d):
            divisor = list(tail) + [1]
            if _poly_mod(coeffs, divisor, p) == [0]:
                return False
    return True


# ---------------------------------------------------------------------------
# the field GF(p^s)
# ---------------------------------------------------------------------------
class PrimeExtensionField:
    """``GF(p^s)`` with table arithmetic; elements are ints ``0 .. p^s - 1``."""

    def __init__(self, p: int, s: int, modulus: tuple[int, ...] | None = None) -> None:
        if p < 2 or any(p % d == 0 for d in range(2, int(p**0.5) + 1)):
            raise ValueError("p must be prime")
        if s < 1:
            raise ValueError("s must be positive")
        self.p = int(p)
        self.s = int(s)
        self.order = self.p**self.s
        if modulus is None:
            modulus = self.find_primitive_modulus(self.p, self.s)
        self.modulus = tuple(int(c) % self.p for c in modulus)
        if len(self.modulus) != self.s + 1 or self.modulus[-1] != 1:
            raise ValueError("modulus must be monic of degree s, given low -> high")
        if not self._is_primitive(self.modulus, self.p, self.s):
            raise ValueError("modulus is not primitive")
        weights = np.array([self.p**k for k in range(self.s)], dtype=np.int64)
        self.WEIGHTS = weights
        self.EXP = self._power_table(self.modulus, self.p, self.s)
        log = np.zeros(self.order, dtype=np.int64)
        log[self.EXP] = np.arange(self.order - 1, dtype=np.int64)
        self.LOG = log  # LOG[0] is a placeholder; every user masks zero
        digits = np.zeros((self.order, self.s), dtype=np.int64)
        values = np.arange(self.order, dtype=np.int64)
        for k in range(self.s):
            digits[:, k] = (values // self.p**k) % self.p
        self.DIGITS = digits
        self.generator = self.p  # the element "x"
        for array in (self.EXP, self.LOG, self.DIGITS, self.WEIGHTS):
            array.setflags(write=False)

    # -- fast construction (same results as _power_digits, far less memory) ----
    @staticmethod
    def _polymulmod(a: list[int], b: list[int], modulus: tuple[int, ...], p: int, s: int) -> list[int]:
        prod = [0] * (2 * s - 1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    prod[i + j] = (prod[i + j] + ai * bj) % p
        for k in range(2 * s - 2, s - 1, -1):
            lead = prod[k]
            if lead:
                for t in range(s):
                    prod[k - s + t] = (prod[k - s + t] - lead * modulus[t]) % p
                prod[k] = 0
        return prod[:s]

    @staticmethod
    def _xpow(e: int, modulus: tuple[int, ...], p: int, s: int) -> list[int]:
        result = [1] + [0] * (s - 1)
        base = [0, 1] + [0] * (s - 2) if s > 1 else [0]
        if s == 1:
            base = [(-modulus[0]) % p]
        while e:
            if e & 1:
                result = PrimeExtensionField._polymulmod(result, base, modulus, p, s)
            base = PrimeExtensionField._polymulmod(base, base, modulus, p, s)
            e >>= 1
        return result

    @staticmethod
    def _is_primitive(modulus: tuple[int, ...], p: int, s: int) -> bool:
        """``x`` has order exactly ``p^s - 1`` modulo ``modulus`` (implies irreducible and primitive)."""

        n = p**s - 1
        one = [1] + [0] * (s - 1)
        if PrimeExtensionField._xpow(n, modulus, p, s) != one:
            return False
        m, r, primes = n, 2, []
        while r * r <= m:
            if m % r == 0:
                primes.append(r)
                while m % r == 0:
                    m //= r
            r += 1
        if m > 1:
            primes.append(m)
        return all(PrimeExtensionField._xpow(n // q, modulus, p, s) != one for q in primes)

    @staticmethod
    def _power_table(modulus: tuple[int, ...], p: int, s: int) -> np.ndarray:
        """Integer codes of ``x^k`` for ``k = 0 .. p^s - 2``, streamed into one int64 array."""

        n = p**s - 1
        out = np.empty(n, dtype=np.int64)
        current = [1] + [0] * (s - 1)
        weights = [p**k for k in range(s)]
        mod = list(modulus)
        for idx in range(n):
            out[idx] = sum(c * w for c, w in zip(current, weights))
            lead = current[-1]
            current = [0] + current[:-1]
            if lead:
                for k in range(s):
                    current[k] = (current[k] - lead * mod[k]) % p
        return out

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def _power_digits(modulus: tuple[int, ...], p: int, s: int) -> list[list[int]]:
        """Digits of ``x^k`` for ``k = 0, 1, ...`` until ``1`` recurs, or empty on a short cycle."""

        one = [1] + [0] * (s - 1)
        current = list(one)
        out = [list(current)]
        limit = p**s - 1
        for _ in range(limit):
            lead = current[-1]
            shifted = [0] + current[:-1]
            if lead:
                for k in range(s):
                    shifted[k] = (shifted[k] - lead * modulus[k]) % p
            current = shifted
            if current == one:
                break
            out.append(list(current))
        else:
            return []
        if len(out) != limit:
            return []
        return out

    @staticmethod
    def find_primitive_modulus(p: int, s: int) -> tuple[int, ...]:
        """First monic degree-``s`` polynomial (lexicographic constants) with ``x`` of full order.

        Full order ``p^s - 1`` certifies irreducibility and primitivity at once.
        """

        for tail in itertools.product(range(p), repeat=s):
            if tail[0] == 0:
                continue
            modulus = tuple(tail) + (1,)
            if PrimeExtensionField._is_primitive(modulus, p, s):
                return modulus
        raise ValueError("no primitive polynomial found")  # pragma: no cover

    # -- coordinates -----------------------------------------------------------
    def coordinates(self, a) -> np.ndarray:
        return self.DIGITS[np.asarray(a, dtype=np.int64)]

    def from_coordinates(self, digits) -> np.ndarray:
        digits = np.asarray(digits, dtype=np.int64) % self.p
        return digits @ self.WEIGHTS

    # -- arithmetic ------------------------------------------------------------
    def add(self, a, b) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        if self.p == 2:  # characteristic 2: coordinates are bits, addition is XOR
            return a ^ b
        if self.s == 1:  # prime field: plain modular arithmetic
            return (a + b) % self.p
        return self.from_coordinates(self.DIGITS[a] + self.DIGITS[b])

    def neg(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        if self.p == 2:
            return a.copy()
        if self.s == 1:
            return (-a) % self.p
        return self.from_coordinates(-self.DIGITS[a])

    def sub(self, a, b) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        if self.p == 2:
            return a ^ b
        if self.s == 1:
            return (a - b) % self.p
        return self.from_coordinates(self.DIGITS[a] - self.DIGITS[b])

    def mul(self, a, b) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        zero = (a == 0) | (b == 0)
        logs = (self.LOG[a] + self.LOG[b]) % (self.order - 1)
        return np.where(zero, 0, self.EXP[logs])

    def inv(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        if np.any(a == 0):
            raise ZeroDivisionError("inverse of zero in GF(p^s)")
        return self.EXP[(-self.LOG[a]) % (self.order - 1)]

    def pow(self, a, k: int) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        k = int(k)
        if k == 0:
            return np.ones_like(a)
        if k < 0:
            return self.pow(self.inv(a), -k)
        logs = (self.LOG[a] * k) % (self.order - 1)
        return np.where(a == 0, 0, self.EXP[logs])

    def frobenius(self, a) -> np.ndarray:
        return self.pow(a, self.p)

    def is_subfield(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        return np.all(self.DIGITS[a][..., 1:] == 0, axis=-1)

    def sum(self, terms, axis: int) -> np.ndarray:
        """Field sum of an array of elements along ``axis``."""

        terms = np.asarray(terms, dtype=np.int64)
        return self.from_coordinates(self.DIGITS[terms].sum(axis=axis))


@functools.lru_cache(maxsize=None)
@functools.lru_cache(maxsize=4)
def get_field(p: int, s: int) -> PrimeExtensionField:
    return PrimeExtensionField(p, s)


# ---------------------------------------------------------------------------
# prime-field linear algebra (numpy; float64 matmul is exact while n (p-1)^2 < 2^53)
# ---------------------------------------------------------------------------
def matmul_mod_p(a, b, p: int) -> np.ndarray:
    product = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)
    return (product % p).astype(np.int64)


def _inverse_table(p: int) -> np.ndarray:
    return np.array([0] + [pow(a, p - 2, p) for a in range(1, p)], dtype=np.int64)


def _rref_mod_p(matrix, p: int) -> tuple[np.ndarray, list[int]]:
    a = np.asarray(matrix, dtype=np.int64).copy() % p
    rows, cols = a.shape
    inv = _inverse_table(p)
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
        a[r] = (a[r] * inv[a[r, c]]) % p
        factors = a[:, c].copy()
        factors[r] = 0
        idx = np.nonzero(factors)[0]
        if len(idx):
            a[idx] = (a[idx] - factors[idx, None] * a[r][None, :]) % p
        pivots.append(c)
        r += 1
    return a, pivots


def rank_mod_p(matrix, p: int) -> int:
    _, pivots = _rref_mod_p(matrix, p)
    return len(pivots)


def inverse_mod_p(matrix, p: int) -> np.ndarray | None:
    a = np.asarray(matrix, dtype=np.int64)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("inverse needs a square matrix")
    aug = np.concatenate([a % p, np.eye(n, dtype=np.int64)], axis=1)
    reduced, pivots = _rref_mod_p(aug, p)
    if pivots[:n] != list(range(n)):
        return None
    return reduced[:, n:]


def kernel_basis_mod_p(matrix, p: int) -> np.ndarray:
    """Rows span the right kernel ``{x : matrix x = 0}`` over F_p."""

    a = np.asarray(matrix, dtype=np.int64)
    cols = a.shape[1]
    reduced, pivots = _rref_mod_p(a, p)
    free = [c for c in range(cols) if c not in set(pivots)]
    basis = np.zeros((len(free), cols), dtype=np.int64)
    for index, f in enumerate(free):
        basis[index, f] = 1
        for row, c in enumerate(pivots):
            basis[index, c] = (-reduced[row, f]) % p
    return basis


def random_invertible_mod_p(n: int, p: int, rng) -> np.ndarray:
    while True:
        candidate = rng.integers(0, p, (n, n), dtype=np.int64)
        if inverse_mod_p(candidate, p) is not None:
            return candidate


def solve_square_over_field(field: PrimeExtensionField, matrix, rhs) -> np.ndarray | None:
    """Solve ``matrix y = rhs`` over ``GF(p^s)``; ``None`` if singular."""

    a = np.asarray(matrix, dtype=np.int64).copy()
    b = np.asarray(rhs, dtype=np.int64).copy()
    n = a.shape[0]
    if a.shape != (n, n) or b.shape != (n,):
        raise ValueError("solve needs a square system")
    aug = np.concatenate([a, b[:, None]], axis=1)
    for c in range(n):
        nz = np.nonzero(aug[c:, c])[0]
        if len(nz) == 0:
            return None
        piv = c + int(nz[0])
        if piv != c:
            aug[[c, piv]] = aug[[piv, c]]
        aug[c] = field.mul(aug[c], field.inv(aug[c, c]))
        factors = aug[:, c].copy()
        factors[c] = 0
        idx = np.nonzero(factors)[0]
        if len(idx):
            aug[idx] = field.sub(aug[idx], field.mul(factors[idx, None], aug[c][None, :]))
    return aug[:, n]


# ---------------------------------------------------------------------------
# the instance
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AlternantTrapdoorInstance:
    p: int
    s: int
    m: int
    r: int
    seed: int
    modulus: tuple[int, ...]
    points: tuple[int, ...]
    multipliers: tuple[int, ...]
    H_p: np.ndarray
    S: np.ndarray
    permutation: np.ndarray
    B_pub: np.ndarray
    accept_sets: tuple[frozenset[int], ...]
    build_attempts: int

    @property
    def n(self) -> int:
        return self.s * self.r

    @property
    def alpha(self) -> float:
        return self.n / self.m

    @property
    def ell(self) -> int:
        return (self.r - 1) // 2

    @property
    def designed_distance(self) -> int:
        return self.r + 1

    @property
    def accepted_size(self) -> int:
        return len(self.accept_sets[0])

    @property
    def rho(self) -> float:
        return self.accepted_size / self.p

    @property
    def field(self) -> PrimeExtensionField:
        return get_field(self.p, self.s)

    def to_private_order(self, public_vector) -> np.ndarray:
        """Public row ``i`` carries private position ``permutation[i]``."""

        v = np.asarray(public_vector, dtype=np.int64)
        out = np.zeros_like(v)
        out[self.permutation] = v
        return out

    def to_public_order(self, private_vector) -> np.ndarray:
        v = np.asarray(private_vector, dtype=np.int64)
        return v[self.permutation]


def build_alternant_trapdoor_instance(
    *,
    m: int,
    seed: int,
    p: int = 11,
    s: int = 4,
    alpha: float | None = 0.12,
    r: int | None = None,
    accepted_size: int | None = None,
    max_attempts: int = 5,
) -> AlternantTrapdoorInstance:
    """Build an instance; certifies ``rank(H_p) = s r`` (generic dimension)."""

    field = get_field(p, s)
    q = field.order
    if r is None:
        if alpha is None:
            raise ValueError("give alpha or r")
        r = int(round(alpha * m / s))
    if r < 3:
        raise ValueError("r must be at least 3 so that ell >= 1")
    if s * r >= m:
        raise ValueError("s*r must be below m")
    if m > q - 1:
        raise ValueError("m must be at most p^s - 1 (evaluation points are nonzero)")
    if accepted_size is None:
        accepted_size = (p - 1) // 2
    if not 0 < accepted_size < p:
        raise ValueError("accepted size must lie strictly between 0 and p")

    rng = np.random.default_rng(seed)
    n = s * r
    attempts = 0
    while True:
        attempts += 1
        points = rng.choice(q - 1, m, replace=False).astype(np.int64) + 1
        multipliers = rng.integers(1, q, m, dtype=np.int64)
        log_a = field.LOG[points]
        log_v = field.LOG[multipliers]
        logs = (log_v[None, :] + np.arange(r, dtype=np.int64)[:, None] * log_a[None, :]) % (q - 1)
        H = field.EXP[logs]  # r x m over F
        H_p = field.DIGITS[H].transpose(0, 2, 1).reshape(n, m)  # row j*s + k = coordinate k of row j
        if rank_mod_p(H_p, p) == n:
            break
        if attempts >= max_attempts:
            raise RuntimeError("alternant parity check did not reach full rank")
    S = random_invertible_mod_p(n, p, rng)
    permutation = rng.permutation(m).astype(np.int64)
    B_pub = matmul_mod_p(H_p.T[permutation], S, p)
    accept_sets = tuple(
        frozenset(int(v) for v in rng.choice(p, accepted_size, replace=False)) for _ in range(m)
    )
    for array in (H_p, S, permutation, B_pub):
        array.setflags(write=False)
    return AlternantTrapdoorInstance(
        p=p,
        s=s,
        m=m,
        r=r,
        seed=seed,
        modulus=field.modulus,
        points=tuple(int(a) for a in points),
        multipliers=tuple(int(v) for v in multipliers),
        H_p=H_p,
        S=S,
        permutation=permutation,
        B_pub=B_pub,
        accept_sets=accept_sets,
        build_attempts=attempts,
    )


# ---------------------------------------------------------------------------
# the decoder (key holder only)
# ---------------------------------------------------------------------------
def syndromes(instance: AlternantTrapdoorInstance, word) -> np.ndarray:
    """The ``r`` GRS syndromes ``S_j = sum_i w_i v_i a_i^j`` of a word in private order."""

    field = instance.field
    q = field.order
    w = np.asarray(word, dtype=np.int64) % instance.p
    support = np.nonzero(w)[0]
    if len(support) == 0:
        return np.zeros(instance.r, dtype=np.int64)
    points = np.asarray(instance.points, dtype=np.int64)[support]
    mult = np.asarray(instance.multipliers, dtype=np.int64)[support]
    logs = (
        field.LOG[mult][None, :]
        + field.LOG[w[support]][None, :]
        + np.arange(instance.r, dtype=np.int64)[:, None] * field.LOG[points][None, :]
    ) % (q - 1)
    terms = field.EXP[logs]  # r x |support|
    return field.sum(terms, axis=1)


def berlekamp_massey(field: PrimeExtensionField, sequence) -> tuple[np.ndarray, int]:
    """Shortest linear recurrence ``S_n = -sum_{i=1}^L C_i S_{n-i}``; returns ``(C, L)``.

    ``C`` is given low -> high with ``C[0] = 1`` and is the error locator
    ``prod_k (1 - a_{i_k} x)`` when the sequence is a GRS syndrome sequence of at most
    ``len(sequence)/2`` errors.
    """

    S = np.asarray(sequence, dtype=np.int64)
    N = len(S)
    C = np.zeros(N + 1, dtype=np.int64)
    Bp = np.zeros(N + 1, dtype=np.int64)
    C[0] = 1
    Bp[0] = 1
    L = 0
    shift = 1
    b = 1
    for n in range(N):
        if L:
            products = field.mul(C[1 : L + 1], S[n - np.arange(1, L + 1)])
            d = int(field.add(S[n], field.sum(products, axis=0)))
        else:
            d = int(S[n])
        if d == 0:
            shift += 1
            continue
        coefficient = int(field.mul(d, field.inv(b)))
        correction = np.zeros(N + 1, dtype=np.int64)
        correction[shift:] = Bp[: N + 1 - shift]
        new_C = field.sub(C, field.mul(coefficient, correction))
        if 2 * L <= n:
            Bp = C
            C = new_C
            L = n + 1 - L
            b = d
            shift = 1
        else:
            C = new_C
            shift += 1
    return C[: L + 1], L


def locate_errors(instance: AlternantTrapdoorInstance, locator) -> np.ndarray:
    """Private positions ``i`` with ``locator(a_i^{-1}) = 0`` (Horner over all points)."""

    field = instance.field
    points = np.asarray(instance.points, dtype=np.int64)
    x = field.inv(points)
    value = np.zeros(instance.m, dtype=np.int64)
    for coefficient in reversed(np.asarray(locator, dtype=np.int64).tolist()):
        value = field.add(field.mul(value, x), coefficient)
    return np.nonzero(value == 0)[0]


def error_values(instance: AlternantTrapdoorInstance, positions, synd) -> np.ndarray | None:
    """Solve ``V y = S[:L]`` with ``V[j][k] = v_{i_k} a_{i_k}^j``; ``None`` unless valid F_p values."""

    field = instance.field
    q = field.order
    positions = np.asarray(positions, dtype=np.int64)
    L = len(positions)
    if L == 0:
        return np.zeros(0, dtype=np.int64)
    points = np.asarray(instance.points, dtype=np.int64)[positions]
    mult = np.asarray(instance.multipliers, dtype=np.int64)[positions]
    logs = (field.LOG[mult][None, :] + np.arange(L, dtype=np.int64)[:, None] * field.LOG[points][None, :]) % (q - 1)
    V = field.EXP[logs]
    values = solve_square_over_field(field, V, np.asarray(synd, dtype=np.int64)[:L])
    if values is None:
        return None
    if np.any(values == 0) or not np.all(field.is_subfield(values)):
        return None
    return values


def decode(instance: AlternantTrapdoorInstance, word, *, public_order: bool = False) -> tuple[int, ...] | None:
    """Recover the unique pattern of weight <= ell with the syndrome of ``word``, or ``None``.

    ``word`` may be an error pattern or a codeword plus an error; only its syndrome is used.
    The result is returned in the same order as the input.
    """

    w = np.asarray(word, dtype=np.int64) % instance.p
    if public_order:
        w = instance.to_private_order(w)
    synd = syndromes(instance, w)
    if not np.any(synd):
        e = np.zeros(instance.m, dtype=np.int64)
        return tuple(int(v) for v in (instance.to_public_order(e) if public_order else e))
    locator, L = berlekamp_massey(instance.field, synd)
    if L > instance.ell:
        return None
    positions = locate_errors(instance, locator)
    if len(positions) != L:
        return None
    values = error_values(instance, positions, synd)
    if values is None:
        return None
    e = np.zeros(instance.m, dtype=np.int64)
    e[positions] = values
    if not np.array_equal(syndromes(instance, e), synd):
        return None
    if public_order:
        e = instance.to_public_order(e)
    return tuple(int(v) for v in e)


def random_error_pattern(instance: AlternantTrapdoorInstance, weight: int, rng) -> np.ndarray:
    """Uniform support of the given weight, nonzero F_p values (private order)."""

    e = np.zeros(instance.m, dtype=np.int64)
    if weight:
        support = rng.choice(instance.m, weight, replace=False)
        e[support] = rng.integers(1, instance.p, weight)
    return e


def verify_decoder(instance: AlternantTrapdoorInstance, trials: int, seed: int, *, beyond_trials: int = 10) -> dict:
    """Exact recovery of random patterns of weight <= ell; beyond-radius patterns are only classified."""

    rng = np.random.default_rng(seed)
    t0 = time.time()
    by_weight: dict[int, list[int]] = {}
    recovered = 0
    rejected = 0
    wrong = 0
    for trial in range(trials):
        weight = instance.ell if trial % 2 == 0 else int(rng.integers(0, instance.ell))
        e = random_error_pattern(instance, weight, rng)
        result = decode(instance, e)
        entry = by_weight.setdefault(weight, [0, 0])
        entry[0] += 1
        if result is None:
            rejected += 1
        elif result == tuple(int(v) for v in e):
            recovered += 1
            entry[1] += 1
        else:
            wrong += 1
    beyond = {"weight": instance.ell + 1, "trials": beyond_trials, "rejected": 0, "recovered": 0, "wrong": 0}
    for _ in range(beyond_trials):
        e = random_error_pattern(instance, instance.ell + 1, rng)
        result = decode(instance, e)
        if result is None:
            beyond["rejected"] += 1
        elif result == tuple(int(v) for v in e):
            beyond["recovered"] += 1
        else:
            beyond["wrong"] += 1
    return {
        "trials": trials,
        "ell": instance.ell,
        "exact_recoveries": recovered,
        "all_recovered": recovered == trials,
        "by_weight": {str(k): v for k, v in sorted(by_weight.items())},
        "failures": {"rejected": rejected, "wrong": wrong},
        "beyond_radius": beyond,
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# exporters
# ---------------------------------------------------------------------------
def as_dense_arrays(instance: AlternantTrapdoorInstance) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(B, sets)`` for the numpy controls: ``B`` is ``m x n`` int64, sets sorted int64 arrays."""

    B = np.array(instance.B_pub, dtype=np.int64)
    sets = [np.array(sorted(s), dtype=np.int64) for s in instance.accept_sets]
    return B, sets


def square_code_note(instance: AlternantTrapdoorInstance) -> dict:
    """Counting note on the square-code distinguisher; not a theorem.

    The public optimizer code has dimension ``n = s r`` and a random code of that dimension
    has square dimension ``min(m, n(n+1)/2)``.  Over the extension field the optimizer code
    is a sum of ``s`` Frobenius-conjugate GRS codes of dimension ``r``, so its square is
    spanned by ``s`` codes of dimension ``2r - 1`` and ``C(s,2)`` cross products of dimension
    at most ``r^2``; when that count exceeds ``m`` the square generically fills ``F_p^m`` and
    the square-code distinguisher (Faugere et al.) has nothing to see.
    """

    n = instance.n
    r = instance.r
    s = instance.s
    generic = n * (n + 1) // 2
    bound = s * (2 * r - 1) + comb(s, 2) * r * r
    return {
        "m": instance.m,
        "n": n,
        "generic_square_dimension": generic,
        "trace_grs_square_bound": bound,
        "square_distinguisher_applicable": bound < instance.m,
        "note": (
            "square-code distinguisher needs the structured square to be smaller than m; "
            "here both counts exceed m, so the square generically fills the space (counting "
            "note, not a proof; the trace-GRS bound equals the square-distinguishability "
            "threshold C(rs+1,2) - s C(r-1,2) of Lemoine's tangent-space attack 2505.10184, "
            "which therefore does not apply either; the syzygy distinguisher of "
            "Randriambololona 2407.15740 is the relevant subexponential attack and is not run here)"
        ),
    }


__all__ = [
    "AlternantTrapdoorInstance",
    "PrimeExtensionField",
    "as_dense_arrays",
    "berlekamp_massey",
    "build_alternant_trapdoor_instance",
    "decode",
    "error_values",
    "get_field",
    "inverse_mod_p",
    "kernel_basis_mod_p",
    "locate_errors",
    "matmul_mod_p",
    "polynomial_is_irreducible",
    "random_error_pattern",
    "random_invertible_mod_p",
    "rank_mod_p",
    "solve_square_over_field",
    "square_code_note",
    "syndromes",
    "verify_decoder",
]
