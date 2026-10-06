"""Hidden-key instances whose secret is not a subfield subcode.

The problem in both cases.  Someone publishes a dense matrix ``B`` (``m`` rows, ``n``
columns) over a prime field ``F_p`` and one accepted set of values per row, and asks for
``x`` making ``(Bx)_i`` land in its set for as many rows as possible.  ``B`` is the
transpose of a scrambled parity-check matrix of a Reed--Solomon code that has been hidden
by a secret transformation, so the key holder decodes ``ker(B^T)`` and can run DQI, while
everyone else sees a dense matrix.  The alternant family of ``alternant_trapdoor.py``
hides the code inside a subfield; the two families here hide it in other ways.

Sparse-mixed key (``sparse_mixed``; the BBCRS idea, Baldi--Bianchi--Chiaraluce--Rosenthal--
Schipani, J. Cryptology 2016).  A generalized Reed--Solomon parity check ``H`` (``r x m``)
is multiplied on the right by a secret invertible ``T`` whose columns have weight exactly
``w``: the private coordinates are grouped into blocks of ``w`` and each block is mixed by a
secret invertible ``w x w`` matrix with no zero entries.  The public matrix is
``H_pub = S H T Pi`` for a random invertible ``S`` and a column permutation ``Pi``, and
``B = H_pub^T``.  Every public column mixes ``w`` GRS columns.  For a pattern ``y`` of
weight ``ell``, ``T Pi y`` has weight at most ``w ell``, so the key holder decodes the GRS
syndrome with Berlekamp--Massey and applies ``T^-1``: a worst-case decoder to
``ell = floor((r - 1) / 2) // w`` errors, reach ``alpha / (2 w)`` at ``alpha = r / m``.
The published attack (Couvreur--Gauthier-Umana--Otmani--Tillich, Asiacrypt 2014,
arXiv 1501.03736) needs the average column weight of ``T`` below ``1 + R`` (``R`` the
code rate), i.e. many columns of weight one; here every column has weight ``w >= 2``.

Column-inserted key (``column_inserted``; the RLCE idea, Wang 2016).  A GRS code
``[m0, k]`` is extended by ``u`` coordinates that are dense linear functionals of the
codeword (``r = c Phi`` for a secret dense ``Phi``), each inserted coordinate is paired
with a distinct GRS coordinate and the pair is mixed by a secret invertible ``2 x 2``
matrix, then the coordinates are permuted and the parity check is scrambled.  The public
length is ``m = m0 + u`` and ``n = (m0 - k) + u``.  An error on either coordinate of a
mixed pair becomes at most one GRS error after unmixing, so the key holder decodes the GRS
part, recomputes the inserted coordinates and recovers the full pattern: a worst-case
decoder to ``ell = floor((m0 - k - 1) / 2)`` errors, reach ``(m0 - k) / (2 m)``.  At
``u = m0 - k`` this is ``alpha / 4``.  The published attack (Couvreur--Lequesne--Tillich,
PQCrypto 2019, arXiv 1805.11489) recovers the key for ``u`` small relative to ``m0 - k``.

Conventions follow ``alternant_trapdoor.py``: public row ``i`` carries private position
``permutation[i]``; the GRS decoder is that module's Berlekamp--Massey pipeline on the
prime field (``get_field(p, 1)``).  Nothing here claims hardness; the numeric square-code
battery below reports what a Schur-product distinguisher sees on the instance.
"""

from __future__ import annotations

import dataclasses
import time
from math import comb

import numpy as np

from .alternant_trapdoor import (
    decode as _grs_decode,
    get_field,
    kernel_basis_mod_p,
    matmul_mod_p,
    rank_mod_p,
    random_invertible_mod_p,
)


# ---------------------------------------------------------------------------
# a prime-field GRS key, decodable by the alternant module's pipeline (s = 1)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GRSKey:
    """``H[j][i] = v_i a_i^j`` over ``F_p``, ``r`` rows, distinct nonzero points."""

    p: int
    m: int
    r: int
    points: tuple[int, ...]
    multipliers: tuple[int, ...]

    @property
    def s(self) -> int:
        return 1

    @property
    def ell(self) -> int:
        return (self.r - 1) // 2

    @property
    def field(self):
        return get_field(self.p, 1)

    def parity_check(self) -> np.ndarray:
        field = self.field
        q = field.order
        log_a = field.LOG[np.asarray(self.points, dtype=np.int64)]
        log_v = field.LOG[np.asarray(self.multipliers, dtype=np.int64)]
        logs = (log_v[None, :] + np.arange(self.r, dtype=np.int64)[:, None] * log_a[None, :]) % (q - 1)
        return field.EXP[logs]

    def decode(self, word) -> np.ndarray | None:
        """Unique error pattern of weight <= ell with the syndrome of ``word``, else ``None``."""

        result = _grs_decode(self, word)
        return None if result is None else np.asarray(result, dtype=np.int64)


def random_grs_key(p: int, m: int, r: int, rng) -> GRSKey:
    if m > p - 1:
        raise ValueError("a prime-field GRS code needs m <= p - 1 (nonzero evaluation points)")
    if r < 3 or r >= m:
        raise ValueError("r must satisfy 3 <= r < m")
    points = rng.choice(p - 1, m, replace=False).astype(np.int64) + 1
    multipliers = rng.integers(1, p, m, dtype=np.int64)
    return GRSKey(p=p, m=m, r=r, points=tuple(int(a) for a in points), multipliers=tuple(int(v) for v in multipliers))


# ---------------------------------------------------------------------------
# sparse-mixed key
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SparseMixedKeyInstance:
    kind: str
    p: int
    m: int
    r: int
    w: int
    seed: int
    grs: GRSKey
    blocks: np.ndarray  # (m // w) x w private (T-column) positions
    mixers: np.ndarray  # (m // w) x w x w, invertible, no zero entries
    mixers_inv: np.ndarray
    S: np.ndarray
    permutation: np.ndarray
    B_pub: np.ndarray
    accept_sets: tuple[frozenset[int], ...]

    @property
    def n(self) -> int:
        return self.r

    @property
    def alpha(self) -> float:
        return self.n / self.m

    @property
    def ell(self) -> int:
        return self.grs.ell // self.w

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
    def reach_law(self) -> str:
        return "alpha/(2w)"

    def to_private_order(self, public_vector) -> np.ndarray:
        v = np.asarray(public_vector, dtype=np.int64)
        out = np.zeros_like(v)
        out[self.permutation] = v
        return out

    def to_public_order(self, private_vector) -> np.ndarray:
        return np.asarray(private_vector, dtype=np.int64)[self.permutation]

    def apply_T(self, private_vector) -> np.ndarray:
        """``z = T y`` blockwise: ``z[block] = A_b @ y[block]``."""

        y = np.asarray(private_vector, dtype=np.int64) % self.p
        z = np.zeros_like(y)
        yb = y[self.blocks]  # nb x w
        zb = np.einsum("bij,bj->bi", self.mixers, yb) % self.p
        z[self.blocks] = zb
        return z

    def apply_T_inverse(self, private_vector) -> np.ndarray:
        z = np.asarray(private_vector, dtype=np.int64) % self.p
        y = np.zeros_like(z)
        zb = z[self.blocks]
        yb = np.einsum("bij,bj->bi", self.mixers_inv, zb) % self.p
        y[self.blocks] = yb
        return y

    def decode(self, word, *, public_order: bool = True) -> np.ndarray | None:
        """Unique pattern of weight <= ell with the syndrome of ``word`` (public order), else ``None``."""

        w = np.asarray(word, dtype=np.int64) % self.p
        y = self.to_private_order(w) if public_order else w
        z = self.apply_T(y)
        e_z = self.grs.decode(z)
        if e_z is None:
            return None
        e = self.apply_T_inverse(e_z)
        if int(np.count_nonzero(e)) > self.ell:
            return None
        return self.to_public_order(e) if public_order else e


def _invertible_dense_block(w: int, p: int, rng) -> tuple[np.ndarray, np.ndarray]:
    from .alternant_trapdoor import inverse_mod_p

    while True:
        a = rng.integers(1, p, (w, w), dtype=np.int64)
        inv = inverse_mod_p(a, p)
        if inv is not None:
            return a, inv


def build_sparse_mixed_key_instance(
    *,
    m: int,
    seed: int,
    p: int,
    alpha: float | None = 0.07,
    r: int | None = None,
    w: int = 2,
    accepted_size: int | None = None,
) -> SparseMixedKeyInstance:
    """``m`` is rounded down to a multiple of ``w``; ``r = round(alpha m)`` unless given."""

    if w < 1:
        raise ValueError("w must be positive")
    m = (m // w) * w
    if r is None:
        if alpha is None:
            raise ValueError("give alpha or r")
        r = int(round(alpha * m))
    if accepted_size is None:
        accepted_size = (p - 1) // 2
    if not 0 < accepted_size < p:
        raise ValueError("accepted size must lie strictly between 0 and p")
    rng = np.random.default_rng(seed)
    grs = random_grs_key(p, m, r, rng)
    if grs.ell // w < 1:
        raise ValueError("r too small for a positive licensed radius")
    H = grs.parity_check()  # r x m
    order = rng.permutation(m).astype(np.int64)
    blocks = order.reshape(m // w, w)
    mixers = np.zeros((m // w, w, w), dtype=np.int64)
    mixers_inv = np.zeros_like(mixers)
    for b in range(m // w):
        mixers[b], mixers_inv[b] = _invertible_dense_block(w, p, rng)
    # H T: column i of block b is sum_{i'} H[:, i'] * A_b[i', i]
    HT = np.zeros_like(H)
    for b in range(m // w):
        cols = blocks[b]
        HT[:, cols] = matmul_mod_p(H[:, cols], mixers[b], p)
    S = random_invertible_mod_p(r, p, rng)
    permutation = rng.permutation(m).astype(np.int64)
    H_pub = matmul_mod_p(S, HT, p)[:, permutation]
    B_pub = np.ascontiguousarray(H_pub.T)
    accept_sets = tuple(frozenset(int(v) for v in rng.choice(p, accepted_size, replace=False)) for _ in range(m))
    for array in (blocks, mixers, mixers_inv, S, permutation, B_pub):
        array.setflags(write=False)
    return SparseMixedKeyInstance(
        kind="sparse_mixed",
        p=p,
        m=m,
        r=r,
        w=w,
        seed=seed,
        grs=grs,
        blocks=blocks,
        mixers=mixers,
        mixers_inv=mixers_inv,
        S=S,
        permutation=permutation,
        B_pub=B_pub,
        accept_sets=accept_sets,
    )


# ---------------------------------------------------------------------------
# column-inserted key
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ColumnInsertedKeyInstance:
    kind: str
    p: int
    m: int
    m0: int
    k: int
    u: int
    seed: int
    grs: GRSKey
    Phi: np.ndarray  # m0 x u
    pairs: np.ndarray  # u distinct GRS positions
    mixers: np.ndarray  # u x 2 x 2
    mixers_inv: np.ndarray
    S: np.ndarray
    permutation: np.ndarray
    B_pub: np.ndarray
    accept_sets: tuple[frozenset[int], ...]

    @property
    def r(self) -> int:
        return self.m0 - self.k

    @property
    def n(self) -> int:
        return self.r + self.u

    @property
    def alpha(self) -> float:
        return self.n / self.m

    @property
    def ell(self) -> int:
        return self.grs.ell

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
    def u_ratio(self) -> float:
        return self.u / self.r

    @property
    def reach_law(self) -> str:
        return "(m0-k)/(2m) = alpha/(2(1+u_ratio))"

    def to_private_order(self, public_vector) -> np.ndarray:
        v = np.asarray(public_vector, dtype=np.int64)
        out = np.zeros_like(v)
        out[self.permutation] = v
        return out

    def to_public_order(self, private_vector) -> np.ndarray:
        return np.asarray(private_vector, dtype=np.int64)[self.permutation]

    def mix(self, unmixed) -> np.ndarray:
        """``(y_a, y_ins) -> (y_a, y_ins) A_j`` on every pair (row-vector convention)."""

        y = np.asarray(unmixed, dtype=np.int64) % self.p
        out = y.copy()
        pair_vals = np.stack([y[self.pairs], y[self.m0 + np.arange(self.u)]], axis=1)  # u x 2
        mixed = np.einsum("uj,ujk->uk", pair_vals, self.mixers) % self.p
        out[self.pairs] = mixed[:, 0]
        out[self.m0 + np.arange(self.u)] = mixed[:, 1]
        return out

    def unmix(self, mixed) -> np.ndarray:
        y = np.asarray(mixed, dtype=np.int64) % self.p
        out = y.copy()
        pair_vals = np.stack([y[self.pairs], y[self.m0 + np.arange(self.u)]], axis=1)
        unm = np.einsum("uj,ujk->uk", pair_vals, self.mixers_inv) % self.p
        out[self.pairs] = unm[:, 0]
        out[self.m0 + np.arange(self.u)] = unm[:, 1]
        return out

    def decode(self, word, *, public_order: bool = True) -> np.ndarray | None:
        w = np.asarray(word, dtype=np.int64) % self.p
        ym = self.to_private_order(w) if public_order else w
        y = self.unmix(ym)
        g = y[: self.m0]
        e_g = self.grs.decode(g)
        if e_g is None:
            return None
        c_g = (g - e_g) % self.p
        c_ins = matmul_mod_p(c_g[None, :], self.Phi, self.p)[0]
        c_mixed = self.mix(np.concatenate([c_g, c_ins]))
        e = (ym - c_mixed) % self.p
        if int(np.count_nonzero(e)) > self.ell:
            return None
        return self.to_public_order(e) if public_order else e


def build_column_inserted_key_instance(
    *,
    m: int,
    seed: int,
    p: int,
    alpha: float | None = 0.07,
    u_ratio: float = 1.0,
    r: int | None = None,
    accepted_size: int | None = None,
) -> ColumnInsertedKeyInstance:
    """Public length ``m = m0 + u`` with ``u = round(u_ratio r)`` and ``n = r + u = round(alpha m)``."""

    if accepted_size is None:
        accepted_size = (p - 1) // 2
    if not 0 < accepted_size < p:
        raise ValueError("accepted size must lie strictly between 0 and p")
    if r is None:
        if alpha is None:
            raise ValueError("give alpha or r")
        r = int(round(alpha * m / (1 + u_ratio)))
    u = int(round(u_ratio * r))
    m0 = m - u
    if u > m0:
        raise ValueError("more inserted coordinates than GRS coordinates")
    rng = np.random.default_rng(seed)
    grs = random_grs_key(p, m0, r, rng)
    H = grs.parity_check()  # r x m0
    k = m0 - r
    Phi = rng.integers(0, p, (m0, u), dtype=np.int64)
    pairs = rng.choice(m0, u, replace=False).astype(np.int64)
    mixers = np.zeros((u, 2, 2), dtype=np.int64)
    mixers_inv = np.zeros_like(mixers)
    for j in range(u):
        mixers[j], mixers_inv[j] = _invertible_dense_block(2, p, rng)
    n = r + u
    # unmixed parity check H' = [[H, 0], [-Phi^T, I_u]]
    Hp = np.zeros((n, m0 + u), dtype=np.int64)
    Hp[:r, :m0] = H
    Hp[r:, :m0] = (-Phi.T) % p
    Hp[r:, m0:] = np.eye(u, dtype=np.int64)
    # mixing: the check columns of a pair transform by A_j^{-T}
    for j in range(u):
        a, b = int(pairs[j]), m0 + j
        cols = np.stack([Hp[:, a], Hp[:, b]], axis=1)  # n x 2
        new = matmul_mod_p(cols, mixers_inv[j].T, p)
        Hp[:, a] = new[:, 0]
        Hp[:, b] = new[:, 1]
    S = random_invertible_mod_p(n, p, rng)
    permutation = rng.permutation(m0 + u).astype(np.int64)
    H_pub = matmul_mod_p(S, Hp, p)[:, permutation]
    B_pub = np.ascontiguousarray(H_pub.T)
    accept_sets = tuple(frozenset(int(v) for v in rng.choice(p, accepted_size, replace=False)) for _ in range(m0 + u))
    for array in (Phi, pairs, mixers, mixers_inv, S, permutation, B_pub):
        array.setflags(write=False)
    return ColumnInsertedKeyInstance(
        kind="column_inserted",
        p=p,
        m=m0 + u,
        m0=m0,
        k=k,
        u=u,
        seed=seed,
        grs=grs,
        Phi=Phi,
        pairs=pairs,
        mixers=mixers,
        mixers_inv=mixers_inv,
        S=S,
        permutation=permutation,
        B_pub=B_pub,
        accept_sets=accept_sets,
    )


HiddenKeyInstance = SparseMixedKeyInstance | ColumnInsertedKeyInstance


def build_hidden_key_instance(kind: str, **kwargs) -> HiddenKeyInstance:
    if kind == "sparse_mixed":
        return build_sparse_mixed_key_instance(**kwargs)
    if kind == "column_inserted":
        return build_column_inserted_key_instance(**kwargs)
    raise ValueError(f"unknown hidden-key kind {kind!r}")


# ---------------------------------------------------------------------------
# verification and exporters
# ---------------------------------------------------------------------------
def random_error_pattern(instance: HiddenKeyInstance, weight: int, rng) -> np.ndarray:
    """Uniform support of the given weight, nonzero F_p values (public order)."""

    e = np.zeros(instance.m, dtype=np.int64)
    if weight:
        support = rng.choice(instance.m, weight, replace=False)
        e[support] = rng.integers(1, instance.p, weight)
    return e


def verify_decoder(instance: HiddenKeyInstance, trials: int, seed: int, *, beyond_trials: int = 10) -> dict:
    """Exact recovery of random public-order patterns of weight <= ell; beyond-radius patterns classified.

    Each trial adds a random codeword of ``ker(B^T)`` to the pattern so that the decoder is
    exercised on a received word, not only on the syndrome path.
    """

    rng = np.random.default_rng(seed)
    t0 = time.time()
    p = instance.p
    B = np.asarray(instance.B_pub, dtype=np.int64)
    # Tested on error patterns alone and, when m is small enough for a kernel basis, on pattern + random kernel vector.
    kernel = kernel_basis_mod_p(B.T, p) if instance.m <= 1200 else None
    by_weight: dict[int, list[int]] = {}
    recovered = rejected = wrong = 0
    for trial in range(trials):
        weight = instance.ell if trial % 2 == 0 else int(rng.integers(0, instance.ell))
        e = random_error_pattern(instance, weight, rng)
        word = e
        if kernel is not None and len(kernel):
            coeffs = rng.integers(0, p, len(kernel), dtype=np.int64)
            word = (e + matmul_mod_p(coeffs[None, :], kernel, p)[0]) % p
        result = instance.decode(word)
        entry = by_weight.setdefault(weight, [0, 0])
        entry[0] += 1
        if result is None:
            rejected += 1
        elif np.array_equal(result, e):
            recovered += 1
            entry[1] += 1
        else:
            wrong += 1
    beyond = {"weight": instance.ell + 1, "trials": beyond_trials, "rejected": 0, "recovered": 0, "wrong": 0}
    for _ in range(beyond_trials):
        e = random_error_pattern(instance, instance.ell + 1, rng)
        result = instance.decode(e)
        if result is None:
            beyond["rejected"] += 1
        elif np.array_equal(result, e):
            beyond["recovered"] += 1
        else:
            beyond["wrong"] += 1
    return {
        "trials": trials,
        "ell": instance.ell,
        "with_codewords": kernel is not None,
        "exact_recoveries": recovered,
        "all_recovered": recovered == trials,
        "by_weight": {str(k): v for k, v in sorted(by_weight.items())},
        "failures": {"rejected": rejected, "wrong": wrong},
        "beyond_radius": beyond,
        "seconds": time.time() - t0,
    }


def as_dense_arrays(instance: HiddenKeyInstance) -> tuple[np.ndarray, list[np.ndarray]]:
    B = np.array(instance.B_pub, dtype=np.int64)
    sets = [np.array(sorted(s), dtype=np.int64) for s in instance.accept_sets]
    return B, sets


# ---------------------------------------------------------------------------
# the square-code battery (what a Schur-product distinguisher sees)
# ---------------------------------------------------------------------------
def _square_rank(generator: np.ndarray, p: int, rng, extra: int = 100) -> dict:
    """Rank of ``length + extra`` Schur products of pairs of *random codewords* of the code.

    Products of random pairs of basis vectors are not generic (a systematic basis makes them
    vanish off the pivot columns), so each factor is a uniformly random combination of the
    basis.  The float64 matmul is exact while ``dim (p-1)^2 < 2^53``.
    """

    dim, length = generator.shape
    count = length + extra
    if dim == 0:
        return {"dimension": 0, "length": int(length), "products": 0, "square_rank": 0, "random_expectation": 0, "full": True}
    left = matmul_mod_p(rng.integers(0, p, (count, dim), dtype=np.int64), generator, p)
    right = matmul_mod_p(rng.integers(0, p, (count, dim), dtype=np.int64), generator, p)
    products = (left * right) % p
    rank = int(rank_mod_p(products, p))
    expectation = int(min(length, comb(dim + 1, 2)))
    return {
        "dimension": int(dim),
        "length": int(length),
        "products": int(count),
        "square_rank": rank,
        "random_expectation": expectation,
        "full": rank == expectation,
    }


def _shorten(generator: np.ndarray, positions: np.ndarray, p: int) -> np.ndarray:
    """Generator of the code shortened at ``positions`` (codewords vanishing there, restricted)."""

    if len(positions) == 0:
        return generator
    sub = kernel_basis_mod_p(generator[:, positions].T, p)  # combinations vanishing on positions
    shortened = matmul_mod_p(sub, generator, p)
    keep = np.setdiff1d(np.arange(generator.shape[1]), positions)
    return shortened[:, keep]


def square_battery(instance: HiddenKeyInstance, rng, *, shortenings: tuple[int, ...] | str = "auto", sides=("image", "kernel")) -> dict:
    """Square-code ranks of ``im(B)`` and ``ker(B^T)``, plain and shortened.

    A distinguisher of Schur-product type fires when a rank falls short of the random
    expectation ``min(length, C(dim+1, 2))``.  For a plain hidden GRS (the ``w = 1`` or
    ``u = 0`` control) the image side collapses to ``2 dim - 1``.

    ``shortenings="auto"`` uses ``0`` and, per side, the shortening ``a = D - d`` that brings
    the dimension ``D`` down to the largest ``d`` with ``d (d + 3) / 2 <= L - D`` (length
    ``L``), so that the random square ``C(d+1, 2)`` fits strictly inside the shortened
    length and any structure shows as a deficit (the regime of the shortened-square attacks
    on RLCE, Couvreur--Lequesne--Tillich 2019).
    """

    p = instance.p
    B = np.asarray(instance.B_pub, dtype=np.int64)
    out: dict = {"m": int(instance.m), "n": int(instance.n)}
    generators = {}
    if "image" in sides:
        generators["image"] = np.ascontiguousarray(B.T)  # n x m
    if "kernel" in sides:
        generators["kernel"] = kernel_basis_mod_p(B.T, p)  # (m - n) x m
    for side, gen in generators.items():
        if shortenings == "auto":
            D, L = gen.shape
            d = 0
            while (d + 1) * (d + 4) // 2 <= L - D:
                d += 1
            plan = (0, D - d) if 0 < D - d < D else (0,)
        else:
            plan = tuple(shortenings)
        for a in plan:
            positions = np.sort(rng.choice(instance.m, a, replace=False)) if a else np.zeros(0, dtype=np.int64)
            g = _shorten(gen, positions, p)
            out[f"{side}_shorten_{a}"] = _square_rank(g, p, rng)
    return out


def random_matrix_battery(m: int, n: int, p: int, rng, *, shortenings: tuple[int, ...] | str = "auto", sides=("image", "kernel")) -> dict:
    """The same battery on a uniformly random ``m x n`` matrix of full rank (calibration)."""

    while True:
        B = rng.integers(0, p, (m, n), dtype=np.int64)
        if rank_mod_p(B, p) == n:
            break

    class _Shell:
        pass

    shell = _Shell()
    shell.p, shell.m, shell.n, shell.B_pub = p, m, n, B
    return square_battery(shell, rng, shortenings=shortenings, sides=sides)


__all__ = [
    "ColumnInsertedKeyInstance",
    "GRSKey",
    "HiddenKeyInstance",
    "SparseMixedKeyInstance",
    "as_dense_arrays",
    "build_column_inserted_key_instance",
    "build_hidden_key_instance",
    "build_sparse_mixed_key_instance",
    "random_error_pattern",
    "random_grs_key",
    "random_matrix_battery",
    "square_battery",
    "verify_decoder",
]
