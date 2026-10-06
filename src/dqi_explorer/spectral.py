"""Finite-size canonical DQI spectral optimization."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, nextafter, sqrt


@dataclass(frozen=True)
class QualityEstimate:
    value: float
    lower: float
    upper: float
    method: str
    coefficients: tuple[float, ...]


def _sturm_count_below(
    diagonal: tuple[float, ...],
    off_diagonal: tuple[float, ...],
    point: float,
) -> int:
    """Count eigenvalues below ``point`` using an LDL^T Sturm sequence."""

    scale = max(1.0, *(abs(value) for value in diagonal), *(abs(value) for value in off_diagonal))
    tiny = 1e-15 * scale
    pivot = diagonal[0] - point
    if abs(pivot) < tiny:
        pivot = -tiny
    count = int(pivot < 0)
    for index in range(1, len(diagonal)):
        pivot = diagonal[index] - point - off_diagonal[index - 1] ** 2 / pivot
        if abs(pivot) < tiny:
            pivot = -tiny
        count += int(pivot < 0)
    return count


def largest_tridiagonal_eigenvalue_interval(
    diagonal: tuple[float, ...],
    off_diagonal: tuple[float, ...],
    *,
    iterations: int = 180,
    roundoff_margin: float = 1e-12,
) -> tuple[float, float]:
    if roundoff_margin <= 0:
        raise ValueError("roundoff_margin must be positive")
    if not diagonal:
        raise ValueError("a matrix must have positive dimension")
    if len(off_diagonal) + 1 != len(diagonal):
        raise ValueError("a tridiagonal matrix needs one fewer off-diagonal entries")
    if len(diagonal) == 1:
        return diagonal[0], diagonal[0]

    radii = tuple(
        (off_diagonal[index - 1] if index > 0 else 0.0)
        + (off_diagonal[index] if index < len(off_diagonal) else 0.0)
        for index in range(len(diagonal))
    )
    lower = min(value - radius for value, radius in zip(diagonal, radii, strict=True)) - 1.0
    upper = max(value + radius for value, radius in zip(diagonal, radii, strict=True)) + 1.0
    dimension = len(diagonal)
    for _ in range(iterations):
        middle = (lower + upper) / 2
        if middle == lower or middle == upper:
            break
        if _sturm_count_below(diagonal, off_diagonal, middle) < dimension:
            lower = middle
        else:
            upper = middle
    # Widen the final bracket by a floating-point safety margin.
    margin = roundoff_margin * (1.0 + max(abs(lower), abs(upper)))
    return nextafter(lower - margin, -inf), nextafter(upper + margin, inf)


def principal_tridiagonal_eigenvector(
    diagonal: tuple[float, ...],
    off_diagonal: tuple[float, ...],
    *,
    eigenvalue_upper_bound: float,
    iterations: int = 12,
) -> tuple[float, ...]:
    """Return the positive principal eigenvector by shifted inverse iteration."""

    if len(diagonal) == 1:
        return (1.0,)
    dimension = len(diagonal)
    shift = eigenvalue_upper_bound + 1e-10 * (1.0 + abs(eigenvalue_upper_bound))
    vector = [1 / sqrt(dimension)] * dimension

    def solve_shifted(right_hand_side: list[float]) -> list[float]:
        # Thomas elimination for (shift*I - A), which is positive definite.
        transformed_upper = [0.0] * (dimension - 1)
        transformed_rhs = [0.0] * dimension
        pivot = shift - diagonal[0]
        transformed_upper[0] = -off_diagonal[0] / pivot
        transformed_rhs[0] = right_hand_side[0] / pivot
        for index in range(1, dimension):
            pivot = (
                shift
                - diagonal[index]
                + off_diagonal[index - 1] * transformed_upper[index - 1]
            )
            if index < dimension - 1:
                transformed_upper[index] = -off_diagonal[index] / pivot
            transformed_rhs[index] = (
                right_hand_side[index]
                + off_diagonal[index - 1] * transformed_rhs[index - 1]
            ) / pivot
        solution = [0.0] * dimension
        solution[-1] = transformed_rhs[-1]
        for index in range(dimension - 2, -1, -1):
            solution[index] = transformed_rhs[index] - transformed_upper[index] * solution[index + 1]
        return solution

    for _ in range(iterations):
        product = solve_shifted(vector)
        norm = sqrt(sum(value * value for value in product))
        if norm == 0:
            raise ArithmeticError("principal-eigenvector iteration reached the zero vector")
        vector = [value / norm for value in product]
    if vector[0] < 0:
        vector = [-value for value in vector]
    return tuple(vector)


def canonical_finite_quality(
    clause_count: int,
    field_order: int,
    accepted_set_size: int,
    degree: int,
    *,
    roundoff_margin: float = 1e-12,
) -> QualityEstimate:
    """Optimize the finite canonical coefficient matrix for equal-size sets.

    This computes the principal eigenvalue of the tridiagonal matrix in the
    finite-size canonical DQI performance theorem.  The returned interval is a
    bisection bracket widened by a floating-point safety margin.
    """

    m, p, r, ell = clause_count, field_order, accepted_set_size, degree
    if m < 1 or p < 2 or not 0 < r < p or not 0 <= ell < m:
        raise ValueError("invalid finite canonical DQI parameters")
    if ell == 0:
        baseline = r / p
        return QualityEstimate(
            value=baseline,
            lower=baseline,
            upper=baseline,
            method="finite canonical degree-zero baseline",
            coefficients=(1.0,),
        )
    diagonal_scale = (p - 2 * r) / sqrt(r * (p - r))
    diagonal = tuple(index * diagonal_scale for index in range(ell + 1))
    off_diagonal = tuple(sqrt(index * (m - index + 1)) for index in range(1, ell + 1))
    eigen_lower, eigen_upper = largest_tridiagonal_eigenvalue_interval(
        diagonal,
        off_diagonal,
        roundoff_margin=roundoff_margin,
    )
    coefficients = principal_tridiagonal_eigenvector(
        diagonal,
        off_diagonal,
        eigenvalue_upper_bound=eigen_upper,
    )
    factor = sqrt(r * (p - r)) / (p * m)
    baseline = r / p
    lower = min(1.0, max(0.0, nextafter(baseline + factor * eigen_lower, -inf)))
    upper = min(1.0, max(0.0, nextafter(baseline + factor * eigen_upper, inf)))
    return QualityEstimate(
        value=(lower + upper) / 2,
        lower=lower,
        upper=upper,
        method="finite canonical tridiagonal principal-eigenvalue interval",
        coefficients=coefficients,
    )


def canonical_asymptotic_quality(decode_fraction: float, random_baseline: float) -> float:
    """The semicircle/general equal-size asymptotic projection."""

    if not 0 <= decode_fraction <= 1 or not 0 <= random_baseline <= 1:
        raise ValueError("fractions must lie in [0, 1]")
    if random_baseline > 1 - decode_fraction:
        return 1.0
    value = (
        sqrt(decode_fraction * (1 - random_baseline))
        + sqrt(random_baseline * (1 - decode_fraction))
    ) ** 2
    return min(1.0, max(0.0, value))


def general_objective_asymptotic_quality(
    decode_fraction: float,
    mean: float,
    std: float,
    skewness: float,
) -> float:
    """Asymptotic DQI expectation for a per-constraint objective with the given moments.

    Lemma 9.2 of Jordan et al. expresses the DQI expectation as a quadratic form in a
    tridiagonal matrix whose entries depend on the objective only through the mean,
    the standard deviation and the standardized third moment of ``f_i`` over a uniform
    argument (the state is built from ``g_i = (f_i - mean)/std``, and ``g_i^2`` splits
    into its mean 1, its projection ``skewness * g_i``, and a remainder orthogonal to
    every level).  The principal eigenvalue is ``m (skewness lambda + 2 sqrt(lambda
    (1-lambda)))`` in the limit, so the normalized expectation per constraint is

        mean + std * (skewness * lambda + 2 sqrt(lambda (1 - lambda))).

    For the indicator of a set of density ``rho`` (mean ``rho``, variance
    ``rho(1-rho)``, skewness ``(1-2rho)/sqrt(rho(1-rho))``) this is the semicircle law
    of :func:`canonical_asymptotic_quality` before its saturation at 1.  For
    ``f_i(v) = cos(2 pi v / q)`` (mean 0, ``std = 1/sqrt 2``, skewness 0 for ``q > 3``)
    it is ``sqrt(2 lambda (1 - lambda))``.  The error register of the DQI state holds
    vectors supported on the Fourier support of ``f_i`` (``{+-1}`` for the cosine), so
    ``lambda`` is the reach of a decoder for that restricted error alphabet.  No
    saturation is applied here; the caller compares against the objective's maximum.
    """

    if not 0 <= decode_fraction <= 1:
        raise ValueError("decode fraction must lie in [0, 1]")
    if not std >= 0 or not isfinite(std) or not isfinite(mean) or not isfinite(skewness):
        raise ValueError("moments must be finite with a nonnegative standard deviation")
    return mean + std * (
        skewness * decode_fraction + 2 * sqrt(decode_fraction * (1 - decode_fraction))
    )


def general_objective_finite_quality(
    clause_count: int,
    degree: int,
    mean: float,
    std: float,
    skewness: float,
    *,
    roundoff_margin: float = 1e-12,
) -> QualityEstimate:
    """Finite-``m`` form of :func:`general_objective_asymptotic_quality`.

    Same tridiagonal matrix as :func:`canonical_finite_quality` (diagonal
    ``k * skewness``, off-diagonal ``sqrt(k (m - k + 1))``) with the objective's own
    moments in place of the indicator's; ``canonical_finite_quality(m, p, r, ell)`` is
    recovered exactly at ``mean = r/p``, ``std = sqrt(r (p - r))/p`` and
    ``skewness = (p - 2r)/sqrt(r (p - r))``.
    """

    m, ell = clause_count, degree
    if m < 1 or not 0 <= ell < m:
        raise ValueError("invalid finite parameters")
    if not std >= 0 or not isfinite(std) or not isfinite(mean) or not isfinite(skewness):
        raise ValueError("moments must be finite with a nonnegative standard deviation")
    if ell == 0:
        return QualityEstimate(
            value=mean,
            lower=mean,
            upper=mean,
            method="finite general-objective degree-zero baseline",
            coefficients=(1.0,),
        )
    diagonal = tuple(index * skewness for index in range(ell + 1))
    off_diagonal = tuple(sqrt(index * (m - index + 1)) for index in range(1, ell + 1))
    eigen_lower, eigen_upper = largest_tridiagonal_eigenvalue_interval(
        diagonal,
        off_diagonal,
        roundoff_margin=roundoff_margin,
    )
    coefficients = principal_tridiagonal_eigenvector(
        diagonal,
        off_diagonal,
        eigenvalue_upper_bound=eigen_upper,
    )
    factor = std / m
    lower = nextafter(mean + factor * eigen_lower, -inf)
    upper = nextafter(mean + factor * eigen_upper, inf)
    return QualityEstimate(
        value=(lower + upper) / 2,
        lower=lower,
        upper=upper,
        method="finite general-objective tridiagonal principal-eigenvalue interval",
        coefficients=coefficients,
    )
