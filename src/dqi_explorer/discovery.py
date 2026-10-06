"""Scalar screening quantities: entropies, the Prange baseline, reaches and the cosine rules.

These are the closed-form inputs of the paper's soundness rules.  Rule (R1) compares an upper
bound on the licensed reach with the reach threshold; the functions here supply the entropy
(sphere-packing) bounds, the Prange baseline ``rho + (1 - rho) alpha``, the worst-case reach of
each construction examined in the search, and the corresponding quantities for the cosine
objective.  Docstrings use the labels the scripts print: (D0) to (D3) are the parts of (R1),
(D4) is (R2), (D5) is (R3), (D6) is (R4), and (D7) is the objective-alphabet form of (R1), (R2).
"""

from __future__ import annotations

from math import log2, sqrt

from .spectral import canonical_asymptotic_quality


def binary_entropy(probability: float) -> float:
    """Binary entropy H_2(p), continuously extended at zero and one."""

    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    if probability in (0.0, 1.0):
        return 0.0
    return -probability * log2(probability) - (1 - probability) * log2(1 - probability)


def qary_entropy(probability: float, alphabet_order: int) -> float:
    """Return the base-``q`` entropy used by the declared coding screen.

    This is an analytic helper for checking a published asymptotic rate bound;
    it does not establish that a particular finite code attains that bound.
    """

    if (
        not isinstance(alphabet_order, int)
        or isinstance(alphabet_order, bool)
        or alphabet_order < 2
    ):
        raise ValueError("alphabet_order must be an integer at least two")
    if not 0 <= probability <= 1 - 1 / alphabet_order:
        raise ValueError("probability lies outside the q-ary entropy domain")
    if probability == 0:
        return 0.0
    return (
        probability * log2(alphabet_order - 1)
        - probability * log2(probability)
        - (1 - probability) * log2(1 - probability)
    ) / log2(alphabet_order)


def restricted_qary_entropy(
    probability: float,
    alphabet_order: int,
    support_size: int,
) -> float:
    """Base-``q`` entropy of error patterns whose nonzero entries lie in a set of size ``s``.

    A pattern of weight ``k`` with entries restricted to ``s`` of the ``q - 1`` nonzero
    symbols is one of ``C(m, k) s^k``, so the sphere-packing gate D0 for a decoder that
    only ever sees such errors (the Fourier support of a smooth objective, see
    :func:`dqi_explorer.spectral.general_objective_asymptotic_quality`) reads
    ``alpha >= (H_2(lambda) + lambda log2 s) / log2 q``.  At ``s = q - 1`` this is
    :func:`qary_entropy`; smaller supports lower the ceiling and widen the window.
    """

    if (
        not isinstance(alphabet_order, int)
        or isinstance(alphabet_order, bool)
        or alphabet_order < 2
    ):
        raise ValueError("alphabet_order must be an integer at least two")
    if (
        not isinstance(support_size, int)
        or isinstance(support_size, bool)
        or not 1 <= support_size <= alphabet_order - 1
    ):
        raise ValueError("support_size must be an integer in [1, q - 1]")
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    if probability == 0:
        return 0.0
    return (
        binary_entropy(probability) + probability * log2(support_size)
    ) / log2(alphabet_order)


def prange_quality(syndrome_fraction: float, accepted_density: float) -> float:
    """Quality of the declared information-set/Prange screening baseline."""

    if not 0 <= syndrome_fraction <= 1 or not 0 <= accepted_density <= 1:
        raise ValueError("fractions must lie in [0, 1]")
    return accepted_density + (1 - accepted_density) * syndrome_fraction


def canonical_beats_prange_balanced(syndrome_fraction: float, decode_fraction: float) -> bool:
    """Cheap balanced screen equivalent to lambda(1-lambda)>alpha^2/4."""

    if not 0 <= syndrome_fraction <= 1 or not 0 <= decode_fraction <= 1:
        raise ValueError("fractions must lie in [0, 1]")
    # The closed-form threshold is usually represented by an irrational float;
    # absorb a few ulps so equality is not accidentally reported as a win.
    return decode_fraction * (1 - decode_fraction) > syndrome_fraction**2 / 4 + 1e-15


IN_MEASURE_MARGIN_CAP = (sqrt(5) - 1) / 4


IN_MEASURE_CAP_RATE = (5 - sqrt(5)) / 10


def in_measure_reach_ceiling(syndrome_fraction: float) -> float:
    """The largest reach any decoder can be licensed to in measure: ``alpha``.

    This is the in-measure Singleton bound; it is a strict ceiling, and no efficient
    decoder is asserted to attain it.
    """

    if not 0 <= syndrome_fraction <= 1:
        raise ValueError("syndrome fraction must lie in [0, 1]")
    return syndrome_fraction


def in_measure_margin_cap_at(syndrome_fraction: float, accepted_density: float = 0.5) -> float:
    """Cap on the DQI-over-Prange margin under an in-measure licence at this rate."""

    reach = in_measure_reach_ceiling(syndrome_fraction)
    return canonical_asymptotic_quality(reach, accepted_density) - prange_quality(
        syndrome_fraction, accepted_density
    )


def alternant_half_designed_reach(syndrome_fraction: float, extension_degree: int) -> float:
    """Worst-case reach of an alternant decoder code with generic dimension.

    An alternant code over ``F_p`` of length ``m`` is the subfield subcode of a GRS
    code over ``F_(p^s)`` with ``r`` parity rows; its designed distance is ``r + 1``
    and its dimension is at least ``m - s r``.  With the generic (tight) dimension
    the optimizer rate is ``alpha = s r / m``, so ``delta = alpha / s`` and the
    bounded-distance reach is ``alpha / (2 s)``.
    """

    if not 0 <= syndrome_fraction <= 1:
        raise ValueError("syndrome fraction must lie in [0, 1]")
    if not isinstance(extension_degree, int) or extension_degree < 1:
        raise ValueError("extension degree must be a positive integer")
    return syndrome_fraction / (2 * extension_degree)


def alternant_reach_window(extension_degree: int) -> float:
    """Largest rate loss at which the alternant reach passes gate D2 at ``rho = 1/2``.

    ``lambda(1-lambda) > alpha^2/4`` with ``lambda = alpha/(2s)`` is
    ``alpha < 2s/(s^2+1)``: 0.8 at s=2, 0.6 at s=3, 0.47 at s=4.
    """

    if not isinstance(extension_degree, int) or extension_degree < 1:
        raise ValueError("extension degree must be a positive integer")
    return 2 * extension_degree / (extension_degree**2 + 1)


def sparse_mixed_reach(syndrome_fraction: float, column_weight: int) -> float:
    """Worst-case reach of a GRS key hidden by a secret column mixing of exact weight ``w``.

    ``H_pub = S H T`` with ``T`` invertible of column weight ``w``: a pattern of weight
    ``ell`` becomes a GRS error of weight at most ``w ell``, so Berlekamp--Massey to
    ``floor((r-1)/2)`` GRS errors decodes ``ell = floor((r-1)/2) // w`` public errors, a
    reach of ``alpha / (2 w)`` at ``alpha = r / m`` (``dqi_explorer/hidden_keys.py``).
    """

    if not 0 <= syndrome_fraction <= 1:
        raise ValueError("syndrome fraction must lie in [0, 1]")
    if not isinstance(column_weight, int) or column_weight < 1:
        raise ValueError("column weight must be a positive integer")
    return syndrome_fraction / (2 * column_weight)


def sparse_mixed_reach_window(column_weight: int) -> float:
    """Largest rate loss at which ``alpha/(2w)`` passes gate D2 at ``rho = 1/2``: ``2w/(w^2+1)``."""

    if not isinstance(column_weight, int) or column_weight < 1:
        raise ValueError("column weight must be a positive integer")
    return 2 * column_weight / (column_weight**2 + 1)


def column_inserted_reach(syndrome_fraction: float, insertion_ratio: float) -> float:
    """Worst-case reach of a GRS key with ``u = insertion_ratio * (m0 - k)`` inserted coordinates.

    Public length ``m = m0 + u``, optimizer rate ``alpha = (m0 - k + u)/m``; an error on
    either coordinate of a mixed pair is at most one GRS error, so the decoder reaches
    ``(m0 - k)/(2 m) = alpha / (2 (1 + u/(m0-k)))``.
    """

    if not 0 <= syndrome_fraction <= 1:
        raise ValueError("syndrome fraction must lie in [0, 1]")
    if insertion_ratio < 0:
        raise ValueError("insertion ratio must be nonnegative")
    return syndrome_fraction / (2 * (1 + insertion_ratio))


def log_cauchy_reach(syndrome_fraction: float) -> float:
    """Worst-case reach of the log-Cauchy code for ``{0, +-1}`` errors: ``alpha / 2``.

    ``B[i][j] = log_g(t_i + alpha_j)`` over a Mersenne prime field; a restricted pattern of
    weight ``ell`` exponentiates to a rational function of degree at most ``ell`` in numerator
    and denominator, reconstructed uniquely from ``n >= 2 ell + 1`` values
    (``dqi_explorer/log_cauchy.py``).  In measure (random signs) the reach is about ``alpha``.
    """

    if not 0 <= syndrome_fraction <= 1:
        raise ValueError("syndrome fraction must lie in [0, 1]")
    return syndrome_fraction / 2


COSINE_STD = 1 / sqrt(2)


def cosine_quality(decode_fraction: float) -> float:
    """DQI expectation per row for ``cos(2 pi (Bx)_i / q)``: ``sqrt(2 lambda (1 - lambda))``."""

    from .spectral import general_objective_asymptotic_quality

    return general_objective_asymptotic_quality(decode_fraction, 0.0, COSINE_STD, 0.0)


def cosine_baseline(syndrome_fraction: float) -> float:
    """One information set scores ``cos = 1`` on its ``alpha m`` rows and 0 on average elsewhere."""

    if not 0 <= syndrome_fraction <= 1:
        raise ValueError("syndrome fraction must lie in [0, 1]")
    return syndrome_fraction


def cosine_margin(syndrome_fraction: float, decode_fraction: float) -> float:
    return cosine_quality(decode_fraction) - cosine_baseline(syndrome_fraction)


def cosine_beats_baseline(syndrome_fraction: float, decode_fraction: float) -> bool:
    """Gate D2 for the cosine objective: ``lambda (1 - lambda) > alpha^2 / 2``."""

    return decode_fraction * (1 - decode_fraction) > syndrome_fraction * syndrome_fraction / 2


def restricted_radius_is_realizable(syndrome_fraction: float, decode_fraction: float, alphabet_order: int, support_size: int = 2) -> bool:
    """Gate D0 for a restricted error alphabet: ``alpha >= (H_2(lambda) + lambda log2 s) / log2 q``."""

    return restricted_qary_entropy(decode_fraction, alphabet_order, support_size) <= syndrome_fraction + 1e-12


def cosine_restart_threshold(syndrome_fraction: float, margin: float, draws: float = 2.0**40, failure: float = 0.05) -> float | None:
    """Restart converse for an objective of range ``R = 2``: ``(1-alpha) R^2 ln(N/eta) / (2 mu^2)``."""

    from math import log

    if margin <= 0:
        return None
    return (1 - syndrome_fraction) * 4 * log(draws / failure) / (2 * margin**2)


def existential_cosine(syndrome_fraction: float, alphabet_order: int) -> float:
    """Existential line for the cosine objective: the Gibbs bound at ``alpha ln q`` nats per row.

    The number of ``x`` is ``q^n = q^(alpha m)``, so a first-moment bound puts every achievable
    empirical row distribution within relative entropy ``alpha ln q`` of uniform; the largest
    mean cosine under that budget is attained by the tilted law ``mu(v) ∝ exp(beta cos(2 pi v/q))``
    with ``KL(mu || uniform) = alpha ln q``.  Solved by bisection on ``beta``.
    """

    from math import exp, log, pi, cos

    if not 0 <= syndrome_fraction <= 1:
        raise ValueError("syndrome fraction must lie in [0, 1]")
    budget = syndrome_fraction * log(alphabet_order)
    if budget <= 0:
        return 0.0
    values = [cos(2 * pi * v / alphabet_order) for v in range(alphabet_order)]

    def moments(beta: float) -> tuple[float, float]:
        weights = [exp(beta * c) for c in values]
        z = sum(weights)
        mean = sum(w * c for w, c in zip(weights, values)) / z
        kl = beta * mean - log(z / alphabet_order)
        return mean, kl

    lower, upper = 0.0, 1.0
    while moments(upper)[1] < budget and upper < 1e6:
        upper *= 2
    for _ in range(200):
        middle = (lower + upper) / 2
        if moments(middle)[1] < budget:
            lower = middle
        else:
            upper = middle
    return moments((lower + upper) / 2)[0]


def construction_d_restricted_reach_ceiling(
    level0_relative_distance: float,
    *,
    euclidean_bdd: bool = True,
) -> float:
    """Ceiling on the relative reach of a restricted-error decoder of a code-chain lattice.

    A ``q``-ary lattice of a code ``C`` (Construction A), or a Construction-D lattice
    built on a chain ``C_0 subset ... subset C_{t-1}`` with modulus ``2^t``, contains
    the lift of every codeword of the level-0 code with entries in ``{0, 1}``.  Two
    error patterns with entries in ``{0, +-1}`` and the same syndrome differ by such
    a lattice vector once their union covers its support, so no decoder can uniquely
    correct restricted patterns of weight beyond ``d_H(C_0) / 2``, and a Euclidean
    bounded-distance decoder of radius ``d_E / 2`` with ``d_E^2 <= d_H(C_0)`` handles
    patterns of weight at most ``d_H(C_0) / 4`` (a pattern of weight ``k`` has norm
    ``sqrt k``).  Relative to the length this is ``delta_H(C_0) / 4`` (Euclidean BDD)
    or ``delta_H(C_0) / 2`` (any unique restricted decoder).  The restricted metric
    therefore inherits gate D1 from the level-0 code: Barnes-Wall (Reed-Muller chain,
    ``d_E^2 = 2^floor(n/2)`` at length ``2^n``) and polar lattices have vanishing
    ceilings, Construction-A Reed-Solomon keeps ``alpha / 4``.
    """

    if not 0 <= level0_relative_distance <= 1:
        raise ValueError("relative distance must lie in [0, 1]")
    return level0_relative_distance / (4 if euclidean_bdd else 2)


def licensed_radius_is_realizable(
    syndrome_fraction: float,
    decode_fraction: float,
    metric_alphabet_order: int,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Gate D0: does any code have this rate and this worst-case decoding radius?

    Returns ``True`` when ``H_q(decode_fraction) <= syndrome_fraction``.  A ``False``
    verdict means the declared pair is realized by no code at all, efficiently or
    otherwise, so it refutes the declared guarantee rather than merely leaving it
    unsupported.  Apply it only to worst-case guarantees.
    """

    if not 0 < syndrome_fraction < 1:
        raise ValueError("syndrome fraction must lie in (0, 1)")
    if not 0 <= decode_fraction <= 1:
        raise ValueError("decode fraction must lie in [0, 1]")
    if decode_fraction <= 0:
        return True
    if decode_fraction > 1 - 1 / metric_alphabet_order:
        # Past the maximum-entropy radius the ceiling would need rate one or more, so
        # no code of positive redundancy decodes uniquely that far.
        return False
    return (
        qary_entropy(decode_fraction, metric_alphabet_order)
        <= syndrome_fraction + tolerance
    )
