"""Quasi-Monte Carlo (Chapter 11).

Low-discrepancy points implemented from their definitions, with nothing taken from a
library: the van der Corput sequence (radical inverse), the Halton sequence, and the
Sobol' sequence with the direction numbers of Joe and Kuo (2008); exact star
discrepancies of small point sets and t-values of digital nets; three randomisations
(random shift modulo 1, random digital shift, Owen's nested uniform scramble) with
error estimates from independent replicates; and the tools of the effective-dimension
laboratory: the arithmetic and geometric Asian calls of the course built on the path
constructions of Chapter 7, exact variance shares of the geometric Asian call (a payoff
that depends on the normals through one linear combination), and pick-freeze estimates
of the variance carried by the first coordinates.

Conventions
-----------
* Point sets are arrays of shape (n, d), one point per row, in [0, 1)^d.
* Sequences are indexed from i = 0, so their first point is the origin. The inverse
  normal map sends it to -infinity: skip it (``start=1``) or randomise before using
  ``to_normals``. Randomised points lie in (0, 1)^d with probability one.
* Sobol' points are produced in the natural order x_i = sum_k bit_k(i) v_k (bitwise
  exclusive or of direction numbers), as integers with ``bits`` binary digits.
  scipy.stats.qmc.Sobol produces the same points in Gray-code order (its point i is our
  point i ^ (i >> 1)); the first 2^m points are the same set. scipy is used only in the
  tests, as a cross-check.
* Random functions take a ``seed`` (anything ``numpy.random.default_rng`` accepts).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import ndtr

from .montecarlo import mc_estimate
from .paths import construction_matrix, gbm_paths, time_grid
from .rng import norm_inv
from .variance import gauss_expect

# ---------------------------------------------------------------------------------------
# One dimension: the radical inverse
# ---------------------------------------------------------------------------------------


def primes(d: int) -> list[int]:
    """The first d primes (the Halton bases)."""
    out, k = [], 2
    while len(out) < d:
        if all(k % p for p in out if p * p <= k):
            out.append(k)
        k += 1
    return out


def _digits(i, b: int):
    """Base-b digits a_0, a_1, ... (least significant first) of the integers i, as a list of arrays."""
    i = np.asarray(i, dtype=np.int64).copy()
    if np.any(i < 0):
        raise ValueError("indices must be nonnegative")
    out = []
    while np.any(i > 0):
        out.append(i % b)
        i //= b
    return out


def radical_inverse(i, b: int = 2):
    """phi_b(i) = sum_k a_k(i) b^{-k-1} for i = sum_k a_k(i) b^k (digits mirrored at the point).

    Evaluated by Horner's rule from the most significant digit, which costs one rounding
    per digit; for b = 2 the result is exact.
    """
    i = np.asarray(i, dtype=np.int64)
    x = np.zeros(i.shape)
    for a in reversed(_digits(i, b)):
        x = (x + a) / b
    return x


def radical_inverse_int(i, b: int, K: int):
    """Integer numerators N with phi_b(i) = N / b^K up to digits beyond the K-th (exact cells).

    N = sum_{k<K} a_k(i) b^{K-1-k}, so floor(phi_b(i) b^k) = N // b^{K-k} for k <= K without
    rounding. Requires b^K < 2^63.
    """
    if b ** K >= 2**63:
        raise ValueError("b^K too large for 64-bit integers")
    i = np.asarray(i, dtype=np.int64)
    N = np.zeros(i.shape, dtype=np.int64)
    for k, a in enumerate(_digits(i, b)[:K]):
        N += a * b ** (K - 1 - k)
    return N


def van_der_corput(n: int, b: int = 2, start: int = 0) -> np.ndarray:
    """The points phi_b(start), ..., phi_b(start + n - 1)."""
    return radical_inverse(np.arange(start, start + n), b)


def van_der_corput_bound(n: int, b: int = 2) -> float:
    """The bound (b - 1)(floor(log_b n) + 1) / n on the star discrepancy of the first n points."""
    if n < 1:
        raise ValueError("n must be positive")
    digits = len(_digits(np.array([n]), b))          # floor(log_b n) + 1, computed exactly
    return (b - 1) * digits / n


def halton(n: int, d: int, start: int = 0, bases=None) -> np.ndarray:
    """The Halton points (phi_{b_1}(i), ..., phi_{b_d}(i)), i = start, ..., start + n - 1.

    The bases are the first d primes unless given (they must be pairwise coprime).
    """
    bases = primes(d) if bases is None else list(bases)
    if len(bases) != d:
        raise ValueError("need one base per dimension")
    i = np.arange(start, start + n)
    if d == 0:
        return np.zeros((n, 0))
    return np.column_stack([radical_inverse(i, b) for b in bases])


# ---------------------------------------------------------------------------------------
# Sobol' points
# ---------------------------------------------------------------------------------------

#: Primitive polynomials and initial direction numbers (s, a, (m_1, ..., m_s)) for the
#: dimensions 2, 3, ..., 128, copied from the file new-joe-kuo-6.21201 of Joe and Kuo (2008),
#: https://web.maths.unsw.edu.au/~fkuo/sobol/ . The polynomial of degree s is
#: x^s + c_1 x^{s-1} + ... + c_{s-1} x + 1, with c_1 ... c_{s-1} the binary digits of a.
#: Dimension 1 is the van der Corput sequence in base 2 (all m_k = 1).
JOE_KUO = (
    (1, 0, (1,)), (2, 1, (1, 3)), (3, 1, (1, 3, 1)), (3, 2, (1, 1, 1)), (4, 1, (1, 1, 3, 3)),
    (4, 4, (1, 3, 5, 13)), (5, 2, (1, 1, 5, 5, 17)), (5, 4, (1, 1, 5, 5, 5)),
    (5, 7, (1, 1, 7, 11, 19)), (5, 11, (1, 1, 5, 1, 1)), (5, 13, (1, 1, 1, 3, 11)),
    (5, 14, (1, 3, 5, 5, 31)), (6, 1, (1, 3, 3, 9, 7, 49)), (6, 13, (1, 1, 1, 15, 21, 21)),
    (6, 16, (1, 3, 1, 13, 27, 49)), (6, 19, (1, 1, 1, 15, 7, 5)),
    (6, 22, (1, 3, 1, 15, 13, 25)), (6, 25, (1, 1, 5, 5, 19, 61)),
    (7, 1, (1, 3, 7, 11, 23, 15, 103)), (7, 4, (1, 3, 7, 13, 13, 15, 69)),
    (7, 7, (1, 1, 3, 13, 7, 35, 63)), (7, 8, (1, 3, 5, 9, 1, 25, 53)),
    (7, 14, (1, 3, 1, 13, 9, 35, 107)), (7, 19, (1, 3, 1, 5, 27, 61, 31)),
    (7, 21, (1, 1, 5, 11, 19, 41, 61)), (7, 28, (1, 3, 5, 3, 3, 13, 69)),
    (7, 31, (1, 1, 7, 13, 1, 19, 1)), (7, 32, (1, 3, 7, 5, 13, 19, 59)),
    (7, 37, (1, 1, 3, 9, 25, 29, 41)), (7, 41, (1, 3, 5, 13, 23, 1, 55)),
    (7, 42, (1, 3, 7, 3, 13, 59, 17)), (7, 50, (1, 3, 1, 3, 5, 53, 69)),
    (7, 55, (1, 1, 5, 5, 23, 33, 13)), (7, 56, (1, 1, 7, 7, 1, 61, 123)),
    (7, 59, (1, 1, 7, 9, 13, 61, 49)), (7, 62, (1, 3, 3, 5, 3, 55, 33)),
    (8, 14, (1, 3, 1, 15, 31, 13, 49, 245)), (8, 21, (1, 3, 5, 15, 31, 59, 63, 97)),
    (8, 22, (1, 3, 1, 11, 11, 11, 77, 249)), (8, 38, (1, 3, 1, 11, 27, 43, 71, 9)),
    (8, 47, (1, 1, 7, 15, 21, 11, 81, 45)), (8, 49, (1, 3, 7, 3, 25, 31, 65, 79)),
    (8, 50, (1, 3, 1, 1, 19, 11, 3, 205)), (8, 52, (1, 1, 5, 9, 19, 21, 29, 157)),
    (8, 56, (1, 3, 7, 11, 1, 33, 89, 185)), (8, 67, (1, 3, 3, 3, 15, 9, 79, 71)),
    (8, 70, (1, 3, 7, 11, 15, 39, 119, 27)), (8, 84, (1, 1, 3, 1, 11, 31, 97, 225)),
    (8, 97, (1, 1, 1, 3, 23, 43, 57, 177)), (8, 103, (1, 3, 7, 7, 17, 17, 37, 71)),
    (8, 115, (1, 3, 1, 5, 27, 63, 123, 213)), (8, 122, (1, 1, 3, 5, 11, 43, 53, 133)),
    (9, 8, (1, 3, 5, 5, 29, 17, 47, 173, 479)), (9, 13, (1, 3, 3, 11, 3, 1, 109, 9, 69)),
    (9, 16, (1, 1, 1, 5, 17, 39, 23, 5, 343)), (9, 22, (1, 3, 1, 5, 25, 15, 31, 103, 499)),
    (9, 25, (1, 1, 1, 11, 11, 17, 63, 105, 183)), (9, 44, (1, 1, 5, 11, 9, 29, 97, 231, 363)),
    (9, 47, (1, 1, 5, 15, 19, 45, 41, 7, 383)), (9, 52, (1, 3, 7, 7, 31, 19, 83, 137, 221)),
    (9, 55, (1, 1, 1, 3, 23, 15, 111, 223, 83)), (9, 59, (1, 1, 5, 13, 31, 15, 55, 25, 161)),
    (9, 62, (1, 1, 3, 13, 25, 47, 39, 87, 257)), (9, 67, (1, 1, 1, 11, 21, 53, 125, 249, 293)),
    (9, 74, (1, 1, 7, 11, 11, 7, 57, 79, 323)), (9, 81, (1, 1, 5, 5, 17, 13, 81, 3, 131)),
    (9, 82, (1, 1, 7, 13, 23, 7, 65, 251, 475)), (9, 87, (1, 3, 5, 1, 9, 43, 3, 149, 11)),
    (9, 91, (1, 1, 3, 13, 31, 13, 13, 255, 487)), (9, 94, (1, 3, 3, 1, 5, 63, 89, 91, 127)),
    (9, 103, (1, 1, 3, 3, 1, 19, 123, 127, 237)), (9, 104, (1, 1, 5, 7, 23, 31, 37, 243, 289)),
    (9, 109, (1, 1, 5, 11, 17, 53, 117, 183, 491)),
    (9, 122, (1, 1, 1, 5, 1, 13, 13, 209, 345)), (9, 124, (1, 1, 3, 15, 1, 57, 115, 7, 33)),
    (9, 137, (1, 3, 1, 11, 7, 43, 81, 207, 175)), (9, 138, (1, 3, 1, 1, 15, 27, 63, 255, 49)),
    (9, 143, (1, 3, 5, 3, 27, 61, 105, 171, 305)), (9, 145, (1, 1, 5, 3, 1, 3, 57, 249, 149)),
    (9, 152, (1, 1, 3, 5, 5, 57, 15, 13, 159)), (9, 157, (1, 1, 1, 11, 7, 11, 105, 141, 225)),
    (9, 167, (1, 3, 3, 5, 27, 59, 121, 101, 271)), (9, 173, (1, 3, 5, 9, 11, 49, 51, 59, 115)),
    (9, 176, (1, 1, 7, 1, 23, 45, 125, 71, 419)), (9, 181, (1, 1, 3, 5, 23, 5, 105, 109, 75)),
    (9, 182, (1, 1, 7, 15, 7, 11, 67, 121, 453)), (9, 185, (1, 3, 7, 3, 9, 13, 31, 27, 449)),
    (9, 191, (1, 3, 1, 15, 19, 39, 39, 89, 15)), (9, 194, (1, 1, 1, 1, 1, 33, 73, 145, 379)),
    (9, 199, (1, 3, 1, 15, 15, 43, 29, 13, 483)), (9, 218, (1, 1, 7, 3, 19, 27, 85, 131, 431)),
    (9, 220, (1, 3, 3, 3, 5, 35, 23, 195, 349)), (9, 227, (1, 3, 3, 7, 9, 27, 39, 59, 297)),
    (9, 229, (1, 1, 3, 9, 11, 17, 13, 241, 157)),
    (9, 230, (1, 3, 7, 15, 25, 57, 33, 189, 213)), (9, 234, (1, 1, 7, 1, 9, 55, 73, 83, 217)),
    (9, 236, (1, 3, 3, 13, 19, 27, 23, 113, 249)), (9, 241, (1, 3, 5, 3, 23, 43, 3, 253, 479)),
    (9, 244, (1, 1, 5, 5, 11, 5, 45, 117, 217)), (9, 253, (1, 3, 3, 7, 29, 37, 33, 123, 147)),
    (10, 4, (1, 3, 1, 15, 5, 5, 37, 227, 223, 459)),
    (10, 13, (1, 1, 7, 5, 5, 39, 63, 255, 135, 487)),
    (10, 19, (1, 3, 1, 7, 9, 7, 87, 249, 217, 599)),
    (10, 22, (1, 1, 3, 13, 9, 47, 7, 225, 363, 247)),
    (10, 50, (1, 3, 7, 13, 19, 13, 9, 67, 9, 737)),
    (10, 55, (1, 3, 5, 5, 19, 59, 7, 41, 319, 677)),
    (10, 64, (1, 1, 5, 3, 31, 63, 15, 43, 207, 789)),
    (10, 69, (1, 1, 7, 9, 13, 39, 3, 47, 497, 169)),
    (10, 98, (1, 3, 1, 7, 21, 17, 97, 19, 415, 905)),
    (10, 107, (1, 3, 7, 1, 3, 31, 71, 111, 165, 127)),
    (10, 115, (1, 1, 5, 11, 1, 61, 83, 119, 203, 847)),
    (10, 121, (1, 3, 3, 13, 9, 61, 19, 97, 47, 35)),
    (10, 127, (1, 1, 7, 7, 15, 29, 63, 95, 417, 469)),
    (10, 134, (1, 3, 1, 9, 25, 9, 71, 57, 213, 385)),
    (10, 140, (1, 3, 5, 13, 31, 47, 101, 57, 39, 341)),
    (10, 145, (1, 1, 3, 3, 31, 57, 125, 173, 365, 551)),
    (10, 152, (1, 3, 7, 1, 13, 57, 67, 157, 451, 707)),
    (10, 158, (1, 1, 1, 7, 21, 13, 105, 89, 429, 965)),
    (10, 161, (1, 1, 5, 9, 17, 51, 45, 119, 157, 141)),
    (10, 171, (1, 3, 7, 7, 13, 45, 91, 9, 129, 741)),
    (10, 181, (1, 3, 7, 1, 23, 57, 67, 141, 151, 571)),
    (10, 194, (1, 1, 3, 11, 17, 47, 93, 107, 375, 157)),
    (10, 199, (1, 3, 3, 5, 11, 21, 43, 51, 169, 915)),
    (10, 203, (1, 1, 5, 3, 15, 55, 101, 67, 455, 625)),
    (10, 208, (1, 3, 5, 9, 1, 23, 29, 47, 345, 595)),
    (10, 227, (1, 3, 7, 7, 5, 49, 29, 155, 323, 589)),
    (10, 242, (1, 3, 3, 7, 5, 41, 127, 61, 261, 717))
)
MAX_SOBOL_DIM = len(JOE_KUO) + 1


def sobol_degrees(d: int) -> list[int]:
    """Degrees e_1, ..., e_d of the polynomials of the first d Sobol' coordinates.

    Coordinate 1 corresponds to the polynomial x (degree 1); the t-value of the first d
    coordinates is at most sum_j (e_j - 1) (Sobol', 1967).
    """
    if d > MAX_SOBOL_DIM:
        raise ValueError(f"at most {MAX_SOBOL_DIM} dimensions are tabulated")
    return ([1] + [row[0] for row in JOE_KUO])[:d]


def direction_integers(j: int, bits: int = 32) -> list[int]:
    """m_1, ..., m_bits of coordinate j (1-based): odd integers with m_k < 2^k.

    For k > s they follow the recursion of the primitive polynomial over F_2,
        m_k = 2 c_1 m_{k-1} xor 4 c_2 m_{k-2} xor ... xor 2^{s-1} c_{s-1} m_{k-s+1}
              xor 2^s m_{k-s} xor m_{k-s}.
    """
    if j == 1:
        return [1] * bits
    s, a, init = JOE_KUO[j - 2]
    m = list(init[:bits])
    for k in range(s, bits):                         # 0-based: m[k] is m_{k+1}
        new = m[k - s] ^ (m[k - s] << s)
        for l in range(1, s):
            if (a >> (s - 1 - l)) & 1:
                new ^= m[k - l] << l
        m.append(new)
    return m


def sobol_direction_numbers(d: int, bits: int = 32) -> np.ndarray:
    """Integers V[k, j] = m_{k+1, j} 2^{bits-k-1}: the direction numbers v_{k+1, j} times 2^bits."""
    if not 1 <= bits <= 63:
        raise ValueError("bits must be between 1 and 63")
    if d > MAX_SOBOL_DIM:
        raise ValueError(f"at most {MAX_SOBOL_DIM} dimensions are tabulated")
    V = np.zeros((bits, d), dtype=np.uint64)
    for j in range(d):
        m = direction_integers(j + 1, bits)
        V[:, j] = [m[k] << (bits - k - 1) for k in range(bits)]
    return V


def sobol_integers(n: int, d: int, start: int = 0, bits: int = 32) -> np.ndarray:
    """Sobol' points times 2^bits, as unsigned integers of shape (n, d), natural order.

    X_i = xor over k of bit_k(i) V[k]: point i combines the direction numbers selected by
    the binary digits of i. Requires start + n <= 2^bits.
    """
    if start + n > 2**bits:
        raise ValueError("index exceeds 2^bits")
    V = sobol_direction_numbers(d, bits)
    idx = np.arange(start, start + n, dtype=np.uint64)
    X = np.zeros((n, d), dtype=np.uint64)
    k = 0
    while k < bits and np.any(idx >> np.uint64(k)):
        sel = ((idx >> np.uint64(k)) & np.uint64(1)).astype(bool)
        X[sel] ^= V[k]
        k += 1
    return X


def sobol(n: int, d: int, start: int = 0, bits: int = 32) -> np.ndarray:
    """The Sobol' points x_start, ..., x_{start+n-1} in [0, 1)^d (natural order)."""
    return sobol_integers(n, d, start, bits).astype(float) / 2.0**bits


def generator_matrices(d: int, m: int, bits: int = 32) -> list[np.ndarray]:
    """The m x m generator matrices C_j over F_2 of the first 2^m Sobol' points.

    Column k holds the first m binary digits of v_{k+1, j}: digit r (row r, 0-based) of
    coordinate j of point i is sum_k C_j[r, k] bit_k(i) mod 2. Each C_j is upper triangular
    with ones on the diagonal, because m_k is odd and smaller than 2^k.
    """
    V = sobol_direction_numbers(d, bits)
    out = []
    for j in range(d):
        C = np.zeros((m, m), dtype=np.uint8)
        for k in range(m):
            v = int(V[k, j])
            for r in range(m):
                C[r, k] = (v >> (bits - 1 - r)) & 1
        out.append(C)
    return out


def _rank_gf2(rows: list[int]) -> int:
    """Rank over F_2 of the rows given as integer bit masks."""
    basis = {}                                       # leading bit -> row
    for r in rows:
        while r:
            h = r.bit_length() - 1
            if h in basis:
                r ^= basis[h]
            else:
                basis[h] = r
                break
    return len(basis)


def _compositions(total: int, parts: int):
    """All (k_1, ..., k_parts) of nonnegative integers with sum ``total``."""
    if parts == 1:
        yield (total,)
        return
    for k in range(total + 1):
        for rest in _compositions(total - k, parts - 1):
            yield (k,) + rest


def t_value(matrices: list[np.ndarray]) -> int:
    """Smallest t such that the digital net with these m x m generator matrices is a (t, m, d)-net.

    Criterion: for every (k_1, ..., k_d) with sum k_j = m - t, the first k_1 rows of C_1,
    ..., the first k_d rows of C_d are linearly independent over F_2.
    """
    m = matrices[0].shape[0]
    masks = [[int("".join(map(str, C[r])), 2) for r in range(m)] for C in matrices]
    for t in range(m + 1):
        if all(_rank_gf2([row for j, k in enumerate(ks) for row in masks[j][:k]]) == m - t
               for ks in _compositions(m - t, len(matrices))):
            return t
    return m


def elementary_counts(X: np.ndarray, ks, bits: int = 32) -> np.ndarray:
    """Numbers of points in each elementary interval prod_j [a_j 2^{-k_j}, (a_j + 1) 2^{-k_j}).

    X holds integer points (times 2^bits). Returns a flat array of 2^{sum k_j} counts.
    """
    X = np.asarray(X, dtype=np.uint64)
    cell = np.zeros(X.shape[0], dtype=np.int64)
    for j, k in enumerate(ks):
        cell = (cell << k) + (X[:, j] >> np.uint64(bits - k)).astype(np.int64) if k else cell
    return np.bincount(cell, minlength=2 ** int(sum(ks)))


def is_net(X: np.ndarray, t: int, bits: int = 32) -> bool:
    """Brute-force check that the 2^m integer points X form a (t, m, d)-net in base 2."""
    n, d = X.shape
    m = int(round(math.log2(n)))
    if 2**m != n or t > m:
        raise ValueError("need 2^m points and t <= m")
    return all(np.all(elementary_counts(X, ks, bits) == 2**t) for ks in _compositions(m - t, d))


# ---------------------------------------------------------------------------------------
# Discrepancy
# ---------------------------------------------------------------------------------------


def local_discrepancy(P: np.ndarray, a) -> float:
    """delta(a) = #{i : x_i in [0, a)} / n - prod_j a_j for the anchored box [0, a)."""
    P = np.atleast_2d(P)
    a = np.asarray(a, dtype=float)
    return float(np.mean(np.all(P < a, axis=1)) - np.prod(a))


def star_discrepancy(P: np.ndarray, max_work: float = 2e8) -> float:
    """Exact star discrepancy D*_n = sup_a |delta(a)| of the points (rows of P).

    The supremum is attained along the grid Gamma = prod_j ({x_ij} u {1}): it is the larger
    of max (vol - open count / n) and max (closed count / n - vol) over a in Gamma.
    d = 1: sorted points, O(n log n). d = 2: a sweep over the first coordinate,
    O(n^2 log n). d >= 3: enumeration of Gamma, for small n only. d = 0 gives 0.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    n, d = P.shape
    if n == 0:
        raise ValueError("need at least one point")
    if d == 0:
        return 0.0
    if d == 1:
        x = np.sort(P[:, 0])
        i = np.arange(1, n + 1)
        return float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))
    grids = [np.unique(np.append(P[:, j], 1.0)) for j in range(d)]
    if d == 2:
        best = 0.0
        g2 = grids[1]
        for a1 in grids[0]:
            o = np.sort(P[P[:, 0] < a1, 1])
            c = np.sort(P[P[:, 0] <= a1, 1])
            vol = a1 * g2
            open_ = np.searchsorted(o, g2, side="left") / n
            closed = np.searchsorted(c, g2, side="right") / n
            best = max(best, float(np.max(vol - open_)), float(np.max(closed - vol)))
        return best
    work = float(np.prod([len(g) for g in grids])) * n * d
    if work > max_work:
        raise ValueError("too many points for the exact computation in this dimension")
    best = 0.0
    for a in itertools.product(*grids):
        a = np.array(a)
        vol = float(np.prod(a))
        best = max(best, vol - float(np.mean(np.all(P < a, axis=1))),
                   float(np.mean(np.all(P <= a, axis=1))) - vol)
    return best


# ---------------------------------------------------------------------------------------
# Randomisation
# ---------------------------------------------------------------------------------------


def random_shift(P: np.ndarray, seed=None) -> np.ndarray:
    """Cranley--Patterson rotation: y_i = (x_i + U) mod 1 with one U ~ U[0, 1)^d for all points."""
    P = np.asarray(P, dtype=float)
    U = np.random.default_rng(seed).random(P.shape[-1])
    return np.mod(P + U, 1.0)


def digital_shift(X: np.ndarray, bits: int = 32, seed=None) -> np.ndarray:
    """Random digital shift in base 2 of integer points X (times 2^bits).

    y_i = x_i xor U, digit by digit, for one U ~ U[0, 1)^d shared by all points: the first
    ``bits`` digits of U are xor-ed into the points, and its remaining digits (a uniform
    fraction of 2^{-bits}, also shared) are appended.
    """
    X = np.asarray(X, dtype=np.uint64)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    D = rng.integers(0, 2**bits, size=d, dtype=np.uint64)
    tail = rng.random(d)
    return ((X ^ D).astype(float) + tail) / 2.0**bits


def owen_scramble(X: np.ndarray, bits: int = 32, seed=None) -> np.ndarray:
    """Owen's nested uniform scramble in base 2 of integer points X (times 2^bits).

    For each coordinate, digit k of a point is flipped by a random bit attached to the
    first k - 1 digits of that point: points that share their first k - 1 digits get the
    same flip at digit k, and different prefixes get independent flips. Once all prefixes
    of length L are distinct, the flips of every later digit are independent across points,
    so digits L + 1, L + 2, ... of the output are i.i.d. fair bits, and are drawn as one
    uniform fraction. Returns floats in (0, 1) (with probability one).
    """
    X = np.asarray(X, dtype=np.uint64)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    if n == 0 or d == 0:
        return np.zeros((n, d))
    L = max(_distinct_prefix_length(X[:, j], bits) for j in range(d))
    if 2**L <= 4 * n:
        # one random bit per node of each coordinate's tree, all coordinates at once
        top = (X >> np.uint64(bits - L)).astype(np.int64)      # the first L digits of each coordinate
        cols = np.arange(d, dtype=np.int64)
        Y = np.zeros((n, d), dtype=np.int64)
        for ell in range(L):
            node = (top >> (L - ell)) + (cols << ell)           # (coordinate, first ell digits), flattened
            table = rng.integers(0, 2, size=d << ell, dtype=np.uint8)   # one flip bit per node
            Y <<= 1
            Y |= ((top >> (L - 1 - ell)) & 1) ^ np.take(table, node)
        return (Y.astype(float) + rng.random((n, d))) / 2.0**L
    out = np.empty((n, d))                           # sparse trees: index the prefixes present
    for j in range(d):
        x = X[:, j]
        Lj = _distinct_prefix_length(x, bits)
        y = np.zeros(n, dtype=np.uint64)
        for ell in range(Lj):
            prefix = (x >> np.uint64(bits - ell)).astype(np.int64) if ell else np.zeros(n, dtype=np.int64)
            uniq, inv = np.unique(prefix, return_inverse=True)
            flips = rng.integers(0, 2, size=uniq.size, dtype=np.uint64)[inv]
            digit = (x >> np.uint64(bits - 1 - ell)) & np.uint64(1)
            y = (y << np.uint64(1)) | (digit ^ flips)
        out[:, j] = (y.astype(float) + rng.random(n)) / 2.0**Lj
    return out


def _distinct_prefix_length(x: np.ndarray, bits: int) -> int:
    """Smallest L such that the first L binary digits of the integers x are pairwise distinct.

    Two numbers share exactly bits - bit_length(a xor b) leading digits, and the longest
    shared prefix in a set is attained by neighbours in sorted order. Equal numbers never
    separate: then L = bits.
    """
    xs = np.sort(np.asarray(x, dtype=np.uint64))
    if xs.size < 2:
        return 0
    diff = xs[1:] ^ xs[:-1]
    if np.any(diff == 0):
        return bits
    common = bits - np.frexp(diff.astype(float))[1]        # frexp exponent = bit length (exact below 2^53)
    return int(common.max()) + 1


def rqmc_points(n: int, d: int, method: str = "owen", seed=None, sequence: str = "sobol",
                bits: int = 32) -> np.ndarray:
    """One randomised copy of the first n points of a sequence.

    sequence "sobol": method "owen" (nested uniform scramble), "digital" (random digital
    shift), "shift" (random shift modulo 1) or "none". sequence "halton": "shift" or
    "none". sequence "random": n independent uniform points (plain Monte Carlo).
    """
    if sequence == "random":
        return np.random.default_rng(seed).random((n, d))
    if sequence == "sobol":
        X = sobol_integers(n, d, bits=bits)
        if method == "owen":
            return owen_scramble(X, bits, seed)
        if method == "digital":
            return digital_shift(X, bits, seed)
        P = X.astype(float) / 2.0**bits
    elif sequence == "halton":
        P = halton(n, d)
    else:
        raise ValueError("sequence must be 'sobol', 'halton' or 'random'")
    if method == "shift":
        return random_shift(P, seed)
    if method == "none":
        return P
    raise ValueError(f"method {method!r} is not available for {sequence!r}")


def rqmc_nested(f, ns, d: int, R: int = 16, method: str = "owen", seed=None,
                sequence: str = "sobol") -> np.ndarray:
    """Replicate estimates (1/n) sum_{i<n} f(y_i) for each n in ``ns``, shape (R, len(ns)).

    Each replicate randomises the first max(ns) points once and reads the running mean at
    every n, so the estimates along a row share their points. ``f`` maps an (n, d) array of
    points to n values, or to an (n, q) array of q integrands evaluated on the same points
    (then the output has shape (R, len(ns), q)). Replicates use independent seeds spawned
    from ``seed``.
    """
    ns = np.asarray(ns, dtype=int)
    N = int(ns.max())
    seeds = np.random.SeedSequence(seed).spawn(R)
    out = None
    for r in range(R):
        vals = np.asarray(f(rqmc_points(N, d, method, seeds[r], sequence)), dtype=float)
        est = np.cumsum(vals, axis=0)[ns - 1] / (ns if vals.ndim == 1 else ns[:, None])
        if out is None:
            out = np.empty((R,) + est.shape)
        out[r] = est
    return out


def rqmc_estimate(f, n: int, d: int, R: int = 16, method: str = "owen", seed=None,
                  sequence: str = "sobol", level: float = 0.95) -> dict:
    """Randomised QMC estimate of int f from R independent replicates of n points each.

    Returns the mean of the replicate estimates, their standard deviation ``sd``, the
    standard error sd / sqrt(R), a Student t_{R-1} interval and the replicates themselves.
    """
    reps = rqmc_nested(f, [n], d, R, method, seed, sequence)[:, 0]
    out = mc_estimate(reps, level=level, student=True)
    out.update(replicates=reps, R=R, n_points=n)
    return out


# ---------------------------------------------------------------------------------------
# Integrands
# ---------------------------------------------------------------------------------------


def to_normals(U: np.ndarray) -> np.ndarray:
    """Standard normals N^{-1}(U) by the accurate quantile of Chapter 5 (U in (0, 1))."""
    U = np.asarray(U, dtype=float)
    if np.any((U <= 0) | (U >= 1)):
        raise ValueError("uniforms must lie in (0, 1): skip the origin or randomise the points")
    return norm_inv(U)


def asian_call_payoffs(Z: np.ndarray, t, S, K, r, y, sigma, method: str = "increments"):
    """Discounted arithmetic and geometric average-price call payoffs on the dates t_1..t_m.

    The paths are exact GBM paths built from the normals Z (n x m) by the chosen
    construction (``paths.gbm_paths``): "increments", "bridge" or "pca". The averages are
    over the m dates t_1, ..., t_m (the initial price is not included).
    """
    S_path = gbm_paths(S, t, r=r, y=y, sigma=sigma, Z=Z, method=method)[:, 1:]
    T = float(np.asarray(t)[-1])
    disc = math.exp(-r * T)
    A = S_path.mean(axis=1)
    G = np.exp(np.log(S_path).mean(axis=1))
    return disc * np.maximum(A - K, 0.0), disc * np.maximum(G - K, 0.0)


def geometric_average_law(t, S, r, y, sigma):
    """Mean and variance of log G, G the geometric average of S on the dates t_1..t_m."""
    t = np.asarray(t, dtype=float)
    t = t[1:] if t[0] == 0.0 else t
    nu = r - y - 0.5 * sigma**2
    mean = math.log(S) + nu * t.mean()
    var = sigma**2 * np.minimum.outer(t, t).sum() / t.size**2
    return float(mean), float(var)


def geometric_asian_call(t, S, K, r, y, sigma) -> float:
    """Exact price of the discretely monitored geometric average-price call (log G is normal)."""
    mu, v = geometric_average_law(t, S, r, y, sigma)
    T = float(np.asarray(t)[-1])
    sd = math.sqrt(v)
    d2 = (mu - math.log(K)) / sd
    return float(math.exp(-r * T) * (math.exp(mu + 0.5 * v) * ndtr(d2 + sd) - K * ndtr(d2)))


def average_direction(t, method: str = "increments") -> np.ndarray:
    """Unit vector w with sum_i W_{t_i} = |A^T 1| (w . Z) under the construction W = A Z.

    The geometric average depends on the normals only through w . Z, a standard normal.
    """
    A = construction_matrix(t, method)
    w = A.T @ np.ones(A.shape[0])
    return w / np.linalg.norm(w)


def lognormal_call_share(c, t, S, K, r, y, sigma) -> np.ndarray:
    """Corr(g(X), g(X')) for the geometric Asian call g(X) = e^{-rT}(e^{mu + s X} - K)^+.

    (X, X') is a standard bivariate normal pair with correlation c in [0, 1]. By the
    pick-freeze identity this is the share Var(E[g(w.Z) | Z_v]) / Var g(w.Z) of the
    variance explained by the normals in v, where c = sum_{j in v} w_j^2. The inner
    expectation E[g(c x + sqrt(1 - c^2) Y)] is a Black--Scholes-type closed form; the outer
    one is computed by quadrature.
    """
    mu, v = geometric_average_law(t, S, r, y, sigma)
    s = math.sqrt(v)
    T = float(np.asarray(t)[-1])
    D = math.exp(-r * T)
    xstar = (math.log(K) - mu) / s

    def g(x):
        return D * max(math.exp(mu + s * x) - K, 0.0)

    m1 = gauss_expect(g, [xstar])
    var = gauss_expect(lambda x: g(x) ** 2, [xstar]) - m1 * m1

    def corr(cc):
        if cc >= 1.0:
            return 1.0
        q = s * math.sqrt(1.0 - cc * cc)

        def inner(x):                    # E[g(cc x + sqrt(1 - cc^2) Y)]
            m = mu + s * cc * x
            dm = (m - math.log(K)) / q
            return D * (math.exp(m + 0.5 * q * q) * ndtr(dm + q) - K * ndtr(dm))

        return (gauss_expect(lambda x: g(x) * inner(x), [xstar]) - m1 * m1) / var

    c = np.atleast_1d(np.asarray(c, dtype=float))
    return np.array([corr(float(cc)) for cc in c])


def ridge_shares(w, share_fn, ks=None) -> dict:
    """Variance shares of a payoff g(w . Z), |w| = 1, from its correlation function share_fn(c).

    Returns the shares of the first k normals, share_fn(sum_{j<=k} w_j^2), for k in ``ks``
    (all k by default), and the additive share sum_j share_fn(w_j^2).
    """
    w = np.asarray(w, dtype=float)
    w2 = w**2 / np.sum(w**2)
    ks = np.arange(1, w.size + 1) if ks is None else np.asarray(ks)
    cum = np.cumsum(w2)[ks - 1]
    return dict(k=ks, truncation=share_fn(np.minimum(cum, 1.0)), additive=float(np.sum(share_fn(w2))),
                linear=cum)


def pick_freeze_shares(f, m: int, ks, n: int = 2**16, seed=None) -> np.ndarray:
    """Pick-freeze estimates of Var(E[f(Z) | Z_1..Z_k]) / Var f(Z), Z ~ N(0, I_m), for k in ks.

    For each k, f is evaluated at Z and at Z' = (Z_1..Z_k, fresh Z'_{k+1..m}); the share is
    estimated by the sample covariance of the two over their pooled variance.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, m))
    Y = np.asarray(f(Z), dtype=float)
    out = []
    for k in ks:
        Zk = Z.copy()
        Zk[:, k:] = rng.standard_normal((n, m - k))
        Yk = np.asarray(f(Zk), dtype=float)
        mean = 0.5 * (Y.mean() + Yk.mean())
        var = 0.5 * (np.mean(Y**2) + np.mean(Yk**2)) - mean**2
        out.append((np.mean(Y * Yk) - mean**2) / var)
    return np.array(out)


def asian_grid(m: int, T: float = 1.0) -> np.ndarray:
    """Uniform monitoring grid 0, T/m, ..., T (the course uses monthly dates, m = 12)."""
    return time_grid(T, m)


def asian_rqmc(m: int, ns, R: int, seed, S, K, r, y, sigma, T: float = 1.0,
               methods=("increments", "bridge", "pca"), geometric: bool = False,
               square: bool = False) -> np.ndarray:
    """Randomised QMC replicates of the Asian call on m equally spaced dates, all constructions at once.

    Each replicate scrambles max(ns) Sobol' points in dimension m (nested uniform scramble),
    maps them to normals and prices the arithmetic call (and, if ``geometric``, the geometric
    one) under every construction in ``methods`` on the same points. Returns an array of
    shape (R, len(ns), q): column c is the arithmetic call under methods[c], followed, if
    ``geometric``, by the geometric calls in the same order, and, if ``square``, by the square
    of the arithmetic payoff under the last construction (an unbiased estimate of its second
    moment, from which the Monte Carlo variance sigma^2 follows).
    """
    t = asian_grid(m, T)

    def f(U):
        Z = to_normals(U)
        cols = [asian_call_payoffs(Z, t, S, K, r, y, sigma, method=me) for me in methods]
        out = [A for A, _ in cols] + ([G for _, G in cols] if geometric else [])
        if square:
            out.append(cols[-1][0] ** 2)
        return np.column_stack(out)

    return rqmc_nested(f, ns, m, R=R, method="owen", seed=seed)
