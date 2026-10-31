"""Random numbers (Chapter 5): uniform generators, their tests, and samplers.

Contents
--------
* Linear congruential generators (LCGs): sequences, periods, the Hull--Dobell conditions,
  jump-ahead, and the lattice ("spectral") view of their d-tuples.
* PCG32, a small permuted congruential generator (an LCG on 64 bits with a permutation
  of the output), implemented from its definition. NumPy's ``default_rng`` (PCG64) is the
  reference generator everywhere else in the course.
* Statistical tests of a uniform sequence: moments, chi-square (one and several
  dimensions), Kolmogorov--Smirnov, autocorrelations, binomial error bars for histograms.
* Samplers: inversion (exponential, semicircle law with Newton and Halley, the normal
  quantile by Acklam's and Beasley--Springer--Moro's approximations with a Halley polish),
  acceptance--rejection (semicircle law with flat and tangent envelopes), Box--Muller and
  the polar method, Gaussian vectors by Cholesky and by principal components.

Uniforms are returned in the open interval (0, 1) unless stated otherwise, so that
``log(u)`` and quantile functions never see 0 or 1. All functions that draw random
numbers take a ``seed`` or a NumPy ``Generator``.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import ndtr
from scipy.stats import chi2

# ----------------------------------------------------------------------------------------
# Linear congruential generators
# ----------------------------------------------------------------------------------------

#: The worked example of the course: x_{i+1} = (9 x_i + 5) mod 16.
COURSE_LCG = dict(m=16, a=9, c=5)
#: IBM's RANDU (1960s): multiplicative, m = 2^31, a = 2^16 + 3. Its triples lie on 15 planes.
RANDU = dict(m=2**31, a=65539, c=0)
#: Park and Miller's "minimal standard": multiplicative, m = 2^31 - 1 (prime), a = 7^5.
MINSTD = dict(m=2**31 - 1, a=16807, c=0)
#: The full-period LCG modulo 2^32 of Numerical Recipes ("ranqd1").
NR32 = dict(m=2**32, a=1664525, c=1013904223)


def lcg_jump(m: int, a: int, c: int, k: int) -> tuple[int, int]:
    """Coefficients (A_k, C_k) with x_{i+k} = (A_k x_i + C_k) mod m.

    A_k = a^k mod m and C_k = c (1 + a + ... + a^{k-1}) mod m, computed by repeated
    squaring of the affine map x -> a x + c in O(log k) steps (exact Python integers).
    """
    A, C = 1, 0                     # the identity map
    a_pow, c_pow = a % m, c % m     # the map applied 2^j times
    while k > 0:
        if k & 1:
            A, C = (a_pow * A) % m, (a_pow * C + c_pow) % m
        a_pow, c_pow = (a_pow * a_pow) % m, (a_pow * c_pow + c_pow) % m
        k >>= 1
    return A, C


def lcg_sequence(n: int, m: int, a: int, c: int = 0, x0: int = 1, block: int = 1024) -> np.ndarray:
    """The first n states x_0 = x0, x_1, ..., x_{n-1} of x_{i+1} = (a x_i + c) mod m.

    Vectorised by jump-ahead: the first ``block`` states are computed one by one, and each
    later block is the previous one pushed forward by (A_B, C_B) = lcg_jump(m, a, c, B).
    Requires m <= 2^32 (products then fit in 64 unsigned bits) or m = 2^64 (wrap-around).
    """
    if not (m <= 2**32 or m == 2**64):
        raise ValueError("modulus must be at most 2^32, or exactly 2^64")
    n = int(n)
    out = np.empty(n, dtype=np.uint64)
    if n == 0:
        return out
    B = min(block, n)
    x = x0 % m
    for i in range(B):
        out[i] = x
        x = (a * x + c) % m
    A, C = lcg_jump(m, a, c, B)
    A64, C64 = np.uint64(A), np.uint64(C)
    for start in range(B, n, B):
        stop = min(start + B, n)
        prev = out[start - B:stop - B]
        if m == 2**64:
            out[start:stop] = prev * A64 + C64                      # wraps modulo 2^64
        else:
            M = np.uint64(m)                    # both factors < 2^32: no overflow
            out[start:stop] = ((prev * A64) % M + C64) % M
    return out


def lcg_uniforms(n: int, m: int, a: int, c: int = 0, seed: int = 1) -> np.ndarray:
    """u_i = x_i / m for i = 1, ..., n (the seed x_0 itself is not returned).

    The values lie in [0, 1); 0 occurs only if some x_i = 0 (never for a multiplicative
    generator with x_0 coprime to m).
    """
    return lcg_sequence(n + 1, m, a, c, seed)[1:].astype(float) / m


def lcg_period(m: int, a: int, c: int, x0: int) -> int:
    """Length of the cycle that the orbit of x0 eventually enters (brute force; small m)."""
    seen = {}
    x, i = x0 % m, 0
    while x not in seen:
        seen[x] = i
        x, i = (a * x + c) % m, i + 1
    return i - seen[x]


def _prime_factors(n: int) -> set[int]:
    out, p = set(), 2
    while p * p <= n:
        while n % p == 0:
            out.add(p)
            n //= p
        p += 1
    if n > 1:
        out.add(n)
    return out


def hull_dobell(m: int, a: int, c: int) -> bool:
    """True iff (m, a, c) satisfies the Hull--Dobell conditions for full period m.

    (i) gcd(c, m) = 1; (ii) every prime factor of m divides a - 1; (iii) if 4 | m then
    4 | a - 1. (For m = 1 every generator trivially has full period.)
    """
    if m == 1:
        return True
    if math.gcd(c, m) != 1:
        return False
    if any((a - 1) % p for p in _prime_factors(m)):
        return False
    return not (m % 4 == 0 and (a - 1) % 4)


# ----------------------------------------------------------------------------------------
# The lattice of d-tuples and the spectral test
# ----------------------------------------------------------------------------------------

def tuples(u: np.ndarray, d: int, overlapping: bool = True) -> np.ndarray:
    """The d-tuples (u_i, ..., u_{i+d-1}) of a sequence, overlapping or not."""
    u = np.asarray(u)
    if overlapping:
        return np.lib.stride_tricks.sliding_window_view(u, d)
    k = len(u) // d
    return u[:k * d].reshape(k, d)


def spectral_test(a: int, m: int, d: int) -> tuple[np.ndarray, float]:
    """Shortest nonzero h in Z^d with h_0 + h_1 a + ... + h_{d-1} a^{d-1} = 0 (mod m).

    Returns (h, nu_d) with nu_d = |h|_2. The d-tuples of an LCG with multiplier a and
    modulus m lie on parallel hyperplanes h.x = const at distance 1/nu_d, the widest spacing
    among the families h.x = const with h admissible for the full lattice mod m. A generator
    that does not visit all residues fills only part of that lattice, and coarser families
    exist: for multiplicative generators modulo 2^e, run the test with a reduced modulus
    (Knuth, TAOCP 2, Sec. 3.3.4). For RANDU (odd states), m = 2^30 gives the true spacing of
    the pairs, 4.3e-5, against 2.2e-5 with m = 2^31; nu_3 = sqrt(118) either way.

    d = 2: Lagrange--Gauss reduction of the basis (m, 0), (-a, 1).
    d >= 3: enumeration of (h_1, ..., h_{d-1}) in a box, with h_0 the residue of
    -(h_1 a + ...) closest to 0. A first pass over the box of radius floor(m^{1/d}) is
    certain to find a dual vector (pigeonhole, as in part (iii) of the lattice proposition);
    a second pass over the box whose radius is the norm of the best vector found cannot
    miss anything shorter. Feasible for d = 3 and m up to about 2^32 (m <= 2^32 required).
    """
    if d == 2:
        b1, b2 = np.array([m, 0], dtype=object), np.array([-(a % m), 1], dtype=object)
        n1, n2 = b1.dot(b1), b2.dot(b2)
        if n1 < n2:
            b1, b2, n1, n2 = b2, b1, n2, n1
        while True:                             # keep |b2| <= |b1|
            mu = round(b1.dot(b2) / n2)
            b1 = b1 - mu * b2
            n1 = b1.dot(b1)
            if n1 >= n2:
                break
            b1, b2, n1, n2 = b2, b1, n2, n1
        h = np.array([int(v) for v in b2])
        if h[np.flatnonzero(h)[0]] < 0:
            h = -h
        return h, math.sqrt(int(n2))
    if m > 2**32:
        raise ValueError("enumeration implemented for m <= 2^32")
    powers = np.array([pow(a, j, m) for j in range(d)], dtype=np.int64)
    best, best_norm2 = None, None

    def search(R, best, best_norm2):
        h1 = np.arange(-R, R + 1, dtype=np.int64)
        others = itertools.product(range(-R, R + 1), repeat=d - 2)   # (h_2, ..., h_{d-1})
        for rest in others:
            rest = np.array(rest, dtype=np.int64)
            base = int(rest @ powers[2:]) % m if d > 2 else 0
            s = (h1 * powers[1] + base) % m                 # |h1| * a < 2^63
            h0 = (-s) % m
            h0 = np.where(h0 > m // 2, h0 - m, h0)
            norm2 = h0.astype(float) ** 2 + h1.astype(float) ** 2 + float(rest @ rest)
            if not rest.any():
                norm2[R] = np.inf                           # exclude h = 0
            k = int(np.argmin(norm2))
            if best_norm2 is None or norm2[k] < best_norm2:
                best_norm2 = int(round(norm2[k]))
                best = np.concatenate([[h0[k], h1[k]], rest]).astype(np.int64)
        return best, best_norm2

    best, best_norm2 = search(int(round(m ** (1.0 / d))) + 1, best, best_norm2)
    best, best_norm2 = search(int(math.isqrt(best_norm2)), best, best_norm2)
    if best[np.flatnonzero(best)[0]] < 0:
        best = -best
    return best, math.sqrt(best_norm2)


def plane_coordinates(P: np.ndarray, h: np.ndarray) -> np.ndarray:
    """h.P / |h| for points P (one per row): the coordinate along the normal of the planes."""
    h = np.asarray(h, dtype=float)
    return P @ h / np.linalg.norm(h)


# ----------------------------------------------------------------------------------------
# PCG32: an LCG on 64 bits with a permuted 32-bit output (O'Neill, 2014)
# ----------------------------------------------------------------------------------------

PCG_MULT = 6364136223846793005


class PCG32:
    """The generator pcg32 of O'Neill's reference implementation (XSH-RR output).

    State: a 64-bit LCG s_{i+1} = s_i * PCG_MULT + inc mod 2^64 with odd increment inc
    (the "stream"). Output: from the old state s, the 32-bit word
    rotr32(((s >> 18) ^ s) >> 27, s >> 59). Seeding follows pcg32_srandom_r.
    """

    def __init__(self, initstate: int = 42, initseq: int = 54):
        self.inc = ((initseq << 1) | 1) % 2**64
        self.state = 0
        self._step()
        self.state = (self.state + initstate) % 2**64
        self._step()

    def _step(self):
        self.state = (self.state * PCG_MULT + self.inc) % 2**64

    @staticmethod
    def _output(old: np.ndarray) -> np.ndarray:
        old = old.astype(np.uint64)
        xorshifted = (((old >> np.uint64(18)) ^ old) >> np.uint64(27)) & np.uint64(0xFFFFFFFF)
        rot = old >> np.uint64(59)
        left = (np.uint64(32) - rot) & np.uint64(31)
        out = (xorshifted >> rot) | ((xorshifted << left) & np.uint64(0xFFFFFFFF))
        return out.astype(np.uint32)

    def random_raw(self, n: int) -> np.ndarray:
        """The next n 32-bit outputs (uint32)."""
        states = lcg_sequence(n + 1, 2**64, PCG_MULT, self.inc, self.state)
        self.state = int(states[-1])
        return self._output(states[:-1])

    def random(self, n: int) -> np.ndarray:
        """n uniforms (k + 1/2) / 2^32 in the open interval (0, 1), resolution 2^-32."""
        return (self.random_raw(n).astype(float) + 0.5) / 2.0**32


def uniforms(n: int, generator: str = "pcg64", seed: int = 0) -> np.ndarray:
    """n uniforms from a named generator: "pcg64" (NumPy), "pcg32", "randu", "minstd", "nr32".

    LCG seeds are mapped to a valid state (odd for RANDU, nonzero for MINSTD).
    """
    if generator == "pcg64":
        u = np.random.default_rng(seed).random(n)
        return np.where(u == 0.0, 2.0**-54, u)        # [0, 1) -> (0, 1)
    if generator == "pcg32":
        return PCG32(initstate=seed, initseq=54).random(n)
    if generator == "randu":
        return lcg_uniforms(n, **RANDU, seed=2 * seed + 1)
    if generator == "minstd":
        return lcg_uniforms(n, **MINSTD, seed=seed % (2**31 - 2) + 1)
    if generator == "nr32":
        return (lcg_sequence(n + 1, **NR32, x0=seed)[1:].astype(float) + 0.5) / 2.0**32
    raise ValueError(f"unknown generator {generator!r}")


# ----------------------------------------------------------------------------------------
# Tests of a uniform sequence
# ----------------------------------------------------------------------------------------

def sample_moments(x: np.ndarray) -> dict:
    """Mean, variance, skewness and excess kurtosis (moments about the sample mean)."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    d = x - m
    m2, m3, m4 = (d**2).mean(), (d**3).mean(), (d**4).mean()
    return dict(mean=m, var=m2, skew=m3 / m2**1.5, exkurt=m4 / m2**2 - 3.0)


def mean_test_uniform(u: np.ndarray) -> tuple[float, float]:
    """z = (mean - 1/2) / sqrt(1 / (12 n)) and its two-sided normal p-value."""
    n = len(u)
    z = (np.mean(u) - 0.5) / math.sqrt(1.0 / (12.0 * n))
    return z, 2.0 * ndtr(-abs(z))


def chi_square_uniform(u: np.ndarray, k: int) -> tuple[float, float]:
    """Pearson's statistic for equidistribution over k equal cells; p-value from chi^2_{k-1}."""
    u = np.asarray(u)
    counts = np.bincount(np.minimum((u * k).astype(np.int64), k - 1), minlength=k)
    e = len(u) / k
    stat = float(((counts - e) ** 2).sum() / e)
    return stat, float(chi2.sf(stat, k - 1))


def serial_test(u: np.ndarray, d: int, k: int) -> tuple[float, float]:
    """Chi-square test on the k^d cells of [0,1)^d for non-overlapping d-tuples."""
    P = tuples(u, d, overlapping=False)
    idx = np.minimum((P * k).astype(np.int64), k - 1)
    cell = np.zeros(len(P), dtype=np.int64)
    for j in range(d):
        cell = cell * k + idx[:, j]
    counts = np.bincount(cell, minlength=k**d)
    e = len(P) / k**d
    stat = float(((counts - e) ** 2).sum() / e)
    return stat, float(chi2.sf(stat, k**d - 1))


def kolmogorov_sf(t):
    """P(K > t) for Kolmogorov's limit law: 2 sum_{j>=1} (-1)^{j-1} exp(-2 j^2 t^2)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    j = np.arange(1, 101)[:, None]
    s = 2.0 * np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * j**2 * t**2), axis=0)
    s = np.where(t < 0.2, 1.0, np.clip(s, 0.0, 1.0))     # the series is useless near 0
    return s if s.size > 1 else float(s[0])


def ks_uniform(u: np.ndarray) -> tuple[float, float]:
    """Kolmogorov--Smirnov distance D_n = sup |F_n(x) - x| and its asymptotic p-value."""
    v = np.sort(np.asarray(u, dtype=float))
    n = len(v)
    i = np.arange(1, n + 1)
    D = float(max(np.max(i / n - v), np.max(v - (i - 1) / n)))
    return D, float(kolmogorov_sf(math.sqrt(n) * D))


def autocorrelation(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Sample autocorrelations r_1, ..., r_max_lag (the usual biased estimator)."""
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    den = d @ d
    return np.array([d[:-k] @ d[k:] / den for k in range(1, max_lag + 1)])


def binned_counts(x: np.ndarray, edges: np.ndarray, cdf) -> dict:
    """Counts in bins, their expectations n p_j and binomial standard deviations.

    Under the null hypothesis the count in bin j is Binomial(n, p_j) with
    p_j = cdf(edges[j+1]) - cdf(edges[j]). Also returns the counts below and above the range.
    """
    x = np.asarray(x)
    n = len(x)
    counts, _ = np.histogram(x, bins=edges)
    p = np.diff(cdf(np.asarray(edges, dtype=float)))
    expected = n * p
    sd = np.sqrt(n * p * (1 - p))
    return dict(counts=counts, expected=expected, sd=sd, z=(counts - expected) / sd,
                below=int(np.sum(x < edges[0])), above=int(np.sum(x >= edges[-1])))


def run_battery(u: np.ndarray) -> dict:
    """The tests of the table in Section 5.2 of the notes on one uniform sequence: z-scores and p-values."""
    n = len(u)
    z_mean, p_mean = mean_test_uniform(u)
    _, p_ks = ks_uniform(u)
    _, p_chi1 = chi_square_uniform(u, 1000)
    r1 = autocorrelation(u, 1)[0]
    _, p_s2 = serial_test(u, 2, 100)
    _, p_s3 = serial_test(u, 3, 20)
    return dict(z_mean=z_mean, p_mean=p_mean, p_ks=p_ks, p_chi1=p_chi1,
                z_lag1=r1 * math.sqrt(n), p_serial2=p_s2, p_serial3=p_s3)


# ----------------------------------------------------------------------------------------
# Inversion
# ----------------------------------------------------------------------------------------

def exponential_inverse(u, lam: float = 1.0):
    """Exponential(lam) by inversion: -log(u) / lam (1 - U has the law of U)."""
    return -np.log(np.asarray(u, dtype=float)) / lam


def semicircle_pdf(x):
    """Wigner's semicircle density (2/pi) sqrt(1 - x^2) on [-1, 1]."""
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < 1, 2.0 / np.pi * np.sqrt(np.clip(1 - x * x, 0, None)), 0.0)


def semicircle_cdf(x):
    """F(x) = 1/2 + (x sqrt(1 - x^2) + arcsin x) / pi on [-1, 1]."""
    x = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
    return 0.5 + (x * np.sqrt(1 - x * x) + np.arcsin(x)) / np.pi


def semicircle_inverse(u, iterations: int = 6, method: str = "newton", history: bool = False):
    """Solve F(x) = u by Newton or Halley iteration from the course's guess x_0 = 2u - 1.

    Newton: x <- x - (F(x) - u) / f(x). Halley: x <- x - 2 g f / (2 f^2 - g f'), with
    g = F(x) - u, f the density and f' its derivative. With ``history=True`` returns the
    array of iterates x_0, ..., x_iterations (shape (iterations + 1,) + u.shape).
    """
    u = np.asarray(u, dtype=float)
    x = 2.0 * u - 1.0
    out = [x]
    for _ in range(iterations):
        g = semicircle_cdf(x) - u
        f = semicircle_pdf(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            if method == "newton":
                step = g / f
            elif method == "halley":
                fp = -2.0 / np.pi * x / np.sqrt(1 - x * x)
                step = 2 * g * f / (2 * f * f - g * fp)
            else:
                raise ValueError(method)
        x = np.where(f > 0, np.clip(x - np.nan_to_num(step), -1.0, 1.0), x)
        out.append(x)
    return np.array(out) if history else x


# Acklam's rational approximation of the normal quantile (relative error < 1.15e-9).
_ACK_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACK_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
_ACK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)


def _horner(coef, x):
    out = np.zeros_like(x) + coef[0]
    for c in coef[1:]:
        out = out * x + c
    return out


def norm_inv_acklam(u):
    """Acklam's approximation of N^{-1}(u): rational in u - 1/2 centrally, in sqrt(-2 log u) in the tails."""
    u = np.asarray(u, dtype=float)
    lo = np.minimum(u, 1.0 - u)
    x = np.empty_like(u)
    central = lo >= 0.02425
    q = u[central] - 0.5
    r = q * q
    x[central] = _horner(_ACK_A, r) * q / (_horner(_ACK_B, r) * r + 1.0)
    tail = ~central
    t = np.sqrt(-2.0 * np.log(lo[tail]))
    xt = _horner(_ACK_C, t) / (_horner(_ACK_D, t) * t + 1.0)     # the lower-tail value
    x[tail] = np.where(u[tail] < 0.5, xt, -xt)
    return x


# Beasley--Springer (central) and Moro (tails), as in Glasserman (2003), Fig. 2.13.
_BSM_A = (2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637)
_BSM_B = (-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833)
_BSM_C = (0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863,
          0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002888167364,
          0.0000003960315187)


def norm_inv_bsm(u):
    """Beasley--Springer--Moro approximation of N^{-1}(u)."""
    u = np.asarray(u, dtype=float)
    y = u - 0.5
    x = np.empty_like(u)
    central = np.abs(y) < 0.42
    r = y[central] ** 2
    x[central] = y[central] * _horner(_BSM_A[::-1], r) / (_horner(_BSM_B[::-1], r) * r + 1.0)
    tail = ~central
    r = np.where(y[tail] > 0, 1.0 - u[tail], u[tail])
    s = np.log(-np.log(r))
    xt = _horner(_BSM_C[::-1], s)
    x[tail] = np.where(y[tail] < 0, -xt, xt)
    return x


def halley_polish(x, u):
    """One Halley step on N(x) = u, done in the lower half to keep relative accuracy.

    With t = (N(x) - u) / N'(x): x <- x - t / (1 + x t / 2). Upper-half points use the
    symmetry N^{-1}(u) = -N^{-1}(1 - u), where 1 - u is exact in floating point.
    """
    u = np.asarray(u, dtype=float)
    x = np.asarray(x, dtype=float)
    upper = u > 0.5
    v = np.where(upper, 1.0 - u, u)
    y = np.where(upper, -x, x)
    t = (ndtr(y) - v) * math.sqrt(2.0 * math.pi) * np.exp(0.5 * y * y)
    y = y - t / (1.0 + 0.5 * y * t)
    return np.where(upper, -y, y)


def norm_inv(u, polish: bool = True):
    """N^{-1}(u) for u in (0, 1): Acklam's approximation, then (by default) one Halley step."""
    x = norm_inv_acklam(u)
    return halley_polish(x, u) if polish else x


# ----------------------------------------------------------------------------------------
# Acceptance--rejection
# ----------------------------------------------------------------------------------------

def envelope_acceptance(theta: float) -> float:
    """Acceptance probability for the semicircle law under the envelope min(2/pi, tangents at +-sin(theta)).

    pi (1 + t) / (4 (1 + t^2)) with t = tan(theta / 2); theta = 0 is the flat envelope 2/pi
    (acceptance pi/4). Maximal at theta = pi/4, where it equals (1 + sqrt 2) pi / 8.
    """
    t = math.tan(theta / 2.0)
    return math.pi * (1.0 + t) / (4.0 * (1.0 + t * t))


def semicircle_envelope(x, theta: float):
    """The envelope chi(x) = (2/pi) min(1, (1 - sin(theta)|x|) / cos(theta)) on [-1, 1]."""
    x = np.asarray(x, dtype=float)
    a, ca = math.sin(theta), math.cos(theta)
    return np.where(np.abs(x) <= 1, 2.0 / np.pi * np.minimum(1.0, (1.0 - a * np.abs(x)) / ca), 0.0)


def _envelope_inverse(w, theta: float):
    """Inverse of the normalised envelope CDF (piecewise linear density), w in (0, 1)."""
    w = np.asarray(w, dtype=float)
    if theta == 0.0:
        return 2.0 * w - 1.0
    a = math.sin(theta)
    t = math.tan(theta / 2.0)
    side = (1.0 - t) / (1.0 + t)                 # mass of each side piece / (2/pi)
    total = 2.0 * (t + side)
    v = w * total                                # position in [0, total]
    s = np.where(v < total / 2, 1.0, -1.0)       # left half -> mirror of right half
    v = np.where(v < total / 2, total - v, v) - total / 2          # in [0, total/2]
    # on [0, t]: density 1 (scaled); on [t, 1]: (1 - a x)/cos(theta)
    ca = math.cos(theta)
    x_mid = v
    g = (v - t) * ca                             # = (x - t) - a (x^2 - t^2) / 2
    disc = np.clip((1.0 - a * t) ** 2 - 2.0 * a * g, 0.0, None)
    x_side = (1.0 - np.sqrt(disc)) / a
    x = np.where(v <= t, x_mid, x_side)
    return np.where(s > 0, -x, x)                # monotone in w


def semicircle_rejection(n: int, theta: float = 0.0, seed=None):
    """n draws from the semicircle law by rejection from the envelope with tangent angle theta.

    Each proposal uses two uniforms: one for X by inversion of the envelope, one for the
    test U chi(X) <= f(X). Returns (samples, number of proposals used).
    """
    rng = np.random.default_rng(seed)
    out, tried = [], 0
    p = envelope_acceptance(theta)
    while sum(len(o) for o in out) < n:
        need = n - sum(len(o) for o in out)
        m = int(need / p * 1.1) + 16
        x = _envelope_inverse(rng.random(m), theta)
        u = rng.random(m)
        ok = u * semicircle_envelope(x, theta) <= semicircle_pdf(x)
        acc = np.flatnonzero(ok)
        if len(acc) > need:                      # count proposals only up to the n-th acceptance
            m = acc[need - 1] + 1
            acc = acc[:need]
        tried += m
        out.append(x[acc])
    return np.concatenate(out), tried


def rejection_sample(n: int, target_pdf, proposal_rvs, proposal_pdf, c: float, seed=None):
    """Generic acceptance--rejection: accept Y ~ g when U c g(Y) <= f(Y).

    ``proposal_rvs(rng, m)`` draws m proposals. Requires f <= c g. Returns (samples, proposals).
    """
    rng = np.random.default_rng(seed)
    out, tried, have = [], 0, 0
    while have < n:
        need = n - have
        m = int(need * c * 1.1) + 16
        y = proposal_rvs(rng, m)
        u = rng.random(m)
        acc = np.flatnonzero(u * c * proposal_pdf(y) <= target_pdf(y))
        if len(acc) > need:
            m = acc[need - 1] + 1
            acc = acc[:need]
        tried += m
        out.append(y[acc])
        have += len(acc)
    return np.concatenate(out), tried


# ----------------------------------------------------------------------------------------
# Normal variates
# ----------------------------------------------------------------------------------------

def box_muller(u1, u2):
    """Z_1 = sqrt(-2 log u1) cos(2 pi u2), Z_2 = sqrt(-2 log u1) sin(2 pi u2); u1 in (0, 1]."""
    u1, u2 = np.asarray(u1, dtype=float), np.asarray(u2, dtype=float)
    r = np.sqrt(-2.0 * np.log(u1))
    return r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)


def normals_box_muller(u: np.ndarray) -> np.ndarray:
    """Standard normals from consecutive pairs (u_{2i}, u_{2i+1}), using both members of each pair."""
    u = np.asarray(u, dtype=float)
    k = len(u) // 2
    z1, z2 = box_muller(u[0:2 * k:2], u[1:2 * k:2])
    return np.column_stack([z1, z2]).ravel()


def polar_marsaglia(u: np.ndarray):
    """The polar method on consecutive pairs of uniforms.

    V = 2u - 1 in (-1, 1)^2; keep pairs with 0 < S = V_1^2 + V_2^2 < 1 and return
    V sqrt(-2 log S / S), both coordinates. Returns (normals, number of pairs used).
    """
    u = np.asarray(u, dtype=float)
    k = len(u) // 2
    v1, v2 = 2 * u[0:2 * k:2] - 1, 2 * u[1:2 * k:2] - 1
    s = v1 * v1 + v2 * v2
    ok = (s > 0) & (s < 1)
    f = np.sqrt(-2.0 * np.log(s[ok]) / s[ok])
    return np.column_stack([v1[ok] * f, v2[ok] * f]).ravel(), k


# ----------------------------------------------------------------------------------------
# Gaussian vectors
# ----------------------------------------------------------------------------------------

def cholesky(S: np.ndarray) -> np.ndarray:
    """Lower-triangular L with positive diagonal and L L^T = S, by the column recursion.

    L_jj = sqrt(S_jj - sum_{k<j} L_jk^2),  L_ij = (S_ij - sum_{k<j} L_ik L_jk) / L_jj  (i > j).
    Raises ValueError if a pivot is not positive, i.e. if S is not positive definite.
    """
    S = np.asarray(S, dtype=float)
    d = S.shape[0]
    if S.shape != (d, d) or not np.allclose(S, S.T):
        raise ValueError("S must be a symmetric square matrix")
    L = np.zeros_like(S)
    for j in range(d):
        pivot = S[j, j] - L[j, :j] @ L[j, :j]
        if pivot <= 0:
            raise ValueError(f"not positive definite: pivot {pivot:.3g} at column {j}")
        L[j, j] = math.sqrt(pivot)
        L[j + 1:, j] = (S[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    return L


def pca_factor(S: np.ndarray, r: int | None = None):
    """A = V_r Lambda_r^{1/2} from the r largest eigenpairs of S (all of them if r is None).

    Returns (A, eigenvalues in decreasing order). A A^T = S when r = d and S is positive
    semidefinite; for r < d, A A^T is the best rank-r approximation and the mean-square
    error of the truncated vector is the sum of the discarded eigenvalues.
    """
    lam, V = np.linalg.eigh(np.asarray(S, dtype=float))
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    if lam[-1] < -1e-10 * max(1.0, abs(lam[0])):
        raise ValueError(f"not positive semidefinite: smallest eigenvalue {lam[-1]:.3g}")
    lam = np.clip(lam, 0.0, None)
    r = len(lam) if r is None else r
    return V[:, :r] * np.sqrt(lam[:r]), lam


def correlated_normals(n: int, S: np.ndarray, mean=None, method: str = "cholesky", seed=None) -> np.ndarray:
    """n draws (rows) of N(mean, S) as mean + A Z with A from Cholesky or PCA."""
    S = np.asarray(S, dtype=float)
    A = cholesky(S) if method == "cholesky" else pca_factor(S)[0]
    Z = np.random.default_rng(seed).standard_normal((n, A.shape[1]))
    X = Z @ A.T
    return X if mean is None else X + np.asarray(mean, dtype=float)
