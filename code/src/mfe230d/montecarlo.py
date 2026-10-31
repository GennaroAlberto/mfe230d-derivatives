"""Plain Monte Carlo and its error (Chapter 6).

The estimator, its standard error and confidence intervals; batch ("binned") standard
errors; the bare-bones Black--Scholes Monte Carlo of the course; efficiency and the
bias-variance trade-off; the delta method and the ratio estimator; and deterministic
quadrature rules, to compare with Monte Carlo as the dimension grows.

Every sampler takes a ``seed`` (anything ``numpy.random.default_rng`` accepts) so that
figures and tests are reproducible. Nothing here reduces variance: that is the subject
of Chapters 8-12.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import ndtri
from scipy.stats import t as student_t

from . import bs


# ---------------------------------------------------------------------------
# The estimator and its error
# ---------------------------------------------------------------------------

def normal_quantile(level: float = 0.95) -> float:
    """z with N(z) = (1 + level) / 2: the half-width multiplier of a two-sided interval."""
    return float(ndtri(0.5 * (1.0 + level)))


def t_quantile(level: float, dof: int) -> float:
    """Two-sided Student-t multiplier t_{dof, (1 + level)/2}."""
    return float(student_t.ppf(0.5 * (1.0 + level), dof))


def mc_estimate(x, level: float = 0.95, student: bool = False) -> dict:
    """Monte Carlo estimate from samples ``x`` (one sample per entry, last axis).

    Returns a dict with the estimate ``mean`` (the sample mean), the sample standard
    deviation ``sd`` (divisor n - 1), the standard error ``se = sd / sqrt(n)`` and the
    confidence interval ``lo, hi`` with normal (or, if ``student``, t_{n-1}) quantiles.
    Works along the last axis, so a 2-D array gives one estimate per row.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if n < 2:
        raise ValueError("need at least two samples for a standard error")
    mean = x.mean(axis=-1)
    sd = x.std(axis=-1, ddof=1)
    se = sd / np.sqrt(n)
    q = t_quantile(level, n - 1) if student else normal_quantile(level)
    return dict(mean=mean, sd=sd, se=se, lo=mean - q * se, hi=mean + q * se, n=n)


def batch_estimate(x, m: int = 1024, level: float = 0.95) -> dict:
    """Batch-means estimate: split the n samples (last axis) into m consecutive batches.

    n must be a multiple of m. Returns the estimate (equal to the overall sample mean),
    the batch standard error sqrt( sum_j (Y_j - mean)^2 / (m (m - 1)) ) with Y_j the batch
    means, and a t_{m-1} confidence interval (Proposition "batch means" of the notes).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if m < 2 or n % m:
        raise ValueError(f"need m >= 2 batches dividing n; got n={n}, m={m}")
    Y = x.reshape(x.shape[:-1] + (m, n // m)).mean(axis=-1)
    mean = Y.mean(axis=-1)
    se = Y.std(axis=-1, ddof=1) / np.sqrt(m)
    q = t_quantile(level, m - 1)
    return dict(mean=mean, se=se, lo=mean - q * se, hi=mean + q * se, m=m, batch_means=Y)


def nested_estimates(x, ns, m: int = 1024) -> dict:
    """Estimates from the first n samples of one stream, for each n in ``ns``.

    For each n: the sample mean, the CLT standard error and the batch standard error with
    min(m, n) batches (with n <= m every batch holds one sample and the two standard
    errors coincide). This is how the convergence plots of the notes are drawn: one
    sequence of draws, read at n = 2, 4, 8, ...
    """
    x = np.asarray(x, dtype=float)
    out = dict(n=np.asarray(ns), mean=[], se=[], se_batch=[])
    for n in ns:
        e = mc_estimate(x[:n])
        b = batch_estimate(x[:n], m=min(m, n))
        out["mean"].append(e["mean"])
        out["se"].append(e["se"])
        out["se_batch"].append(b["se"])
    for k in ("mean", "se", "se_batch"):
        out[k] = np.asarray(out[k])
    return out


def coverage(sampler, n: int, reps: int, true_value: float, level: float = 0.95,
             seed=None, chunk: int = 2**22) -> float:
    """Fraction of ``reps`` independent nominal-``level`` CLT intervals that contain ``true_value``.

    ``sampler(rng, shape)`` must return i.i.d. samples of the given shape. Replications
    are processed in chunks of about ``chunk`` samples to bound memory.
    """
    rng = np.random.default_rng(seed)
    z = normal_quantile(level)
    hits, done = 0, 0
    per = max(1, chunk // n)
    while done < reps:
        b = min(per, reps - done)
        x = sampler(rng, (b, n))
        mean = x.mean(axis=1)
        se = x.std(axis=1, ddof=1) / np.sqrt(n)
        hits += int(np.sum(np.abs(mean - true_value) <= z * se))
        done += b
    return hits / reps


# ---------------------------------------------------------------------------
# The bare-bones Black--Scholes Monte Carlo
# ---------------------------------------------------------------------------

def terminal_prices(S, T, r, y, sigma, z):
    """S_T = S exp(nu T + sigma sqrt(T) z), nu = r - y - sigma^2/2, for standard normals z."""
    nu = r - y - 0.5 * sigma**2
    return S * np.exp(nu * T + sigma * np.sqrt(T) * np.asarray(z, dtype=float))


def bs_call_bbmc(S, K, T, r, y, sigma, n: int, seed=None, z=None) -> dict:
    """Bare-bones Monte Carlo for the Black--Scholes call, as in the course.

    Draws n standard normals (or uses ``z``), builds S_T exactly, and returns the
    estimates with CLT standard errors of the call price C_t = e^{-rT} E[(S_T-K)^+]
    (``price``), of the undiscounted E[(S_T-K)^+] (``payoff``) and of E[S_T] (``ST``),
    each as a pair (estimate, standard error), plus the samples themselves.
    """
    if z is None:
        z = np.random.default_rng(seed).standard_normal(n)
    ST = terminal_prices(S, T, r, y, sigma, z)
    CT = np.maximum(ST - K, 0.0)
    disc = np.exp(-r * T)
    eS, eC = mc_estimate(ST), mc_estimate(CT)
    return dict(price=(disc * eC["mean"], disc * eC["se"]),
                payoff=(eC["mean"], eC["se"]),
                ST=(eS["mean"], eS["se"]),
                samples=dict(ST=ST, CT=CT, z=np.asarray(z)))


def payoff_moments_quadrature(payoff, S, T, r, y, sigma, k: int = 200_000, width: float = 12.0,
                              shape: bool = False):
    """Mean and variance of the discounted payoff X = e^{-rT} f(S_T) by the midpoint rule.

    Integrates against the normal density on [-width, width] with k cells; this is the
    'numerical integration' route to the exact variance of a payoff (no closed form used).
    With ``shape=True`` also returns the Berry--Esseen ratio rho / sigma^3, with
    rho = E|X - mu|^3, and the kurtosis kappa = E(X - mu)^4 / sigma^4.
    """
    h = 2 * width / k
    z = -width + h * (np.arange(k) + 0.5)
    w = h * bs.norm_pdf(z)
    X = np.exp(-r * T) * payoff(terminal_prices(S, T, r, y, sigma, z))
    mean = float(np.sum(w * X))
    var = float(np.sum(w * (X - mean) ** 2))
    if not shape:
        return mean, var
    rho = float(np.sum(w * np.abs(X - mean) ** 3))
    mu4 = float(np.sum(w * (X - mean) ** 4))
    return mean, var, rho / var**1.5, mu4 / var**2


# ---------------------------------------------------------------------------
# Heavy tails
# ---------------------------------------------------------------------------

def pareto(alpha: float, size, seed=None, rng=None) -> np.ndarray:
    """Pareto samples with P(X > x) = x^{-alpha} for x >= 1, by inversion X = U^{-1/alpha}.

    The mean alpha/(alpha-1) is finite for alpha > 1; the variance is finite only for alpha > 2.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    u = 1.0 - rng.random(size)          # in (0, 1]: avoids U = 0
    return u ** (-1.0 / alpha)


def pareto_mean(alpha: float) -> float:
    return alpha / (alpha - 1.0) if alpha > 1 else np.inf


def berry_esseen_bound(ratio: float, n, constant: float = 0.4748):
    """Berry--Esseen bound C rho / (sigma^3 sqrt(n)) on the Kolmogorov distance to N(0,1).

    ``ratio`` is rho / sigma^3 with rho = E|X - mu|^3; the constant 0.4748 is Shevtsova's.
    """
    return constant * ratio / np.sqrt(np.asarray(n, dtype=float))


def bernoulli_skewness_ratio(p: float) -> float:
    """rho / sigma^3 for an indicator with success probability p: ((1-p)^2 + p^2) / sqrt(p (1-p))."""
    return ((1 - p) ** 2 + p**2) / np.sqrt(p * (1 - p))


# ---------------------------------------------------------------------------
# Efficiency and the bias-variance trade-off
# ---------------------------------------------------------------------------

def efficiency(var: float, cost: float) -> float:
    """Efficiency 1 / (variance per sample x cost per sample) (Glynn--Whitt)."""
    return 1.0 / (var * cost)


def relative_efficiency(var1: float, cost1: float, var2: float, cost2: float) -> float:
    """E_{2,1} = sigma_1^2 tau_1 / (sigma_2^2 tau_2): efficiency of method 2 relative to method 1."""
    return (var1 * cost1) / (var2 * cost2)


def optimal_allocation(budget: float, c_bias: float, beta: float, var: float,
                       c_cost: float, gamma: float) -> dict:
    """Minimise MSE(h) = c_bias^2 h^{2 beta} + var c_cost h^{-gamma} / budget over h > 0.

    Model: bias c_bias h^beta, per-sample variance ``var``, cost per sample c_cost h^{-gamma},
    and n = budget / (c_cost h^{-gamma}) samples. Returns the optimal h, n, the MSE and the
    share of squared bias in it (gamma / (2 beta + gamma)).
    """
    h = (gamma * var * c_cost / (2 * beta * c_bias**2 * budget)) ** (1.0 / (2 * beta + gamma))
    n = budget / (c_cost * h ** (-gamma))
    bias2 = c_bias**2 * h ** (2 * beta)
    mse = bias2 + var / n
    return dict(h=h, n=n, mse=mse, bias2_share=bias2 / mse)


# ---------------------------------------------------------------------------
# Functions of means: delta method and ratio estimator
# ---------------------------------------------------------------------------

def delta_method(grad, cov, n: int) -> float:
    """Asymptotic standard error sqrt(grad^T cov grad / n) of g(sample mean)."""
    grad = np.atleast_1d(np.asarray(grad, dtype=float))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    return float(np.sqrt(grad @ cov @ grad / n))


def ratio_estimate(yv, xv, level: float = 0.95) -> dict:
    """Ratio estimator of E[Y]/E[X] from paired samples, with its delta-method standard error.

    Returns the estimate R = mean(Y)/mean(X), the standard error from the residuals
    D_i = Y_i - R X_i, a normal confidence interval, and the plug-in estimate of the
    leading bias (R Var X - Cov(X, Y)) / (n mean(X)^2).
    """
    yv, xv = np.asarray(yv, dtype=float), np.asarray(xv, dtype=float)
    n = yv.shape[-1]
    xb, yb = xv.mean(), yv.mean()
    R = yb / xb
    d = yv - R * xv
    se = d.std(ddof=1) / (np.sqrt(n) * abs(xb))
    c = np.cov(xv, yv, ddof=1)
    bias = (R * c[0, 0] - c[0, 1]) / (n * xb**2)
    q = normal_quantile(level)
    return dict(mean=R, se=se, lo=R - q * se, hi=R + q * se, bias=bias)


# ---------------------------------------------------------------------------
# Deterministic quadrature on [0, 1]^d
# ---------------------------------------------------------------------------

def riemann_1d(f, k: int, side: str = "left") -> float:
    """Left or right Riemann sum of f on [0, 1] with k cells ('forward' / 'backward' Euler)."""
    h = 1.0 / k
    x = h * np.arange(k) + (h if side == "right" else 0.0)
    return float(h * np.sum(f(x)))


def trapezoid_1d(f, k: int) -> float:
    """Composite trapezoid rule on [0, 1] with k cells."""
    x = np.linspace(0.0, 1.0, k + 1)
    fx = f(x)
    return float((0.5 * fx[0] + fx[1:-1].sum() + 0.5 * fx[-1]) / k)


def midpoint_1d(f, k: int) -> float:
    """Composite midpoint rule on [0, 1] with k cells."""
    h = 1.0 / k
    return float(h * np.sum(f(h * (np.arange(k) + 0.5))))


def product_midpoint(f, d: int, k: int, chunk: int = 2**20) -> float:
    """Product midpoint rule on [0, 1]^d with k^d points.

    ``f`` maps an (N, d) array of points to N values. Points are generated in chunks of
    about ``chunk`` so that memory stays bounded for large k^d.
    """
    h = 1.0 / k
    nodes = h * (np.arange(k) + 0.5)
    N = k**d
    total = 0.0
    # enumerate the grid in blocks: the first coordinates vary slowest
    inner = 0
    while inner < d and k ** (inner + 1) <= chunk:
        inner += 1
    inner = max(inner, 1)
    tail = np.stack(np.meshgrid(*([nodes] * inner), indexing="ij"), axis=-1).reshape(-1, inner)
    for head in itertools.product(nodes, repeat=d - inner):
        pts = np.empty((tail.shape[0], d))
        pts[:, : d - inner] = head
        pts[:, d - inner:] = tail
        total += float(np.sum(f(pts)))
    return total / N


def mc_integrate(f, d: int, n: int, seed=None) -> dict:
    """Plain Monte Carlo estimate of the integral of f over [0, 1]^d with n uniform points."""
    u = np.random.default_rng(seed).random((n, d))
    return mc_estimate(f(u))


def sine_product(x) -> np.ndarray:
    """f(x) = prod_j (pi/2) sin(pi x_j): the course's 1-D test integrand in d dimensions.

    Its integral over [0, 1]^d is 1 for every d, and its variance under the uniform law
    is (pi^2/8)^d - 1.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return np.prod(0.5 * np.pi * np.sin(np.pi * x), axis=-1)
