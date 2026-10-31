"""Importance sampling (Chapter 9).

Conventions follow the notes. Under the risk-neutral measure Q the log-return
x_T = log(S_T / S) is N(nu T, sigma^2 T) with nu = r - y - sigma^2 / 2. Importance
sampling draws x_T from the sampling law N((nu + alpha) T, beta^2 sigma^2 T) instead and
multiplies the discounted payoff by the likelihood ratio L = q / q~ (target density over
sampling density). In standard units, with m = nu T and s = sigma sqrt(T), the shift is
a = alpha T, or c = a / s standard deviations; d = strike_to_z(K, ...) is the distance from
the centre of the law of Z to the strike, in standard deviations.

Path-dependent payoffs are written as functions of a vector Z of independent standard
normals (one per monitoring date); a mean shift Z ~ N(mu, I) has the weight
exp(-mu.Z + |mu|^2 / 2), which is also the Girsanov density of a piecewise-constant drift.

Every sampler takes a ``seed`` (anything numpy.random.default_rng accepts) and is
vectorised over samples. Closed forms are computed in log space, so that they stay finite
deep in the tails and close to the edge beta = 1/sqrt(2). Standard errors, paths and
control variates come from Chapters 6-8 (montecarlo.mc_estimate, paths.gbm_paths,
variance.control_variate); this module only adds what importance sampling needs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import log_ndtr, ndtr

from .paths import gbm_paths, time_grid
from .variance import strike_to_z  # noqa: F401  (d = strike_to_z(K, S, T, r, y, sigma))

LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Weights and diagnostics
# ---------------------------------------------------------------------------
def gaussian_log_lr(x, m, s, m_is, s_is):
    """log of the N(m, s^2) density over the N(m_is, s_is^2) density at x: the log weight."""
    x = np.asarray(x, dtype=float)
    return np.log(s_is / s) - 0.5 * ((x - m) / s) ** 2 + 0.5 * ((x - m_is) / s_is) ** 2


def effective_sample_size(weights):
    """ESS = (sum w)^2 / sum w^2 of nonnegative weights: between 1 and n, n iff all are equal."""
    w = np.asarray(weights, dtype=float)
    return w.sum() ** 2 / np.sum(w * w)


def self_normalized(Y, L):
    """Self-normalised importance-sampling estimator sum(f L) / sum(L), with Y = f L."""
    return float(np.sum(Y) / np.sum(L))


# ---------------------------------------------------------------------------
# Gaussian mean shift and scale (Proposition "is-gaussian")
# ---------------------------------------------------------------------------
def log_weight_second_moment(c, beta=1.0):
    """log E~[L^2] for target N(0, 1) and sampling law N(c, beta^2).

    Equals log(beta^2 / sqrt(2 beta^2 - 1)) + c^2 / (2 beta^2 - 1) when 2 beta^2 > 1,
    and +inf otherwise (the weight then has infinite variance).
    """
    c, beta = np.broadcast_arrays(np.asarray(c, float), np.asarray(beta, float))
    q = 2.0 * beta**2 - 1.0
    out = np.full(c.shape, np.inf)
    ok = q > 0
    out[ok] = np.log(beta[ok] ** 2 / np.sqrt(q[ok])) + c[ok] ** 2 / q[ok]
    return out if out.ndim else float(out)


def weight_second_moment(c, beta=1.0):
    """E~[L^2] = E_Q[L] for target N(0, 1) and sampling law N(c, beta^2); e^{c^2} if beta = 1."""
    return np.exp(log_weight_second_moment(c, beta))


def mirror_law(m, s, a, beta=1.0):
    """The ingredients of the mirror formula E~[(f L)^2] = E~[L^2] E[f(X~)^2].

    Target N(m, s^2), sampling N(m + a, beta^2 s^2) with 2 beta^2 > 1. Returns
    (log E~[L^2], m~, v~), where X~ ~ N(m~, v~), m~ = m - a / (2 beta^2 - 1) and
    v~ = beta^2 s^2 / (2 beta^2 - 1).
    """
    q = 2.0 * beta**2 - 1.0
    if q <= 0:
        raise ValueError("the mirror law needs beta > 1/sqrt(2)")
    return log_weight_second_moment(a / s, beta), m - a / q, beta**2 * s**2 / q


def _log_signed_sum(logs, signs):
    """log(sum_i sign_i exp(logs_i)) for a positive total, computed stably."""
    logs = np.asarray(logs, float)
    top = np.max(logs)
    return top + np.log(np.sum(np.asarray(signs) * np.exp(logs - top)))


def _log_payoff_moment(kind, S, K, mu, v, power):
    """log E[f^power] for f = (S e^Y - K)^+ ('call') or 1{S e^Y > K} ('digital'), Y ~ N(mu, v)."""
    sd, k = np.sqrt(v), np.log(K / S)
    d = lambda j: (mu + j * v - k) / sd  # noqa: E731
    if kind == "digital":
        return log_ndtr(d(0))
    if kind != "call":
        raise ValueError("kind must be 'call' or 'digital'")
    if power == 1:
        return _log_signed_sum([np.log(S) + mu + v / 2 + log_ndtr(d(1)), np.log(K) + log_ndtr(d(0))],
                               [1, -1])
    return _log_signed_sum([2 * np.log(S) + 2 * mu + 2 * v + log_ndtr(d(2)),
                            np.log(2 * K * S) + mu + v / 2 + log_ndtr(d(1)),
                            2 * np.log(K) + log_ndtr(d(0))], [1, -1, 1])


def is_moments(kind, S, K, T, r, y, sigma, alpha=0.0, beta=1.0):
    """Price and second moment per sample of the IS estimator of a European call or digital.

    The estimator is e^{-rT} f(S_T) L with log(S_T / S) ~ N((nu + alpha) T, beta^2 sigma^2 T)
    under the sampling law. Returns (price, E~[(e^{-rT} f L)^2]). For 2 beta^2 <= 1 the
    second moment is reported as +inf: it is infinite for 2 beta^2 < 1, and at 2 beta^2 = 1
    exactly it is finite for the call only when alpha > sigma^2 (see the notes).
    """
    m, s = (r - y - 0.5 * sigma**2) * T, sigma * np.sqrt(T)
    price = np.exp(-r * T + _log_payoff_moment(kind, S, K, m, s * s, 1))
    if 2 * beta**2 <= 1:
        return price, np.inf
    log_el, mt, vt = mirror_law(m, s, alpha * T, beta)
    return price, np.exp(-2 * r * T + log_el + _log_payoff_moment(kind, S, K, mt, vt, 2))


def variance_ratio(kind, S, K, T, r, y, sigma, alpha, beta=1.0):
    """Variance per sample of plain Monte Carlo over that of importance sampling."""
    p, m2_plain = is_moments(kind, S, K, T, r, y, sigma, 0.0, 1.0)
    _, m2 = is_moments(kind, S, K, T, r, y, sigma, alpha, beta)
    return (m2_plain - p * p) / (m2 - p * p)


def optimal_alpha(kind, S, K, T, r, y, sigma, beta=1.0, bracket=(-1.0, 5.0)):
    """The drift shift alpha that minimises the second moment (hence the variance) at fixed beta."""
    f = lambda a: np.log(is_moments(kind, S, K, T, r, y, sigma, a, beta)[1])  # noqa: E731
    return minimize_scalar(f, bounds=bracket, method="bounded", options={"xatol": 1e-10}).x


def sample_terminal_is(kind, S, K, T, r, y, sigma, alpha=0.0, beta=1.0, n=10_000, seed=None):
    """Samples of the IS estimator e^{-rT} f(S_T) L, together with S_T and L (for controls).

    Returns (Y, S_T, L), three arrays of length n; alpha = 0, beta = 1 is plain Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    m, s = (r - y - 0.5 * sigma**2) * T, sigma * np.sqrt(T)
    x = m + alpha * T + beta * s * rng.standard_normal(n)
    L = np.exp(gaussian_log_lr(x, m, s, m + alpha * T, beta * s))
    ST = S * np.exp(x)
    f = np.maximum(ST - K, 0.0) if kind == "call" else (ST > K).astype(float)
    return np.exp(-r * T) * f * L, ST, L


def ghs_shift_call(d, s):
    """GHS shift (standard units) for the call G(z) = (e^{s(z-d)} - 1)^+: the root mu > d of
    mu (e^{s(mu-d)} - 1) = s e^{s(mu-d)}, i.e. of grad log G(mu) = mu."""
    g = lambda mu: mu * np.expm1(s * (mu - d)) - s * np.exp(s * (mu - d))  # noqa: E731
    hi = d + 1.0
    while g(hi) < 0:
        hi += 1.0
    return brentq(g, d + 1e-12, hi, xtol=1e-14)


def ce_shift_digital(d):
    """Cross-entropy shift for 1{Z > d}: the conditional mean E[Z | Z > d] = N'(d) / N(-d)."""
    return float(np.exp(-0.5 * d * d - LOG_SQRT_2PI - log_tail(d)))


def ce_shift_call(d, s):
    """Cross-entropy shift for G(z) = (e^{s(z-d)} - 1)^+: E[Z G(Z)] / E[G(Z)], Z ~ N(0, 1).

    Uses E[Z e^{sZ} 1{Z>d}] = e^{s^2/2}(s N(s-d) + N'(d-s)) and E[Z 1{Z>d}] = N'(d).
    """
    e = np.exp(-s * d + 0.5 * s * s)
    num = e * (s * ndtr(s - d) + np.exp(-0.5 * (d - s) ** 2 - LOG_SQRT_2PI)) - np.exp(-0.5 * d * d - LOG_SQRT_2PI)
    den = e * ndtr(s - d) - ndtr(-d)
    return float(num / den)


# ---------------------------------------------------------------------------
# Rare events in standard units (Proposition "is-rare")
# ---------------------------------------------------------------------------
def log_tail(d):
    """log P(Z > d) = log N(-d) for Z ~ N(0, 1)."""
    return log_ndtr(-np.asarray(d, dtype=float))


def tail_shift_relative_moment(d, a):
    """E~[Y^2] / p^2 for p = P(Z > d) estimated with the mean shift Z ~ N(a, 1):
    e^{a^2} N(-(d + a)) / N(-d)^2."""
    a = np.asarray(a, float)
    return np.exp(a**2 + log_tail(np.asarray(d) + a) - 2 * log_tail(d))


def tail_exponential_relative_moment(d):
    """E~[Y^2] / p^2 for p = P(Z > d), d > 0, sampled as Z = d + E / d with E ~ Exp(1).

    Closed form: E~[Y^2] = sqrt(pi) e^{-3 d^2 / 4} N(-d / sqrt 2) / (2 pi d).
    """
    d = np.asarray(d, dtype=float)
    log_m2 = -0.75 * d**2 + 0.5 * np.log(np.pi) + log_tail(d / np.sqrt(2.0)) - np.log(2 * np.pi * d)
    return np.exp(log_m2 - 2 * log_tail(d))


def tail_exponential_is(d, n, seed=None):
    """Samples of the exponential IS estimator of P(Z > d): Z = d + E/d, weight N'(Z) / (d e^{-d(Z-d)})."""
    rng = np.random.default_rng(seed)
    z = d + rng.exponential(1.0, n) / d
    return np.exp(-0.5 * z * z - LOG_SQRT_2PI + d * (z - d) - np.log(d))


# ---------------------------------------------------------------------------
# Exponential tilting (Esscher) for sums of i.i.d. variables
# ---------------------------------------------------------------------------
def tilted_sum_estimator(sample_tilted, cgf, theta, n_terms, level, n, seed=None):
    """Samples of 1{S >= level} exp(-theta S + n_terms psi(theta)), S a sum of n_terms tilted draws.

    ``sample_tilted(rng, theta, shape)`` draws from the tilted law; ``cgf(theta)`` is
    psi(theta) = log E[e^{theta X}] under the original law.
    """
    rng = np.random.default_rng(seed)
    total = sample_tilted(rng, theta, (n, n_terms)).sum(axis=1)
    return (total >= level) * np.exp(-theta * total + n_terms * cgf(theta))


def poisson_tilt(lam):
    """(sampler, cgf) for X ~ Poisson(lam): the tilt by theta is Poisson(lam e^theta)."""
    return (lambda rng, th, shape: rng.poisson(lam * np.exp(th), shape),
            lambda th: lam * np.expm1(th))


def poisson_rate_function(lam, a):
    """theta_a = log(a / lam) and I(a) = a log(a / lam) - a + lam for Poisson(lam) summands."""
    theta = np.log(a / lam)
    return theta, a * theta - a + lam


# ---------------------------------------------------------------------------
# Path space: mean shifts of the driving normals and the GHS drift
# ---------------------------------------------------------------------------
def girsanov_weight(Z, mu):
    """exp(-mu.Z + |mu|^2 / 2), row by row: the weight of Z ~ N(mu, I) back to N(0, I)."""
    mu = np.asarray(mu, dtype=float)
    return np.exp(-np.asarray(Z) @ mu + 0.5 * mu @ mu)


def asian_dates_prices(Z, S, T, r, y, sigma):
    """S at the dates t_j = j T / m (j = 1..m) from normals Z of shape (n, m), via paths.gbm_paths."""
    Z = np.atleast_2d(Z)
    return gbm_paths(S, time_grid(T, Z.shape[1]), r, y, sigma, Z=Z)[:, 1:]


def ghs_drift(log_payoff_and_grad, z0):
    """Glasserman-Heidelberger-Shahabuddin drift: a maximiser of F(z) - |z|^2 / 2, F = log G.

    ``log_payoff_and_grad(z)`` returns (F(z), grad F(z)), with F = -inf where G = 0, and z0
    must satisfy G(z0) > 0. An interior maximiser solves grad F(mu) = mu. Uses BFGS.
    """
    def obj(z):
        F, g = log_payoff_and_grad(z)
        if not np.isfinite(F):          # outside {G > 0}: a large value pushes the search back
            return 1e12, -z
        return -(F - 0.5 * z @ z), -(g - z)
    return minimize(obj, np.asarray(z0, float), jac=True, method="BFGS", options={"gtol": 1e-10}).x


def asian_log_payoff(S, K, T, r, y, sigma, m):
    """F(z) = log(A(z) - K) and its gradient; A = arithmetic average of S at t_j = j T / m."""
    dt = T / m
    def F(z):
        Sj = asian_dates_prices(z, S, T, r, y, sigma)[0]
        A = Sj.mean()
        if A <= K:
            return -np.inf, None
        dA = np.cumsum(Sj[::-1])[::-1] * sigma * np.sqrt(dt) / m   # dA/dz_i = sum_{j>=i} S_j sigma sqrt(dt) / m
        return np.log(A - K), dA / (A - K)
    return F


def asian_ghs_drift(S, K, T, r, y, sigma, m):
    """GHS drift for the arithmetic Asian call on m equally spaced dates."""
    F = asian_log_payoff(S, K, T, r, y, sigma, m)
    z0 = np.zeros(m)
    while not np.isfinite(F(z0)[0]):    # start inside {A > K}
        z0 += 0.1
    return ghs_drift(F, z0)


def geometric_asian_call(S, K, T, r, y, sigma, m):
    """Exact price of the call on the geometric average of S at t_j = j T / m (the control)."""
    t = time_grid(T, m)[1:]
    mg = np.log(S) + (r - y - 0.5 * sigma**2) * t.mean()
    vg = sigma**2 * np.minimum.outer(t, t).sum() / m**2
    sd = np.sqrt(vg)
    return np.exp(-r * T) * (np.exp(mg + 0.5 * vg) * ndtr((mg + vg - np.log(K)) / sd)
                             - K * ndtr((mg - np.log(K)) / sd))


def asian_is_samples(S, K, T, r, y, sigma, m, mu, n, seed=None):
    """Samples of the IS estimator of the arithmetic Asian call with Z ~ N(mu, I).

    Returns (Y, C, L): the weighted discounted payoff, the weighted discounted geometric
    Asian payoff (a control whose mean is geometric_asian_call) and the weight.
    mu = 0 is plain Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (m,))
    Z = rng.standard_normal((n, m)) + mu
    L = girsanov_weight(Z, mu)
    P = asian_dates_prices(Z, S, T, r, y, sigma)
    disc = np.exp(-r * T)
    Y = disc * np.maximum(P.mean(axis=1) - K, 0.0) * L
    C = disc * np.maximum(np.exp(np.log(P).mean(axis=1)) - K, 0.0) * L
    return Y, C, L


# ---------------------------------------------------------------------------
# Cross-entropy (Proposition "is-cross-entropy")
# ---------------------------------------------------------------------------
def cross_entropy_drift(score, target, dim, n=10_000, n_iter=8, rho=0.1, kind="call", seed=None, v0=None):
    """Multilevel cross-entropy for the mean v of N(v, I), to estimate E[f(score(Z))].

    f is (score - level)^+ for kind='call' and 1{score >= level} for kind='digital'.
    Each iteration samples Z ~ N(v, I), sets level = min(target, (1 - rho)-quantile of the
    scores), weights w = f L_v(Z) and updates v = sum w Z / sum w, the cross-entropy optimum
    in the Gaussian mean family. Returns (array of the successive v, list of levels).
    """
    rng = np.random.default_rng(seed)
    v = np.zeros(dim) if v0 is None else np.asarray(v0, float)
    hist, levels = [v.copy()], []
    for _ in range(n_iter):
        Z = rng.standard_normal((n, dim)) + v
        sc = score(Z)
        level = min(target, float(np.quantile(sc, 1.0 - rho)))
        f = np.maximum(sc - level, 0.0) if kind == "call" else (sc >= level).astype(float)
        w = f * girsanov_weight(Z, v)
        v = (w @ Z) / w.sum()
        hist.append(v.copy())
        levels.append(level)
    return np.array(hist), levels


# ---------------------------------------------------------------------------
# Laboratory: importance sampling and control variates on the Asian call
# ---------------------------------------------------------------------------
def asian_laboratory(strikes=(100.0, 120.0, 140.0, 160.0), m=16, n=2**17, seed=2026, params=None):
    """Plain, control variate, IS (GHS drift) and IS + control for the arithmetic Asian call.

    For each strike: plain samples (mu = 0) and IS samples (GHS drift) are drawn with
    independent streams spawned from ``seed``; the control is the weighted geometric Asian
    payoff with its exact mean, fitted with the in-sample coefficient
    (variance.control_variate). Because the plain sample sees few payoffs deep out of the
    money, the variances per sample of plain Monte Carlo (``plain_var``) and of plain Monte
    Carlo with the optimal control (``cv_var``) are also estimated from the IS sample,
    through E_Q[h] = E~[h L], with the IS+CV price as the mean.
    Returns a list of dicts with the estimates, standard errors and the GHS drift.
    """
    from .bs import TEST_CASE
    from .variance import control_variate, plain
    p = dict(TEST_CASE) if params is None else dict(params)
    p.pop("K", None)
    seeds = np.random.SeedSequence(seed).spawn(2 * len(strikes))
    out = []
    for i, K in enumerate(strikes):
        geo = geometric_asian_call(K=K, m=m, **p)
        mu = asian_ghs_drift(K=K, m=m, **p)
        row = dict(K=K, mu=mu, geometric=geo)
        for name, drift, sd in (("plain", np.zeros(m), seeds[2 * i]), ("is", mu, seeds[2 * i + 1])):
            Y, C, L = asian_is_samples(K=K, m=m, mu=drift, n=n, seed=sd, **p)
            row[name] = plain(Y)
            row[name + "_cv"] = control_variate(Y, C, geo, beta="estimated")
            if name == "is":
                # Variances per sample of plain MC and of plain MC with the optimal control,
                # estimated from the IS sample: E_Q[h] = E~[h L] with h = (Y - b C)^2 / L^2.
                price = row["is_cv"]["mean"]
                A, B, D = np.mean(Y * Y / L), np.mean(Y * C / L), np.mean(C * C / L)
                b = (B - geo * price) / (D - geo**2)
                row["plain_var"] = float(A - price**2)
                row["cv_var"] = float(A - 2 * b * B + b * b * D - (price - b * geo) ** 2)
        row["n"] = n
        out.append(row)
    return out
