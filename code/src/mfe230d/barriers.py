"""Barrier and lookback options (Chapter 16).

Closed forms for continuously monitored barriers by the method of images (single and
double barriers, rebates paid at expiry, in-out parity), lookback options from the law of
the running maximum, exact prices of discretely monitored barriers by recursive
quadrature over the monitoring dates, the Broadie-Glasserman-Kou continuity correction,
and Monte Carlo with Brownian-bridge corrections (crossing probabilities and exact
sampling of the extreme of the bridge between two grid dates).

Conventions
-----------
* Under the risk-neutral measure S_t = S exp(X_t) with X_t = nu t + sigma W_t and
  nu = r - y - sigma^2 / 2 (``log_drift``); lam = 2 nu / sigma^2 is the exponent of the
  image weight (B / S)^lam.
* Barrier names: "down-out", "down-in", "up-out", "up-in"; payoff kinds "call", "put".
  A down barrier is below the spot, an up barrier above it. If the spot is already on the
  far side of the barrier (or on it), a continuously monitored out option is knocked out at
  once and the in option is the vanilla.
* Discrete monitoring: the dates are t_j = j T / m, j = 1, ..., m (time 0 is not a
  monitoring date), as in the course's table of discrete down-and-out calls.
* Rebates are paid at expiry: by an out option if it is knocked out, by an in option if it
  is never knocked in.
* Every sampler takes ``seed`` (anything numpy.random.default_rng accepts) or the normals.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr, zeta

from . import bs
from .paths import gbm_paths, time_grid
from .variance import control_variate, regression_control_variates

# beta = -zeta(1/2) / sqrt(2 pi): the Broadie-Glasserman-Kou constant, 0.5825971579...
BETA_BGK = float(-zeta(0.5) / np.sqrt(2.0 * np.pi))
SQRT2PI = np.sqrt(2.0 * np.pi)


def log_drift(r, y, sigma):
    """nu = r - y - sigma^2 / 2, the risk-neutral drift of log S."""
    return r - y - 0.5 * sigma**2


def _split(barrier):
    direction, io = barrier.split("-")
    if direction not in ("down", "up") or io not in ("in", "out"):
        raise ValueError("barrier must be 'down-out', 'down-in', 'up-out' or 'up-in'")
    return direction, io


# ---------------------------------------------------------------------------------------
# The law of the extremes (reflection principle with drift)
# ---------------------------------------------------------------------------------------


def extreme_tail(b, T, nu, sigma):
    """Q(max_{t<=T} X_t >= b) for b >= 0 and Q(min_{t<=T} X_t <= b) for b < 0.

    X_t = nu t + sigma W_t, X_0 = 0. Equals 1 at b = 0 (a barrier at the spot is reached at
    once). Computed in log space, so it stays finite for distant barriers.
    """
    b = np.asarray(b, dtype=float)
    s = sigma * np.sqrt(T)
    lam = 2.0 * nu / sigma**2
    with np.errstate(over="ignore", invalid="ignore"):
        up = ndtr((nu * T - b) / s) + np.exp(lam * b + log_ndtr((-b - nu * T) / s))
        dn = ndtr((b - nu * T) / s) + np.exp(lam * b + log_ndtr((b + nu * T) / s))
    return np.where(b >= 0, up, dn)


def log_density(x, T, nu, sigma):
    """Density p_T of X_T ~ N(nu T, sigma^2 T)."""
    s = sigma * np.sqrt(T)
    return np.exp(-0.5 * ((np.asarray(x, dtype=float) - nu * T) / s) ** 2) / (s * SQRT2PI)


def knockout_density(x, b, T, nu, sigma):
    """Density of X_T on the event that X has not reached b on [0, T] (b != 0).

    The image formula p_T(x) - exp(lam b) p_T(x - 2 b) on the side of b where X starts, and
    0 on the other side.
    """
    x = np.asarray(x, dtype=float)
    s = sigma * np.sqrt(T)
    lam = 2.0 * nu / sigma**2
    image = np.exp(lam * b - 0.5 * ((x - 2.0 * b - nu * T) / s) ** 2) / (s * SQRT2PI)
    alive = x < b if b > 0 else x > b
    return np.where(alive, np.maximum(log_density(x, T, nu, sigma) - image, 0.0), 0.0)


# ---------------------------------------------------------------------------------------
# European building blocks and single barriers by images
# ---------------------------------------------------------------------------------------


def corridor_price(x, K, lo, hi, T, r, y, sigma, kind="call"):
    """Black-Scholes price of g(S_T) 1{lo < S_T < hi}, g the call or put payoff, spot x.

    lo may be 0 and hi may be np.inf. Built from vanilla and digital prices (bs.py).
    """
    x = np.asarray(x, dtype=float)
    if kind == "call":
        a = max(K, lo)
        if a >= hi:
            return np.zeros_like(x)
        v = bs.call(x, a, T, r, y, sigma) + (a - K) * bs.digital_call(x, a, T, r, y, sigma)
        if np.isfinite(hi):
            v = v - bs.call(x, hi, T, r, y, sigma) - (hi - K) * bs.digital_call(x, hi, T, r, y, sigma)
        return v
    if kind == "put":
        c = min(K, hi)
        if c <= lo:
            return np.zeros_like(x)
        v = bs.put(x, c, T, r, y, sigma) + (K - c) * bs.digital_put(x, c, T, r, y, sigma)
        if lo > 0:
            v = v - bs.put(x, lo, T, r, y, sigma) - (K - lo) * bs.digital_put(x, lo, T, r, y, sigma)
        return v
    raise ValueError("kind must be 'call' or 'put'")


def _log_ndtr_diff(u1, u2):
    """log(N(u2) - N(u1)) for u1 <= u2, accurate in both tails (-inf when the difference is 0)."""
    u1, u2 = np.broadcast_arrays(np.asarray(u1, dtype=float), np.asarray(u2, dtype=float))
    upper = u1 > 0                                # both in the upper tail: use N(-u1) - N(-u2)
    a = np.where(upper, log_ndtr(-u1), log_ndtr(u2))
    c = np.where(upper, log_ndtr(-u2), log_ndtr(u1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = a + np.log1p(-np.exp(c - a))
    return np.where((u1 >= u2) | ~np.isfinite(a), -np.inf, out)


def weighted_corridor_price(x, logw, K, lo, hi, T, r, y, sigma, kind="call"):
    """exp(logw) times ``corridor_price(x, ...)``, computed in log space.

    The image terms of the barrier formulas multiply a tiny corridor price by a huge weight
    (B / S)^lam when sigma is small (lam is of order 1 / sigma^2); the product is computed
    here from Gaussian integrals whose logarithms are combined before exponentiating, so it
    stays accurate (and finite) as sigma -> 0.
    """
    x = np.asarray(x, dtype=float)
    logw = np.asarray(logw, dtype=float)
    s = sigma * np.sqrt(T)
    m = log_drift(r, y, sigma) * T
    lx = np.log(x)
    if kind == "call":
        a = max(K, lo)
        if a >= hi:
            return np.zeros(np.broadcast(x, logw).shape)
        z1 = np.log(a) - lx
        z2 = np.log(hi) - lx if np.isfinite(hi) else np.full_like(lx, np.inf)
        stock = np.exp(logw + lx - y * T + _log_ndtr_diff((z1 - m) / s - s, (z2 - m) / s - s))
        cash = K * np.exp(logw - r * T + _log_ndtr_diff((z1 - m) / s, (z2 - m) / s))
        return stock - cash
    if kind == "put":
        c = min(K, hi)
        if c <= lo:
            return np.zeros(np.broadcast(x, logw).shape)
        z1 = np.log(lo) - lx if lo > 0 else np.full_like(lx, -np.inf)
        z2 = np.log(c) - lx
        cash = K * np.exp(logw - r * T + _log_ndtr_diff((z1 - m) / s, (z2 - m) / s))
        stock = np.exp(logw + lx - y * T + _log_ndtr_diff((z1 - m) / s - s, (z2 - m) / s - s))
        return cash - stock
    raise ValueError("kind must be 'call' or 'put'")


def vanilla(S, K, T, r, y, sigma, kind="call"):
    return bs.call(S, K, T, r, y, sigma) if kind == "call" else bs.put(S, K, T, r, y, sigma)


def barrier_price(S, K, B, T, r, y, sigma, kind="call", barrier="down-out", rebate=0.0):
    """Continuously monitored single-barrier option (the eight Rubinstein-Reiner cases).

    Out price = G(S) - (B / S)^lam G(B^2 / S), with G the price of the European claim
    g(S_T) 1{S_T on the side of B where the spot starts}, the image term computed in log space
    (``weighted_corridor_price``); in price = vanilla - out (in-out parity). ``rebate`` is paid at T by the out option if it is knocked out, and by the in
    option if it is never knocked in. Vectorised over S.
    """
    direction, io = _split(barrier)
    S = np.asarray(S, dtype=float)
    nu = log_drift(r, y, sigma)
    lam = 2.0 * nu / sigma**2
    lo, hi = (B, np.inf) if direction == "down" else (0.0, B)
    knocked0 = S <= B if direction == "down" else S >= B
    Ss = np.where(knocked0, B * (2.0 if direction == "down" else 0.5), S)   # placeholder spot
    out = (weighted_corridor_price(Ss, 0.0, K, lo, hi, T, r, y, sigma, kind)
           - weighted_corridor_price(B * B / Ss, lam * np.log(B / Ss), K, lo, hi, T, r, y, sigma, kind))
    hit = extreme_tail(np.log(B / Ss), T, nu, sigma)
    D = np.exp(-r * T)
    if io == "out":
        return np.where(knocked0, rebate * D, out + rebate * D * hit)
    van = vanilla(S, K, T, r, y, sigma, kind)
    return np.where(knocked0, van, van - out + rebate * D * (1.0 - hit))


# ---------------------------------------------------------------------------------------
# Double barriers: the image series
# ---------------------------------------------------------------------------------------


def double_barrier_terms(w, T, sigma, tol=1e-14):
    """Smallest n such that the truncation bound sum_{j >= n} exp(-2 j^2 w^2 / (sigma^2 T))
    of the image series (Proposition bar-double) is below tol; w = log(U / L)."""
    c = 2.0 * w**2 / (sigma**2 * T)
    n = 1
    while True:
        j = np.arange(n, n + 200)
        if np.sum(np.exp(-c * j**2)) < tol:
            return n
        n += 1


def double_barrier_density(x, l, u, T, nu, sigma, n_terms=None):
    """Density of X_T on {l < X_t < u for all t <= T}, l < 0 < u, by the image series
    summed over |k| <= n_terms (default from ``double_barrier_terms``)."""
    x = np.asarray(x, dtype=float)
    w = u - l
    lam = 2.0 * nu / sigma**2
    n = double_barrier_terms(w, T, sigma) if n_terms is None else int(n_terms)
    s = sigma * np.sqrt(T)
    q = np.zeros_like(x)
    for k in range(-n, n + 1):
        q += np.exp(lam * k * w - 0.5 * ((x - 2 * k * w - nu * T) / s) ** 2)
        q -= np.exp(lam * (u + k * w) - 0.5 * ((x - 2 * u - 2 * k * w - nu * T) / s) ** 2)
    q /= s * SQRT2PI
    return np.where((x > l) & (x < u), q, 0.0)


def double_barrier_truncation_bound(n, l, u, T, nu, sigma):
    """Bound on sup |q - q_n| over (l, u) for the image series truncated at |k| <= n."""
    w = u - l
    c = 2.0 * w**2 / (sigma**2 * T)
    j = np.arange(n, n + 400)
    return 4.0 * np.exp(abs(nu) * w / sigma**2) / (sigma * np.sqrt(2 * np.pi * T)) * np.sum(np.exp(-c * j**2))


def double_barrier_price(S, K, L, U, T, r, y, sigma, kind="call", n_terms=None):
    """Continuously monitored double knock-out (L < S < U) by the image series:
    sum_k (U/L)^{lam k} [G(S (U/L)^{2k}) - (U/S)^lam G((U^2/S) (U/L)^{2k})],
    G the price of g(S_T) 1{L < S_T < U}. Returns 0 if S is outside (L, U)."""
    S = float(S)
    if not L < S < U:
        return 0.0
    nu = log_drift(r, y, sigma)
    lam = 2.0 * nu / sigma**2
    ratio = U / L
    n = double_barrier_terms(np.log(ratio), T, sigma) if n_terms is None else int(n_terms)
    lr = np.log(ratio)

    def G(x, logw):
        return float(weighted_corridor_price(x, logw, K, L, U, T, r, y, sigma, kind))

    total = 0.0
    for k in range(-n, n + 1):          # weights in log space: lam is of order 1 / sigma^2
        total += (G(S * ratio ** (2 * k), lam * k * lr)
                  - G(U * U / S * ratio ** (2 * k), lam * k * lr + lam * np.log(U / S)))
    return total


def double_barrier_sine_density(x, l, u, T, nu, sigma, n_terms=50):
    """The same density by the eigenfunction (sine) series, which converges fast for large T."""
    x = np.asarray(x, dtype=float)
    w = u - l
    mu = nu / sigma
    k = np.arange(1, n_terms + 1)[:, None]
    # driftless, unit-volatility series in y = x / sigma, then the Girsanov factor
    wy, ly, yy = w / sigma, l / sigma, x[None, :] / sigma
    series = (2.0 / wy) * np.sum(np.exp(-0.5 * (k * np.pi / wy) ** 2 * T)
                                 * np.sin(k * np.pi * (-ly) / wy) * np.sin(k * np.pi * (yy - ly) / wy), axis=0)
    q = np.exp(mu * x / sigma - 0.5 * mu**2 * T) * series / sigma
    return np.where((x > l) & (x < u), q, 0.0)


# ---------------------------------------------------------------------------------------
# Lookbacks
# ---------------------------------------------------------------------------------------


def _J(kappa, c, d, s):
    """int_c^inf exp(kappa z) N((d - z) / s) dz (finite for every real kappa)."""
    a = (d - c) / s
    if abs(kappa) < 1e-10:
        return s * bs.norm_pdf(a) + (d - c) * ndtr(a)
    return (np.exp(kappa * d + 0.5 * kappa**2 * s**2) * ndtr(a + kappa * s) - np.exp(kappa * c) * ndtr(a)) / kappa


def exp_max_moment(kappa, c, T, nu, sigma):
    """E[exp(kappa max(c, M_T))], M_T = max_{t<=T} X_t, for c >= 0 and real kappa."""
    s = sigma * np.sqrt(T)
    lam = 2.0 * nu / sigma**2
    return np.exp(kappa * c) + kappa * (_J(kappa, c, nu * T, s) + _J(kappa + lam, c, -nu * T, s))


def max_call_tail(c, T, nu, sigma):
    """E[(exp(M_T) - exp(c))^+] = int_c^inf e^z Q(M_T > z) dz for c >= 0."""
    s = sigma * np.sqrt(T)
    lam = 2.0 * nu / sigma**2
    return _J(1.0, c, nu * T, s) + _J(1.0 + lam, c, -nu * T, s)


def lookback_floating(S, T, r, y, sigma, kind="put", extreme=None):
    """Floating-strike lookback: put pays max S - S_T, call pays S_T - min S.

    ``extreme`` is the maximum (put) or minimum (call) already reached, default S
    (inception).
    """
    nu = log_drift(r, y, sigma)
    if kind == "put":
        c = 0.0 if extreme is None else np.log(extreme / S)
        if c < 0:
            raise ValueError("the running maximum cannot be below the spot")
        return S * np.exp(-r * T) * exp_max_moment(1.0, c, T, nu, sigma) - S * np.exp(-y * T)
    if kind == "call":
        c = 0.0 if extreme is None else np.log(S / extreme)          # -log(min/S) >= 0
        if c < 0:
            raise ValueError("the running minimum cannot be above the spot")
        # min(log(m/S), min X) = -max(c, max(-X)), and -X has drift -nu
        return S * np.exp(-y * T) - S * np.exp(-r * T) * exp_max_moment(-1.0, c, T, -nu, sigma)
    raise ValueError("kind must be 'call' or 'put'")


def lookback_fixed(S, K, T, r, y, sigma, kind="call", extreme=None):
    """Fixed-strike lookback: call pays (max S - K)^+, put pays (K - min S)^+."""
    nu = log_drift(r, y, sigma)
    if kind == "call":
        Mx = S if extreme is None else extreme
        if K <= Mx:
            return (lookback_floating(S, T, r, y, sigma, "put", Mx) + S * np.exp(-y * T)
                    - K * np.exp(-r * T))
        return S * np.exp(-r * T) * max_call_tail(np.log(K / S), T, nu, sigma)
    if kind == "put":
        Mn = S if extreme is None else extreme
        if K >= Mn:
            return (lookback_floating(S, T, r, y, sigma, "call", Mn) - S * np.exp(-y * T)
                    + K * np.exp(-r * T))
        # (K - S e^{min X})^+ = S (k - e^{-max(-X)})^+, k = K/S < 1: integrate e^{-z} Q(max(-X) > z)
        c = -np.log(K / S)
        s = sigma * np.sqrt(T)
        lam = -2.0 * nu / sigma**2
        tail = _J(-1.0, c, -nu * T, s) + _J(-1.0 + lam, c, nu * T, s)
        return S * np.exp(-r * T) * tail
    raise ValueError("kind must be 'call' or 'put'")


# ---------------------------------------------------------------------------------------
# Discrete monitoring: exact prices by recursive quadrature, and the BGK correction
# ---------------------------------------------------------------------------------------


def discrete_corridor_price(S, K, lo, hi, T, r, y, sigma, m, kind="call",
                            order=8, panels_per_sd=1.0, width=8.0):
    """Price of g(S_T) 1{lo < S_{t_j} < hi for j = 1..m}, t_j = j T / m, by recursion.

    v_{m-1}(x) is the one-step price of the corridor claim (closed form); then
    v_{j}(x) = 1{alive} int p_h(x' - x) v_{j+1}(x') dx' over the alive interval, by
    composite Gauss-Legendre (``order`` nodes per panel, panels of width sd / panels_per_sd,
    sd = sigma sqrt(h)), and the price is exp(-rT) int p_{h}(x') v_1(x') dx'.
    lo may be 0 and hi np.inf (single barriers). The log domain is truncated at
    +-``width`` standard deviations sigma sqrt(T) around the drift.
    """
    h = T / m
    nu = log_drift(r, y, sigma)
    sd = sigma * np.sqrt(h)
    disc = np.exp(-r * T)

    def last_step(x):   # undiscounted one-step value of the corridor claim
        return np.exp(r * h) * corridor_price(S * np.exp(x), K, lo, hi, h, r, y, sigma, kind)

    if m == 1:
        return float(disc * last_step(0.0))
    a = min(0.0, nu * T) - width * sigma * np.sqrt(T)
    c = max(0.0, nu * T) + width * sigma * np.sqrt(T)
    if lo > 0:
        a = max(a, np.log(lo / S))
    if np.isfinite(hi):
        c = min(c, np.log(hi / S))
    if c <= a:
        return 0.0
    npan = int(np.ceil((c - a) * panels_per_sd / sd))
    g, wts = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, c, npan + 1)
    left, right = edges[:-1, None], edges[1:, None]
    x = (0.5 * (left + right) + 0.5 * (right - left) * g).ravel()
    wx = (0.5 * (right - left) * wts).ravel()
    v = last_step(x)                                       # v_{m-1} on the alive nodes
    P = np.exp(-0.5 * ((x[None, :] - x[:, None] - nu * h) / sd) ** 2) * (wx[None, :] / (sd * SQRT2PI))
    for _ in range(m - 2):
        v = P @ v
    p0 = np.exp(-0.5 * ((x - nu * h) / sd) ** 2) * (wx / (sd * SQRT2PI))
    return float(disc * (p0 @ v))


def discrete_barrier_price(S, K, B, T, r, y, sigma, m, kind="call", barrier="down-out", **quad):
    """Discretely monitored barrier option (dates j T / m, j = 1..m) by recursive quadrature;
    in options by in-out parity with the vanilla."""
    direction, io = _split(barrier)
    lo, hi = (B, np.inf) if direction == "down" else (0.0, B)
    out = discrete_corridor_price(S, K, lo, hi, T, r, y, sigma, m, kind, **quad)
    if io == "out":
        return out
    return float(vanilla(S, K, T, r, y, sigma, kind)) - out


def bgk_barrier(B, sigma, h, direction):
    """Shifted barrier B exp(+beta sigma sqrt(h)) for an up barrier, exp(-...) for a down one."""
    sign = 1.0 if direction == "up" else -1.0
    return B * np.exp(sign * BETA_BGK * sigma * np.sqrt(h))


def bgk_price(S, K, B, T, r, y, sigma, m, kind="call", barrier="down-out"):
    """Continuous closed form evaluated at the BGK-shifted barrier (m equally spaced dates)."""
    direction, _ = _split(barrier)
    return barrier_price(S, K, bgk_barrier(B, sigma, T / m, direction), T, r, y, sigma, kind, barrier)


def spitzer_expected_max(m, T, nu, sigma):
    """E[max_{0<=j<=m} X_{jT/m}] exactly, by Spitzer's identity sum_{k=1}^m E[X_{kh}^+] / k."""
    k = np.arange(1, m + 1)
    h = T / m
    mean, sd = nu * k * h, sigma * np.sqrt(k * h)
    return float(np.sum((mean * ndtr(mean / sd) + sd * bs.norm_pdf(mean / sd)) / k))


def expected_max(T, nu, sigma):
    """E[max_{t<=T} X_t] = int_0^inf Q(M_T > z) dz."""
    s = sigma * np.sqrt(T)
    lam = 2.0 * nu / sigma**2
    return float(_J(0.0, 0.0, nu * T, s) + _J(lam, 0.0, -nu * T, s))


# ---------------------------------------------------------------------------------------
# The Brownian bridge between grid dates
# ---------------------------------------------------------------------------------------


def crossing_probability(x0, x1, b, sigma, h, direction="up"):
    """Probability that X reaches b between two dates h apart, given X = x0 and x1 there.

    up: exp(-2 (b - x0)(b - x1) / (sigma^2 h)) if both are below b, else 1;
    down: exp(-2 (x0 - b)(x1 - b) / (sigma^2 h)) if both are above b, else 1.
    The drift does not enter. Works on log prices (or on Brownian paths with sigma = 1).
    """
    x0, x1 = np.asarray(x0, dtype=float), np.asarray(x1, dtype=float)
    if direction == "up":
        d0, d1 = b - x0, b - x1
    else:
        d0, d1 = x0 - b, x1 - b
    with np.errstate(over="ignore"):
        p = np.exp(-2.0 * np.maximum(d0, 0.0) * np.maximum(d1, 0.0) / (sigma**2 * h))
    return np.where((d0 > 0) & (d1 > 0), p, 1.0)


def bridge_extreme(x0, x1, sigma, h, u, which="max"):
    """Exact draw of the max (or min) of the bridge from x0 to x1 over a step of length h,
    by inversion of its conditional distribution with the uniforms u in (0, 1]."""
    x0, x1 = np.asarray(x0, dtype=float), np.asarray(x1, dtype=float)
    root = np.sqrt((x1 - x0) ** 2 - 2.0 * sigma**2 * h * np.log(u))
    return 0.5 * (x0 + x1 + root) if which == "max" else 0.5 * (x0 + x1 - root)


# ---------------------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------------------


def barrier_mc_samples(S, K, B, T, r, y, sigma, m, n=None, seed=None, kind="call", barrier="down-out",
                       monitor="grid", Z=None, U=None):
    """Discounted payoff samples of a single-barrier out option from paths on m equal steps.

    monitor = "grid": the barrier is checked at the m dates (the discrete contract exactly;
              a biased estimator of the continuous one);
              "bridge": continuous monitoring, the path between dates integrated out:
              the payoff is weighted by prod_j (1 - p_j), p_j = ``crossing_probability``;
              "bridge-sample": continuous monitoring, the extreme of the bridge on each step
              drawn exactly (``bridge_extreme``) with uniforms U (shape (n, m)).
    Either seed (with n) or the normals Z (shape (n, m)) is given; U is drawn from the
    same generator after Z when not supplied. In options: use parity with the vanilla.
    """
    direction, io = _split(barrier)
    if io != "out":
        raise ValueError("simulate the out option and use in-out parity")
    rng = np.random.default_rng(seed)
    if Z is None:
        Z = rng.standard_normal((int(n), int(m)))
    t = time_grid(T, m)
    X = np.log(gbm_paths(1.0, t, r, y, sigma, Z=Z))
    b = np.log(B / S)
    ST = S * np.exp(X[:, -1])
    pay = np.exp(-r * T) * (np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0))
    h = T / m
    if monitor == "grid":
        alive = (X[:, 1:].min(axis=1) > b) if direction == "down" else (X[:, 1:].max(axis=1) < b)
        return pay * alive
    if monitor == "bridge":
        p = crossing_probability(X[:, :-1], X[:, 1:], b, sigma, h, direction)
        return pay * np.prod(1.0 - p, axis=1)
    if monitor == "bridge-sample":
        if U is None:
            U = 1.0 - rng.random(Z.shape)          # in (0, 1]
        which = "min" if direction == "down" else "max"
        ext = bridge_extreme(X[:, :-1], X[:, 1:], sigma, h, U, which)
        alive = (ext.min(axis=1) > b) if direction == "down" else (ext.max(axis=1) < b)
        return pay * alive
    raise ValueError("monitor must be 'grid', 'bridge' or 'bridge-sample'")


def discrete_barrier_cv_study(S=100.0, K=100.0, B=95.0, T=1.0, r=0.01, y=0.02, sigma=0.2, m=12,
                              n=1_000_000, seed=2026, chunk=200_000):
    """The course's Monte Carlo study of the discrete down-and-out call (m dates).

    On n paths (and their n antithetic partners): plain estimator; antithetic pairs; control
    variates with known means: the continuously monitored down-and-out call (bridge
    minimum sampled exactly on each step, or survival-probability weighted), and the
    vanilla call; and both controls together (regression). Returns a dict of results (mean,
    se, and for controls the coefficient and correlation) plus exact reference values.
    """
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in ("disc", "disc_a", "cont_s", "cont_p", "van")}
    done = 0
    while done < n:
        k = min(chunk, n - done)
        Z = rng.standard_normal((k, m))
        U = 1.0 - rng.random((k, m))
        common = dict(S=S, K=K, B=B, T=T, r=r, y=y, sigma=sigma, m=m, kind="call", barrier="down-out")
        parts["disc"].append(barrier_mc_samples(Z=Z, monitor="grid", **common))
        parts["disc_a"].append(barrier_mc_samples(Z=-Z, monitor="grid", **common))
        parts["cont_s"].append(barrier_mc_samples(Z=Z, U=U, monitor="bridge-sample", **common))
        parts["cont_p"].append(barrier_mc_samples(Z=Z, monitor="bridge", **common))
        ST = S * np.exp(log_drift(r, y, sigma) * T + sigma * np.sqrt(T / m) * Z.sum(axis=1))
        parts["van"].append(np.exp(-r * T) * np.maximum(ST - K, 0.0))
        done += k
    d = {k: np.concatenate(v) for k, v in parts.items()}
    cont = float(barrier_price(S, K, B, T, r, y, sigma, "call", "down-out"))
    van = float(bs.call(S, K, T, r, y, sigma))

    def plain(v):
        return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(v.size)))

    out = dict(exact_discrete=discrete_barrier_price(S, K, B, T, r, y, sigma, m),
               exact_continuous=cont, exact_vanilla=van, n=n)
    out["plain"] = plain(d["disc"])
    out["antithetic"] = plain(0.5 * (d["disc"] + d["disc_a"]))
    out["antithetic"]["rho"] = float(np.corrcoef(d["disc"], d["disc_a"])[0, 1])
    out["cv_cont_sampled_beta1"] = control_variate(d["disc"], d["cont_s"], cont, beta=1.0)
    out["cv_cont_sampled"] = control_variate(d["disc"], d["cont_s"], cont)
    out["cv_cont_prob"] = control_variate(d["disc"], d["cont_p"], cont)
    out["cv_vanilla_beta1"] = control_variate(d["disc"], d["van"], van, beta=1.0)
    out["cv_vanilla"] = control_variate(d["disc"], d["van"], van)
    out["cv_both"] = regression_control_variates(d["disc"], np.column_stack([d["cont_s"], d["van"]]),
                                                 [cont, van])
    out["cv_both_prob"] = regression_control_variates(d["disc"], np.column_stack([d["cont_p"], d["van"]]),
                                                      [cont, van])
    out["control_samples_mean"] = dict(cont_sampled=plain(d["cont_s"]), cont_prob=plain(d["cont_p"]),
                                       vanilla=plain(d["van"]))
    return out
