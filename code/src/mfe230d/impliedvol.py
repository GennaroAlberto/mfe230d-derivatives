"""Implied volatility by Newton's method (Chapter 2).

Everything is done in normalised variables. With the forward F = S e^{(r-y)T}, the discount factor
D = e^{-rT}, the log-moneyness m = log(F/K) and the total volatility w = sigma sqrt(T),

    C / (D K) = c(m, w) = e^m N(d_+) - N(d_-),    d_pm = m/w +- w/2.

For fixed m, c is increasing in w with derivative N'(d_-) (the normalised vega). It is convex for
w < w* and concave for w > w*, where w* = sqrt(2|m|) is the inflection point, and for m <= 0 its
logarithm is concave on the whole half-line (Proposition bsf-implied-vol of the notes). Hence:

* ``newton_price``: Newton on c itself, started at w* (the start of Manaster and Koehler, 1982, with the
  yield included), converges monotonically; from other starts it can fail (``newton_iterates``).
* ``implied_total_vol``: Newton on log c, with a halving safeguard, converges from every positive start,
  monotonically once an iterate falls below the root. This is the solver used by ``implied_vol``.

Prices are first reduced to out-of-the-money options (put-call parity, then put-call symmetry), so the solvers
only see m <= 0 and a target 0 < beta < e^m. ``bs.implied_vol`` (Brent's method from SciPy) is used only to
test ours.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfcx

from mfe230d import bs

SQRT2PI = np.sqrt(2.0 * np.pi)
LOG_SQRT2PI = 0.5 * np.log(2.0 * np.pi)


def normalised_call(m, w):
    """c(m, w) = e^m N(m/w + w/2) - N(m/w - w/2); limits (e^m - 1)^+ at w = 0 and e^m at w = inf."""
    m, w = np.broadcast_arrays(np.asarray(m, dtype=float), np.asarray(w, dtype=float))
    out = np.where(w == np.inf, np.exp(m), np.maximum(np.expm1(m), 0.0))
    pos = (w > 0) & np.isfinite(w)
    mm, ww = m[pos], w[pos]
    dp = mm / ww + 0.5 * ww
    out[pos] = np.exp(mm) * bs.norm_cdf(dp) - bs.norm_cdf(dp - ww)
    return out[()] if out.ndim == 0 else out


def normalised_vega(m, w):
    """dc/dw = N'(d_-) = e^m N'(d_+), with d_- = m/w - w/2."""
    m, w = np.asarray(m, dtype=float), np.asarray(w, dtype=float)
    return bs.norm_pdf(m / w - 0.5 * w)


def normalised_volga(m, w):
    """d^2c/dw^2 = d_+ d_- N'(d_-) / w = (m^2/w^2 - w^2/4) N'(d_-) / w."""
    m, w = np.asarray(m, dtype=float), np.asarray(w, dtype=float)
    return (m**2 / w**2 - 0.25 * w**2) * bs.norm_pdf(m / w - 0.5 * w) / w


def inflection_point(m):
    """w* = sqrt(2 |m|): c(m, .) is convex on (0, w*) and concave on (w*, inf)."""
    return np.sqrt(2.0 * np.abs(np.asarray(m, dtype=float)))


def mills(x):
    """Mills ratio R(x) = N(-x) / N'(x) = sqrt(pi/2) erfcx(x / sqrt 2)."""
    return np.sqrt(0.5 * np.pi) * erfcx(np.asarray(x, dtype=float) / np.sqrt(2.0))


def log_normalised_call(m, w):
    """log c(m, w) and the logarithmic derivative c'/c, for m <= 0 and 0 < w < inf.

    Below the inflection point (d_+ < 0) we use c = N'(d_-) [R(-d_+) - R(-d_-)] with the Mills ratio R,
    which avoids underflow for tiny prices; above it, the direct formula.
    """
    m, w = np.broadcast_arrays(np.asarray(m, dtype=float), np.asarray(w, dtype=float))
    dp = m / w + 0.5 * w
    dm = dp - w
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        gap = mills(-dp) - mills(-dm)                   # R(a) - R(b), a = -d_+ < b = -d_-
        low_log, low_dlog = -0.5 * dm**2 - LOG_SQRT2PI + np.log(gap), 1.0 / gap
        c = np.exp(m) * bs.norm_cdf(dp) - bs.norm_cdf(dm)
        high_log, high_dlog = np.log(c), bs.norm_pdf(dm) / c
    low = dp < 0
    return np.where(low, low_log, high_log), np.where(low, low_dlog, high_dlog)


def initial_guess(m, beta):
    """Starting value for m <= 0 and an out-of-the-money target 0 < beta < e^m.

    m = 0: sqrt(2 pi) beta (Brenner--Subrahmanyam), which lies below the root.
    m < 0, beta >= c(m, w*): the inflection point w*, which lies below the root.
    m < 0, beta <  c(m, w*): min(w*, w_g) with w_g = -2m / (l + sqrt(l^2 + 2m)), l = sqrt(-2 log beta), the root of
    d_-(w) = -l (the leading term of log c for small prices); either side of the root is fine for the solver.
    """
    m, beta = np.broadcast_arrays(np.asarray(m, dtype=float), np.asarray(beta, dtype=float))
    ws = inflection_point(m)
    c_star = 0.5 * np.exp(m) - bs.norm_cdf(-ws)          # at w*, d_+ = 0 and d_- = -w*
    with np.errstate(invalid="ignore", divide="ignore"):
        ell = np.sqrt(-2.0 * np.log(beta))
        wg = -2.0 * m / (ell + np.sqrt(np.maximum(ell**2 + 2.0 * m, 0.0)))
    w0 = np.where(beta >= c_star, ws, np.minimum(ws, wg))
    return np.where(m == 0, SQRT2PI * beta, w0)


def newton_iterates(m, beta, w0, n_steps=20):
    """Plain Newton iterates w_0, ..., w_n on the price, w <- w - (c(m, w) - beta) / N'(d_-), without any
    safeguard (for illustration). Stops early, padding with NaN, once an iterate is not a positive finite number
    (the last entry kept is that iterate)."""
    w = [float(w0)]
    for _ in range(n_steps):
        x = w[-1]
        if not (np.isfinite(x) and x > 0):
            break
        w.append(x - (float(normalised_call(m, x)) - beta) / float(normalised_vega(m, x)))
    return np.array(w + [np.nan] * (n_steps + 1 - len(w)))


def newton_price(m, beta, w0=None, xtol=1e-14, maxiter=200):
    """Newton on the price, w <- w - (c(m, w) - beta) / N'(d_-), from w0 (default: the inflection point w*,
    the start of Manaster and Koehler). Vectorised, no safeguard.

    Stops when the update is below xtol * w, when an update points the other way than the first one (from w*
    the iterates are monotone, so this only happens at rounding level), or when an iterate leaves (0, inf)
    (then w is NaN). Returns (w, iterations); entries still running after ``maxiter`` report maxiter.
    """
    m, beta = np.broadcast_arrays(np.asarray(m, dtype=float), np.asarray(beta, dtype=float))
    scalar = m.ndim == 0
    m, beta = np.atleast_1d(m), np.atleast_1d(beta)
    w = np.array(inflection_point(m) if w0 is None else np.broadcast_to(w0, m.shape), dtype=float)
    its = np.zeros(m.shape, dtype=int)
    first = np.zeros(m.shape)                       # sign of the first update
    active = np.ones(m.shape, dtype=bool)
    for _ in range(maxiter):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        ww = w[idx]
        f = normalised_call(m[idx], ww) - beta[idx]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            new = ww - f / normalised_vega(m[idx], ww)
        sgn = np.sign(new - ww)
        first[idx] = np.where(first[idx] == 0, sgn, first[idx])
        bad = ~(np.isfinite(new) & (new > 0))
        reverse = ~bad & ((sgn != first[idx]) | (f == 0))
        done = bad | reverse | (np.abs(new - ww) <= xtol * np.abs(new))
        w[idx] = np.where(bad, np.nan, np.where(reverse, ww, new))
        its[idx] += np.where(reverse, 0, 1)
        active[idx[done]] = False
    return (w[0], its[0]) if scalar else (w, its)


def iteration_study(n=20_000, seed=2026, m_max=2.0, w_range=(0.02, 2.0), floor=1e-14):
    """Random out-of-the-money cases: m uniform on [-m_max, 0], w log-uniform on w_range, kept when
    c(m, w) > floor * e^m. Returns a dict with the cases and, for ``implied_total_vol`` (default start) and
    ``newton_price`` from w*, the solutions and iteration counts."""
    rng = np.random.default_rng(seed)
    m = -m_max * rng.random(n)
    w = np.exp(np.log(w_range[0]) + np.log(w_range[1] / w_range[0]) * rng.random(n))
    beta = normalised_call(m, w)
    keep = beta > floor * np.exp(m)
    m, w, beta = m[keep], w[keep], beta[keep]
    w_log, it_log = implied_total_vol(m, beta)
    w_mk, it_mk = newton_price(m, beta)
    return dict(m=m, w=w, beta=beta, w_log=w_log, it_log=it_log, w_mk=w_mk, it_mk=it_mk)


def implied_total_vol(m, beta, w0=None, xtol=1e-15, maxiter=100):
    """Solve c(m, w) = beta for w > 0, for m <= 0 and 0 < beta < e^m (vectorised).

    Newton's method on G(w) = log c(m, w) - log beta, which is increasing and concave:
        w <- w - G(w) c(m, w) / N'(d_-(w)),   and w <- w/2 if that is not positive.
    From any start the iterates fall below the root after finitely many steps and then increase to it.
    Stops when the update is below xtol * w, or when an iterate below the root fails to increase (rounding).
    ``w0`` defaults to ``initial_guess``. Returns (w, number of iterations).
    """
    m, beta = np.broadcast_arrays(np.asarray(m, dtype=float), np.asarray(beta, dtype=float))
    scalar = m.ndim == 0
    m, beta = np.atleast_1d(m), np.atleast_1d(beta)
    if np.any(m > 0):
        raise ValueError("reduce to m <= 0 first (out-of-the-money option)")
    w = np.array(initial_guess(m, beta) if w0 is None else np.broadcast_to(w0, m.shape), dtype=float)
    log_beta = np.log(beta)
    its = np.zeros(m.shape, dtype=int)
    below = np.zeros(m.shape, dtype=bool)            # an iterate has been at or below the root
    active = np.ones(m.shape, dtype=bool)
    for _ in range(maxiter):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        ww = w[idx]
        lc, dlc = log_normalised_call(m[idx], ww)
        G = lc - log_beta[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            new = ww - G / dlc
        new = np.where(np.isfinite(new) & (new > 0), new, 0.5 * ww)
        was_below = below[idx] | (G <= 0)
        stall = was_below & (new <= ww)              # monotone phase: no increase means rounding level
        small = np.abs(new - ww) <= xtol * np.abs(new)
        new = np.where(stall, ww, new)
        w[idx] = new
        below[idx] = was_below
        its[idx] += np.where(stall, 0, 1)
        active[idx[stall | small | (G == 0)]] = False
    return (w[0], its[0]) if scalar else (w, its)


def otm_target(price, S, K, T, r, y, kind="call"):
    """Reduce a call or put price to (m, beta): m = -|log(F/K)| <= 0 and beta the normalised price of the
    out-of-the-money option, so that the implied total volatility solves c(m, w) = beta."""
    price, S, K, T = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (price, S, K, T)))
    m = np.log(S / K) + (r - y) * T
    p = price / (K * np.exp(-r * T))                # price in units of the discounted strike
    if kind == "call":
        call_hat, put_hat = p, p - np.expm1(m)        # put-call parity: c - p = e^m - 1
    elif kind == "put":
        call_hat, put_hat = p + np.expm1(m), p
    else:
        raise ValueError("kind must be 'call' or 'put'")
    # m <= 0: the call is out of the money. m > 0: the put is, and put-call symmetry gives
    # p_hat(m, w) = e^m c(-m, w).
    beta = np.where(m <= 0, call_hat, np.exp(-m) * put_hat)
    return -np.abs(m), beta


def _rounding_scale(price, S, K, T, r, y):
    """Size of the rounding error of the OTM target computed by ``otm_target`` (a few ulps of the input,
    normalised as beta is)."""
    price, S, K, T = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (price, S, K, T)))
    m = np.log(S / K) + (r - y) * T
    p = np.abs(price) / (K * np.exp(-r * T))
    return 8 * np.finfo(float).eps * (p + np.abs(np.expm1(m)) + 1.0) * np.exp(-np.maximum(m, 0.0))


def implied_vol(price, S, K, T, r, y, kind="call", w0=None, xtol=1e-15, maxiter=100, full_output=False):
    """Black--Scholes implied volatility of call or put prices (vectorised): our Newton solver on log prices.

    Conventions at the edges: a price equal to its lower no-arbitrage bound (S e^{-yT} - K e^{-rT})^+ for a call,
    (K e^{-rT} - S e^{-yT})^+ for a put, gives 0.0 (the limit sigma -> 0); a price equal to the upper bound
    S e^{-yT} (call) or K e^{-rT} (put) gives inf; a price outside the bounds, T <= 0, K <= 0 or S <= 0 gives NaN.
    With ``full_output``, also returns the iteration counts.
    """
    price, S, K, T = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (price, S, K, T)))
    sigma = np.full(price.shape, np.nan)
    its = np.zeros(price.shape, dtype=int)
    ok = (T > 0) & (K > 0) & (S > 0) & np.isfinite(price)
    if ok.any():
        m, beta = otm_target(price[ok], S[ok], K[ok], T[ok], r, y, kind)
        tol = _rounding_scale(price[ok], S[ok], K[ok], T[ok], r, y)
        s = np.full(m.shape, np.nan)
        it = np.zeros(m.shape, dtype=int)
        s[np.abs(beta) <= tol] = 0.0                    # at the lower bound, up to rounding
        s[np.abs(beta - np.exp(m)) <= tol] = np.inf      # at the upper bound, up to rounding
        inside = (beta > tol) & (beta < np.exp(m) - tol)
        if inside.any():
            start = None if w0 is None else np.broadcast_to(w0, price.shape)[ok][inside]
            w, n = implied_total_vol(m[inside], beta[inside], start, xtol, maxiter)
            s[inside] = w / np.sqrt(T[ok][inside])
            it[inside] = n
        sigma[ok], its[ok] = s, it
    sigma, its = (sigma[()], its[()]) if sigma.ndim == 0 else (sigma, its)
    return (sigma, its) if full_output else sigma
