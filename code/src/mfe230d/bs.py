"""Black--Scholes closed forms with a continuous dividend yield (or foreign rate) y.

Every function is vectorised over its arguments (NumPy broadcasting). Conventions follow
the notes: spot S, strike K, time to maturity T (years), interest rate r, yield y,
volatility sigma, and

    d_pm = [log(S/K) + (r - y +- sigma^2/2) T] / (sigma sqrt(T)).

The course's test case S = K = 100, T = 1, r = 1%, y = 2%, sigma = 20% gives the call
price 7.36429 (see ``TEST_CASE``).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

TEST_CASE = dict(S=100.0, K=100.0, T=1.0, r=0.01, y=0.02, sigma=0.20)


def norm_pdf(x):
    """Standard normal density N'(x)."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def norm_cdf(x):
    """Standard normal distribution function N(x)."""
    return ndtr(x)


def d_pm(S, K, T, r, y, sigma):
    """The pair (d_+, d_-)."""
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    sd = sigma * np.sqrt(T)
    dp = (np.log(S / K) + (r - y + 0.5 * sigma**2) * T) / sd
    return dp, dp - sd


def forward(S, T, r, y):
    """Forward price S e^{(r-y)T}."""
    return np.asarray(S, dtype=float) * np.exp((r - y) * np.asarray(T, dtype=float))


def call(S, K, T, r, y, sigma):
    """European call: S e^{-yT} N(d_+) - K e^{-rT} N(d_-)."""
    dp, dm = d_pm(S, K, T, r, y, sigma)
    return S * np.exp(-y * T) * ndtr(dp) - K * np.exp(-r * T) * ndtr(dm)


def put(S, K, T, r, y, sigma):
    """European put: K e^{-rT} N(-d_-) - S e^{-yT} N(-d_+)."""
    dp, dm = d_pm(S, K, T, r, y, sigma)
    return K * np.exp(-r * T) * ndtr(-dm) - S * np.exp(-y * T) * ndtr(-dp)


def digital_call(S, K, T, r, y, sigma):
    """Cash-or-nothing call paying 1 if S_T > K: e^{-rT} N(d_-)."""
    _, dm = d_pm(S, K, T, r, y, sigma)
    return np.exp(-r * T) * ndtr(dm)


def digital_put(S, K, T, r, y, sigma):
    """Cash-or-nothing put paying 1 if S_T < K: e^{-rT} N(-d_-)."""
    _, dm = d_pm(S, K, T, r, y, sigma)
    return np.exp(-r * T) * ndtr(-dm)


def greeks(S, K, T, r, y, sigma, kind="call"):
    """Delta, gamma, vega, theta (per year, dV/dt = -dV/dT), rho_r and rho_y.

    Returns a dict of arrays. ``kind`` is "call" or "put".
    """
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    dp, dm = d_pm(S, K, T, r, y, sigma)
    dq, dr = np.exp(-y * T), np.exp(-r * T)
    pdf = norm_pdf(dp)
    gamma = dq * pdf / (S * sigma * np.sqrt(T))
    vega = S * dq * pdf * np.sqrt(T)
    if kind == "call":
        delta = dq * ndtr(dp)
        theta = (-S * dq * pdf * sigma / (2 * np.sqrt(T))
                 + y * S * dq * ndtr(dp) - r * K * dr * ndtr(dm))
        rho_r = K * T * dr * ndtr(dm)
        rho_y = -T * S * dq * ndtr(dp)
    elif kind == "put":
        delta = -dq * ndtr(-dp)
        theta = (-S * dq * pdf * sigma / (2 * np.sqrt(T))
                 - y * S * dq * ndtr(-dp) + r * K * dr * ndtr(-dm))
        rho_r = -K * T * dr * ndtr(-dm)
        rho_y = T * S * dq * ndtr(-dp)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho_r=rho_r, rho_y=rho_y)


def implied_vol(price, S, K, T, r, y, kind="call", lo=1e-6, hi=5.0):
    """Implied volatility of one option price by Brent's method on [lo, hi].

    Raises ValueError if the price is outside the no-arbitrage bounds for that bracket.
    """
    f = call if kind == "call" else put
    g = lambda s: float(f(S, K, T, r, y, s)) - price
    if g(lo) > 0 or g(hi) < 0:
        raise ValueError("price outside the range spanned by volatilities in [lo, hi]")
    return brentq(g, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
