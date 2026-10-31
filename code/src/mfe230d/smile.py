"""Arbitrage in the strike and the smile (Chapter 3).

Call prices C(K) at one maturity T, as functions of the strike:

* ``strike_checks`` tests quotes at finitely many strikes against the strike conditions
  (bounds, monotonicity, slopes in [-e^{-rT}, 0], convexity);
* ``second_difference`` and ``fd_density`` estimate the Breeden--Litzenberger density
  e^{rT} C''(K) from prices, and ``fd_density_error_bound`` / ``fd_optimal_step`` give the
  bias-plus-noise bound for quotes with errors of size eps;
* ``smile_call``, ``smile_digital`` and ``smile_density`` price a call, a digital call and
  the discounted density C''(K) from an implied-volatility smile K -> sigma(K) and its
  strike derivatives (the skew term -vega * sigma'(K) and the second-order terms);
* ``call_spread_bounds`` gives the cheapest super- and the dearest sub-replication of a
  digital call by call spreads on a strike grid;
* two synthetic smiles: ``course_smile`` (the course's near-the-money skew
  sigma_atm + xi ln(S/K) + kappa ln(S/K)^2) and ``ssvi_smile`` (the SSVI surface of
  Gatheral and Jacquier with the power-law function phi), with ``durrleman_g``;
* ``atom_call``: a law of S_T with an atom (a lognormal mixed with a point mass), whose
  call price has a kink.

Black--Scholes closed forms come from ``bs``. Every function is vectorised (NumPy
broadcasting) unless it says otherwise. Smile functions return the triple
(sigma, d sigma/dK, d^2 sigma/dK^2).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from . import bs


# ---------------------------------------------------------------------------------------
# Strike conditions on quotes
# ---------------------------------------------------------------------------------------

def strike_checks(K, C, S, T, r, y, tol=1e-12):
    """Check call quotes C at strikes 0 < K_1 < ... < K_n against the strike conditions.

    With K_0 = 0 and C_0 = S e^{-yT} (the call with strike 0 is the prepaid forward), the
    slopes s_i = (C_i - C_{i-1}) / (K_i - K_{i-1}) must satisfy

        -e^{-rT} <= s_1 <= s_2 <= ... <= s_n <= 0   and   C_n >= 0.

    Returns a dict with the slopes, the butterfly weights s_{i+1} - s_i (i = 1..n-1), and
    booleans 'slope_low' (s_1 >= -e^{-rT}), 'monotone' (s_n <= 0), 'convex' (one per
    interior strike), 'nonneg' (C_n >= 0) and 'ok' (all of them).
    """
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    if K.ndim != 1 or K.shape != C.shape or np.any(np.diff(K) <= 0) or K[0] <= 0:
        raise ValueError("K must be increasing and positive, with one quote per strike")
    k = np.concatenate([[0.0], K])
    c = np.concatenate([[S * np.exp(-y * T)], C])
    s = np.diff(c) / np.diff(k)
    fly = np.diff(s)
    out = dict(slopes=s, butterflies=fly,
               slope_low=bool(s[0] >= -np.exp(-r * T) - tol),
               monotone=bool(s[-1] <= tol),
               convex=fly >= -tol,
               nonneg=bool(C[-1] >= -tol))
    out["ok"] = out["slope_low"] and out["monotone"] and bool(np.all(out["convex"])) and out["nonneg"]
    return out


# ---------------------------------------------------------------------------------------
# Breeden--Litzenberger by finite differences
# ---------------------------------------------------------------------------------------

def second_difference(f, K, h):
    """(f(K - h) - 2 f(K) + f(K + h)) / h^2 for a vectorised function f."""
    K = np.asarray(K, dtype=float)
    return (f(K - h) - 2.0 * f(K) + f(K + h)) / h**2


def fd_density(call_fn, K, h, r, T):
    """Estimate of the risk-neutral density of S_T at K: e^{rT} times the second difference of C."""
    return np.exp(r * T) * second_difference(call_fn, K, h)


def fd_density_error_bound(h, M4, eps):
    """Bound h^2 M4 / 12 + 4 eps / h^2 on |second difference of noisy prices - C''(K)|.

    M4 bounds |C''''| on [K - h, K + h]; eps bounds the price errors.
    """
    h = np.asarray(h, dtype=float)
    return h**2 * M4 / 12.0 + 4.0 * eps / h**2


def fd_optimal_step(M4, eps):
    """The step (48 eps / M4)^{1/4} that minimises ``fd_density_error_bound``; the minimum is 2 sqrt(M4 eps / 3)."""
    return (48.0 * eps / M4) ** 0.25


# ---------------------------------------------------------------------------------------
# Prices from a smile
# ---------------------------------------------------------------------------------------

def smile_call(K, S, T, r, y, sig):
    """C(K) = C^BS(K, sigma(K)): the call price whose implied volatility at K is sig."""
    return bs.call(S, K, T, r, y, sig)


def smile_digital(K, S, T, r, y, sig, dsig):
    """Digital call -C'(K) = e^{-rT} N(d_-) - vega * sigma'(K), all at sigma = sigma(K).

    Returns a dict: 'price', 'bs' (the Black--Scholes digital e^{-rT} N(d_-) at sigma(K)),
    'skew' (the skew term -vega * sigma'(K)) and 'vega'.
    """
    K = np.asarray(K, dtype=float)
    dp, dm = bs.d_pm(S, K, T, r, y, sig)
    vega = S * np.exp(-y * T) * bs.norm_pdf(dp) * np.sqrt(T)
    plain = np.exp(-r * T) * ndtr(dm)
    skew = -vega * dsig
    return dict(price=plain + skew, bs=plain, skew=skew, vega=vega)


def smile_asset_digital(K, S, T, r, y, sig, dsig):
    """Asset-or-nothing call e^{-rT} E[S_T 1{S_T > K}] = C(K) + K * digital(K)."""
    return smile_call(K, S, T, r, y, sig) + np.asarray(K, dtype=float) * smile_digital(K, S, T, r, y, sig, dsig)["price"]


def smile_density(K, S, T, r, y, sig, dsig, d2sig):
    """Discounted density C''(K) from the smile:

        e^{-rT} N'(d_-) / (K sigma sqrt T) * [1 + 2 K sqrt(T) d_+ sigma' + K^2 T d_+ d_- sigma'^2
                                              + K^2 T sigma sigma''].
    """
    K = np.asarray(K, dtype=float)
    dp, dm = bs.d_pm(S, K, T, r, y, sig)
    sq = np.sqrt(T)
    bracket = 1.0 + 2.0 * K * sq * dp * dsig + K**2 * T * dp * dm * dsig**2 + K**2 * T * sig * d2sig
    return np.exp(-r * T) * bs.norm_pdf(dm) / (K * sig * sq) * bracket


def smile_bracket(K, S, T, r, y, sig, dsig, d2sig):
    """The bracket of ``smile_density``: C''(K) divided by its value with the same sigma(K) and no skew."""
    K = np.asarray(K, dtype=float)
    dp, dm = bs.d_pm(S, K, T, r, y, sig)
    return 1.0 + 2.0 * K * np.sqrt(T) * dp * dsig + K**2 * T * dp * dm * dsig**2 + K**2 * T * sig * d2sig


# ---------------------------------------------------------------------------------------
# Two synthetic smiles
# ---------------------------------------------------------------------------------------

def course_smile(K, S, sigma_atm=0.25, xi=0.28, kappa=0.0):
    """The course's near-the-money smile sigma(K) = sigma_atm + xi L + kappa L^2, L = ln(S/K).

    Returns (sigma, d sigma/dK, d^2 sigma/dK^2). Only meaningful where sigma > 0.
    """
    K = np.asarray(K, dtype=float)
    L = np.log(S / K)
    sig = sigma_atm + xi * L + kappa * L**2
    a = xi + 2.0 * kappa * L
    dsig = -a / K
    d2sig = (a + 2.0 * kappa) / K**2
    return sig, dsig, d2sig


def phi_power_law(theta, eta=1.0, gamma=0.5):
    """phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma)) (Gatheral--Jacquier 2014, eq. (4.5))."""
    theta = np.asarray(theta, dtype=float)
    return eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))


def ssvi_total_variance(k, theta, rho, phi):
    """SSVI slice w(k) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)) and w', w''."""
    k = np.asarray(k, dtype=float)
    u = phi * k + rho
    R = np.sqrt(u * u + 1.0 - rho**2)
    w = 0.5 * theta * (1.0 + rho * phi * k + R)
    w1 = 0.5 * theta * phi * (rho + u / R)
    w2 = 0.5 * theta * phi**2 * (1.0 - rho**2) / R**3
    return w, w1, w2


def ssvi_smile(K, S, T, r, y, sigma_atm=0.20, rho=-0.7, eta=1.0, gamma=0.5):
    """Implied volatility of an SSVI slice at maturity T and its strike derivatives.

    The at-the-money-forward total variance is theta = sigma_atm^2 T, phi is the power law,
    and k = ln(K / F) with F = S e^{(r-y)T}. Returns (sigma, d sigma/dK, d^2 sigma/dK^2).
    """
    K = np.asarray(K, dtype=float)
    F = bs.forward(S, T, r, y)
    theta = sigma_atm**2 * T
    w, w1, w2 = ssvi_total_variance(np.log(K / F), theta, rho, phi_power_law(theta, eta, gamma))
    sig = np.sqrt(w / T)
    dsig = w1 / (2.0 * sig * T * K)                       # d sigma/dk = w'/(2 sigma T), dk/dK = 1/K
    d2sig = (w2 - w1) / (2.0 * sig * T * K**2) - w1 * dsig / (2.0 * sig**2 * T * K)
    return sig, dsig, d2sig


def ssvi_conditions(theta, rho, eta=1.0, gamma=0.5):
    """Left sides of the SSVI no-arbitrage conditions of Gatheral--Jacquier (2014), at theta.

    Returns a dict: 'calendar' = d(theta phi)/dtheta / phi (Theorem 4.1 needs it in
    [0, (1 + sqrt(1 - rho^2)) / rho^2]); 'fly1' = theta phi (1 + |rho|) (Theorem 4.2 needs
    < 4); 'fly2' = theta phi^2 (1 + |rho|) (needs <= 4).
    """
    theta = np.asarray(theta, dtype=float)
    phi = phi_power_law(theta, eta, gamma)
    # theta phi = eta (theta/(1+theta))^(1-gamma), whose theta-derivative divided by phi is (1-gamma)/(1+theta)
    return dict(calendar=(1.0 - gamma) / (1.0 + theta),
                calendar_max=np.inf if rho == 0 else (1.0 + np.sqrt(1.0 - rho**2)) / rho**2,
                fly1=theta * phi * (1.0 + abs(rho)),
                fly2=theta * phi**2 * (1.0 + abs(rho)))


def durrleman_g(k, w, w1, w2):
    """g(k) = (1 - k w'/(2w))^2 - w'^2/4 (1/w + 1/4) + w''/2 (Gatheral--Jacquier 2014, eq. (2.1))."""
    k = np.asarray(k, dtype=float)
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1**2 * (1.0 / w + 0.25) + 0.5 * w2


# ---------------------------------------------------------------------------------------
# Digitals by call spreads
# ---------------------------------------------------------------------------------------

def call_spread_bounds(call_fn, K, h):
    """(sub, super) = ((C(K) - C(K+h))/h, (C(K-h) - C(K))/h): the dearest sub- and the cheapest
    super-replication of the digital call 1{S_T > K} by calls with strikes on a grid of spacing h
    that contains K (K >= h)."""
    K = np.asarray(K, dtype=float)
    c = call_fn(K)
    return (c - call_fn(K + h)) / h, (call_fn(K - h) - c) / h


# ---------------------------------------------------------------------------------------
# A law with an atom
# ---------------------------------------------------------------------------------------

def atom_call(K, S, T, r, y, sigma, p, a):
    """Call price when S_T = a with probability p and S_T = F1 exp(sigma sqrt(T) Z - sigma^2 T/2)
    otherwise, with F1 = (F - p a)/(1 - p) so that E[S_T] = F = S e^{(r-y)T} (needs p a < F).

    e^{-rT} [p (a - K)^+ + (1 - p) Black(F1, K)]; a = 0 is a default (jump to ruin)."""
    K = np.asarray(K, dtype=float)
    F = bs.forward(S, T, r, y)
    if not (0.0 <= p < 1.0) or p * a >= F or a < 0:
        raise ValueError("need 0 <= p < 1, a >= 0 and p a < F")
    F1 = (F - p * a) / (1.0 - p)
    black = bs.call(F1, K, T, 0.0, 0.0, sigma)   # with r = y = 0: the undiscounted Black formula on F1
    return np.exp(-r * T) * (p * np.maximum(a - K, 0.0) + (1.0 - p) * black)


def atom_tail(K, S, T, r, y, sigma, p, a):
    """Q(S_T > K) in the model of ``atom_call``."""
    K = np.asarray(K, dtype=float)
    F = bs.forward(S, T, r, y)
    F1 = (F - p * a) / (1.0 - p)
    _, dm = bs.d_pm(F1, K, T, 0.0, 0.0, sigma)
    return p * (a > K) + (1.0 - p) * ndtr(dm)
