"""Asian options (Chapter 15).

Discretely monitored average-price options on one asset that follows geometric Brownian
motion with deterministic term structures of r, y and sigma (Chapter 7):

    dS_t / S_t = (r(t) - y(t)) dt + sigma(t) dW_t   under Q.

Contract
--------
* Fixing dates ``dates`` 0 <= t_1 < ... < t_m <= T (a fixing at t = 0 is allowed: it fixes
  S_0), payment date ``T`` (default t_m).
* Weights w_j > 0, normalised to sum to one (equal weights 1/m by default).
* Arithmetic average A = sum_j w_j S_{t_j}; geometric average G = prod_j S_{t_j}^{w_j}.
* Average-price call (A - K)^+ (``kind="call"``) or put (K - A)^+, paid at T and discounted
  with D = exp(-int_0^T r).
* ``r``, ``y`` and ``sigma`` are numbers, ``paths.PiecewiseConstant`` curves or vectorised
  functions of time, as in ``paths.py``; only their integrals over the fixing intervals
  matter, and integrals of ``PiecewiseConstant`` curves are exact.

Notation as in the notes: F_j = E[S_{t_j}] (forwards), Theta(t) = int_0^t sigma^2 (the
variance clock), C_ij = Theta(min(t_i, t_j)) = Cov(log S_{t_i}, log S_{t_j}).

Contents
--------
law          ``fixing_law``, ``discount``, ``black``
exact        ``geometric_law``, ``geometric_price``, ``geometric_greeks``
bounds       ``strip_bound`` (upper), ``threshold_bound``, ``curran_first_term``,
             ``conditioning_lower_bound``, ``conditional_upper_bound``
approx.      ``arithmetic_moments``, ``lognormal_price`` (two moments: Levy; Turnbull-Wakeman),
             ``shifted_lognormal_price`` (three moments), ``edgeworth_price`` and
             ``edgeworth_density`` (Turnbull-Wakeman correction), ``curran_refined`` (conditional lognormal law of A - G given G)
Monte Carlo  ``fixing_paths``, ``payoff_samples``, ``control_means``, ``mc_price``, ``cv_laboratory``,
             ``dispersion_bound`` (a priori bound for the geometric control),
             ``conditional_samples`` (integrate out the geometric direction),
             ``rqmc_reference``, ``geometric_pair_moments``
reference    ``recursion_price`` (recursion on the law of the normalised sum)
seasoned     ``seasoned_strike``
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import comb, ndtr

from . import paths
from .bs import norm_pdf
from .variance import control_variate, regression_control_variates

TEST_DATES = np.arange(1, 13) / 12.0          # the course's monthly fixings over one year


# ---------------------------------------------------------------------------------------
# The contract and the joint law of the fixings
# ---------------------------------------------------------------------------------------


def _contract(dates, T=None, weights=None):
    t = np.asarray(dates, dtype=float).ravel()
    if t.size == 0 or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("fixing dates must satisfy 0 <= t_1 < ... < t_m")
    T = float(t[-1]) if T is None else float(T)
    if T < t[-1]:
        raise ValueError("the payment date T must not precede the last fixing")
    if weights is None:
        w = np.full(t.size, 1.0 / t.size)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != t.size or np.any(w <= 0):
            raise ValueError("need one positive weight per fixing date")
        w = w / w.sum()
    return t, T, w


def discount(r, T) -> float:
    """Discount factor exp(-int_0^T r)."""
    return float(np.exp(-paths.integral(r, 0.0, T)))


def fixing_law(S, dates, r=0.0, y=0.0, sigma=0.2) -> dict:
    """Joint law of (log S_{t_1}, ..., log S_{t_m}): a Gaussian vector.

    Returns a dict with ``t`` (dates), ``F`` (forwards E S_{t_j}), ``theta`` (Theta(t_j)),
    ``mean`` (E log S_{t_j} = log F_j - Theta(t_j)/2), ``cov`` (C_ij = Theta(t_i ^ t_j)), and,
    for the intervals (t_{j-1}, t_j] with t_0 = 0, ``ell`` (mean of log S_{t_j}/S_{t_{j-1}})
    and ``dtheta`` (its variance); for a fixing at t_1 = 0 the first interval is empty.
    """
    t = np.asarray(dates, dtype=float).ravel()
    grid = paths.as_grid(t)
    p = paths.effective_parameters(grid, r, y, sigma)
    off = grid.size - t.size                       # 1 if 0 was prepended, else 0
    drift = np.concatenate([[0.0], np.cumsum((p["r"] - p["y"]) * p["dt"])])[off:]
    theta = p["clock"][off:]
    logF = math.log(S) + drift
    dtheta = np.diff(np.concatenate([[0.0], theta]))
    ddrift = np.diff(np.concatenate([[0.0], drift]))
    return dict(t=t, F=np.exp(logF), theta=theta, mean=logF - 0.5 * theta,
                cov=np.minimum.outer(theta, theta), ell=ddrift - 0.5 * dtheta,
                dtheta=np.maximum(dtheta, 0.0))


def black(F, K, v, D=1.0, kind="call"):
    """D E[(X - K)^+] (or the put) for a lognormal X with mean F and log-variance v >= 0.

    The Black formula D (F N(d_+) - K N(d_-)), d_pm = (log(F/K) +- v/2)/sqrt(v). Degenerate
    cases: v = 0 gives the intrinsic value D (F - K)^+; K <= 0 gives the forward D (F - K)
    for the call and 0 for the put. Vectorised over F, K and v.
    """
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    F, K, v = (np.array(a, dtype=float) for a in np.broadcast_arrays(F, K, v))
    shape = F.shape
    F, K, v = F.ravel(), K.ravel(), v.ravel()
    sd = np.sqrt(np.maximum(v, 0.0))
    call = np.maximum(F - K, 0.0)                 # v = 0: intrinsic value of the forward
    ok = (K > 0) & (sd > 0)
    if np.any(ok):
        dp = (np.log(F[ok] / K[ok]) + 0.5 * v[ok]) / sd[ok]
        call[ok] = F[ok] * ndtr(dp) - K[ok] * ndtr(dp - sd[ok])
    call = np.where(K <= 0, F - K, call)
    return (D * (call if kind == "call" else call - (F - K))).reshape(shape)


# ---------------------------------------------------------------------------------------
# The geometric average: exact prices and Greeks
# ---------------------------------------------------------------------------------------


def geometric_law(S, dates, r=0.0, y=0.0, sigma=0.2, weights=None):
    """Mean and variance (mu_G, v_G) of log G, G = prod_j S_{t_j}^{w_j}.

    mu_G = sum_j w_j (log F_j - Theta(t_j)/2) and
    v_G = sum_{i,j} w_i w_j Theta(t_i ^ t_j) = sum_k (Theta(t_k) - Theta(t_{k-1})) W_k^2,
    with the tail sums W_k = w_k + ... + w_m (Theta(t_0) = 0).
    """
    t, _, w = _contract(dates, None, weights)
    L = fixing_law(S, t, r, y, sigma)
    tails = np.cumsum(w[::-1])[::-1]
    return float(w @ L["mean"]), float(L["dtheta"] @ tails**2)


def geometric_price(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None, kind="call") -> float:
    """Exact price of the call (or put) on the geometric average: D times the Black formula
    with forward F_G = exp(mu_G + v_G/2) and log-variance v_G (Proposition as-geometric)."""
    t, T, w = _contract(dates, T, weights)
    mu, v = geometric_law(S, t, r, y, sigma, w)
    return float(black(math.exp(mu + 0.5 * v), K, v, discount(r, T), kind))


def geometric_greeks(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None) -> dict:
    """Price, delta, gamma and vega of the geometric average-price call.

    Delta and gamma: F_G is proportional to S, so delta = D F_G N(d_+)/S and
    gamma = D F_G N'(d_+)/(S^2 sqrt(v_G)). Vega is the derivative with respect to a parallel
    shift sigma(t) -> sigma(t) + eps of the whole volatility curve, at eps = 0:
    d Theta(t)/d eps = 2 int_0^t sigma.
    """
    t, T, w = _contract(dates, T, weights)
    mu, v = geometric_law(S, t, r, y, sigma, w)
    D = discount(r, T)
    F = math.exp(mu + 0.5 * v)
    if v <= 0 or K <= 0:
        raise ValueError("Greeks need v_G > 0 and K > 0")
    sd = math.sqrt(v)
    d1 = (math.log(F / K) + 0.5 * v) / sd
    price = D * (F * ndtr(d1) - K * ndtr(d1 - sd))
    dP_dmu = D * F * ndtr(d1)
    dP_dv = 0.5 * D * F * (ndtr(d1) + norm_pdf(d1) / sd)
    dtheta = 2.0 * np.asarray(paths.integral(sigma, np.zeros_like(t), t), dtype=float)
    tails = np.cumsum(w[::-1])[::-1]
    dmu = -0.5 * w @ dtheta
    dv = np.diff(np.concatenate([[0.0], dtheta])) @ tails**2
    return dict(price=float(price), delta=float(dP_dmu / S),
                gamma=float(D * F * norm_pdf(d1) / (S * S * sd)), vega=float(dP_dmu * dmu + dP_dv * dv))


# ---------------------------------------------------------------------------------------
# Moments of the arithmetic average and moment-matching approximations
# ---------------------------------------------------------------------------------------


def arithmetic_moments(S, dates, r=0.0, y=0.0, sigma=0.2, weights=None, order: int = 4) -> np.ndarray:
    """Raw moments E[A^p], p = 0..order, in closed form, in O(m order^2) operations.

    With X_k = S_{t_k}/S_{t_{k-1}} (independent lognormals, X_1 = S_{t_1}/S_0) and the
    backward recursion Z_m = w_m, Z_k = w_k + X_{k+1} Z_{k+1}, one has A = S X_1 Z_1 and
    E[Z_k^p] = sum_q binom(p, q) w_k^(p-q) E[X_{k+1}^q] E[Z_{k+1}^q].
    """
    t, _, w = _contract(dates, None, weights)
    L = fixing_law(S, t, r, y, sigma)
    q = np.arange(order + 1)
    EX = np.exp(np.outer(L["ell"], q) + 0.5 * np.outer(L["dtheta"], q * q))   # E[X_k^q], shape (m, p+1)
    M = w[-1] ** q.astype(float)
    B = comb(q[:, None], q[None, :])                                          # binom(p, q)
    for k in range(t.size - 2, -1, -1):
        Wp = np.where(q[:, None] >= q[None, :], w[k] ** np.maximum(q[:, None] - q[None, :], 0), 0.0)
        M = (B * Wp) @ (EX[k + 1] * M)
    return S ** q * EX[0] * M


def _central(mom):
    m1 = mom[1]
    mu2 = mom[2] - m1**2
    mu3 = mom[3] - 3 * m1 * mom[2] + 2 * m1**3
    mu4 = mom[4] - 4 * m1 * mom[3] + 6 * m1**2 * mom[2] - 3 * m1**4 if mom.size > 4 else np.nan
    return m1, max(mu2, 0.0), mu3, mu4


def lognormal_price(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None, kind="call") -> float:
    """Two-moment lognormal approximation: replace A by the lognormal with the same mean and
    variance (Levy 1992; the base of Turnbull and Wakeman 1991). Exact for m = 1."""
    t, T, w = _contract(dates, T, weights)
    mom = arithmetic_moments(S, t, r, y, sigma, w, order=2)
    v = math.log(mom[2] / mom[1] ** 2)
    return float(black(mom[1], K, max(v, 0.0), discount(r, T), kind))


def shifted_lognormal_fit(m1, mu2, mu3):
    """Shift tau and lognormal parameters of tau + e^X, X ~ N(a, b^2), with mean m1, variance
    mu2 > 0 and third central moment mu3 > 0. The skewness gamma = mu3/mu2^1.5 of e^X equals
    (u + 2) sqrt(u - 1) with u = e^{b^2}; with s = sqrt(u - 1) this is s^3 + 3 s = gamma,
    whose unique real root is given by Cardano's formula. Returns (tau, mean of e^X, b^2)."""
    g = mu3 / mu2**1.5
    h = math.sqrt(0.25 * g * g + 1.0)
    s = np.cbrt(0.5 * g + h) + np.cbrt(0.5 * g - h)
    b2 = math.log1p(s * s)
    mY = math.sqrt(mu2) / s
    return m1 - mY, mY, b2


def shifted_lognormal_price(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None, kind="call") -> float:
    """Three-moment approximation: A ~ tau + lognormal, matching mean, variance and skewness
    (the skewness of A is positive whenever v_A > 0). Exact for m = 1 (then tau = 0)."""
    t, T, w = _contract(dates, T, weights)
    m1, mu2, mu3, _ = _central(arithmetic_moments(S, t, r, y, sigma, w, order=3))
    D = discount(r, T)
    if mu2 <= 1e-14 * m1 * m1 or mu3 <= 0:
        return float(black(m1, K, 0.0, D, kind))
    tau, mY, b2 = shifted_lognormal_fit(m1, mu2, mu3)
    return float(black(mY, K - tau, b2, D, kind))


def _lognormal_density_derivatives(x, a, b2):
    """Lognormal density l(x) with log-mean a and log-variance b2, and l', l''."""
    b = math.sqrt(b2)
    z = (math.log(x) - a) / b
    l0 = math.exp(-0.5 * z * z) / (x * b * math.sqrt(2 * math.pi))
    g = -(1.0 + z / b) / x                       # (log l)'
    gp = (1.0 + z / b - 1.0 / b2) / x**2         # (log l)''
    return l0, l0 * g, l0 * (g * g + gp)


def edgeworth_density(x, S, dates, r=0.0, y=0.0, sigma=0.2, weights=None):
    """The Jarrow-Rudd corrected density l - (k3 - k3_LN)/3! l3 + (k4 - k4_LN)/4! l4 behind
    ``edgeworth_price``, at the points x > 0, where l is the two-moment lognormal density and
    l3, l4 its third and fourth derivatives. It integrates to one and has the first four moments
    of A, but it can be negative."""
    t, _, w = _contract(dates, None, weights)
    m1, mu2, mu3, mu4 = _central(arithmetic_moments(S, t, r, y, sigma, w, order=4))
    b2 = math.log1p(mu2 / m1**2)
    a, b, e = math.log(m1) - 0.5 * b2, math.sqrt(b2), math.exp(b2)
    k3_ln = (e + 2) * math.sqrt(e - 1) * mu2**1.5
    k4_ln = (e**4 + 2 * e**3 + 3 * e**2 - 6) * mu2**2
    x = np.asarray(x, dtype=float)
    z = (np.log(x) - a) / b
    l0 = np.exp(-0.5 * z * z) / (x * b * math.sqrt(2 * math.pi))
    g1, g2 = -(1 + z / b) / x, (1 + z / b - 1 / b2) / x**2           # derivatives of log l
    g3, g4 = (3 / b2 - 2 - 2 * z / b) / x**3, (6 + 6 * z / b - 11 / b2) / x**4
    l3 = l0 * (g3 + 3 * g1 * g2 + g1**3)
    l4 = l0 * (g4 + 4 * g1 * g3 + 3 * g2**2 + 6 * g1**2 * g2 + g1**4)
    return l0 - (mu3 - k3_ln) / 6 * l3 + ((mu4 - 3 * mu2**2) - k4_ln) / 24 * l4


def edgeworth_price(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None) -> float:
    """Turnbull-Wakeman approximation: the two-moment lognormal call plus the Edgeworth
    (Jarrow-Rudd) correction for the third and fourth cumulants,

        C ~ C_LN + D [ -(k3 - k3_LN)/3! l'(K) + (k4 - k4_LN)/4! l''(K) ],

    where l is the matched lognormal density and k_n the cumulants of A and of the
    lognormal. An approximation: the corrected density can be negative."""
    t, T, w = _contract(dates, T, weights)
    m1, mu2, mu3, mu4 = _central(arithmetic_moments(S, t, r, y, sigma, w, order=4))
    D = discount(r, T)
    b2 = math.log1p(mu2 / m1**2)
    a = math.log(m1) - 0.5 * b2
    c0 = float(black(m1, K, b2, D))
    if K <= 0 or b2 <= 0:
        return c0
    e = math.exp(b2)
    k3_ln = (e + 2) * math.sqrt(e - 1) * mu2**1.5
    k4_ln = (e**4 + 2 * e**3 + 3 * e**2 - 6) * mu2**2
    k3, k4 = mu3, mu4 - 3 * mu2**2
    _, l1, l2 = _lognormal_density_derivatives(K, a, b2)
    return float(c0 + D * (-(k3 - k3_ln) / 6.0 * l1 + (k4 - k4_ln) / 24.0 * l2))


# ---------------------------------------------------------------------------------------
# Bounds; conditioning on the geometric average (Curran; Rogers-Shi)
# ---------------------------------------------------------------------------------------


def strip_bound(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None) -> float:
    """Upper bound D E[sum_j w_j (S_{t_j} - K)^+] >= D E[(A - K)^+] (convexity of x -> (x - K)^+)."""
    t, T, w = _contract(dates, T, weights)
    L = fixing_law(S, t, r, y, sigma)
    return float(discount(r, T) * w @ black(L["F"], K, L["theta"]))


def dispersion_bound(S, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None) -> float:
    """D^2 E[(A - G)^2] in closed form: an upper bound on Var(Y - X) for the arithmetic payoff
    Y = D (A - K)^+ and the geometric control X = D (G - K)^+, whatever the strike, because
    0 <= Y - X <= D (A - G). Uses E[S_{t_j} G] = F_j E[G] e^{c_j}, c_j = Cov(log S_{t_j}, log G)."""
    t, T, w = _contract(dates, T, weights)
    g = geometric_conditioning(S, t, r, y, sigma, w)
    EA2 = arithmetic_moments(S, t, r, y, sigma, w, order=2)[2]
    EG = math.exp(g["mu_G"] + 0.5 * g["v_G"])
    EAG = EG * (w * g["F"]) @ np.exp(g["c"])
    return float(discount(r, T) ** 2 * (EA2 - 2 * EAG + EG * EG * math.exp(g["v_G"])))


def geometric_conditioning(S, dates, r=0.0, y=0.0, sigma=0.2, weights=None) -> dict:
    """Ingredients of the conditional law of the fixings given X = log G.

    Returns w, F, cov, mu_G, v_G and c_j = Cov(log S_{t_j}, log G) = sum_i w_i C_ij >= 0. Given
    X = x, log S_{t_j} is normal with mean E log S_{t_j} + c_j (x - mu_G)/v_G and covariance
    C_ij - c_i c_j / v_G, so E[S_{t_j} | X = x] = F_j exp(c_j (x - mu_G)/v_G - c_j^2/(2 v_G)).
    """
    t, _, w = _contract(dates, None, weights)
    L = fixing_law(S, t, r, y, sigma)
    c = L["cov"] @ w
    mu_G, v_G = float(w @ L["mean"]), float(w @ c)
    return dict(w=w, F=L["F"], cov=L["cov"], c=c, mu_G=mu_G, v_G=v_G)


def conditional_mean(x, g) -> np.ndarray:
    """E[A | log G = x] for an array x (g from ``geometric_conditioning``)."""
    x = np.asarray(x, dtype=float)[..., None]
    e = g["w"] * g["F"] * np.exp(g["c"] * (x - g["mu_G"]) / g["v_G"] - 0.5 * g["c"] ** 2 / g["v_G"])
    return e.sum(axis=-1)


def conditional_variance(x, g) -> np.ndarray:
    """Var(A | log G = x) = sum_ij w_i w_j e_i e_j (exp(C_ij - c_i c_j / v_G) - 1), e_i = E[S_i | x]."""
    x = np.asarray(x, dtype=float)[..., None]
    e = g["w"] * g["F"] * np.exp(g["c"] * (x - g["mu_G"]) / g["v_G"] - 0.5 * g["c"] ** 2 / g["v_G"])
    R = np.expm1(g["cov"] - np.outer(g["c"], g["c"]) / g["v_G"])
    return np.maximum(np.einsum("...i,ij,...j->...", e, R, e), 0.0)


def threshold_bound(S, K, dates, x, r=0.0, y=0.0, sigma=0.2, T=None, weights=None) -> float:
    """D E[(A - K) 1{log G > x}] in closed form; a lower bound on the call for every x
    (x = -inf gives D (E A - K))."""
    t, T, w = _contract(dates, T, weights)
    g = geometric_conditioning(S, t, r, y, sigma, w)
    sd = math.sqrt(g["v_G"])
    if not np.isfinite(x):
        val = (w @ g["F"] - K) if x < 0 else 0.0
    else:
        z = (g["mu_G"] - x) / sd
        val = w @ (g["F"] * ndtr(z + g["c"] / sd)) - K * ndtr(z)
    return float(discount(r, T) * val)


def curran_first_term(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None) -> float:
    """The exact part of Curran's decomposition, D E[(A - K) 1{G > K}] (K > 0)."""
    return threshold_bound(S, K, dates, math.log(K), r, y, sigma, T, weights)


def curran_threshold(S, K, dates, r=0.0, y=0.0, sigma=0.2, weights=None) -> float:
    """The root x* of E[A | log G = x] = K (-inf when the conditional mean stays above K)."""
    g = geometric_conditioning(S, dates, r, y, sigma, weights)
    floor = float(np.sum((g["w"] * g["F"] * np.exp(-0.5 * g["c"] ** 2 / g["v_G"]))[g["c"] <= 0]))
    if K <= floor:
        return -math.inf
    sd = math.sqrt(g["v_G"])
    f = lambda x: math.log(float(conditional_mean(x, g))) - math.log(K)
    lo, hi = g["mu_G"] - sd, g["mu_G"] + sd
    while f(lo) > 0:
        lo -= 4 * sd
    while f(hi) < 0:
        hi += 4 * sd
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=500)


def conditioning_lower_bound(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None) -> float:
    """D E[(E[A | G] - K)^+] = D E[(A - K) 1{log G > x*}]: the largest threshold bound
    (Rogers and Shi 1995, in discrete time; Curran's "naive" approximation)."""
    x = curran_threshold(S, K, dates, r, y, sigma, weights)
    return threshold_bound(S, K, dates, x, r, y, sigma, T, weights)


def _gauss_legendre(a, b, n):
    x, wq = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * wq


def conditional_upper_bound(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None,
                            nodes: int = 400) -> float:
    """Upper bound LB + (D/2) E[sqrt(Var(A | G)) 1{G < K}], LB = ``conditioning_lower_bound``.

    The expectation is a one-dimensional integral over z = (log G - mu_G)/sqrt(v_G) < z_K,
    computed by Gauss-Legendre quadrature on [-12, z_K] (the integrand is smooth).
    """
    t, T, w = _contract(dates, T, weights)
    g = geometric_conditioning(S, t, r, y, sigma, w)
    lb = conditioning_lower_bound(S, K, t, r, y, sigma, T, w)
    sd = math.sqrt(g["v_G"])
    zK = (math.log(K) - g["mu_G"]) / sd
    if zK <= -12:
        return lb
    z, wq = _gauss_legendre(-12.0, zK, nodes)
    integrand = np.sqrt(conditional_variance(g["mu_G"] + sd * z, g)) * norm_pdf(z)
    return float(lb + 0.5 * discount(r, T) * wq @ integrand)


def curran_refined(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None, freeze: bool = False,
                   nodes: int = 400) -> float:
    """Curran's decomposition with the second term approximated: given G = g < K, A - G is
    taken lognormal with its exact conditional mean E[A | g] - g and variance Var(A | g)
    (``freeze=True``: both frozen at g = K, as in the course). An approximation, not a bound."""
    t, T, w = _contract(dates, T, weights)
    g = geometric_conditioning(S, t, r, y, sigma, w)
    first = curran_first_term(S, K, t, r, y, sigma, T, w)
    sd = math.sqrt(g["v_G"])
    zK = (math.log(K) - g["mu_G"]) / sd
    if zK <= -12 or t.size == 1:
        return first
    z, wq = _gauss_legendre(-12.0, zK, nodes)
    x = g["mu_G"] + sd * z
    xs = np.full_like(x, math.log(K)) if freeze else x
    mean = conditional_mean(xs, g) - np.exp(xs)
    var = conditional_variance(xs, g)
    v = np.log1p(var / mean**2)
    second = black(mean, K - np.exp(x), v) * norm_pdf(z)
    return float(first + discount(r, T) * wq @ second)


# ---------------------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------------------


def fixing_paths(S, dates, r=0.0, y=0.0, sigma=0.2, n_paths: int = 1, seed=None, method: str = "increments",
                 Z=None) -> np.ndarray:
    """Exact samples of (S_{t_1}, ..., S_{t_m}), shape (n_paths, m), by ``paths.gbm_paths`` on the
    grid of fixing dates (one normal per interval; m - 1 normals if t_1 = 0)."""
    t = np.asarray(dates, dtype=float).ravel()
    grid = paths.as_grid(t)
    P = paths.gbm_paths(S, grid, r, y, sigma, n_paths, seed, method, Z)
    return P[:, grid.size - t.size:]


def payoff_samples(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None, n_paths: int = 1, seed=None,
                   method: str = "increments", Z=None) -> dict:
    """Discounted payoffs on the same paths: ``arithmetic`` D (A - K)^+, ``geometric`` D (G - K)^+,
    ``vanilla`` D (S_{t_m} - K)^+ and ``strip`` D sum_j w_j (S_{t_j} - K)^+."""
    t, T, w = _contract(dates, T, weights)
    P = fixing_paths(S, t, r, y, sigma, n_paths, seed, method, Z)
    D = discount(r, T)
    A = P @ w
    G = np.exp(np.log(P) @ w)
    return dict(arithmetic=D * np.maximum(A - K, 0.0), geometric=D * np.maximum(G - K, 0.0),
                vanilla=D * np.maximum(P[:, -1] - K, 0.0), strip=D * np.maximum(P - K, 0.0) @ w)


def control_means(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None) -> dict:
    """Exact means of the ``geometric``, ``vanilla`` and ``strip`` payoffs of ``payoff_samples``."""
    t, T, w = _contract(dates, T, weights)
    L = fixing_law(S, t, r, y, sigma)
    D = discount(r, T)
    return dict(geometric=geometric_price(S, K, t, r, y, sigma, T, w),
                vanilla=float(black(L["F"][-1], K, L["theta"][-1], D)),
                strip=strip_bound(S, K, t, r, y, sigma, T, w))


def mc_price(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None, n: int = 2**16, seed=None,
             controls=("geometric",), beta="estimated", method: str = "increments", Z=None) -> dict:
    """Monte Carlo price of the arithmetic call, plain (``controls=()``) or with control variates.

    One control: ``variance.control_variate`` (``beta`` a number, "estimated" or "split").
    Several: least-squares regression (``variance.regression_control_variates``), whose
    output includes the sample R^2. Returns mean, se and the fitted coefficients.
    """
    smp = payoff_samples(S, K, dates, r, y, sigma, T, weights, n, seed, method, Z)
    Y = smp["arithmetic"]
    if not controls:
        return dict(mean=float(Y.mean()), se=float(Y.std(ddof=1) / math.sqrt(Y.size)), n=Y.size,
                    sd=float(Y.std(ddof=1)))
    mu = control_means(S, K, dates, r, y, sigma, T, weights)
    if len(controls) == 1:
        c = controls[0]
        return control_variate(Y, smp[c], mu[c], beta)
    X = np.column_stack([smp[c] for c in controls])
    return regression_control_variates(Y, X, [mu[c] for c in controls])


def cv_laboratory(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None, n: int = 2**20, seed=2026,
                  chunk: int = 2**18) -> dict:
    """The course's control-variate experiment on n paths (simulated in chunks): plain Monte
    Carlo, the geometric control with the in-sample coefficient, the coefficient-one control,
    and the three controls (geometric, vanilla, strip) by regression. Returns a dict of dicts
    with mean, se and the coefficients, plus the plain standard deviation per path ``sd``."""
    rng = np.random.default_rng(seed)
    t, T, w = _contract(dates, T, weights)
    d = t.size - (1 if t[0] == 0 else 0)
    parts = []
    for k in range(0, n, chunk):
        smp = payoff_samples(S, K, t, r, y, sigma, T, w, Z=rng.standard_normal((min(chunk, n - k), d)))
        parts.append(np.column_stack([smp[c] for c in ("arithmetic", "geometric", "vanilla", "strip")]))
    V = np.concatenate(parts)
    mu = control_means(S, K, t, r, y, sigma, T, w)
    Y = V[:, 0]
    return dict(sd=float(Y.std(ddof=1)),
                plain=dict(mean=float(Y.mean()), se=float(Y.std(ddof=1) / math.sqrt(n))),
                geometric=control_variate(Y, V[:, 1], mu["geometric"], "estimated"),
                beta_one=control_variate(Y, V[:, 1], mu["geometric"], 1.0),
                triple=regression_control_variates(Y, V[:, 1:], [mu[c] for c in ("geometric", "vanilla", "strip")]))


def _geometric_direction(S, dates, r, y, sigma, weights):
    """Decomposition log S_{t_j} = mean_j + b_j xi + (M Z')_j with xi = (log G - mu_G)/sqrt(v_G)
    standard normal and independent of Z' ~ N(0, I_{m-1}); b_j = c_j / sqrt(v_G) >= 0."""
    t, _, w = _contract(dates, None, weights)
    L = fixing_law(S, t, r, y, sigma)
    m = t.size
    Bm = np.tril(np.ones((m, m))) * np.sqrt(L["dtheta"])       # log S - mean = Bm Z, Z ~ N(0, I_m)
    u = Bm.T @ w
    v_G = float(u @ u)
    if v_G <= 0:
        raise ValueError("the geometric average is deterministic (v_G = 0)")
    e = u / math.sqrt(v_G)
    Qp = np.linalg.svd(e[None, :])[2][1:].T                     # orthonormal basis of e^perp
    return dict(w=w, mean=L["mean"], b=Bm @ e, M=Bm @ Qp, mu_G=float(w @ L["mean"]), v_G=v_G)


def _exp_sum_root(a, b, K, iters: int = 100):
    """Root xi of sum_j a_j exp(b_j xi) = K, row by row (a > 0, b >= 0, max b > 0), assuming
    sum_{b_j = 0} a_j < K. Newton on the convex increasing log-sum-exp, which converges
    monotonically from the right after at most one step."""
    la = np.log(a)
    x = np.zeros(a.shape[0])
    lk = math.log(K)
    for _ in range(iters):
        s = la + b * x[:, None]
        smax = s.max(axis=1, keepdims=True)
        e = np.exp(s - smax)
        tot = e.sum(axis=1)
        f = smax[:, 0] + np.log(tot) - lk
        fp = (e @ b) / tot
        step = f / fp
        x = np.clip(x - step, -200.0, 200.0)
        if np.max(np.abs(step)) < 1e-13:
            break
    return x


def conditional_samples(S, K, dates, Zp, r=0.0, y=0.0, sigma=0.2, T=None, weights=None) -> np.ndarray:
    """Conditional Monte Carlo: E[D (A - K)^+ | Z'] for normals Zp of shape (n, m - 1).

    Given Z', the average is A = sum_j a_j e^{b_j xi} with a_j = w_j exp(mean_j + (M Z')_j),
    increasing in the standard normal xi, so with xi* the root of A = K,
    E[(A - K)^+ | Z'] = sum_j a_j e^{b_j^2/2} N(b_j - xi*) - K N(-xi*). The conditional value is
    a smooth function of Z' with the same mean as the payoff and a smaller variance.
    """
    t, T, w = _contract(dates, T, weights)
    g = _geometric_direction(S, t, r, y, sigma, w)
    Zp = np.atleast_2d(np.asarray(Zp, dtype=float))
    a = w * np.exp(g["mean"] + Zp @ g["M"].T)
    b = g["b"]
    eb = np.exp(0.5 * b * b)
    out = (a * eb).sum(axis=1) - K                              # always in the money
    floor = a[:, b <= 1e-15].sum(axis=1)
    live = floor < K
    if np.any(live):
        xs = _exp_sum_root(a[live], b, K)
        out[live] = (a[live] * eb * ndtr(b - xs[:, None])).sum(axis=1) - K * ndtr(-xs)
    return discount(r, T) * out


def rqmc_reference(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None, n: int = 2**14, R: int = 16,
                   seed=2026) -> dict:
    """Reference price: ``conditional_samples`` averaged over R independent nested-scrambled
    Sobol' point sets of size n (qmc.py). Returns mean, se (from the R replicates) and the
    replicates. Unbiased: each replicate is an unbiased estimate of the price."""
    from . import qmc
    t, T, w = _contract(dates, T, weights)
    d = t.size - 1
    if d == 0:                                                   # one fixing: a European call
        L = fixing_law(S, t, r, y, sigma)
        return dict(mean=float(black(L["F"][0], K, L["theta"][0], discount(r, T))), se=0.0,
                    replicates=np.array([]))
    reps = np.array([conditional_samples(S, K, t, qmc.to_normals(qmc.rqmc_points(n, d, "owen", s)),
                                         r, y, sigma, T, w).mean()
                     for s in np.random.SeedSequence(seed).spawn(R)])
    return dict(mean=float(reps.mean()), se=float(reps.std(ddof=1) / math.sqrt(R)), replicates=reps)


def geometric_pair_moments(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None, n: int = 2**14,
                           R: int = 8, seed=11) -> dict:
    """Var Y, Cov(Y, X), Var X, rho and beta* for Y = D (A - K)^+ and the geometric control
    X = D (G - K)^+, to high accuracy: E[Y^2 | Z'] and E[YX | Z'] are closed forms in the
    geometric direction xi (as in ``conditional_samples``), averaged by randomised QMC
    over Z'. Var X is exact. Returns the estimates and the standard errors of E Y^2, E YX."""
    from . import qmc
    t, T, w = _contract(dates, T, weights)
    g = _geometric_direction(S, t, r, y, sigma, w)
    D = discount(r, T)
    b, sG = g["b"], math.sqrt(g["v_G"])
    xG = (math.log(K) - g["mu_G"]) / sG
    E = lambda beta, x: np.exp(0.5 * beta * beta) * ndtr(beta - x)    # E[e^{beta xi} 1{xi > x}]
    m2, mx = [], []
    for s in np.random.SeedSequence(seed).spawn(R):
        Zp = qmc.to_normals(qmc.rqmc_points(n, t.size - 1, "owen", s))
        a = w * np.exp(g["mean"] + Zp @ g["M"].T)
        xs = np.full((a.shape[0], 1), -np.inf)
        live = a[:, b <= 1e-15].sum(axis=1) < K
        if np.any(live):
            xs[live, 0] = _exp_sum_root(a[live], b, K)
        bb = b[:, None] + b[None, :]
        y2 = (np.einsum("ni,nj,nij->n", a, a, E(bb[None], xs[..., None]))
              - 2 * K * (a * E(b[None], xs)).sum(axis=1) + K * K * ndtr(-xs[:, 0]))
        gG = math.exp(g["mu_G"])
        yx = (gG * (a * E(b + sG, xG)).sum(axis=1) - K * (a * E(b, xG)).sum(axis=1)
              - K * gG * E(sG, xG) + K * K * ndtr(-xG))
        m2.append(D * D * y2.mean())
        mx.append(D * D * yx.mean())
    m2, mx = np.array(m2), np.array(mx)
    EY = rqmc_reference(S, K, t, r, y, sigma, T, w, n, R, seed)["mean"]
    EX = geometric_price(S, K, t, r, y, sigma, T, w)
    v = g["v_G"]
    F = math.exp(g["mu_G"] + 0.5 * v)
    EX2 = D * D * (F * F * math.exp(v) * ndtr(-xG + 2 * sG) - 2 * K * F * ndtr(-xG + sG) + K * K * ndtr(-xG))
    varY, cov, varX = m2.mean() - EY**2, mx.mean() - EY * EX, EX2 - EX**2
    return dict(varY=float(varY), cov=float(cov), varX=float(varX), rho=float(cov / math.sqrt(varX * varY)),
                beta=float(cov / varX), se_m2=float(m2.std(ddof=1) / math.sqrt(R)),
                se_mx=float(mx.std(ddof=1) / math.sqrt(R)))


# ---------------------------------------------------------------------------------------
# A reference price by recursion on the law of the normalised sum
# ---------------------------------------------------------------------------------------


def recursion_price(S, K, dates, r=0.0, y=0.0, sigma=0.2, T=None, weights=None, N: int = 4001,
                    nq: int = 64, width: float = 14.0) -> float:
    """Price of the arithmetic call by a recursion on distribution functions.

    With X_k = S_{t_k}/S_{t_{k-1}} and Z_m = w_m, Z_k = w_k + X_{k+1} Z_{k+1}, the average is
    A = S X_1 Z_1 and X_{k+1} is independent of Z_{k+1}. So the distribution function of
    U_k = log Z_k follows from that of U_{k+1} by
        P(U_k <= u) = E[ P(U_{k+1} <= log(e^u - w_k) - log X_{k+1}) ],   u > log w_k,
    an expectation over one normal variable (Gauss-Hermite rule with nq nodes); each
    distribution function is stored on a uniform grid of N points (natural cubic spline in
    between) covering width log-standard deviations either side of the centre. Finally
    E[(S X_1 Z_1 - K)^+] = int (1 - F_{U_1}(u)) dc(e^u) with c(z) = E[(S X_1 z - K)^+] the
    Black formula (or, if X_1 is deterministic, int_{log(K/(S X_1))} S X_1 e^u (1 - F(u)) du).
    The only errors are the grid, the interpolation and the quadrature: refine N and nq to
    check convergence (the defaults give about ten digits in the test case).
    """
    t, T, w = _contract(dates, T, weights)
    L = fixing_law(S, t, r, y, sigma)
    D = discount(r, T)
    ell, dth = L["ell"], L["dtheta"]
    m = t.size
    if L["theta"][-1] <= 0:                                      # no randomness at all
        return float(D * max(w @ L["F"] - K, 0.0))
    x, wq = np.polynomial.hermite_e.hermegauss(nq)
    wq = wq / wq.sum()
    # moments of Z_k for the grids: E Z, E Z^2 (backward, as in arithmetic_moments)
    EZ1, EZ2 = w[-1], w[-1] ** 2
    law = ("point", math.log(w[-1]))
    for k in range(m - 2, -1, -1):
        mu_x, s_x = ell[k + 1], math.sqrt(dth[k + 1])
        EX1, EX2 = math.exp(mu_x + 0.5 * s_x**2), math.exp(2 * mu_x + 2 * s_x**2)
        EZ1, EZ2 = w[k] + EX1 * EZ1, w[k] ** 2 + 2 * w[k] * EX1 * EZ1 + EX2 * EZ2
        if law[0] == "point" and s_x == 0.0:
            law = ("point", math.log(w[k] + math.exp(mu_x + law[1])))
            continue
        s_ln = math.sqrt(max(math.log(EZ2 / EZ1**2), 1e-12))
        lo = max(math.log(w[k]) + 1e-13, math.log(EZ1) - width * s_ln)
        hi = math.log(EZ1) + width * s_ln
        u = np.linspace(lo, hi, N)
        v = np.log(np.expm1(u - math.log(w[k]))) + math.log(w[k])   # log(e^u - w_k)
        if law[0] == "point":
            F = ndtr((v - mu_x - law[1]) / s_x)
        else:
            spl, ulo, uhi = law[1], law[2], law[3]
            arg = v[:, None] - mu_x - s_x * x[None, :] if s_x > 0 else (v - mu_x)[:, None]
            Fa = np.where(arg < ulo, 0.0, np.where(arg > uhi, 1.0, spl(np.clip(arg, ulo, uhi))))
            F = Fa @ wq if s_x > 0 else Fa[:, 0]
        F = np.clip(F, 0.0, 1.0)
        law = ("cdf", CubicSpline(u, F, bc_type="natural"), lo, hi)
    mu1, s1 = ell[0], math.sqrt(dth[0])
    if law[0] == "point":
        z = math.exp(law[1])
        return float(black(S * z * math.exp(mu1 + 0.5 * s1**2), K, s1**2, D))
    spl, lo, hi = law[1], law[2], law[3]
    surv = lambda uu: 1.0 - np.clip(spl(np.clip(uu, lo, hi)), 0.0, 1.0) * (uu >= lo)
    F1 = S * math.exp(mu1 + 0.5 * s1**2)
    if s1 > 0:
        c_lo = float(black(F1 * math.exp(lo), K, s1**2))
        uu, ww = _composite_gl(lo, hi, N // 8)
        dp = (np.log(F1 * np.exp(uu) / K) + 0.5 * s1**2) / s1 if K > 0 else np.inf
        dc = F1 * np.exp(uu) * ndtr(dp)                          # d/du c(e^u)
        return float(D * (c_lo + ww @ (surv(uu) * dc)))
    a = S * math.exp(mu1)
    uK = math.log(K / a) if K > 0 else -math.inf
    start = max(lo, uK)
    val = max(a * math.exp(lo) - K, 0.0)
    if start < hi:
        uu, ww = _composite_gl(start, hi, N // 8)
        val += ww @ (a * np.exp(uu) * surv(uu))
    return float(D * val)


def _composite_gl(a, b, panels: int, order: int = 8):
    """Nodes and weights of a composite Gauss-Legendre rule on [a, b]."""
    xg, wg = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, panels + 1)
    h = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    return (mid[:, None] + h[:, None] * xg).ravel(), (h[:, None] * wg).ravel()


# ---------------------------------------------------------------------------------------
# Seasoned options
# ---------------------------------------------------------------------------------------


def seasoned_strike(K, past_prices, past_weights, future_weights):
    """Strike shift for an Asian option some of whose fixings are known.

    With A = sum_past w_j s_j + W A', W = sum of the future weights and A' the average of the
    future fixings with renormalised weights, (A - K)^+ = W (A' - K')^+ where
    K' = (K - sum_past w_j s_j)/W. Returns (W, K'); K' <= 0 means the call will surely be
    exercised and is worth W times a forward on A'.
    """
    past = float(np.dot(past_weights, past_prices))
    W = float(np.sum(future_weights))
    return W, (K - past) / W
