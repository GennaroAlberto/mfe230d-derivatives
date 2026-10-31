"""Monte Carlo Greeks (Chapter 13).

Finite differences ("bumping") with and without common random numbers, with their exact bias
and variance; pathwise and likelihood-ratio estimators of the Black--Scholes Greeks; the three
mixed estimators of the gamma; Greeks of an Asian call along simulated paths; and a hand-coded
adjoint (reverse-mode) pricer for an Asian call with one volatility per period.

Conventions follow the notes. Under the risk-neutral measure

    S_T = S exp(nu T + sigma sqrt(T) Z),   nu = r - y - sigma^2 / 2,   Z ~ N(0, 1),

and every per-sample estimator below takes the standard normal draws ``z`` and returns an array
of the same shape, one value per draw: average it (``montecarlo.mc_estimate``) for an estimate
and a standard error. The same draws can be passed to several estimators (common random
numbers). ``bs.py`` supplies the closed forms used as exact values. Payoffs are ``"call"``,
``"put"`` and ``"digital"`` (cash-or-nothing call paying 1 if S_T > K).

Every method of the chapter is implemented here from scratch; no automatic-differentiation
library is used.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

from . import bs
from .paths import brownian_paths, gbm_from_brownian, time_grid
from .variance import gauss_expect, strike_to_z

KINDS = ("call", "put", "digital")
PARAMS = ("S", "K", "T", "r", "y", "sigma")


# ------------------------------------------------------------------------------ the model
def growth(z, T, r, y, sigma):
    """G = S_T / S = exp(nu T + sigma sqrt(T) z) for standard normal draws z."""
    return np.exp((r - y - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.asarray(z, dtype=float))


def payoff(kind, s, K):
    """Undiscounted payoff f(s) of a call, a put or a cash-or-nothing digital call."""
    s = np.asarray(s, dtype=float)
    if kind == "call":
        return np.maximum(s - K, 0.0)
    if kind == "put":
        return np.maximum(K - s, 0.0)
    if kind == "digital":
        return (s > K).astype(float)
    raise ValueError(f"kind must be one of {KINDS}")


def payoff_derivative(kind, s, K):
    """f'(s) wherever it exists (every s != K): 1{s > K} for the call, -1{s < K} for the put,
    and 0 for the digital (whose derivative vanishes off the strike, which is why the pathwise
    estimator of a digital Greek is biased)."""
    s = np.asarray(s, dtype=float)
    if kind == "call":
        return (s > K).astype(float)
    if kind == "put":
        return -(s < K).astype(float)
    if kind == "digital":
        return np.zeros_like(s)
    raise ValueError(f"kind must be one of {KINDS}")


def price(kind, S, K, T, r, y, sigma):
    """Black--Scholes price of the payoff (closed forms of bs.py)."""
    f = dict(call=bs.call, put=bs.put, digital=bs.digital_call)[kind]
    return f(S, K, T, r, y, sigma)


def discounted_payoffs(z, kind, S, K, T, r, y, sigma):
    """Samples e^{-rT} f(S_T) of the discounted payoff."""
    return np.exp(-r * T) * payoff(kind, S * growth(z, T, r, y, sigma), K)


def exact_greek(kind, greek, S, K, T, r, y, sigma):
    """Closed-form Greek: delta, gamma, vega, theta (= dV/dt = -dV/dT), rho_r, rho_y.

    Calls and puts come from ``bs.greeks``; the digital call e^{-rT} N(d_-) is differentiated
    by hand (the formulas are checked against finite differences in the tests).
    """
    if kind in ("call", "put"):
        return bs.greeks(S, K, T, r, y, sigma, kind=kind)[greek]
    if kind != "digital":
        raise ValueError(f"kind must be one of {KINDS}")
    dp, dm = bs.d_pm(S, K, T, r, y, sigma)
    D = bs.digital_call(S, K, T, r, y, sigma)
    w = np.exp(-r * T) * bs.norm_pdf(dm)
    sd = sigma * np.sqrt(T)
    nu = r - y - 0.5 * sigma**2
    out = dict(delta=w / (S * sd),
               gamma=-w * dp / (S**2 * sd**2),
               vega=-w * dp / sigma,
               rho_r=-T * D + w * np.sqrt(T) / sigma,
               rho_y=-w * np.sqrt(T) / sigma,
               theta=r * D - w * (nu * T - np.log(S / K)) / (2 * sigma * T**1.5))
    return out[greek]


def _params(S, K, T, r, y, sigma):
    return dict(S=S, K=K, T=T, r=r, y=y, sigma=sigma)


# ------------------------------------------------------------------- finite differences
SCHEMES = {
    "forward": ((+1, 1.0), (0, -1.0)),
    "backward": ((0, 1.0), (-1, -1.0)),
    "central": ((+1, 0.5), (-1, -0.5)),
    "second": ((+1, 1.0), (0, -2.0), (-1, 1.0)),
}


def fd_samples(f, theta, h, z, scheme="central", z_indep=None):
    """Per-sample finite-difference quotients of theta -> f(theta, z).

    ``scheme`` is "forward" (f(t+h) - f(t))/h, "backward" (f(t) - f(t-h))/h, "central"
    (f(t+h) - f(t-h))/(2h) or "second" (f(t+h) - 2 f(t) + f(t-h))/h^2. With ``z_indep`` None,
    every evaluation uses the draws ``z`` (common random numbers). Otherwise ``z_indep`` is a
    sequence of further arrays of draws, one for each evaluation after the first, and the
    evaluations are independent.
    """
    pts = SCHEMES[scheme]
    draws = [z] * len(pts) if z_indep is None else [z] + list(z_indep)
    if len(draws) < len(pts):
        raise ValueError(f"scheme {scheme!r} needs {len(pts) - 1} independent extra draws")
    total = sum(c * np.asarray(f(theta + k * h, w), dtype=float) for (k, c), w in zip(pts, draws))
    return total / (h * h if scheme == "second" else h)


def payoff_function(kind, param, S, K, T, r, y, sigma):
    """theta -> (z -> discounted payoff samples) with the parameter ``param`` set to theta."""
    base = _params(S, K, T, r, y, sigma)

    def f(theta, z):
        p = dict(base, **{param: theta})
        return discounted_payoffs(z, kind, **p)
    return f


def fd_spot_greeks(z, kind, S, K, T, r, y, sigma, eps, z_indep=None):
    """The course's spot Greeks with relative bumps S(1 +- eps), per sample.

    Returns Delta_+ = (C_+ - C_0)/(eps S), Delta_- = (C_0 - C_-)/(eps S),
    Delta_c = (C_+ - C_-)/(2 eps S) and Gamma_c = (C_+ - 2 C_0 + C_-)/(eps S)^2. With
    ``z_indep`` = (z_plus, z_minus), C_+ and C_- use their own draws (independent sampling);
    by default all three use ``z`` (common random numbers).
    """
    p = _params(S, K, T, r, y, sigma)
    zp, zm = (z, z) if z_indep is None else z_indep
    C0 = discounted_payoffs(z, kind, **p)
    Cp = discounted_payoffs(zp, kind, **dict(p, S=S * (1 + eps)))
    Cm = discounted_payoffs(zm, kind, **dict(p, S=S * (1 - eps)))
    h = eps * S
    return dict(delta_plus=(Cp - C0) / h, delta_minus=(C0 - Cm) / h,
                delta_c=(Cp - Cm) / (2 * h), gamma_c=(Cp - 2 * C0 + Cm) / h**2)


_GREEK_OF = {("S", 1): "delta", ("S", 2): "gamma", ("sigma", 1): "vega", ("r", 1): "rho_r",
             ("y", 1): "rho_y"}


def fd_moments(kind, param, h, scheme="central", crn=True, *, S, K, T, r, y, sigma):
    """Exact mean and per-sample variance of a finite-difference quotient (Black--Scholes payoff).

    The mean is the same finite difference of the closed-form price, so the bias is exact. The
    variance of one quotient (one draw, or one draw per evaluation if ``crn`` is False) is
    computed by Gaussian quadrature split at the strike crossings. Returns a dict with ``mean``,
    ``var``, ``exact`` (the closed-form Greek, or nan if not available) and ``bias``.
    """
    base = _params(S, K, T, r, y, sigma)
    pts = SCHEMES[scheme]
    power = 2 if scheme == "second" else 1
    thetas = [base[param] + k * h for k, _ in pts]
    pars = [dict(base, **{param: t}) for t in thetas]
    mean = sum(c * price(kind, **p) for (_, c), p in zip(pts, pars)) / h**power
    kinks = [float(strike_to_z(p["K"], p["S"], p["T"], p["r"], p["y"], p["sigma"])) for p in pars]

    def Y(p, u):
        return float(np.exp(-p["r"] * p["T"]) * payoff(kind, p["S"] * growth(u, p["T"], p["r"], p["y"], p["sigma"]), p["K"]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)   # tiny bumps: harmless roundoff warnings
        if crn:
            D = lambda u: sum(c * Y(p, u) for (_, c), p in zip(pts, pars)) / h**power  # noqa: E731
            var = gauss_expect(lambda u: D(u) ** 2, kinks) - mean**2
        else:
            var = 0.0
            for (_, c), p, kz in zip(pts, pars, kinks):
                m1 = price(kind, **p)
                var += c * c * (gauss_expect(lambda u, p=p: Y(p, u) ** 2, [kz]) - m1 * m1)
            var /= h ** (2 * power)
    name = _GREEK_OF.get((param, power))
    exact = float(exact_greek(kind, name, **base)) if name else float("nan")
    return dict(mean=float(mean), var=float(max(var, 0.0)), exact=exact, bias=float(mean) - exact)


def fd_rmse(kind, param, h, n, scheme="central", crn=True, **P):
    """Root mean squared error sqrt(bias^2 + var/n) of the n-sample finite-difference estimator."""
    m = fd_moments(kind, param, h, scheme, crn, **P)
    return float(np.sqrt(m["bias"] ** 2 + m["var"] / n))


def optimal_bump(kind, param, n, scheme="central", crn=True, bounds=(1e-6, 1.0), **P):
    """The bump h (absolute, in the units of ``param``) minimising the exact RMSE at n samples.

    Minimises over log h on ``bounds`` (a bounded scalar search, so the result may sit on a
    bound when the RMSE is monotone, as in the Lipschitz case with common random numbers).
    Returns dict(h, rmse).
    """
    obj = lambda lh: np.log(fd_rmse(kind, param, float(np.exp(lh)), n, scheme, crn, **P))  # noqa: E731
    res = minimize_scalar(obj, bounds=tuple(np.log(bounds)), method="bounded",
                          options=dict(xatol=1e-4))
    return dict(h=float(np.exp(res.x)), rmse=float(np.exp(res.fun)))


# ------------------------------------------------------------------- pathwise estimators
def pathwise_samples(z, kind, S, K, T, r, y, sigma):
    """Pathwise (analytical-differencing) estimators of the Black--Scholes Greeks, per sample.

    With G = S_T / S and f' the payoff derivative (``payoff_derivative``):
      delta  e^{-rT} f'(S_T) G,               vega   e^{-rT} f'(S_T) S_T (sqrt(T) z - sigma T),
      rho_r  -T e^{-rT} f(S_T) + T e^{-rT} f'(S_T) S_T,   rho_y  -T e^{-rT} f'(S_T) S_T,
      theta  r e^{-rT} f(S_T) - e^{-rT} f'(S_T) S_T (nu + sigma z / (2 sqrt(T))),
      dual_delta (d/dK)  -e^{-rT} 1{S_T > K} (call),  e^{-rT} 1{S_T < K} (put).
    Unbiased for the call and the put. For the digital f' = 0 off the strike and the estimators
    miss the density term: they are biased (the point of the notes' discussion).
    """
    z = np.asarray(z, dtype=float)
    G = growth(z, T, r, y, sigma)
    ST = S * G
    disc = np.exp(-r * T)
    f = payoff(kind, ST, K)
    fp = payoff_derivative(kind, ST, K)
    nu = r - y - 0.5 * sigma**2
    dual = dict(call=-(ST > K).astype(float), put=(ST < K).astype(float),
                digital=np.zeros_like(ST))[kind]
    return dict(delta=disc * fp * G,
                vega=disc * fp * ST * (np.sqrt(T) * z - sigma * T),
                rho_r=-T * disc * f + T * disc * fp * ST,
                rho_y=-T * disc * fp * ST,
                theta=r * disc * f - disc * fp * ST * (nu + sigma * z / (2 * np.sqrt(T))),
                dual_delta=disc * dual)


# --------------------------------------------------------------- likelihood-ratio estimators
def lr_samples(z, kind, S, K, T, r, y, sigma):
    """Likelihood-ratio estimators of the Black--Scholes Greeks, per sample.

    The payoff Y = e^{-rT} f(S_T) is multiplied by the score of the law of log S_T, which is
    N(log S + nu T, sigma^2 T); in terms of the draw z:
      delta  Y z / (S sigma sqrt T),          gamma  Y (z^2 - 1 - sigma sqrt(T) z) / (S^2 sigma^2 T),
      vega   Y ((z^2 - 1)/sigma - sqrt(T) z),  rho_r  -T Y + Y sqrt(T) z / sigma,
      rho_y  -Y sqrt(T) z / sigma,            theta  r Y - Y (nu z/(sigma sqrt T) + (z^2 - 1)/(2T)).
    Valid for every payoff of polynomial growth, digitals included. Needs S, sigma, T > 0: the
    weights grow like 1/(sigma sqrt T).
    """
    z = np.asarray(z, dtype=float)
    Y = discounted_payoffs(z, kind, S, K, T, r, y, sigma)
    sd = sigma * np.sqrt(T)
    nu = r - y - 0.5 * sigma**2
    return dict(delta=Y * z / (S * sd),
                gamma=Y * (z * z - 1.0 - sd * z) / (S * S * sd * sd),
                vega=Y * ((z * z - 1.0) / sigma - np.sqrt(T) * z),
                rho_r=-T * Y + Y * np.sqrt(T) * z / sigma,
                rho_y=-Y * np.sqrt(T) * z / sigma,
                theta=r * Y - Y * (nu * z / sd + (z * z - 1.0) / (2 * T)))


def gamma_samples(z, kind, S, K, T, r, y, sigma):
    """The three gamma estimators of the notes, per sample.

      lr_lr  e^{-rT} f(S_T) (z^2 - 1 - sigma sqrt(T) z) / (S^2 sigma^2 T)   (any payoff),
      pw_lr  e^{-rT} f'(S_T) S_T (z - sigma sqrt T) / (S^2 sigma sqrt T)    (LR applied to the pathwise delta),
      lr_pw  e^{-rT} (f'(S_T) S_T - f(S_T)) z / (S^2 sigma sqrt T)           (pathwise applied to the LR delta),
    and pw_pw = 0, the pathwise derivative of the pathwise delta of a call or put (biased).
    pw_lr and lr_pw are unbiased for payoffs that are continuous and piecewise C^1 with f and
    s f'(s) of polynomial growth in s and 1/s (calls and puts), not for the digital.
    """
    z = np.asarray(z, dtype=float)
    ST = S * growth(z, T, r, y, sigma)
    disc = np.exp(-r * T)
    f = payoff(kind, ST, K)
    fp = payoff_derivative(kind, ST, K)
    sd = sigma * np.sqrt(T)
    return dict(lr_lr=disc * f * (z * z - 1.0 - sd * z) / (S * S * sd * sd),
                pw_lr=disc * fp * ST * (z - sd) / (S * S * sd),
                lr_pw=disc * (fp * ST - f) * z / (S * S * sd),
                pw_pw=np.zeros_like(z))


def estimator_moments(sample_fn, name, kind, S, K, T, r, y, sigma):
    """Exact mean and variance of one per-sample estimator, by quadrature over z.

    ``sample_fn`` is ``pathwise_samples``, ``lr_samples`` or ``gamma_samples`` and ``name`` the
    key of the estimator. Returns (mean, variance).
    """
    P = _params(S, K, T, r, y, sigma)
    g = lambda u: float(sample_fn(np.array([u]), kind, **P)[name][0])  # noqa: E731
    kz = [float(strike_to_z(K, S, T, r, y, sigma))]
    m1 = gauss_expect(g, kz)
    m2 = gauss_expect(lambda u: g(u) ** 2, kz)
    return m1, m2 - m1 * m1


def greeks_table(n=100_000, seed=13, kind="call", eps=0.01, S=100.0, K=100.0, T=1.0, r=0.01,
                 y=0.02, sigma=0.20):
    """The laboratory of the notes: every Greek of the payoff by pathwise, likelihood ratio and
    central finite differences with common random numbers, with n draws from default_rng(seed).

    Bumps: eps * S for delta and gamma (the second difference), eps * sigma for vega, eps * T
    for theta (= -dV/dT), and eps * 1% for the rates (one basis point for eps = 0.01).

    Returns a dict greek -> dict(exact, pw, lr, fd), each estimator a pair (mean, se); for the
    gamma, ``pw`` is the PW-LR estimator and ``lr_pw`` is added.
    """
    P = _params(S, K, T, r, y, sigma)
    z = np.random.default_rng(seed).standard_normal(n)
    pw, lr, gm = pathwise_samples(z, kind, **P), lr_samples(z, kind, **P), gamma_samples(z, kind, **P)
    est = lambda x: (float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))))  # noqa: E731
    out = {}
    for g, param, scheme in (("delta", "S", "central"), ("gamma", "S", "second"), ("vega", "sigma", "central"),
                             ("rho_r", "r", "central"), ("rho_y", "y", "central"), ("theta", "T", "central")):
        h = eps * (P[param] if param in ("S", "sigma", "T") else 0.01)
        fd = fd_samples(payoff_function(kind, param, **P), P[param], h, z, scheme)
        if g == "theta":            # theta = dV/dt = -dV/dT
            fd = -fd
        row = dict(exact=float(exact_greek(kind, g, **P)), lr=est(lr[g]), fd=est(fd))
        if g == "gamma":
            row.update(pw=est(gm["pw_lr"]), lr_pw=est(gm["lr_pw"]))
        else:
            row["pw"] = est(pw[g])
        out[g] = row
    return out


# ------------------------------------------------------------------ an Asian call on a path
def asian_samples(Z, S, K, T, r, y, sigma, average="arithmetic"):
    """Discounted payoff, pathwise and likelihood-ratio delta and vega of an Asian call, per path.

    Fixings t_j = j T / m, j = 1..m, with m = Z.shape[1]; the path is built with ``paths.py``
    (W from the increments, S_t = S exp((r - y - sigma^2/2) t + sigma W_t)). With A the
    arithmetic (or geometric) average and Y = e^{-rT}(A - K)^+:
      pw_delta  e^{-rT} 1{A > K} A / S,
      pw_vega   e^{-rT} 1{A > K} dA/dsigma,  dS_{t_j}/dsigma = S_{t_j} (W_{t_j} - sigma t_j),
      lr_delta  Y Z_1 / (S sigma sqrt(t_1))                       (only the first step depends on S),
      lr_vega   Y sum_j ((Z_j^2 - 1)/sigma - sqrt(dt) Z_j)         (every step depends on sigma).
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    m = Z.shape[1]
    t = time_grid(T, m)
    dt = T / m
    W = brownian_paths(t, Z=Z)[:, 1:]
    Sj = gbm_from_brownian(S, t[1:], W, r - y, sigma)
    if average == "arithmetic":
        A = Sj.mean(axis=1)
        dA = (Sj * (W - sigma * t[1:])).mean(axis=1)
    elif average == "geometric":
        A = np.exp(np.log(Sj).mean(axis=1))
        dA = A * (W - sigma * t[1:]).mean(axis=1)
    else:
        raise ValueError("average must be 'arithmetic' or 'geometric'")
    disc = np.exp(-r * T)
    Y = disc * np.maximum(A - K, 0.0)
    ind = (A > K).astype(float)
    return dict(price=Y, pw_delta=disc * ind * A / S, pw_vega=disc * ind * dA,
                lr_delta=Y * Z[:, 0] / (S * sigma * np.sqrt(dt)),
                lr_vega=Y * ((Z * Z - 1.0).sum(axis=1) / sigma - np.sqrt(dt) * Z.sum(axis=1)))


def geometric_asian_greeks(S, K, T, r, y, sigma, m):
    """Exact price, delta and vega of the call on the geometric average at t_j = j T / m.

    log G ~ N(mg, vg) with mg = log S + nu tbar, vg = sigma^2 sum_{i,j} min(t_i, t_j) / m^2;
    price e^{-rT}(e^{mg + vg/2} N(d1) - K N(d1 - sqrt vg)), d1 = (mg + vg - log K)/sqrt(vg).
    """
    t = time_grid(T, m)[1:]
    tbar = t.mean()
    c = np.minimum.outer(t, t).sum() / m**2
    mg = np.log(S) + (r - y - 0.5 * sigma**2) * tbar
    vg = sigma**2 * c
    sd = np.sqrt(vg)
    d1 = (mg + vg - np.log(K)) / sd
    disc, F = np.exp(-r * T), np.exp(mg + 0.5 * vg)
    price_ = disc * (F * ndtr(d1) - K * ndtr(d1 - sd))
    dP_dm = disc * F * ndtr(d1)
    dP_dv = 0.5 * disc * F * (ndtr(d1) + bs.norm_pdf(d1) / sd)
    return dict(price=float(price_), delta=float(dP_dm / S),
                vega=float(dP_dm * (-sigma * tbar) + dP_dv * 2 * sigma * c))


# --------------------------------------------------------------- adjoint (reverse) mode
def asian_adjoint(Z, S, K, T, r, y, sigmas, average="arithmetic"):
    """Price and gradient of the Asian call along each path, by a hand-coded adjoint sweep.

    Volatility sigmas[j-1] on the period (t_{j-1}, t_j], t_j = j T / m (a flat volatility is
    sigmas = sigma * ones(m)). Forward sweep, storing the path (the "tape"):
        X_0 = log S,  X_j = X_{j-1} + (r - y - sigma_j^2 / 2) dt + sigma_j sqrt(dt) Z_j,
        S_j = exp(X_j),  A = mean_j S_j (or exp(mean_j X_j)),  P = e^{-rT} (A - K)^+.
    Reverse sweep, with bar v = dP/dv:
        bar A = e^{-rT} 1{A > K},  bar X_j = bar A dA/dX_j + bar X_{j+1} (j = m, ..., 1),
        bar sigma_j = bar X_j (sqrt(dt) Z_j - sigma_j dt),
        bar r = -T P + dt sum_j bar X_j,  bar y = -dt sum_j bar X_j,  bar S = bar X_1 / S.
    One reverse sweep gives all m + 3 sensitivities. Returns dict(price, delta, rho_r, rho_y,
    vega) with vega of shape (n, m): the bucketed vegas dP/dsigma_j.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n, m = Z.shape
    sig = np.broadcast_to(np.asarray(sigmas, dtype=float), (m,))
    dt = T / m
    sq = np.sqrt(dt)
    # forward sweep: the tape is the array X of log-prices
    X = np.empty((n, m))
    x = np.full(n, np.log(S))
    for j in range(m):
        x = x + (r - y - 0.5 * sig[j] ** 2) * dt + sig[j] * sq * Z[:, j]
        X[:, j] = x
    Sj = np.exp(X)
    A = Sj.mean(axis=1) if average == "arithmetic" else np.exp(X.mean(axis=1))
    disc = np.exp(-r * T)
    P = disc * np.maximum(A - K, 0.0)
    # reverse sweep
    Abar = disc * (A > K)
    dA = Sj / m if average == "arithmetic" else A[:, None] / m * np.ones((1, m))
    Xbar = np.empty((n, m))
    acc = np.zeros(n)
    for j in range(m - 1, -1, -1):
        acc = acc + Abar * dA[:, j]
        Xbar[:, j] = acc
    vega = Xbar * (sq * Z - sig * dt)
    s = Xbar.sum(axis=1) * dt
    return dict(price=P, delta=Xbar[:, 0] / S, rho_r=-T * P + s, rho_y=-s, vega=vega)


def asian_tangent(Z, S, K, T, r, y, sigmas, direction, average="arithmetic"):
    """Directional derivative of the Asian payoff along one input direction (forward mode).

    ``direction`` is a dict with optional keys S, r, y (numbers) and sigma (array of length m):
    the tangent of each input. One sweep per direction, so all m + 3 sensitivities cost m + 3
    sweeps; used to check the adjoint and to compare costs.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n, m = Z.shape
    sig = np.broadcast_to(np.asarray(sigmas, dtype=float), (m,))
    dsig = np.broadcast_to(np.asarray(direction.get("sigma", 0.0), dtype=float), (m,))
    dr, dy = direction.get("r", 0.0), direction.get("y", 0.0)
    dt, sq = T / m, np.sqrt(T / m)
    x = np.full(n, np.log(S))
    dx = np.full(n, direction.get("S", 0.0) / S)
    A = np.zeros(n)
    dA = np.zeros(n)
    xs, dxs = np.zeros(n), np.zeros(n)
    for j in range(m):
        x = x + (r - y - 0.5 * sig[j] ** 2) * dt + sig[j] * sq * Z[:, j]
        dx = dx + (dr - dy - sig[j] * dsig[j]) * dt + dsig[j] * sq * Z[:, j]
        if average == "arithmetic":
            s = np.exp(x)
            A += s / m
            dA += s * dx / m
        else:
            xs += x / m
            dxs += dx / m
    if average == "geometric":
        A = np.exp(xs)
        dA = A * dxs
    disc = np.exp(-r * T)
    return disc * (A > K) * dA - T * dr * disc * np.maximum(A - K, 0.0)
