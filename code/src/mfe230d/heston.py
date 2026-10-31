"""The Heston stochastic volatility model (Chapter 19).

Under the risk-neutral measure,

    dS_t = S_t ((r - y) dt + sqrt(V_t) dW^S_t),     dV_t = kappa (theta - V_t) dt + xi sqrt(V_t) dW_t,

with d<W^S, W>_t = rho dt. Parameters are passed as keywords ``v0, kappa, theta, xi, rho``.

Characteristic function. ``riccati`` returns (C, D) with
E[exp(i u ln S_T)] = exp(i u (ln S + (r - y) T) + C(u, T) + D(u, T) v0), in the stable form of the notes
(``form="stable"``: g = (beta - d)/(beta + d) and exp(-d T), principal branches) or in Heston's original
form (``form="heston"``: g_H = 1/g and exp(+d T)), which jumps when the principal logarithm crosses its
branch cut. ``riccati_rk4`` integrates the Riccati equations numerically, as an independent check.
``explosion_time`` is the time at which E[S_T^a] becomes infinite; ``moment_bound`` the largest moment
that is finite at T.

Pricing. ``gil_pelaez_price``, ``carr_madan_price`` and ``cos_price`` implement the three inversions of
the notes, each with its own quadrature (midpoint rule on the half line, which is the trapezoidal rule on
the whole line for these even integrands; cosine series for COS). ``implied_vols`` inverts prices with
the Newton solver of ``impliedvol``.

Simulation. ``simulate`` runs the full-truncation Euler scheme (log-Euler for the asset) or Andersen's
quadratic-exponential (QE) scheme, with or without the martingale correction, and returns per-path
estimates of a put or a call: the plain discounted payoff, or the conditional (mixing) value given the
variance path, which has the same expectation (Proposition st-mixing of Chapter 10), optionally with the
forward as a control variate. ``bias_study`` measures the bias against the Fourier price.

Every method is implemented from scratch; ``bs.py`` supplies Black--Scholes formulas and
``stratified.py`` the mixing helpers and the exact (noncentral chi-square) variance sampler of Chapter 10.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .bs import call as bs_call, put as bs_put
from .impliedvol import implied_vol
from .stratified import bs_conditional, mixing_parameters

# The laboratory of Chapter 10 (same as stratified.HESTON_LAB), with the test case S = K = 100, T = 1,
# r = 1%, y = 2%. The ATM put is 8.1534 for rho = 0 and 7.9480 for rho = -0.7.
TEST = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3)
MARKET = dict(S=100.0, K=100.0, T=1.0, r=0.01, y=0.02)
# Fang--Oosterlee (2008, Section 5.3) and Albrecher et al. (2007): a 2005 calibration to Eurostoxx 50.
FO = dict(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
# The parameters of Andersen's (2008) Case I: Feller's condition badly violated.
CASE_I = dict(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9)


# ---------------------------------------------------------------------------------------
# The characteristic function
# ---------------------------------------------------------------------------------------

def _clog(z):
    """Principal logarithm, Arg in (-pi, pi]."""
    return np.log(z)


def riccati(u, T, kappa, theta, xi, rho, form: str = "stable"):
    """(C(u, T), D(u, T)) solving D' = xi^2 D^2 / 2 - beta D - (u^2 + i u)/2, C' = kappa theta D, C(0) = D(0) = 0,
    with beta = kappa - i rho xi u, in closed form. u may be complex (an array).

    ``form="stable"`` uses d = principal sqrt(beta^2 + xi^2 (u^2 + i u)), g = (beta - d)/(beta + d),
    D = (beta - d)/xi^2 (1 - e^{-dT})/(1 - g e^{-dT}),
    C = kappa theta/xi^2 [(beta - d) T - 2 Log((1 - g e^{-dT})/(1 - g))];
    ``form="heston"`` uses Heston's (1993) g_H = 1/g with e^{+dT} and the principal logarithm.
    xi = 0 is handled exactly (deterministic variance: D = -(u^2 + iu)(1 - e^{-kappa T})/(2 kappa)). The
    isolated points where d = 0 (never on the real axis, where |d| >= kappa) are not treated.
    """
    u = np.asarray(u, dtype=complex)
    a = u * u + 1j * u
    if xi == 0.0:
        e = -np.expm1(-kappa * T) / kappa
        return -0.5 * a * theta * (T - e), -0.5 * a * e
    beta = kappa - 1j * rho * xi * u
    d = np.sqrt(beta * beta + xi * xi * a)
    if form == "stable":
        g = (beta - d) / (beta + d)
        e = np.exp(-d * T)
        D = (beta - d) / xi**2 * (1 - e) / (1 - g * e)
        C = kappa * theta / xi**2 * ((beta - d) * T - 2 * _clog((1 - g * e) / (1 - g)))
    elif form == "heston":
        with np.errstate(divide="ignore", invalid="ignore"):   # beta = d at u = 0: Heston's g_H is infinite
            return _heston_form(u, T, kappa, theta, xi, beta, d)
    else:
        raise ValueError("form must be 'stable' or 'heston'")
    return C, D


def _heston_form(u, T, kappa, theta, xi, beta, d):
    """Heston's original form, g_H = (beta + d)/(beta - d) and e^{+dT}, with the principal logarithm."""
    gh = (beta + d) / (beta - d)
    em = np.exp(-d * T)                           # e^{-dT}; e^{+dT} itself may overflow
    D = (beta + d) / xi**2 * (em - 1) / (em - gh)
    # principal Log of z = (1 - gh e^{dT}) / (1 - gh) = e^{dT} w, computed without overflow:
    # log|z| = Re(d) T + log|w| and Arg z = Arg(e^{i Im(d) T} w), wrapped to (-pi, pi].
    w = (em - gh) / (1 - gh)
    logz = d.real * T + np.log(np.abs(w)) + 1j * np.angle(np.exp(1j * d.imag * T) * w)
    C = kappa * theta / xi**2 * ((beta + d) * T - 2 * logz)
    return C, D


def riccati_rk4(u, T, kappa, theta, xi, rho, n: int = 2000):
    """Integrate the Riccati equations with the classical fourth-order Runge--Kutta method (n steps)."""
    u = np.asarray(u, dtype=complex)
    beta = kappa - 1j * rho * xi * u
    a = u * u + 1j * u
    f = lambda D: 0.5 * xi * xi * D * D - beta * D - 0.5 * a
    D = np.zeros_like(u)
    C = np.zeros_like(u)
    h = T / n
    for _ in range(n):
        k1 = f(D)
        k2 = f(D + 0.5 * h * k1)
        k3 = f(D + 0.5 * h * k2)
        k4 = f(D + h * k3)
        C = C + kappa * theta * h / 6 * (D + 2 * (D + 0.5 * h * k1) + 2 * (D + 0.5 * h * k2) + D + h * k3)
        D = D + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return C, D


def log_charfn(u, S, T, r, y, v0, kappa, theta, xi, rho, form: str = "stable"):
    """log E[exp(i u ln S_T)]."""
    C, D = riccati(u, T, kappa, theta, xi, rho, form)
    return 1j * np.asarray(u, dtype=complex) * (np.log(S) + (r - y) * T) + C + D * v0


def charfn(u, S, T, r, y, v0, kappa, theta, xi, rho, form: str = "stable"):
    """phi(u) = E[exp(i u ln S_T)] (the convention of Chapter 2: phi_Y(z) = E[e^{izY}])."""
    return np.exp(log_charfn(u, S, T, r, y, v0, kappa, theta, xi, rho, form))


def explosion_time(a, kappa, xi, rho):
    """T*(a) = sup{T : E[S_T^a] < infinity} for real a (Andersen--Piterbarg 2007; Keller-Ressel 2011, (6.2)).

    With kappa_a = kappa - rho xi a and Delta = kappa_a^2 - xi^2 a (a - 1): T* = inf for a in [0, 1]; for
    a(a - 1) > 0, T* = inf if Delta >= 0 and kappa_a > 0; T* = log((kappa_a - sqrt Delta)/(kappa_a + sqrt Delta))
    / sqrt Delta if Delta > 0 >= kappa_a (2/|kappa_a| if Delta = 0); and T* = (2/gamma)(pi/2 + arctan(kappa_a/gamma)),
    gamma = sqrt(-Delta), if Delta < 0. These are the blow-up times of the real Riccati equation for D(-ia, .).
    """
    a = float(a)
    if 0.0 <= a <= 1.0:
        return np.inf
    ka = kappa - rho * xi * a
    Dl = ka * ka - xi * xi * a * (a - 1.0)
    if Dl >= 0:
        if ka > 0:
            return np.inf
        s = np.sqrt(Dl)
        return 2.0 / abs(ka) if s == 0 else float(np.log((ka - s) / (ka + s)) / s)
    g = np.sqrt(-Dl)
    return float(2.0 / g * (0.5 * np.pi + np.arctan(ka / g)))


def moment_bound(T, kappa, xi, rho, side: int = 1, a_max: float = 200.0, tol: float = 1e-10):
    """a_+(T) = sup{a >= 1 : E[S_T^a] < infinity} (side=1) or a_-(T) = inf{a <= 0 : E[S_T^a] < infinity}
    (side=-1), by bisection on the explosion time; +-inf if beyond +-a_max.

    Uses that T*(a) is nonincreasing in |a - 1/2| outside [0, 1] (checked on a grid in the tests)."""
    lo, hi = (1.0, a_max) if side > 0 else (0.0, a_max)
    f = (lambda a: a) if side > 0 else (lambda a: -a)
    if explosion_time(f(hi), kappa, xi, rho) > T:
        return side * np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if explosion_time(f(mid), kappa, xi, rho) > T else (lo, mid)
    return f(lo)


def tail_bounds(x_lo, x_hi, S, T, r, y, v0, kappa, theta, xi, rho, n_grid: int = 400):
    """Chernoff bounds Q(ln S_T > x_hi) <= min_p e^{-p x_hi} E[S_T^p] and Q(ln S_T < x_lo) <= min_p e^{p x_lo}
    E[S_T^{-p}], the minimum over a grid of p inside the interval where the moments are finite."""
    out = []
    for side, x in ((1, x_hi), (-1, x_lo)):
        a = moment_bound(T, kappa, xi, rho, side)
        pmax = min(abs(a), 60.0) * (1 - 1e-6)    # moments of order p in (0, |a|) are finite
        ps = np.linspace(pmax / n_grid, pmax, n_grid)
        lm = np.real(log_charfn(-1j * side * ps, S, T, r, y, v0, kappa, theta, xi, rho))
        out.append(float(np.exp(np.min(lm - side * ps * x))))
    return out[1], out[0]


def cos_interval(S, K, T, r, y, eps: float = 1e-14, **params):
    """[a, b] for Y = ln(S_T/K) such that the Chernoff bounds of Q(Y < a) and Q(Y > b) are at most eps."""
    p = _model(params)
    a_p = moment_bound(T, p["kappa"], p["xi"], p["rho"], 1)
    a_m = moment_bound(T, p["kappa"], p["xi"], p["rho"], -1)
    res = []
    for side, a in ((1, a_p), (-1, a_m)):
        pmax = min(abs(a), 60.0) * (1 - 1e-6)
        ps = np.linspace(pmax / 400, pmax, 400)
        lm = np.real(log_charfn(-1j * side * ps, S, T, r, y, **p)) - side * ps * np.log(K)
        res.append(np.min((lm - np.log(eps)) / ps))      # smallest x with e^{-p x} M(p) <= eps
    return -res[1], res[0]


def laplace_integrated_variance(lam, T, v0, kappa, theta, xi):
    """L(lam) = E[exp(-lam Q_T)], Q_T = int_0^T V dt, lam >= 0: the bond-price formula of Cox, Ingersoll and Ross,
    A e^{-B v0} with g = sqrt(kappa^2 + 2 xi^2 lam), B = 2 lam (e^{gT} - 1)/((g + kappa)(e^{gT} - 1) + 2g),
    A = (2 g e^{(kappa + g) T/2} / ((g + kappa)(e^{gT} - 1) + 2g))^{2 kappa theta / xi^2} (written in e^{-gT})."""
    lam = np.asarray(lam, dtype=float)
    g = np.sqrt(kappa * kappa + 2 * xi * xi * lam)
    em = np.exp(-g * T)
    den = (g + kappa) * (1 - em) + 2 * g * em
    B = 2 * lam * (1 - em) / den
    logA = 2 * kappa * theta / xi**2 * (np.log(2 * g) + 0.5 * (kappa - g) * T - np.log(den))
    return np.exp(logA - B * v0)


def moments_log(S, T, r, y, v0, kappa, theta, xi, rho, h: float = 1e-2):
    """Mean c1 and variance c2 of ln S_T: c1 exact; c2 = -(log phi)''(0) by a five-point difference."""
    EQ = theta * T + (v0 - theta) * (-np.expm1(-kappa * T) / kappa)
    c1 = np.log(S) + (r - y) * T - 0.5 * EQ
    lp = lambda u: log_charfn(u, S, T, r, y, v0, kappa, theta, xi, rho)
    d2 = (-lp(2 * h) + 16 * lp(h) - 30 * lp(0.0) + 16 * lp(-h) - lp(-2 * h)) / (12 * h * h)
    return float(c1), float(-d2.real)


# ---------------------------------------------------------------------------------------
# Fourier pricing
# ---------------------------------------------------------------------------------------

def _model(params):
    return {k: params[k] for k in ("v0", "kappa", "theta", "xi", "rho")}


def gil_pelaez_price(K, S, T, r, y, kind: str = "call", h: float = 0.05, n: int = 2000, **params):
    """Heston's P1/P2 form: C = S e^{-yT} P1 - K e^{-rT} P2, with
    P2 = 1/2 + (1/pi) int_0^inf Re[e^{-iuk} phi(u)/(iu)] du and P1 the same with phi(u - i)/phi(-i).
    Midpoint rule with step h and n nodes u_j = (j + 1/2) h. Puts by parity."""
    p = _model(params)
    k = np.log(K)
    u = (np.arange(n) + 0.5) * h
    fwd = S * np.exp((r - y) * T)
    phi2 = charfn(u, S, T, r, y, **p)
    phi1 = charfn(u - 1j, S, T, r, y, **p) / fwd
    w = np.exp(-1j * u * np.log(K)) / (1j * u)
    P1 = 0.5 + h / np.pi * np.sum((w * phi1).real)
    P2 = 0.5 + h / np.pi * np.sum((w * phi2).real)
    c = S * np.exp(-y * T) * P1 - K * np.exp(-r * T) * P2
    return float(c if kind == "call" else c - S * np.exp(-y * T) + K * np.exp(-r * T))


def carr_madan_price(K, S, T, r, y, kind: str = "call", alpha: float = 1.0, h: float = 0.05, n: int = 2000,
                     **params):
    """Carr--Madan: C(k) = e^{-alpha k}/pi int_0^inf Re[e^{-ivk} psi(v)] dv,
    psi(v) = e^{-rT} phi(v - (alpha + 1) i) / (alpha^2 + alpha - v^2 + i (2 alpha + 1) v).
    Requires E[S_T^{1 + alpha}] < infinity (see ``moment_bound``). Midpoint rule; puts by parity."""
    p = _model(params)
    k = np.log(K)
    v = (np.arange(n) + 0.5) * h
    psi = np.exp(-r * T) * charfn(v - (alpha + 1) * 1j, S, T, r, y, **p) / (
        alpha * alpha + alpha - v * v + 1j * (2 * alpha + 1) * v)
    c = np.exp(-alpha * k) / np.pi * h * np.sum((np.exp(-1j * v * k) * psi).real)
    return float(c if kind == "call" else c - S * np.exp(-y * T) + K * np.exp(-r * T))


def _chi_psi(k, a, b, c, d):
    """Cosine coefficients of e^x and 1 on [c, d] relative to [a, b] (Fang--Oosterlee (20)--(23))."""
    w = k * np.pi / (b - a)
    chi = (np.cos(w * (d - a)) * np.exp(d) - np.cos(w * (c - a)) * np.exp(c)
           + w * np.sin(w * (d - a)) * np.exp(d) - w * np.sin(w * (c - a)) * np.exp(c)) / (1 + w * w)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(k == 0, d - c, (np.sin(w * (d - a)) - np.sin(w * (c - a))) / np.where(k == 0, 1, w))
    return chi, psi


def cos_range(S, K, T, r, y, L: float = 12.0, **params):
    """Truncation interval [c1 - L sqrt(c2), c1 + L sqrt(c2)] for Y = ln(S_T/K) (Fang--Oosterlee, Heston case)."""
    c1, c2 = moments_log(S, T, r, y, **_model(params))
    c1 -= np.log(K)
    return c1 - L * np.sqrt(c2), c1 + L * np.sqrt(c2)


def cos_price(K, S, T, r, y, kind: str = "put", N: int = 256, L: float = 12.0, interval=None, **params):
    """COS method (Fang--Oosterlee 2008) for Y = ln(S_T/K): the put from its cosine coefficients, the call
    by parity. ``interval`` overrides the truncation range [a, b] (which must contain 0)."""
    p = _model(params)
    a, b = interval if interval is not None else cos_range(S, K, T, r, y, L, **p)
    a, b = min(a, -1e-12), max(b, 1e-12)
    k = np.arange(N)
    w = k * np.pi / (b - a)
    phiY = charfn(w, S, T, r, y, **p) * np.exp(-1j * w * np.log(K))
    chi, psi = _chi_psi(k, a, b, a, 0.0)
    Vk = 2.0 / (b - a) * K * (psi - chi)
    terms = (phiY * np.exp(-1j * w * a)).real * Vk
    terms[0] *= 0.5
    put = float(np.exp(-r * T) * np.sum(terms))
    return put if kind == "put" else put + S * np.exp(-y * T) - K * np.exp(-r * T)


def price(K, S, T, r, y, kind: str = "call", method: str = "cos", eps: float = 1e-14, **params):
    """A Heston price to about 1e-10 or better: COS on the interval of ``cos_interval`` with enough terms that
    |phi| has decayed below eps at the last frequency, or Gil-Pelaez / Carr--Madan with the matching step
    h = pi / (b - a) and range."""
    p = _model(params)
    a, b = cos_interval(S, K, T, r, y, eps, **p)
    # frequency at which |phi| < eps (scan on a doubling grid)
    U = 1.0
    while abs(charfn(U, S, T, r, y, **p)) > eps and U < 1e4:     # cap: |phi| may decay slowly as |rho| -> 1
        U *= 1.25
    if method == "cos":
        N = int(np.ceil(U * (b - a) / np.pi)) + 1
        return cos_price(K, S, T, r, y, kind, N=N, interval=(a, b), **p)
    h = np.pi / (b - a)
    n = int(np.ceil(U / h)) + 1
    if method == "gil-pelaez":
        return gil_pelaez_price(K, S, T, r, y, kind, h=h, n=n, **p)
    if method == "carr-madan":
        # the damped price e^{alpha k} C(k) decays like e^{alpha k} as k -> -inf and like e^{-(a_+ - 1 - alpha) k}
        # as k -> inf: the period 2 pi / h of the trapezoidal rule must cover both tails as well as [a, b]
        ap = moment_bound(T, p["kappa"], p["xi"], p["rho"])
        alpha = min(4.0, 0.25 * (ap - 1))          # Carr and Madan: a quarter of the upper bound
        width = (b - a) + abs(np.log(eps)) * (1 / alpha + 1 / max(ap - 1 - alpha, 1e-3))
        h = 2 * np.pi / width
        return carr_madan_price(K, S, T, r, y, kind, alpha=alpha, h=h, n=int(np.ceil(U / h)) + 1, **p)
    raise ValueError(method)


def implied_vols(Ks, S, T, r, y, **params):
    """Black--Scholes implied volatilities of Heston prices (out-of-the-money options, ``price`` with COS)."""
    Ks = np.atleast_1d(np.asarray(Ks, dtype=float))
    fwd = S * np.exp((r - y) * T)
    out = np.empty(Ks.size)
    for i, K in enumerate(Ks):
        kind = "call" if K >= fwd else "put"
        out[i] = implied_vol(price(K, S, T, r, y, kind, **_model(params)), S, K, T, r, y, kind)
    return out


# ---------------------------------------------------------------------------------------
# The variance process: conditional moments and the QE step
# ---------------------------------------------------------------------------------------

def cir_moments(v, h, kappa, theta, xi):
    """Conditional mean m and variance s^2 of V_{t+h} given V_t = v (exact)."""
    v = np.asarray(v, dtype=float)
    e = np.exp(-kappa * h)
    m = theta + (v - theta) * e
    s2 = v * xi * xi * e * (1 - e) / kappa + theta * xi * xi * (1 - e) ** 2 / (2 * kappa)
    return m, s2


def qe_parameters(v, h, kappa, theta, xi, psi_c: float = 1.5):
    """Parameters of Andersen's QE step: dict with m, s2, psi, the quadratic (a, b) and exponential (p, beta)
    branches (NaN where not used), and the mask ``quad`` of psi <= psi_c."""
    m, s2 = cir_moments(v, h, kappa, theta, xi)
    psi = s2 / (m * m)
    quad = psi <= psi_c
    with np.errstate(divide="ignore", invalid="ignore"):
        ip = 2.0 / psi
        b2 = np.where(quad, ip - 1 + np.sqrt(ip) * np.sqrt(np.maximum(ip - 1, 0.0)), np.nan)
        a = np.where(quad, m / (1 + b2), np.nan)
        p = np.where(quad, np.nan, (psi - 1) / (psi + 1))
        beta = np.where(quad, np.nan, (1 - p) / m)
    return dict(m=m, s2=s2, psi=psi, quad=quad, a=a, b=np.sqrt(b2), p=p, beta=beta)


def qe_step(v, h, kappa, theta, xi, z, psi_c: float = 1.5):
    """One QE step from V_t = v with the standard normal z (the exponential branch uses U = N(z))."""
    q = qe_parameters(v, h, kappa, theta, xi, psi_c)
    u = ndtr(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        quad_v = q["a"] * (q["b"] + z) ** 2
        expo_v = np.where(u <= q["p"], 0.0, np.log((1 - q["p"]) / (1 - u)) / q["beta"])
    return np.where(q["quad"], quad_v, expo_v), q


def andersen_k(h, kappa, theta, xi, rho, gamma1: float = 0.5, gamma2: float = 0.5):
    """K0..K4 of Andersen's discretisation of ln S (without the drift (r - y) h)."""
    K0 = -rho * kappa * theta * h / xi
    K1 = gamma1 * h * (kappa * rho / xi - 0.5) - rho / xi
    K2 = gamma2 * h * (kappa * rho / xi - 0.5) + rho / xi
    K3 = gamma1 * h * (1 - rho * rho)
    K4 = gamma2 * h * (1 - rho * rho)
    return K0, K1, K2, K3, K4


def martingale_k0(q, v, K1, K2, K3, K4):
    """Andersen's corrected K0* = -log M - (K1 + K3/2) v, M = E[exp(A V_{t+h}) | V_t = v], A = K2 + K4/2.
    M is finite iff A a < 1/2 on the quadratic branch and A < beta on the exponential one; NaN otherwise."""
    A = K2 + 0.5 * K4
    with np.errstate(divide="ignore", invalid="ignore"):
        one = 1 - 2 * A * q["a"]
        logM_q = np.where(one > 0, A * q["b"] ** 2 * q["a"] / one - 0.5 * np.log(one), np.nan)
        logM_e = np.where(A < q["beta"], np.log(q["p"] + q["beta"] * (1 - q["p"]) / (q["beta"] - A)), np.nan)
    logM = np.where(q["quad"], logM_q, logM_e)
    return -logM - (K1 + 0.5 * K3) * v


# ---------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------

def simulate(S, K, T, r, y, n_steps: int, n_paths: int, scheme: str = "qe", kind: str = "put",
             estimator: str = "conditional", martingale: bool = True, psi_c: float = 1.5, seed=None, **params):
    """Per-path estimates of the discounted put (or call) under a discretisation with n_steps equal steps.

    scheme: "euler" (full truncation for V, log-Euler for S, Lord--Koekkoek--van Dijk 2010) or "qe" (Andersen
    2008, gamma1 = gamma2 = 1/2, with the martingale correction if ``martingale``).
    estimator: "plain" (one more normal per step for S) or "conditional" (the exact conditional expectation
    given the variance path; same mean, smaller variance).
    Returns dict(values=..., forward=...), where ``forward`` is the conditional expectation of S_T e^{-(r-y)T}
    (mean S under the scheme), usable as a control variate.
    """
    p = _model(params)
    v0, kappa, theta, xi, rho = (p[k] for k in ("v0", "kappa", "theta", "xi", "rho"))
    rng = np.random.default_rng(seed)
    n = int(n_paths)
    h = T / n_steps
    rb = np.sqrt(max(1.0 - rho * rho, 0.0))
    if scheme == "euler":
        Vt = np.full(n, float(v0))
        Q = np.zeros(n)
        I = np.zeros(n)
        Wp = np.zeros(n)
        for _ in range(n_steps):
            dW = rng.standard_normal(n) * np.sqrt(h)
            Vp = np.maximum(Vt, 0.0)
            sv = np.sqrt(Vp)
            Q += Vp * h
            I += sv * dW
            if estimator == "plain":
                Wp += sv * rng.standard_normal(n) * np.sqrt(h)
            Vt = Vt + kappa * (theta - Vp) * h + xi * sv * dW
        St, sg = mixing_parameters(S, T, rho, Q, I)        # Proposition st-mixing (iv)
        mean_log = np.log(St) + (r - y - 0.5 * sg * sg) * T
        var_log = sg * sg * T
        if estimator == "plain":
            lnST = np.log(S) + (r - y) * T - 0.5 * Q + rho * I + rb * Wp
    elif scheme == "qe":
        K0, K1, K2, K3, K4 = andersen_k(h, kappa, theta, xi, rho)
        V = np.full(n, float(v0))
        mean_log = np.full(n, np.log(S) + (r - y) * T)
        var_log = np.zeros(n)
        for _ in range(n_steps):
            Vn, q = qe_step(V, h, kappa, theta, xi, rng.standard_normal(n), psi_c)
            k0 = martingale_k0(q, V, K1, K2, K3, K4) if martingale else K0
            mean_log += k0 + K1 * V + K2 * Vn
            var_log += K3 * V + K4 * Vn
            V = Vn
        if estimator == "plain":
            lnST = mean_log + np.sqrt(var_log) * rng.standard_normal(n)
    else:
        raise ValueError("scheme must be 'euler' or 'qe'")
    fwd = np.exp(mean_log + 0.5 * var_log - (r - y) * T)   # E[S_T | V path] e^{-(r - y) T}
    if estimator == "conditional":
        sig = np.sqrt(var_log / T)
        vals = bs_conditional(kind, fwd, K, T, r, y, sig)
    else:
        ST = np.exp(lnST)
        pay = np.maximum(K - ST, 0.0) if kind == "put" else np.maximum(ST - K, 0.0)
        vals = np.exp(-r * T) * pay
    return dict(values=vals, forward=fwd)


def estimate(sim, S, control: bool = True):
    """Mean and standard error of the simulated values, with the forward control variate (known mean S,
    in-sample coefficient) if ``control``."""
    yv = sim["values"]
    n = yv.size
    if not control:
        return float(yv.mean()), float(yv.std(ddof=1) / np.sqrt(n))
    x = sim["forward"] - S
    xc = x - x.mean()
    b = float(np.dot(xc, yv - yv.mean()) / np.dot(xc, xc))
    adj = yv - b * x
    return float(adj.mean()), float(np.sqrt(np.sum((adj - adj.mean()) ** 2) / (n - 2) / n))


def bias_study(steps_per_year=(1, 2, 4, 8, 16, 32), n_paths: int = 2**18, seed: int = 19, S=100.0, K=100.0, T=1.0,
               r=0.01, y=0.02, kind="put", schemes=("euler", "qe"), **params):
    """Bias of each scheme against the Fourier price (``price``), for time steps h = 1/m, m in ``steps_per_year``.

    Each (scheme, m) is an independent run with seed [seed, i, j] (i the index of m, j = 0 for Euler, 1 for QE),
    the conditional estimator and the forward control. Returns {"reference": price, scheme: dict(h, bias, se)}."""
    p = _model(params)
    ref = price(K, S, T, r, y, kind, **p)
    out = {"reference": ref}
    for j, sch in enumerate(schemes):
        b, s = [], []
        for i, m in enumerate(steps_per_year):
            sim = simulate(S, K, T, r, y, int(round(m * T)), n_paths, scheme=sch, kind=kind, seed=[seed, i, j], **p)
            est, se = estimate(sim, S)
            b.append(est - ref)
            s.append(se)
        out[sch] = dict(h=1.0 / np.asarray(steps_per_year, dtype=float), bias=np.asarray(b), se=np.asarray(s))
    return out
