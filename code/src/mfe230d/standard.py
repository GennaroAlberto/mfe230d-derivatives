"""The standard model, the Feynman--Kac formula and Green's functions (Chapter 1).

Geometric Brownian motion with a dividend yield y,

    dS_t / S_t = g dt + sigma dW_t,     g = mu - y under P,   g = r - y under Q,

its log-normal moments and transition density p(tau; x, xi), the Green's function
G(t, x; T, xi) = e^{-r (T - t)} p(T - t; x, xi) of the pricing equation

    v_t + (r - y) x v_x + (1/2) sigma^2 x^2 v_xx - r v = 0,

finite-difference residuals of the backward and forward equations, prices as superpositions
of Arrow--Debreu prices, Girsanov reweighting from P to Q, and a Feynman--Kac check of a
closed-form PDE solution (a zero-coupon bond with an Ornstein--Uhlenbeck short rate) against
Monte Carlo.

Black--Scholes closed forms are imported from ``bs``; paths come from ``paths`` and Monte
Carlo estimates (mean, standard error, confidence interval) from ``montecarlo``. Every
function is vectorised (NumPy broadcasting) unless it says otherwise; every sampler takes a
``seed``.
"""
from __future__ import annotations

import numpy as np

from . import bs, montecarlo, paths


# ---------------------------------------------------------------------------------------
# The model under P and Q
# ---------------------------------------------------------------------------------------

def log_drift(g, sigma):
    """Drift of log S for dS/S = g dt + sigma dW: g - sigma^2/2 (nu = r - y - sigma^2/2 under Q)."""
    return np.asarray(g, dtype=float) - 0.5 * np.asarray(sigma, dtype=float) ** 2


def market_price_of_risk(mu, r, sigma):
    """theta = (mu - r) / sigma, the drift removed from W by the change of measure."""
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    return (np.asarray(mu, dtype=float) - r) / np.asarray(sigma, dtype=float)


def lognormal_moment(q, x, tau, g, sigma):
    """E[S_{t+tau}^q | S_t = x] = x^q exp(q g tau + q (q - 1) sigma^2 tau / 2), any real q."""
    q, x, tau = (np.asarray(a, dtype=float) for a in (q, x, tau))
    return x**q * np.exp(q * g * tau + 0.5 * q * (q - 1.0) * np.asarray(sigma) ** 2 * tau)


def sup_moment_bound(q, x, T, g, sigma):
    """Doob bound E[sup_{s<=T} S_s^q] <= 2 x^q exp((|q m + q^2 sigma^2/2| + q^2 sigma^2/2) T), m = g - sigma^2/2."""
    q = np.asarray(q, dtype=float)
    m = log_drift(g, sigma)
    s2 = np.asarray(sigma, dtype=float) ** 2
    return 2.0 * np.asarray(x, dtype=float) ** q * np.exp((np.abs(q * m + 0.5 * q * q * s2) + 0.5 * q * q * s2) * T)


def transition_density(xi, x, tau, g, sigma):
    """Density at xi > 0 of S_{t+tau} given S_t = x, for dS/S = g dt + sigma dW (tau, sigma > 0).

    p(tau; x, xi) = N'(z) / (xi sigma sqrt(tau)),  z = (log(xi/x) - (g - sigma^2/2) tau) / (sigma sqrt(tau)).
    Returns 0 for xi <= 0.
    """
    xi, x, tau, sigma = (np.asarray(a, dtype=float) for a in (xi, x, tau, sigma))
    if np.any(tau <= 0) or np.any(sigma <= 0):
        raise ValueError("tau and sigma must be positive (the limit tau -> 0 is a Dirac delta)")
    sd = sigma * np.sqrt(tau)
    pos = xi > 0
    xs = np.where(pos, xi, 1.0)
    z = (np.log(xs / x) - log_drift(g, sigma) * tau) / sd
    return np.where(pos, bs.norm_pdf(z) / (xs * sd), 0.0)


def green(xi, x, tau, r, y, sigma):
    """Green's function of the pricing equation: G = e^{-r tau} p(tau; x, xi) under Q (drift r - y).

    G(t, x; T, xi) d xi is the time-t price, when S_t = x, of the claim paying 1 if S_T is in
    [xi, xi + d xi]: the Arrow--Debreu price density. ``tau`` is T - t > 0.
    """
    return np.exp(-r * np.asarray(tau, dtype=float)) * transition_density(xi, x, tau, r - y, sigma)


def risk_neutral_density_ratio(ST, S0, T, mu, r, y, sigma):
    """dQ/dP on F_T as a function of S_T: Z_T = exp(-theta W_T - theta^2 T / 2), W_T read off S_T."""
    theta = market_price_of_risk(mu, r, sigma)
    W_T = (np.log(np.asarray(ST, dtype=float) / S0) - log_drift(mu - y, sigma) * T) / sigma
    return np.exp(-theta * W_T - 0.5 * theta**2 * T)


def girsanov_price(f, S0, T, mu, r, y, sigma, n: int, seed=None) -> dict:
    """Price e^{-rT} E^Q[f(S_T)] estimated from samples drawn under P, weighted by Z_T.

    Samples S_T = S0 exp((mu - y - sigma^2/2) T + sigma sqrt(T) Z) (the real-world law) and
    averages e^{-rT} Z_T f(S_T). Returns ``montecarlo.mc_estimate`` of the weighted samples.
    """
    z = np.random.default_rng(seed).standard_normal(int(n))
    ST = S0 * np.exp(log_drift(mu - y, sigma) * T + sigma * np.sqrt(T) * z)
    weights = risk_neutral_density_ratio(ST, S0, T, mu, r, y, sigma)
    return montecarlo.mc_estimate(np.exp(-r * T) * weights * f(ST))


# ---------------------------------------------------------------------------------------
# Prices as superpositions of Arrow--Debreu prices
# ---------------------------------------------------------------------------------------

def superposition_price(f, x, tau, r, y, sigma, xi) -> float:
    """v(t, x) = int G(t, x; T, xi) f(xi) d xi by the trapezoid rule on the grid ``xi``.

    ``x`` is a number. Put grid points at the kinks and jumps of f (strikes) and let the grid
    cover the region where G is not negligible. At a jump point, f should return the midpoint
    of its one-sided limits (e.g. 1/2 for a digital at its strike): the trapezoid rule is then
    second-order accurate on both sides of the jump.
    """
    xi = np.asarray(xi, dtype=float)
    vals = green(xi, x, tau, r, y, sigma) * f(xi)
    return float(np.sum(0.5 * (vals[1:] + vals[:-1]) * np.diff(xi)))


def log_grid(x, tau, sigma, width: float = 12.0, n: int = 20_001, points=()):
    """Grid in xi, uniform in log xi over x exp(+-width sigma sqrt(tau) + small drift), with extra points."""
    sd = sigma * np.sqrt(tau)
    grid = x * np.exp(np.linspace(-width * sd, width * sd, int(n)))
    return np.unique(np.concatenate([grid, np.asarray(points, dtype=float)]))


# ---------------------------------------------------------------------------------------
# Finite-difference checks of the backward and forward equations
# ---------------------------------------------------------------------------------------

def fk_residual(u, t, x, b, sig, rate, cost=None, ht=1e-4, hx=1e-4):
    """Central-difference approximation of u_t + b u_x + (1/2) sig^2 u_xx - rate u + cost at (t, x).

    ``u(t, x)``, ``b(t, x)``, ``sig(t, x)``, ``rate(t, x)`` and ``cost(t, x)`` are vectorised
    callables (``rate`` and ``cost`` may also be numbers); ``ht``, ``hx`` are absolute steps
    (arrays broadcast). The truncation error is O(ht^2 + hx^2); rounding adds about
    eps |u| / hx^2.
    """
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    u0 = u(t, x)
    ut = (u(t + ht, x) - u(t - ht, x)) / (2 * ht)
    ux = (u(t, x + hx) - u(t, x - hx)) / (2 * hx)
    uxx = (u(t, x + hx) - 2 * u0 + u(t, x - hx)) / hx**2
    rr = rate(t, x) if callable(rate) else rate
    c = 0.0 if cost is None else (cost(t, x) if callable(cost) else cost)
    return ut + b(t, x) * ux + 0.5 * sig(t, x) ** 2 * uxx - rr * u0 + c


def backward_residual(t, x, T, xi, r, y, sigma, h=1e-3):
    """Residual of the pricing equation for (t, x) -> G(t, x; T, xi), relative steps h.

    Time step h (T - t), space step h x. Returns (residual, scale) with scale = |G_t|,
    so that residual / scale measures the relative error of the check.
    """
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    G = lambda s, z: green(xi, z, T - s, r, y, sigma)  # noqa: E731
    res = fk_residual(G, t, x, lambda s, z: (r - y) * z, lambda s, z: sigma * z, r,
                      ht=h * (T - t), hx=h * x)
    ht = h * (T - t)
    scale = np.abs((G(t + ht, x) - G(t - ht, x)) / (2 * ht))
    return res, scale


def forward_residual(T, xi, t, x, r, y, sigma, h=1e-3):
    """Residual of the forward (adjoint) equation for (T, xi) -> G(t, x; T, xi), relative steps h.

    G_T - [ (1/2) (sigma^2 xi^2 G)_{xi xi} - ((r - y) xi G)_xi - r G ], by central differences
    with time step h (T - t) and space step h xi. Returns (residual, scale = |G_T|).
    """
    T, xi = np.asarray(T, dtype=float), np.asarray(xi, dtype=float)
    G = lambda s, z: green(z, x, s - t, r, y, sigma)  # noqa: E731
    hT, hz = h * (T - t), h * xi
    GT = (G(T + hT, xi) - G(T - hT, xi)) / (2 * hT)
    A = lambda z: 0.5 * sigma**2 * z**2 * G(T, z)  # noqa: E731
    Bf = lambda z: (r - y) * z * G(T, z)  # noqa: E731
    Azz = (A(xi + hz) - 2 * A(xi) + A(xi - hz)) / hz**2
    Bz = (Bf(xi + hz) - Bf(xi - hz)) / (2 * hz)
    return GT - (Azz - Bz - r * G(T, xi)), np.abs(GT)


def max_relative_residuals(hs, tau=0.25, S=100.0, r=0.01, y=0.02, sigma=0.2, n=61):
    """Largest relative residuals of the backward and forward equations over a grid, for each h.

    Backward: (t, x) -> G(t, x; T, S) at T - t = tau, x on n points of S exp([-2, 2] sigma sqrt(tau)).
    Forward:  (T, xi) -> G(0, S; T, xi) at T = tau, xi on the same points.
    Residuals are divided by max |G_t| (resp. max |G_T|) over the grid. Returns two arrays.
    """
    pts = S * np.exp(np.linspace(-2.0, 2.0, n) * sigma * np.sqrt(tau))
    back, fwd = [], []
    for h in np.atleast_1d(hs):
        rb, sb = backward_residual(0.0, pts, tau, S, r, y, sigma, h)
        rf, sf = forward_residual(tau, pts, 0.0, S, r, y, sigma, h)
        back.append(np.max(np.abs(rb)) / np.max(sb))
        fwd.append(np.max(np.abs(rf)) / np.max(sf))
    return np.array(back), np.array(fwd)


# ---------------------------------------------------------------------------------------
# Feynman--Kac by simulation
# ---------------------------------------------------------------------------------------

def discounted_payoffs(t, X, rate, payoff, cost=None) -> np.ndarray:
    """Samples of D_T g(X_T) + int_0^T D_s c(s, X_s) ds along paths X on the grid t.

    D_s = exp(-int_0^s r(u, X_u) du). Both time integrals use the trapezoid rule on the grid.
    ``X`` has shape (n_paths, m + 1); ``rate(t, x)`` and ``cost(t, x)`` are vectorised callables
    (``cost`` may be None), ``payoff(x)`` acts on the terminal values.
    """
    t = paths.as_grid(t)
    X = np.atleast_2d(X)
    dt = np.diff(t)
    rv = rate(t[None, :], X)
    A = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(0.5 * (rv[:, 1:] + rv[:, :-1]) * dt, axis=1)], axis=1)
    D = np.exp(-A)
    out = D[:, -1] * payoff(X[:, -1])
    if cost is not None:
        cv = D * cost(t[None, :], X)
        out = out + np.sum(0.5 * (cv[:, 1:] + cv[:, :-1]) * dt, axis=1)
    return out


def vasicek_bond(x, tau, kappa, theta, xi):
    """u = exp(A(tau) - B(tau) x): price of a zero-coupon bond when the short rate is OU.

    Solves u_t + kappa (theta - x) u_x + (1/2) xi^2 u_xx - x u = 0 with u = 1 at tau = 0, where
    B = (1 - e^{-kappa tau}) / kappa and A = (theta - xi^2 / (2 kappa^2)) (B - tau) - xi^2 B^2 / (4 kappa).
    The notes write x-bar for the long-run level ``theta`` and gamma for the volatility ``xi``.
    """
    x, tau = np.asarray(x, dtype=float), np.asarray(tau, dtype=float)
    B = -np.expm1(-kappa * tau) / kappa
    A = (theta - xi**2 / (2 * kappa**2)) * (B - tau) - xi**2 * B**2 / (4 * kappa)
    return np.exp(A - B * x)


def fk_vasicek_check(x0, T, kappa, theta, xi, n_steps: int = 50, n_paths: int = 100_000, seed=None,
                     chunk: int = 50_000) -> dict:
    """Bond price E[exp(-int_0^T X_s ds)] by Monte Carlo against the PDE solution ``vasicek_bond``.

    X is the OU process dX = kappa (theta - X) dt + xi dW from x0, simulated exactly on a
    uniform grid by ``paths.ou_paths``; the integral uses the trapezoid rule. Paths are drawn
    in chunks from one generator seeded by ``seed``. Returns the ``mc_estimate`` dict plus
    ``pde`` (the closed form) and ``error`` (estimate - pde).
    """
    rng = np.random.default_rng(seed)
    t = paths.time_grid(T, n_steps)
    samples = []
    left = int(n_paths)
    while left > 0:
        k = min(chunk, left)
        Z = rng.standard_normal((k, n_steps))
        X = paths.ou_paths(x0, t, kappa, theta, xi, Z=Z)
        samples.append(discounted_payoffs(t, X, lambda s, z: z, lambda z: np.ones_like(z)))
        left -= k
    est = montecarlo.mc_estimate(np.concatenate(samples))
    pde = float(vasicek_bond(x0, T, kappa, theta, xi))
    est.update(pde=pde, error=float(est["mean"]) - pde)
    return est


def fk_gbm_check(f, pde_value, S0, T, r, y, sigma, n_paths: int = 100_000, seed=None) -> dict:
    """e^{-rT} E^Q[f(S_T)] by Monte Carlo (exact GBM step from ``paths.gbm_paths``) against a PDE value.

    Returns the ``mc_estimate`` dict plus ``pde`` and ``error``.
    """
    S = paths.gbm_paths(S0, [T], r=r, y=y, sigma=sigma, n_paths=n_paths, seed=seed)
    est = montecarlo.mc_estimate(np.exp(-r * T) * f(S[:, -1]))
    est.update(pde=float(pde_value), error=float(est["mean"]) - float(pde_value))
    return est
