"""American and Bermudan options by simulation (Chapter 14).

Conventions follow the notes. Exercise is allowed at the dates t_k = k T / m, k = 0..m,
and Z_k = e^{-r t_k} g(S_{t_k}) is the payoff discounted to time 0. Every simulated
quantity below is in these time-0 units: cash flows, continuation values (estimates of
E[Z_{tau_{k+1}} | S_{t_k}]), martingales and dual bounds.

Contents:
  * payoffs and regression bases (monomials and the weighted Laguerre functions of
    Longstaff and Schwartz, both in the scaled variable x = S / K);
  * reference values computed without simulation: a CRR binomial tree (European,
    Bermudan or American exercise; the American price with the Broadie--Detemple smoothing
    and Richardson extrapolation) and backward induction on a fine grid in log-price with
    the exact Gaussian transition (Bermudan options, Richardson extrapolation in the mesh);
  * the Longstaff--Schwartz algorithm: fit on one set of paths, evaluate the frozen policy
    on an independent set (out-of-sample), exercise boundaries, a European control variate;
  * upper bounds by duality: perfect foresight, the discounted European price as a
    martingale, the Andersen--Broadie martingale of a policy (nested simulation) and the
    Doob martingale of the grid value function (nearly optimal);
  * the 20 cases of Table 1 of Longstaff and Schwartz (2001).

Every random function takes a ``seed`` (anything numpy.random.default_rng accepts).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from . import bs
from .paths import gbm_paths, time_grid

# ---------------------------------------------------------------------------------------
# Payoffs and regression bases
# ---------------------------------------------------------------------------------------


def put_payoff(K):
    """g(s) = (K - s)^+ (vectorised)."""
    return lambda s: np.maximum(K - np.asarray(s, dtype=float), 0.0)


def call_payoff(K):
    """g(s) = (s - K)^+ (vectorised)."""
    return lambda s: np.maximum(np.asarray(s, dtype=float) - K, 0.0)


def payoff_of(kind, K):
    """The put or call payoff function with strike K (kind is "put" or "call")."""
    if kind == "put":
        return put_payoff(K)
    if kind == "call":
        return call_payoff(K)
    raise ValueError("kind must be 'put' or 'call'")


def poly_basis(degree: int = 3, scale: float = 1.0):
    """Monomials 1, x, ..., x^degree in the scaled variable x = s / scale."""
    def psi(s):
        x = np.asarray(s, dtype=float) / scale
        return np.vander(x, degree + 1, increasing=True)
    psi.size = degree + 1
    psi.label = f"monomials of degree {degree}"
    return psi


def laguerre_basis(n: int = 3, scale: float = 1.0):
    """A constant and e^{-x/2} L_j(x), j = 0..n-1 (Laguerre polynomials L_j), x = s / scale.

    With n = 3 these are the four functions of Longstaff and Schwartz (2001), Table 1:
    1, e^{-x/2}, e^{-x/2}(1 - x), e^{-x/2}(1 - 2x + x^2/2).
    """
    def psi(s):
        x = np.asarray(s, dtype=float) / scale
        out = np.empty((x.size, n + 1))
        out[:, 0] = 1.0
        if n:
            w = np.exp(-0.5 * x)
            prev, cur = np.zeros_like(x), np.ones_like(x)      # L_{-1} := 0, L_0 = 1
            for j in range(n):
                out[:, j + 1] = w * cur
                prev, cur = cur, ((2 * j + 1 - x) * cur - j * prev) / (j + 1)
        return out
    psi.size = n + 1
    psi.label = f"constant and {n} weighted Laguerre functions"
    return psi


def design_condition(basis, s) -> float:
    """2-norm condition number of the regression matrix basis(s)."""
    return float(np.linalg.cond(basis(s)))


# ---------------------------------------------------------------------------------------
# Reference values without simulation
# ---------------------------------------------------------------------------------------


def binomial(S, K, T, r, y, sigma, n: int, kind: str = "put", exercise="american",
             smooth: bool = False) -> float:
    """Cox--Ross--Rubinstein tree with n steps.

    exercise: "european", "american" (every step), or an integer m: Bermudan exercise at
    the m dates t_k = k T / m, k = 1..m (n must be a multiple of m), and at t_0 = 0.
    smooth=True replaces the values one step before maturity by the Black--Scholes
    European price (maximised with the payoff where exercise is allowed): the "binomial
    Black--Scholes" tree of Broadie and Detemple (1996).
    """
    g = payoff_of(kind, K)
    if exercise == "european":
        every = 0
    elif exercise == "american":
        every = 1
    else:
        m = int(exercise)
        if n % m:
            raise ValueError("n must be a multiple of the number of exercise dates")
        every = n // m
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    q = (np.exp((r - y) * dt) - d) / (u - d)
    if not 0.0 < q < 1.0:
        raise ValueError("risk-neutral probability outside (0, 1): take more steps")
    disc = np.exp(-r * dt)
    price = bs.put if kind == "put" else bs.call

    def nodes(i):                       # S u^{i - 2j}, j = 0..i (j down moves)
        return S * u ** (i - 2.0 * np.arange(i + 1))

    if smooth:
        i = n - 1
        V = price(nodes(i), K, dt, r, y, sigma)
        if every and i % every == 0:
            V = np.maximum(V, g(nodes(i)))
        start = n - 2
    else:
        V = g(nodes(n))
        start = n - 1
    for i in range(start, -1, -1):
        V = disc * (q * V[:-1] + (1.0 - q) * V[1:])
        if every and i % every == 0:
            V = np.maximum(V, g(nodes(i)))
    return float(V[0])


def american_binomial(S, K, T, r, y, sigma, n: int = 4000, kind: str = "put") -> float:
    """American price by the smoothed tree with Richardson extrapolation (BBSR):
    2 V(n) - V(n/2), with V the "binomial Black--Scholes" tree. n must be even."""
    a = binomial(S, K, T, r, y, sigma, n, kind, "american", smooth=True)
    b = binomial(S, K, T, r, y, sigma, n // 2, kind, "american", smooth=True)
    return 2.0 * a - b


def _grid_backward(S, K, T, r, y, sigma, m, kind, h, width, nsd, exercise=None):
    """Backward induction on the log-price grid shared by ``bermudan_grid`` and ``policy_grid``.

    exercise=None: V_k = max(g, C_k) (the optimal rule). Otherwise exercise(k, s) returns the
    mask of grid points where a given rule exercises at date k, and V_k = g there, C_k elsewhere.
    Returns x, s, g(s), the lists V (k = 1..m) and C (k = 0..m-1), and the index L of S.
    """
    g = payoff_of(kind, K)
    dt = T / m
    nu = r - y - 0.5 * sigma**2
    sd = sigma * np.sqrt(dt)
    if width is None:
        width = 10.0 * sigma * np.sqrt(T) + 1.0
    L = int(np.ceil(width / h))
    J = int(np.ceil(nsd * sd / h))
    x0 = np.log(S / K)
    x = x0 + h * np.arange(-L, L + 1)
    s = K * np.exp(x)
    gx = g(s)
    z = h * np.arange(-J, J + 1)
    kern = np.exp(-0.5 * ((z - nu * dt) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi)) * h
    kern = kern[::-1]                                # correlation written as a convolution
    left = g(K * np.exp(x0 + h * np.arange(-L - J, -L)))
    right = g(K * np.exp(x0 + h * np.arange(L + 1, L + J + 1)))
    disc = np.exp(-r * dt)
    Vs = [None] * (m + 1)
    Cs = [None] * (m + 1)
    V = gx.copy()
    Vs[m] = V
    for k in range(m - 1, -1, -1):
        C = disc * fftconvolve(np.concatenate([left, V, right]), kern, mode="valid")
        Cs[k] = C
        if k > 0:
            V = np.maximum(C, gx) if exercise is None else np.where(exercise(k, s), gx, C)
            Vs[k] = V
    return x, s, gx, Vs, Cs, L


def bermudan_grid(S, K, T, r, y, sigma, m: int, kind: str = "put", h: float = 5e-4,
                  width: float | None = None, nsd: float = 10.0, details: bool = False):
    """Bermudan option (exercise at t_k = k T / m, k = 0..m) by backward induction on a grid.

    The grid is x_i = log(S/K) + i h in log-moneyness, |i| <= L. Between two dates,
    log S moves by N(nu dt, sigma^2 dt), nu = r - y - sigma^2/2; the continuation value
    C_k(x) = e^{-r dt} E[V_{k+1}(x + nu dt + sigma sqrt(dt) Z)] is computed by the trapezoid
    rule against the exact Gaussian density (an FFT convolution), and V_k = max(g, C_k).
    Beyond the grid, which extends ``width`` in log space on each side (by default
    10 sigma sqrt(T) + 1), the values are set to the payoff; that far out they do not
    matter. The error is O(h^2); ``bermudan_reference`` extrapolates it away.

    Returns the price, or with details=True a dict with the price, the grids x and s, the
    arrays V[k] (value at date k, k = 1..m) and C[k] (continuation value at date k,
    k = 0..m-1), both in time-t_k units, and the critical prices boundary[k], k = 1..m.
    """
    x, s, gx, Vs, Cs, L = _grid_backward(S, K, T, r, y, sigma, m, kind, h, width, nsd)
    price = float(max(gx[L], Cs[0][L]))
    if not details:
        return price
    b = np.full(m + 1, np.nan)
    for k in range(1, m):
        b[k] = _boundary(s, gx, Cs[k], kind)
    b[m] = K
    return dict(price=price, s=s, x=x, V=Vs, C=Cs, boundary=b, t=time_grid(T, m), h=h, K=K)


def policy_grid(fit, basis, S, K, T, r, y, sigma, kind: str = "put", h: float = 1e-3,
                width: float | None = None, nsd: float = 10.0) -> dict:
    """Exact value of a frozen Longstaff--Schwartz policy, by backward induction on the grid.

    Same grid and transition as ``bermudan_grid``, but at date k the value is g where the
    policy ``fit`` exercises (Z_k > 0 and Z_k >= fitted continuation) and the continuation
    value C_k of the policy elsewhere. Returns the policy value at t_0 (the mean of the
    out-of-sample estimator given the training paths, up to O(h^2)) and the grid arrays.
    """
    m = fit["m"]
    D = discount(T, m, r)
    g = payoff_of(kind, K)

    def exercise(k, s):
        return _exercise_mask(s, D[k] * g(s), fit["coefs"][k], basis)

    x, s, gx, Vs, Cs, L = _grid_backward(S, K, T, r, y, sigma, m, kind, h, width, nsd, exercise)
    price = float(gx[L]) if fit["ex0"] else float(Cs[0][L])
    return dict(price=price, s=s, x=x, V=Vs, C=Cs, t=time_grid(T, m), h=h, K=K)


def _boundary(s, gx, C, kind):
    """Critical price at one date: where the exercise value g crosses the continuation C.

    For the put, the largest s in the money with g(s) >= C(s); for the call, the smallest.
    Linear interpolation of g - C between the two grid points around the crossing.
    """
    diff = gx - C
    ex = (diff >= 0) & (gx > 0)
    if not ex.any():
        return np.nan
    if kind == "put":
        i = np.flatnonzero(ex).max()
        if i + 1 >= s.size:
            return s[i]
        d0, d1 = diff[i], diff[i + 1]
        return float(s[i] + (s[i + 1] - s[i]) * d0 / (d0 - d1))
    i = np.flatnonzero(ex).min()
    if i == 0:
        return s[0]
    d0, d1 = diff[i - 1], diff[i]
    return float(s[i - 1] + (s[i] - s[i - 1]) * (-d0) / (d1 - d0))


def bermudan_reference(S, K, T, r, y, sigma, m: int, kind: str = "put", h: float = 1e-3) -> float:
    """Richardson extrapolation (4 V(h/2) - V(h)) / 3 of ``bermudan_grid`` (error O(h^2))."""
    a = bermudan_grid(S, K, T, r, y, sigma, m, kind, h=h)
    b = bermudan_grid(S, K, T, r, y, sigma, m, kind, h=h / 2)
    return (4.0 * b - a) / 3.0


# ---------------------------------------------------------------------------------------
# Simulation and the Longstaff--Schwartz algorithm
# ---------------------------------------------------------------------------------------


def simulate(S0, T, m, r, y, sigma, n_paths: int, seed=None, antithetic: bool = False):
    """Exact GBM paths at the dates t_k = k T / m, shape (n_paths, m + 1).

    With antithetic=True, the second half of the paths uses the negated normals of the
    first half (n_paths must be even).
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_paths must be even for antithetic pairs")
        Z = rng.standard_normal((n_paths // 2, m))
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal((n_paths, m))
    return gbm_paths(S0, time_grid(T, m), r, y, sigma, Z=Z)


def _substream(seed, j: int) -> list:
    """The seed list [*seed, j]: an independent stream for each j (numpy SeedSequence)."""
    return [int(v) for v in np.atleast_1d(seed).ravel()] + [int(j)]


def discount(T, m, r):
    """Discount factors e^{-r t_k}, k = 0..m."""
    return np.exp(-r * time_grid(T, m))


def lsm_fit(S, payoff, basis, r, T, itm_only: bool = True, min_points: int | None = None,
            antithetic: bool = False) -> dict:
    """Longstaff--Schwartz on the training paths S (shape (n, m+1), or (n, m+1, d)).

    Backward over k = m-1, ..., 1: regress the realised discounted cash flows Y (time-0
    units) on basis(S_k) over the in-the-money paths (all paths if itm_only=False), by
    least squares (numpy.linalg.lstsq: SVD, so rank deficiency does no harm); exercise
    where Z_k > 0 and Z_k >= the fitted continuation value, and update Y there. A date
    with fewer than ``min_points`` (default: number of basis functions + 1) usable paths
    gets no regression and no exercise: the rule is still a stopping time. At t_0 all
    paths share S_0, so the continuation value is the sample mean c0 of Y, and the policy
    exercises at once if Z_0 > 0 and Z_0 >= c0.

    Returns a dict: coefs (list, None where no regression), c0, ex0, the in-sample
    estimate ``value`` = max(Z_0, c0) and its naive standard error ``se`` (that of the
    mean of Y, from the pair averages if the paths are antithetic pairs; it ignores the
    dependence created by the regression; 0 if the policy exercises at t_0), the samples
    Y, the exercise dates tau and the number of in-the-money paths at each date.
    """
    S = np.asarray(S, dtype=float)
    n, m1 = S.shape[:2]
    m = m1 - 1
    D = discount(T, m, r)
    J = basis.size
    min_points = J + 1 if min_points is None else min_points
    Y = D[m] * payoff(S[:, m])
    tau = np.full(n, m)
    coefs = [None] * (m + 1)
    n_itm = np.zeros(m + 1, dtype=int)
    for k in range(m - 1, 0, -1):
        Zk = D[k] * payoff(S[:, k])
        sel = Zk > 0 if itm_only else np.ones(n, dtype=bool)
        n_itm[k] = int((Zk > 0).sum())
        idx = np.flatnonzero(sel)
        if idx.size < min_points:
            continue
        X = basis(S[idx, k])
        coef, *_ = np.linalg.lstsq(X, Y[idx], rcond=None)
        C = X @ coef
        ex = (Zk[idx] > 0) & (Zk[idx] >= C)
        j = idx[ex]
        Y[j] = Zk[j]
        tau[j] = k
        coefs[k] = coef
    Z0 = float(payoff(S[:1, 0])[0])
    c0 = float(Y.mean())
    ex0 = bool(Z0 > 0 and Z0 >= c0)
    se = 0.0 if ex0 else mean_se(Y, antithetic)[1]
    return dict(coefs=coefs, c0=c0, ex0=ex0, Z0=Z0, value=Z0 if ex0 else c0, se=se, samples=Y,
                tau=np.where(ex0, 0, tau), n_itm=n_itm, m=m, T=T, r=r)


def mean_se(Y, antithetic: bool = False):
    """Sample mean and standard error; with antithetic=True the two halves of Y are pairs
    (path i and path i + n/2), and the standard error is that of the n/2 pair averages."""
    Y = np.asarray(Y, dtype=float)
    if antithetic:
        h = Y.size // 2
        P = 0.5 * (Y[:h] + Y[h:2 * h])
        return float(P.mean()), float(P.std(ddof=1) / np.sqrt(h))
    return float(Y.mean()), float(Y.std(ddof=1) / np.sqrt(Y.size))


def _exercise_mask(Sk, Zk, coef, basis):
    """Where the frozen rule exercises at one date: Z_k > 0 and Z_k >= fitted continuation."""
    ex = np.zeros(Zk.shape, dtype=bool)
    if coef is None:
        return ex
    itm = np.flatnonzero(Zk > 0)
    if itm.size:
        ex[itm] = Zk[itm] >= basis(Sk[itm]) @ coef
    return ex


def policy_cashflows(S, fit, payoff, basis, start: int = 0):
    """Discounted cash flows of the frozen policy ``fit`` along the paths S.

    S[:, 0] is the date ``start``; the rule is applied at the dates start+1, ..., m (and at
    t_0 too when start == 0, where it exercises everything if fit["ex0"]). Returns (Y, tau):
    the discounted payoff Z_tau (time-0 units) and the exercise date of each path.
    """
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    m, T, r = fit["m"], fit["T"], fit["r"]
    D = discount(T, m, r)
    if start == 0 and fit["ex0"]:
        return np.full(n, fit["Z0"]), np.zeros(n, dtype=int)
    Y = D[m] * payoff(S[:, m - start])
    tau = np.full(n, m)
    alive = np.ones(n, dtype=bool)
    for k in range(start + 1, m):
        coef = fit["coefs"][k]
        if coef is None:
            continue
        a = np.flatnonzero(alive)
        if a.size == 0:
            break
        Sk = S[a, k - start]
        Zk = D[k] * payoff(Sk)
        ex = _exercise_mask(Sk, Zk, coef, basis)
        j = a[ex]
        Y[j] = Zk[ex]
        tau[j] = k
        alive[j] = False
    return Y, tau


def lsm_price(S0, K, T, r, y, sigma, m: int, n_paths: int, basis, kind: str = "put",
              seed=0, antithetic: bool = False, itm_only: bool = True, control: bool = True) -> dict:
    """Longstaff--Schwartz in-sample and out-of-sample estimates of a Bermudan put or call.

    Training paths use seed ``(seed, 0)``, evaluation paths seed ``(seed, 1)`` (independent
    streams), both with n_paths paths. The out-of-sample estimate applies the frozen policy
    to the evaluation paths. With control=True it also reports the out-of-sample estimate
    with the discounted European payoff e^{-rT} g(S_T) as a control variate, its known
    mean being the Black--Scholes price; the coefficient is estimated on the training
    paths (a pilot sample), so the controlled estimator stays unbiased for the policy value.
    """
    g = payoff_of(kind, K)
    train = simulate(S0, T, m, r, y, sigma, n_paths, seed=_substream(seed, 0), antithetic=antithetic)
    fit = lsm_fit(train, g, basis, r, T, itm_only=itm_only, antithetic=antithetic)
    test = simulate(S0, T, m, r, y, sigma, n_paths, seed=_substream(seed, 1), antithetic=antithetic)
    Yt, tau = policy_cashflows(test, fit, g, basis)
    mean, se = mean_se(Yt, antithetic)
    out = dict(fit=fit, in_sample=fit["value"], in_se=fit["se"], out_sample=mean, out_se=se, tau=tau,
               samples=Yt)
    if control:
        price = bs.put if kind == "put" else bs.call
        mu = float(price(S0, K, T, r, y, sigma))
        xtr = np.exp(-r * T) * g(train[:, -1])
        xte = np.exp(-r * T) * g(test[:, -1])
        ytr = fit["samples"]
        if antithetic:                      # the coefficient that is optimal for pair averages
            h = n_paths // 2
            xtr, ytr = 0.5 * (xtr[:h] + xtr[h:]), 0.5 * (ytr[:h] + ytr[h:])
        xc = xtr - xtr.mean()
        beta = float(xc @ (ytr - ytr.mean()) / (xc @ xc)) if xc @ xc > 0 else 0.0
        adj = Yt - beta * (xte - mu)
        cv, cv_se = mean_se(adj, antithetic)
        out.update(cv=cv, cv_se=cv_se, beta=beta,
                   rho=float(np.corrcoef(Yt, xte)[0, 1]) if Yt.std() > 0 else float("nan"))
    return out


def cash_flows_at(S, fit, payoff, basis, k: int):
    """The discounted cash flows Y that the backward pass of ``lsm_fit`` regresses at date k.

    Replays the fitted policy on the training paths S over the dates m-1, ..., k+1 (the
    cash flows at date k depend only on those decisions). Time-0 units.
    """
    S = np.asarray(S, dtype=float)
    m, T, r = fit["m"], fit["T"], fit["r"]
    D = discount(T, m, r)
    Y = D[m] * payoff(S[:, m])
    for j in range(m - 1, k, -1):
        Zj = D[j] * payoff(S[:, j])
        ex = _exercise_mask(S[:, j], Zj, fit["coefs"][j], basis)
        Y[ex] = Zj[ex]
    return Y


def continuation(fit, basis, k: int, s):
    """Fitted continuation value at date k (time-0 units); nan where no regression."""
    coef = fit["coefs"][k]
    s = np.asarray(s, dtype=float)
    return np.full(s.shape, np.nan) if coef is None else basis(s) @ coef


def lsm_boundary(fit, payoff, basis, s_grid, kind: str = "put"):
    """Exercise boundary of the fitted policy at k = 1..m-1, on the price grid s_grid.

    For the put, the largest s in s_grid at which the rule exercises (its exercise set
    need not be an interval); for the call, the smallest. nan where it never exercises.
    """
    m, T, r = fit["m"], fit["T"], fit["r"]
    D = discount(T, m, r)
    s_grid = np.asarray(s_grid, dtype=float)
    b = np.full(m + 1, np.nan)
    for k in range(1, m):
        Zk = D[k] * payoff(s_grid)
        ex = _exercise_mask(s_grid, Zk, fit["coefs"][k], basis)
        if ex.any():
            b[k] = s_grid[ex].max() if kind == "put" else s_grid[ex].min()
    return b


# ---------------------------------------------------------------------------------------
# Duality: upper bounds
# ---------------------------------------------------------------------------------------


def discounted_payoffs(S, payoff, r, T):
    """Z_k = e^{-r t_k} g(S_{t_k}) along the paths, shape (n, m+1)."""
    S = np.asarray(S, dtype=float)
    return discount(T, S.shape[1] - 1, r) * payoff(S)


def dual_samples(Z, M):
    """max_k (Z_k - M_k) path by path: the samples of the dual (upper) bound."""
    return np.max(np.asarray(Z) - np.asarray(M), axis=1)


def european_martingale(S, K, T, r, y, sigma, kind: str = "put"):
    """M_k = e^{-r t_k} P(t_k, S_{t_k}) - P(0, S_0), with P the Black--Scholes European
    price (P(T, s) = g(s)): a martingale with M_0 = 0, in closed form."""
    S = np.asarray(S, dtype=float)
    m = S.shape[1] - 1
    t = time_grid(T, m)
    price = bs.put if kind == "put" else bs.call
    P = np.empty_like(S)
    for k in range(m):
        P[:, k] = price(S[:, k], K, T - t[k], r, y, sigma)
    P[:, m] = payoff_of(kind, K)(S[:, m])
    U = np.exp(-r * t) * P
    return U - U[:, :1]


def andersen_broadie(S, fit, payoff, basis, y, sigma, n_inner: int, seed=None, chunk: int = 50) -> dict:
    """Andersen--Broadie upper bound for the policy ``fit``, by nested simulation.

    For each outer path (rows of S) and each date k < m, the continuation value of the
    policy, c_k = E[Z_{tau_{k+1}} | S_{t_k}], is estimated by the mean of n_inner
    independent inner paths started at S_{t_k} and stopped by the policy from k+1 on.
    Then L_k = Z_k where the policy exercises at k and c_k otherwise (L_m = Z_m), and
        M_0 = 0,   M_{k+1} = M_k + L_{k+1} - c_k,
    with the estimates in place of c_k. Returns the samples max_k (Z_k - M_k), their mean
    and standard error, and the arrays of estimated c_k, L_k and M_k. ``chunk`` outer paths
    have their inner paths simulated together; it affects memory and speed, not the result.
    The cost is of order n * n_inner * m^2 / 2 simulated steps.
    """
    S = np.asarray(S, dtype=float)
    n, m1 = S.shape
    m, T, r = fit["m"], fit["T"], fit["r"]
    if m1 != m + 1:
        raise ValueError("outer paths and policy have different exercise dates")
    D = discount(T, m, r)
    dt = T / m
    Z = D * payoff(S)
    ex = np.zeros((n, m + 1), dtype=bool)
    ex[:, 0] = fit["ex0"]
    for k in range(1, m):
        ex[:, k] = _exercise_mask(S[:, k], Z[:, k], fit["coefs"][k], basis)
    ex[:, m] = True
    rng = np.random.default_rng(seed)
    c = np.empty((n, m))
    for k in range(m):
        steps = m - k
        tk = dt * np.arange(steps + 1)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            s0 = np.repeat(S[a:b, k], n_inner)[:, None]
            Zn = rng.standard_normal((s0.shape[0], steps))
            inner = gbm_paths(s0, tk, r, y, sigma, Z=Zn)
            Y, _ = policy_cashflows(inner, fit, payoff, basis, start=k)
            c[a:b, k] = Y.reshape(b - a, n_inner).mean(axis=1)
    L = np.empty((n, m + 1))
    L[:, m] = Z[:, m]
    L[:, :m] = np.where(ex[:, :m], Z[:, :m], c)
    M = np.zeros((n, m + 1))
    M[:, 1:] = np.cumsum(L[:, 1:] - c, axis=1)
    H = dual_samples(Z, M)
    return dict(samples=H, mean=float(H.mean()), se=float(H.std(ddof=1) / np.sqrt(n)), c=c, L=L, M=M,
                exercise=ex)


def grid_doob_martingale(S, details: dict, r):
    """The Doob martingale of the grid value function, along simulated paths.

    With U_k = e^{-r t_k} V_k(S_{t_k}) the (grid) Snell envelope,
        M*_{k+1} - M*_k = e^{-r t_{k+1}} V_{k+1}(S_{t_{k+1}}) - e^{-r t_k} C_k(S_{t_k}),
    V_k and C_k interpolated linearly in log-price from ``bermudan_grid(..., details=True)``.
    """
    S = np.asarray(S, dtype=float)
    m = len(details["C"]) - 1
    x = details["x"]
    lx = np.log(S / details["K"])
    D = np.exp(-r * details["t"])
    M = np.zeros_like(S)
    for k in range(m):
        Vn = np.interp(lx[:, k + 1], x, details["V"][k + 1])
        Ck = np.interp(lx[:, k], x, details["C"][k])
        M[:, k + 1] = M[:, k] + D[k + 1] * Vn - D[k] * Ck
    return M


def policy_martingale(S, fit, payoff, basis, pol: dict):
    """The Andersen--Broadie martingale of the policy ``fit`` with exact continuation values.

    c_k = e^{-r t_k} C_k(S_{t_k}) is interpolated (linearly in log-price) from the policy's
    continuation values on the grid, ``pol = policy_grid(...)``; L_k = Z_k where the policy
    exercises and c_k elsewhere (L_m = Z_m), and M_{k+1} - M_k = L_{k+1} - c_k. This is the
    martingale that ``andersen_broadie`` estimates by nested simulation.
    """
    S = np.asarray(S, dtype=float)
    m, T, r = fit["m"], fit["T"], fit["r"]
    D = discount(T, m, r)
    Z = D * payoff(S)
    lx = np.log(S / pol["K"])
    c = np.empty((S.shape[0], m))
    for k in range(m):
        c[:, k] = D[k] * np.interp(lx[:, k], pol["x"], pol["C"][k])
    L = np.empty_like(S)
    L[:, m] = Z[:, m]
    for k in range(m):
        ex = np.full(S.shape[0], fit["ex0"]) if k == 0 else _exercise_mask(S[:, k], Z[:, k], fit["coefs"][k], basis)
        L[:, k] = np.where(ex, Z[:, k], c[:, k])
    M = np.zeros_like(S)
    M[:, 1:] = np.cumsum(L[:, 1:] - c, axis=1)
    return M


# ---------------------------------------------------------------------------------------
# The studies of the notes: bias of in-sample and out-of-sample LSM; lower and upper bounds
# ---------------------------------------------------------------------------------------


def bias_study(S0=40.0, sigma=0.20, T=1.0, m=50, degree=2, n_train=100, reps=150, seed=8, h=2e-3,
               K=40.0, r=0.06) -> dict:
    """In-sample LSM estimates and exact values of the fitted policies, over replications.

    Replication j fits LSM (monomials of the given degree in S/K, in-the-money regression)
    on n_train paths with seed [seed, degree, n_train, j]. Its in-sample estimate is
    recorded, and the value of its frozen policy is computed exactly on the grid
    (``policy_grid``): that value is the mean of the out-of-sample estimator given the
    training paths. Returns both arrays and the Bermudan price (``bermudan_reference``).
    """
    g = put_payoff(K)
    basis = poly_basis(degree, scale=K)
    true = bermudan_reference(S0, K, T, r, 0.0, sigma, m, "put", h=h)
    ins, outs = np.empty(reps), np.empty(reps)
    for j in range(reps):
        train = simulate(S0, T, m, r, 0.0, sigma, n_train, seed=[seed, degree, n_train, j])
        fit = lsm_fit(train, g, basis, r, T)
        ins[j] = fit["value"]
        outs[j] = policy_grid(fit, basis, S0, K, T, r, 0.0, sigma, "put", h=h)["price"]
    return dict(true=true, in_sample=ins, policy=outs)


def duality_study(S0=36.0, sigma=0.20, T=1.0, m=50, n_train=100_000, n_test=100_000, n_outer=2000,
                  n_cheap=100_000, nested=((400, 100),), seed=14, h=5e-4, K=40.0, r=0.06) -> dict:
    """Lower and upper bounds for one Bermudan put (by default the LS case S0 = 36, 50 dates).

    LSM with the LS basis is fitted on n_train antithetic paths (seed [seed, 0]); its value is
    computed exactly on the grid and estimated out-of-sample on n_test paths ([seed, 1]).
    On n_outer outer paths ([seed, 2]) the dual samples max_k (Z_k - M_k) are computed for
    M = 0 (perfect foresight), the European put martingale, the Andersen--Broadie martingale
    of the policy with exact (grid) continuation values, and the Doob martingale of the grid
    value function; and, for each (n_out, n_in) in ``nested``, the nested Andersen--Broadie
    estimator on the first n_out outer paths with n_in inner paths ([seed, 3, n_in]).
    The summary of the two closed-form bounds (M = 0 and the European martingale) uses
    n_cheap further paths ([seed, 4]) instead, as they cost almost nothing; ``samples`` keeps
    the values on the n_outer outer paths (for the figure).
    """
    g = put_payoff(K)
    basis = laguerre_basis(3, scale=K)
    det = bermudan_grid(S0, K, T, r, 0.0, sigma, m, "put", h=h, details=True)
    train = simulate(S0, T, m, r, 0.0, sigma, n_train, seed=[seed, 0], antithetic=True)
    fit = lsm_fit(train, g, basis, r, T, antithetic=True)
    pol = policy_grid(fit, basis, S0, K, T, r, 0.0, sigma, "put", h=h)
    test = simulate(S0, T, m, r, 0.0, sigma, n_test, seed=[seed, 1])
    Y, tau = policy_cashflows(test, fit, g, basis)
    outer = simulate(S0, T, m, r, 0.0, sigma, n_outer, seed=[seed, 2])
    Z = discounted_payoffs(outer, g, r, T)
    samples = dict(zero=Z.max(axis=1),
                   european=dual_samples(Z, european_martingale(outer, K, T, r, 0.0, sigma, "put")),
                   ab_exact=dual_samples(Z, policy_martingale(outer, fit, g, basis, pol)),
                   doob=dual_samples(Z, grid_doob_martingale(outer, det, r)))
    for n_out, n_in in nested:
        ab = andersen_broadie(outer[:n_out], fit, g, basis, 0.0, sigma, n_in, seed=[seed, 3, n_in])
        samples[f"ab_{n_in}"] = ab["samples"]
    summary = {k: mean_se(v) for k, v in samples.items()}
    cheap = simulate(S0, T, m, r, 0.0, sigma, n_cheap, seed=[seed, 4])
    Zc = discounted_payoffs(cheap, g, r, T)
    summary["zero"] = mean_se(Zc.max(axis=1))
    summary["european"] = mean_se(dual_samples(Zc, european_martingale(cheap, K, T, r, 0.0, sigma, "put")))
    return dict(true=det["price"], policy=pol["price"], out_sample=mean_se(Y), samples=samples,
                summary=summary, fit=fit)


# ---------------------------------------------------------------------------------------
# Table 1 of Longstaff and Schwartz (2001)
# ---------------------------------------------------------------------------------------

# (S, sigma, T, European (closed form), American (finite difference), LSM, s.e.) as printed
# in Longstaff and Schwartz (2001), Table 1: K = 40, r = 6%, 50 exercise dates per year,
# 100,000 paths (50,000 plus antithetic), a constant and three weighted Laguerre functions.
LS_TABLE1 = (
    (36, 0.20, 1, 3.844, 4.478, 4.472, 0.010), (36, 0.20, 2, 3.763, 4.840, 4.821, 0.012),
    (36, 0.40, 1, 6.711, 7.101, 7.091, 0.020), (36, 0.40, 2, 7.700, 8.508, 8.488, 0.024),
    (38, 0.20, 1, 2.852, 3.250, 3.244, 0.009), (38, 0.20, 2, 2.991, 3.745, 3.735, 0.011),
    (38, 0.40, 1, 5.834, 6.148, 6.139, 0.019), (38, 0.40, 2, 6.979, 7.670, 7.669, 0.022),
    (40, 0.20, 1, 2.066, 2.314, 2.313, 0.009), (40, 0.20, 2, 2.356, 2.885, 2.879, 0.010),
    (40, 0.40, 1, 5.060, 5.312, 5.308, 0.018), (40, 0.40, 2, 6.326, 6.920, 6.921, 0.022),
    (42, 0.20, 1, 1.465, 1.617, 1.617, 0.007), (42, 0.20, 2, 1.841, 2.212, 2.206, 0.010),
    (42, 0.40, 1, 4.379, 4.582, 4.588, 0.017), (42, 0.40, 2, 5.736, 6.248, 6.243, 0.021),
    (44, 0.20, 1, 1.017, 1.110, 1.118, 0.007), (44, 0.20, 2, 1.429, 1.690, 1.675, 0.009),
    (44, 0.40, 1, 3.783, 3.948, 3.957, 0.017), (44, 0.40, 2, 5.202, 5.647, 5.622, 0.021),
)
LS_CASE = dict(K=40.0, r=0.06, y=0.0, dates_per_year=50)


def ls_table1_references(h: float = 2e-3, n_tree: int = 2000) -> list:
    """For each LS case: European (closed form), Bermudan (grid) and American (tree).

    The defaults are the settings of Table 14.1 of the notes (errors below 1e-6 for the
    Bermudan prices and about 4e-5 for the American ones).
    """
    K, r = LS_CASE["K"], LS_CASE["r"]
    rows = []
    for S0, sig, T, *_ in LS_TABLE1:
        m = int(LS_CASE["dates_per_year"] * T)
        rows.append(dict(S=S0, sigma=sig, T=T,
                         european=float(bs.put(S0, K, T, r, 0.0, sig)),
                         bermudan=bermudan_reference(S0, K, T, r, 0.0, sig, m, "put", h=h),
                         american=american_binomial(S0, K, T, r, 0.0, sig, n=n_tree, kind="put")))
    return rows


def ls_table1_lsm(n_paths: int = 100_000, seed: int = 2001, rows=None) -> list:
    """Our LSM estimates for the LS cases, with their setup (antithetic paths, the LS basis).

    Row i uses the seed [seed, i] for training and evaluation streams (see ``lsm_price``).
    """
    K, r = LS_CASE["K"], LS_CASE["r"]
    basis = laguerre_basis(3, scale=K)
    out = []
    for i, (S0, sig, T, *_) in enumerate(LS_TABLE1):
        if rows is not None and i not in rows:
            continue
        m = int(LS_CASE["dates_per_year"] * T)
        res = lsm_price(S0, K, T, r, 0.0, sig, m, n_paths, basis, "put", seed=[seed, i],
                        antithetic=True, control=False)
        out.append(dict(row=i, S=S0, sigma=sig, T=T, in_sample=res["in_sample"], in_se=res["in_se"],
                        out_sample=res["out_sample"], out_se=res["out_se"]))
    return out
