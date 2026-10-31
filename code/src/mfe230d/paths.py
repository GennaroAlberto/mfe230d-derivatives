"""Simulating paths (Chapter 7).

Exact simulation of Brownian motion, geometric Brownian motion (with term structures of
r, y and sigma) and the Ornstein-Uhlenbeck process on arbitrary time grids; three
constructions of Brownian paths (increments, Brownian bridge, principal components); the
Euler and Milstein schemes for scalar SDEs dX = a(t, X) dt + b(t, X) dW; Richardson
extrapolation; and the Duffie-Glynn allocation of effort between time steps and paths.

Conventions
-----------
* Time arguments ``t`` are either a grid starting at 0 (see ``time_grid``) or increasing
  observation dates t_1 < ... < t_m with t_1 > 0, to which 0 is prepended (``as_grid``).
  Below, "the grid" means ``as_grid(t)``, of length m + 1 for m steps.
* Path arrays have shape ``(n_paths, m + 1)``: one row per path, first column the
  initial value (time 0).
* Every sampler takes either ``seed`` (anything accepted by ``numpy.random.default_rng``)
  or the standard normals ``Z`` of shape ``(n_paths, m)`` (a 1-D array of length m is
  read as one path) to use. Passing ``Z``
  lets several methods share the same randomness (common random numbers, antithetic
  pairs ``Z`` and ``-Z``, stratified or quasi-random normals). The schemes take the
  Brownian increments ``dW`` instead, so that coarse and fine grids can share one path
  (``coarsen``).
* Term-structure inputs ``r``, ``y``, ``sigma`` may be a number, a ``PiecewiseConstant``
  curve, or a vectorised function of time; only their integrals over the steps matter.
  Integrals of ``PiecewiseConstant`` curves are exact; functions are integrated by one
  Gauss-Legendre panel per interval, which is accurate for smooth functions but NOT
  across a jump or kink inside the interval: pass curves with jumps as
  ``PiecewiseConstant``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

# ---------------------------------------------------------------------------------------
# Time grids and term structures
# ---------------------------------------------------------------------------------------


def time_grid(T: float, n_steps: int) -> np.ndarray:
    """Uniform grid 0 = t_0 < t_1 < ... < t_n = T with n = n_steps."""
    return np.linspace(0.0, float(T), int(n_steps) + 1)


def as_grid(times) -> np.ndarray:
    """Turn observation dates into a grid starting at 0 (0 is prepended if missing)."""
    t = np.asarray(times, dtype=float).ravel()
    if t.size == 0 or t[0] != 0.0:
        t = np.concatenate([[0.0], t])
    if np.any(np.diff(t) <= 0):
        raise ValueError("time grid must be strictly increasing and start at 0")
    return t


class PiecewiseConstant:
    """Right-continuous step function: value ``values[i]`` on [breaks[i-1], breaks[i]).

    ``breaks`` has one element fewer than ``values``; the first value applies before
    ``breaks[0]`` and the last one after ``breaks[-1]``. Integrals are exact.
    """

    def __init__(self, breaks, values):
        self.breaks = np.asarray(breaks, dtype=float).ravel()
        self.values = np.asarray(values, dtype=float).ravel()
        if self.values.size != self.breaks.size + 1:
            raise ValueError("need len(values) == len(breaks) + 1")
        self._knots = np.concatenate([[0.0], self.breaks])
        # cumulative integral from 0 up to each knot
        self._cum = np.concatenate([[0.0], np.cumsum(self.values[:-1] * np.diff(self._knots))])

    def __call__(self, t):
        return self.values[np.searchsorted(self.breaks, np.asarray(t, dtype=float), side="right")]

    def _primitive(self, x):
        x = np.asarray(x, dtype=float)
        i = np.searchsorted(self._knots, x, side="right") - 1
        i = np.clip(i, 0, self._knots.size - 1)
        return self._cum[i] + self.values[i] * (x - self._knots[i])

    def integral(self, a, b):
        """Integral of the curve over [a, b] (a, b >= 0; arrays broadcast)."""
        return self._primitive(b) - self._primitive(a)

    def squared(self) -> "PiecewiseConstant":
        return PiecewiseConstant(self.breaks, self.values**2)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


def integral(f, a, b):
    """Integral of f over [a, b] for a number, a ``PiecewiseConstant`` or a callable.

    Callables are integrated by 32-point Gauss-Legendre quadrature on each interval
    (exact for polynomials of degree up to 63). For curves with kinks or jumps inside
    an interval, pass a ``PiecewiseConstant`` instead.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if isinstance(f, PiecewiseConstant):
        return f.integral(a, b)
    if not callable(f):
        return float(f) * (b - a)
    half, mid = 0.5 * (b - a), 0.5 * (b + a)
    nodes = mid[..., None] + half[..., None] * _GL_X
    return half * np.sum(_GL_W * np.asarray(f(nodes), dtype=float), axis=-1)


def _integral_of_square(sigma, a, b):
    if isinstance(sigma, PiecewiseConstant):
        return sigma.squared().integral(a, b)
    if callable(sigma):
        return integral(lambda u: np.asarray(sigma(u), dtype=float) ** 2, a, b)
    return float(sigma) ** 2 * (np.asarray(b, dtype=float) - np.asarray(a, dtype=float))


def effective_parameters(t, r=0.0, y=0.0, sigma=0.0) -> dict:
    """Interval ("forward") averages of r, y and sigma^2 over each step of the grid.

    ``t`` is a grid or observation dates (see ``as_grid``); with m steps, returns a dict
    of arrays of length m:
      dt     step lengths t_j - t_{j-1};
      r, y   (1/dt) * integral of r (resp. y) over the step;
      var    (1/dt) * integral of sigma^2 over the step (interval forward variance);
      sigma  sqrt(var);
      nu     r - y - var/2, the log-drift of the step;
    and ``clock``, the integrated variance Theta(t_j) = int_0^{t_j} sigma^2, of length m + 1.
    Callables are integrated by one quadrature panel per step: see the module docstring
    about jumps.
    """
    t = as_grid(t)
    a, b = t[:-1], t[1:]
    dt = b - a
    rbar = integral(r, a, b) / dt
    ybar = integral(y, a, b) / dt
    var = _integral_of_square(sigma, a, b) / dt
    var = np.maximum(var, 0.0)
    return dict(dt=dt, r=rbar, y=ybar, var=var, sigma=np.sqrt(var), nu=rbar - ybar - 0.5 * var,
                clock=np.concatenate([[0.0], np.cumsum(var * dt)]))


def term_rate(f, t):
    """Term average (1/t) int_0^t f(u) du, for t > 0 (a term rate r_{0,t} or yield y_{0,t}).

    Exact for numbers and ``PiecewiseConstant`` curves. A callable is integrated by a single
    Gauss-Legendre panel on [0, t], which is wrong if f jumps inside [0, t].
    """
    t = np.asarray(t, dtype=float)
    return integral(f, np.zeros_like(t), t) / t


def term_vol(sigma, t):
    """Term volatility sigma_{0,t} = sqrt((1/t) int_0^t sigma(u)^2 du), for t > 0.

    Exact for numbers and ``PiecewiseConstant`` curves. A callable is integrated by a single
    Gauss-Legendre panel on [0, t], which is wrong if sigma jumps inside [0, t]
    (pass such a curve as ``PiecewiseConstant``).
    """
    t = np.asarray(t, dtype=float)
    return np.sqrt(_integral_of_square(sigma, np.zeros_like(t), t) / t)


# ---------------------------------------------------------------------------------------
# Brownian paths: increments, Brownian bridge, principal components
# ---------------------------------------------------------------------------------------


def standard_normals(n_paths: int, m: int, seed=None) -> np.ndarray:
    """Independent N(0, 1) draws of shape (n_paths, m)."""
    return np.random.default_rng(seed).standard_normal((int(n_paths), int(m)))


def bridge_order(m: int):
    """Order in which the Brownian bridge construction fills the grid points 1..m.

    Returns integer arrays (idx, left, right) of length m: the k-th normal sets the point
    idx[k], given the already-built points left[k] < idx[k] < right[k]; right[k] = -1
    means "no right neighbour" (the first step, which sets the last point from 0).
    Points are filled coarse-to-fine by bisecting index ranges; for m = 2^k this is the
    dyadic construction (T, T/2, T/4, 3T/4, ...).
    """
    idx, left, right = [m], [0], [-1]
    queue = [(0, m)]
    while queue:
        nxt = []
        for lo, hi in queue:
            if hi - lo < 2:
                continue
            mid = (lo + hi) // 2
            idx.append(mid); left.append(lo); right.append(hi)
            nxt += [(lo, mid), (mid, hi)]
        queue = nxt
    return np.array(idx), np.array(left), np.array(right)


def bridge_interpolate(w_left, w_right, s_left, s, s_right, z):
    """Sample W_s given W at s_left < s < s_right (the Brownian bridge law).

    Mean: linear interpolation; variance (s - s_left)(s_right - s)/(s_right - s_left).
    Works on arrays; a zero-length interval returns w_left.
    """
    span = s_right - s_left
    with np.errstate(invalid="ignore", divide="ignore"):
        wl = np.where(span > 0, (s_right - s) / span, 1.0)
        sd = np.sqrt(np.where(span > 0, (s - s_left) * (s_right - s) / span, 0.0))
    return wl * w_left + (1.0 - wl) * w_right + sd * z


def _brownian_on_clock(s, Z, method):
    """Brownian motion B evaluated at the nondecreasing 'clock' s (s[0] = 0) from normals Z."""
    s = np.asarray(s, dtype=float)
    Z = np.atleast_2d(Z)
    n, m = Z.shape
    if m != s.size - 1:
        raise ValueError(f"Z must have {s.size - 1} columns, got {m}")
    W = np.zeros((n, m + 1))
    if method == "increments":
        W[:, 1:] = np.cumsum(np.sqrt(np.diff(s)) * Z, axis=1)
    elif method == "bridge":
        idx, left, right = bridge_order(m)
        for k, (i, lo, hi) in enumerate(zip(idx, left, right)):
            if hi < 0:  # forward step from the left neighbour
                W[:, i] = W[:, lo] + np.sqrt(s[i] - s[lo]) * Z[:, k]
            else:
                W[:, i] = bridge_interpolate(W[:, lo], W[:, hi], s[lo], s[i], s[hi], Z[:, k])
    elif method == "pca":
        A, _ = pca_factors(s)
        W[:, 1:] = Z @ A.T
    else:
        raise ValueError("method must be 'increments', 'bridge' or 'pca'")
    return W


def pca_factors(t):
    """Principal-component square root of the covariance min(t_i, t_j), i, j = 1..m.

    Returns (A, lam): A is m x m with A @ A.T = C and columns ordered by decreasing
    variance lam (the eigenvalues of C).
    """
    t = np.asarray(t, dtype=float)[1:]
    C = np.minimum.outer(t, t)
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0.0, None), V[:, order]
    V = V * np.where(V.sum(axis=0) < 0, -1.0, 1.0)  # sign convention: factors load positively
    return V * np.sqrt(lam), lam


def brownian_paths(t, n_paths: int = 1, seed=None, method: str = "increments", Z=None) -> np.ndarray:
    """Exact samples of Brownian motion on the grid as_grid(t), shape (n_paths, m + 1).

    method: "increments" (cumulative sums), "bridge" (Brownian bridge, coarse to fine) or
    "pca" (principal components). All three have exactly the law of (W_{t_0}, ..., W_{t_m});
    they differ in how the normals Z map to the path.
    """
    t = as_grid(t)
    if Z is None:
        Z = standard_normals(n_paths, t.size - 1, seed)
    return _brownian_on_clock(t, Z, method)


def construction_matrix(t, method: str = "increments") -> np.ndarray:
    """Matrix A with (W_{t_1}, ..., W_{t_m}) = A Z for the chosen construction (A A^T = C)."""
    t = as_grid(t)
    m = t.size - 1
    return _brownian_on_clock(t, np.eye(m), method)[:, 1:].T


def explained_variance(t, method: str = "increments") -> np.ndarray:
    """Share of the total variance sum_i Var(W_{t_i}) carried by the first k normals, k = 1..m."""
    A = construction_matrix(t, method)
    contrib = np.sum(A**2, axis=0)
    return np.cumsum(contrib) / np.sum(contrib)


def brownian_increments(t, n_paths: int = 1, seed=None) -> np.ndarray:
    """Increments W_{t_{k+1}} - W_{t_k} on the grid as_grid(t), shape (n_paths, m)."""
    t = as_grid(t)
    return np.sqrt(np.diff(t)) * standard_normals(n_paths, t.size - 1, seed)


def coarsen(dW: np.ndarray, factor: int) -> np.ndarray:
    """Increments of the same Brownian path on a grid ``factor`` times coarser."""
    n, m = dW.shape
    if m % factor:
        raise ValueError("number of steps must be divisible by factor")
    return dW.reshape(n, m // factor, factor).sum(axis=2)


# ---------------------------------------------------------------------------------------
# Exact simulation: GBM with term structures, Ornstein-Uhlenbeck
# ---------------------------------------------------------------------------------------


def gbm_paths(S0, t, r=0.0, y=0.0, sigma=0.2, n_paths: int = 1, seed=None,
              method: str = "increments", Z=None) -> np.ndarray:
    """Exact paths of dS/S = (r(t) - y(t)) dt + sigma(t) dW on the grid t.

    Uses the interval averages of r, y and sigma^2 (``effective_parameters``):
        S_{t_j} = S_{t_{j-1}} exp(nu_j dt_j + sigma_j sqrt(dt_j) Z_j),
    which is the exact joint law at the grid dates. With method "bridge" or "pca" the
    Gaussian part is built on the variance clock Theta(t_j) by that construction instead.
    Returns an array of shape (n_paths, m + 1) for the grid as_grid(t); the first column
    is S0.
    """
    t = as_grid(t)
    p = effective_parameters(t, r, y, sigma)
    if Z is None:
        Z = standard_normals(n_paths, t.size - 1, seed)
    B = _brownian_on_clock(p["clock"], Z, method)
    drift = np.concatenate([[0.0], np.cumsum(p["nu"] * p["dt"])])
    return S0 * np.exp(drift + B)


def gbm_from_brownian(S0, t, W, mu, sigma):
    """Exact GBM S0 exp((mu - sigma^2/2) t + sigma W_t) along given Brownian paths W."""
    return S0 * np.exp((mu - 0.5 * sigma**2) * np.asarray(t) + sigma * W)


def ou_paths(x0, t, kappa, theta, xi, n_paths: int = 1, seed=None, Z=None) -> np.ndarray:
    """Exact paths of the Ornstein-Uhlenbeck process dX = kappa (theta - X) dt + xi dW.

    X_{t+h} = theta + (X_t - theta) e^{-kappa h} + xi sqrt((1 - e^{-2 kappa h}) / (2 kappa)) Z,
    with the variance factor read as h when kappa = 0 (Brownian motion with volatility xi).
    Returns (n_paths, m + 1) for the grid as_grid(t).
    """
    t = as_grid(t)
    h = np.diff(t)
    if Z is None:
        Z = standard_normals(n_paths, h.size, seed)
    Z = np.atleast_2d(Z)
    if Z.shape[1] != h.size:
        raise ValueError(f"Z must have {h.size} columns, got {Z.shape[1]}")
    decay = np.exp(-kappa * h)
    vfac = h if kappa == 0 else -np.expm1(-2.0 * kappa * h) / (2.0 * kappa)
    X = np.empty((Z.shape[0], t.size))
    X[:, 0] = x0
    for k in range(h.size):
        X[:, k + 1] = theta + (X[:, k] - theta) * decay[k] + xi * np.sqrt(vfac[k]) * Z[:, k]
    return X


# ---------------------------------------------------------------------------------------
# Discretisation schemes for dX = a(t, X) dt + b(t, X) dW
# ---------------------------------------------------------------------------------------


def _increments(t, n_paths, seed, dW):
    t = as_grid(t)
    if dW is None:
        dW = brownian_increments(t, n_paths, seed)
    dW = np.atleast_2d(dW)
    if dW.shape[1] != t.size - 1:
        raise ValueError("dW must have one column per step of as_grid(t)")
    return t, dW


def euler(a, b, x0, t, n_paths: int = 1, seed=None, dW=None) -> np.ndarray:
    """Euler-Maruyama: X_{k+1} = X_k + a(t_k, X_k) h_k + b(t_k, X_k) dW_k.

    a(t, x) and b(t, x) must accept a float t and an array x. ``dW`` (n_paths, m), if
    given, are the Brownian increments on the grid as_grid(t). Returns (n_paths, m + 1).
    """
    t, dW = _increments(t, n_paths, seed, dW)
    X = np.empty((dW.shape[0], t.size))
    X[:, 0] = x0
    for k in range(t.size - 1):
        x, h = X[:, k], t[k + 1] - t[k]
        X[:, k + 1] = x + a(t[k], x) * h + b(t[k], x) * dW[:, k]
    return X


def milstein(a, b, b_x, x0, t, n_paths: int = 1, seed=None, dW=None) -> np.ndarray:
    """Milstein: Euler plus (1/2) b(t_k, X_k) b_x(t_k, X_k) (dW_k^2 - h_k).

    b_x(t, x) is the derivative of b with respect to x.
    """
    t, dW = _increments(t, n_paths, seed, dW)
    X = np.empty((dW.shape[0], t.size))
    X[:, 0] = x0
    for k in range(t.size - 1):
        x, h, d = X[:, k], t[k + 1] - t[k], dW[:, k]
        bx = b(t[k], x)
        X[:, k + 1] = x + a(t[k], x) * h + bx * d + 0.5 * bx * b_x(t[k], x) * (d * d - h)
    return X


def gbm_coefficients(mu, sigma):
    """(a, b, b_x) for dS = mu S dt + sigma S dW, ready for ``euler`` and ``milstein``."""
    return (lambda t, x: mu * x), (lambda t, x: sigma * x), (lambda t, x: sigma + 0.0 * x)


def gbm_scheme_terminal(S0, mu, sigma, T, dW, scheme: str = "euler") -> np.ndarray:
    """Terminal value of the Euler or Milstein scheme for GBM on a uniform grid.

    For linear coefficients each step multiplies by a factor, so the scheme's terminal
    value is S0 times the product of 1 + mu h + sigma dW_k (Euler), plus
    sigma^2 (dW_k^2 - h)/2 inside the factor (Milstein). Same numbers as ``euler`` and
    ``milstein`` with ``gbm_coefficients``, computed without a loop over steps.
    """
    h = T / dW.shape[1]
    factor = 1.0 + mu * h + sigma * dW
    if scheme == "milstein":
        factor = factor + 0.5 * sigma**2 * (dW * dW - h)
    elif scheme != "euler":
        raise ValueError("scheme must be 'euler' or 'milstein'")
    return S0 * np.prod(factor, axis=1)


# ---------------------------------------------------------------------------------------
# Measuring orders, Richardson extrapolation, allocation of effort
# ---------------------------------------------------------------------------------------


def softplus_call(x, K=100.0, eps=5.0):
    """Smooth call-like payoff eps * log(1 + exp((x - K)/eps)); tends to (x - K)^+ as eps -> 0."""
    return eps * np.logaddexp(0.0, (np.asarray(x, dtype=float) - K) / eps)


def gbm_scheme_errors(S0=100.0, mu=-0.01, sigma=0.2, T=1.0, f=softplus_call, levels=range(7),
                      n_paths=10**6, seed=2026, chunk=100_000) -> dict:
    """Strong and weak errors of Euler and Milstein for GBM on the same Brownian paths.

    For each m = 2^l (l in levels) the schemes run with m steps on increments obtained by
    summing those of the finest grid, and are compared with the exact solution
    S0 exp((mu - sigma^2/2) T + sigma W_T) on the same path:
      strong[s][l]  = mean |X_T^h - X_T|                      (scheme s in "euler", "milstein")
      weak[s][l]    = mean f(X_T^h) - f(X_T), an unbiased estimate of E f(X_T^h) - E f(X_T)
      weak_se[s][l] = its standard error
      rich[s][l]    = mean of the Richardson combination 2 f(X^h) - f(X^{2h}) - f(X), l >= 1
      rich_se[s][l] = its standard error
    Also returns ``m``, ``h``, the payoff variance ``var_f`` = Var f(X_T) and the mean of
    f(X_T) (Monte Carlo), ``mean_f``.
    """
    levels = list(levels)
    L = max(levels)
    nf = 2**L
    rng = np.random.default_rng(seed)
    keys = ("euler", "milstein")
    acc = {s: {l: np.zeros(6) for l in levels} for s in keys}  # strong, weak, weak^2, rich, rich^2, count
    fs = np.zeros(3)
    done = 0
    while done < n_paths:
        n = min(chunk, n_paths - done)
        dWf = np.sqrt(T / nf) * rng.standard_normal((n, nf))
        XT = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * dWf.sum(axis=1))
        fX = f(XT)
        fs += (fX.sum(), (fX**2).sum(), n)
        prev = {}
        for l in sorted(levels):
            m = 2**l
            dW = coarsen(dWf, nf // m)
            for s in keys:
                Xh = gbm_scheme_terminal(S0, mu, sigma, T, dW, scheme=s)
                fh = f(Xh)
                d = fh - fX
                A = acc[s][l]
                A[0] += np.abs(Xh - XT).sum(); A[1] += d.sum(); A[2] += (d**2).sum(); A[5] += n
                if (s, l - 1) in prev:
                    R = 2.0 * fh - prev[(s, l - 1)] - fX
                    A[3] += R.sum(); A[4] += (R**2).sum()
                prev[(s, l)] = fh
        done += n
    out = dict(m=np.array([2**l for l in levels]), h=T / np.array([2**l for l in levels]),
               mean_f=fs[0] / fs[2], var_f=fs[1] / fs[2] - (fs[0] / fs[2]) ** 2)
    for name in ("strong", "weak", "weak_se", "rich", "rich_se"):
        out[name] = {s: np.full(len(levels), np.nan) for s in keys}
    for s in keys:
        for i, l in enumerate(levels):
            A = acc[s][l]
            n = A[5]
            out["strong"][s][i] = A[0] / n
            out["weak"][s][i] = A[1] / n
            out["weak_se"][s][i] = np.sqrt(max(A[2] / n - (A[1] / n) ** 2, 0.0) / n)
            if (l - 1) in levels:
                out["rich"][s][i] = A[3] / n
                out["rich_se"][s][i] = np.sqrt(max(A[4] / n - (A[3] / n) ** 2, 0.0) / n)
    return out


def gbm_scheme_weak_error(m: int, scheme: str = "euler", S0=100.0, mu=-0.01, sigma=0.2, T=1.0,
                          f=softplus_call, dx=1e-4, half_width=None) -> float:
    """Weak error E f(Y_T) - E f(X_T) of Euler or Milstein for GBM with m steps, computed exactly.

    Each scheme multiplies by i.i.d. factors, so log Y_T is a sum of m i.i.d. variables whose
    distribution function is explicit. The function discretises that law on a lattice of
    step ``dx`` in log space (cell masses), convolves it m times by FFT, and integrates f;
    the exact solution is treated in the same way, so that the lattice errors cancel in the
    difference (results are stable to about 1e-9 in dx). Two approximations are made: the
    probability N(-(1 + mu h)/(sigma sqrt h)) that an Euler factor is not positive is dropped
    (3.7e-7 per step at h = 1 in the test case; Milstein factors are positive when
    sigma^2 h < 1 + 2 mu h), and log-factors beyond +-half_width are dropped.
    """
    h = T / m
    a, b = 1.0 + mu * h, sigma * np.sqrt(h)
    if half_width is None:
        half_width = max(4.0, 14.0 * sigma * np.sqrt(T) + 2.0 * abs(mu) * T)

    def cdf(kind, y):
        if kind == "exact":
            return ndtr((y - (mu - 0.5 * sigma**2) * h) / b)
        if kind == "euler":
            return ndtr((np.exp(y) - a) / b) - ndtr(-a / b)
        c = 0.5 * sigma**2 * h                      # Milstein factor a - c + b Z + c Z^2
        z0, gmin = b / (2 * c), a - c - b * b / (4 * c)
        r = np.sqrt(np.maximum(np.exp(y) - gmin, 0.0) / c)
        return ndtr(-z0 + r) - ndtr(-z0 - r)

    if scheme not in ("euler", "milstein"):
        raise ValueError("scheme must be 'euler' or 'milstein'")
    J = int(round(half_width / dx))
    y = np.arange(-J, J) * dx
    values = f(S0 * np.exp(y))
    out = []
    for kind in (scheme, "exact"):
        p = np.roll(cdf(kind, y + dx / 2) - cdf(kind, y - dx / 2), -J)   # index 0 <-> y = 0
        law = np.roll(np.fft.irfft(np.fft.rfft(p) ** m, n=p.size), J)
        out.append(np.sum(law * values))
    return float(out[0] - out[1])


def fitted_order(h, err) -> float:
    """Least-squares slope of log|err| against log h."""
    h, err = np.asarray(h, dtype=float), np.abs(np.asarray(err, dtype=float))
    return float(np.polyfit(np.log(h), np.log(err), 1)[0])


def richardson(y_h, y_2h, p: float = 1.0):
    """Richardson extrapolation (2^p y_h - y_2h) / (2^p - 1) for an error c h^p + O(h^q)."""
    return (2.0**p * np.asarray(y_h) - np.asarray(y_2h)) / (2.0**p - 1.0)


def mse_model(m, budget, c, v, beta=1.0, T=1.0, kappa=1.0):
    """Mean-square error c^2 (T/m)^(2 beta) + v kappa m / budget of m steps and budget/(kappa m) paths."""
    m = np.asarray(m, dtype=float)
    return c**2 * (T / m) ** (2 * beta) + v * kappa * m / budget


def optimal_allocation(budget, c, v, beta=1.0, T=1.0, kappa=1.0) -> dict:
    """Duffie-Glynn allocation of a budget (in units of one time step of one path).

    Bias c h^beta with h = T/m, payoff variance v, cost kappa per step. Returns the
    minimiser m (continuous) of ``mse_model``, the number of paths n = budget/(kappa m),
    the step h, and the bias, standard deviation and root-mean-square error there.
    """
    m = (2.0 * beta * c**2 * T ** (2 * beta) * budget / (v * kappa)) ** (1.0 / (2 * beta + 1))
    n = budget / (kappa * m)
    bias = abs(c) * (T / m) ** beta
    sd = np.sqrt(v / n)
    return dict(m=m, n=n, h=T / m, bias=bias, sd=sd, rmse=np.sqrt(bias**2 + sd**2))
