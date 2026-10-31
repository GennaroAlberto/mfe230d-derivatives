"""Antithetic variates, moment matching, common random numbers and control variates (Chapter 8).

Conventions follow the notes. Under the risk-neutral measure the terminal price is

    S_T = S exp(nu T + sigma sqrt(T) Z),   nu = r - y - sigma^2 / 2,   Z ~ N(0, 1),

and every estimator below takes the standard normal draws ``z`` as an argument, so that
the same draws can be reused (common random numbers) and every result is reproducible
from a seed. Estimators return a dict with at least ``mean`` and ``se`` (standard error).

Every method of the chapter is implemented here from scratch; ``bs.py`` supplies only
the Black--Scholes closed forms used as exact values and as known control means.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import ndtr

from .bs import TEST_CASE, call

# The course's "devious" inputs for the square-root option (Lecture 6): with them the
# square-root call has the price of the test-case call, 7.36429, by the pre-wash map.
DEVIOUS = dict(S=1.0e4, K=1.0e4, T=1.0, r=0.01, y=-0.01, sigma=0.40)


# ---------------------------------------------------------------- the lognormal model
def terminal_price(z, S, T, r, y, sigma):
    """S_T = S exp(nu T + sigma sqrt(T) z) for standard normal draws z (any shape)."""
    nu = r - y - 0.5 * sigma**2
    return S * np.exp(nu * T + sigma * np.sqrt(T) * np.asarray(z, dtype=float))


def strike_to_z(K, S, T, r, y, sigma):
    """The value z* with S_T(z*) = K: the call pays off exactly when Z > z*."""
    nu = r - y - 0.5 * sigma**2
    return (np.log(K / S) - nu * T) / (sigma * np.sqrt(T))


def partial_moment(p, K, S, T, r, y, sigma, above=True):
    """E[S_T^p 1{S_T > K}] (or 1{S_T < K} if ``above`` is False), in closed form.

    With ln S_T ~ N(m, v): E[S_T^p 1{S_T > K}] = exp(p m + p^2 v / 2) N((m + p v - ln K) / sqrt v).
    """
    m = np.log(S) + (r - y - 0.5 * sigma**2) * T
    v = sigma**2 * T
    d = (m + p * v - np.log(K)) / np.sqrt(v)
    return np.exp(p * m + 0.5 * p * p * v) * ndtr(d if above else -d)


def call_payoff_moments(S, K, T, r, y, sigma):
    """Mean and variance of the undiscounted call payoff C_T = (S_T - K)^+."""
    pm = lambda p: partial_moment(p, K, S, T, r, y, sigma)
    m1 = pm(1) - K * pm(0)
    m2 = pm(2) - 2 * K * pm(1) + K * K * pm(0)
    return float(m1), float(m2 - m1 * m1)


def gauss_expect(g, kinks=(), lim=12.0):
    """E[g(Z)] for Z ~ N(0, 1) by adaptive quadrature, splitting at the points in ``kinks``.

    ``g`` takes and returns floats. Used for exact (quadrature) values of means,
    variances and correlations of payoffs, against which the simulations are checked.
    """
    phi = lambda u: np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
    edges = [-lim] + sorted(float(k) for k in kinks if -lim < k < lim) + [lim]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        total += integrate.quad(lambda u: g(u) * phi(u), a, b, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    return total


# ------------------------------------------------------------------ plain estimator
def plain(values):
    """Sample mean and CLT standard error of i.i.d. samples."""
    v = np.asarray(values, dtype=float)
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(v.size)), n=v.size)


def efficiency(var1, tau1, var2, tau2):
    """Relative efficiency of method 2 against method 1: sigma_1^2 tau_1 / (sigma_2^2 tau_2)."""
    return (var1 * tau1) / (var2 * tau2)


# ------------------------------------------------------------------ antithetic variates
def antithetic(f, z):
    """Antithetic estimator of E[f(Z)] from n pairs (z_i, -z_i).

    ``f`` maps an array of normal draws (last axis = components of one draw, or a 1-D
    array of scalar draws) to an array of payoffs. Returns the mean of the pair averages
    (f(z_i) + f(-z_i)) / 2, its standard error (from the n i.i.d. pair averages), and the
    sample correlation ``rho`` between f(Z) and f(-Z).
    """
    a, b = np.asarray(f(z), dtype=float), np.asarray(f(-np.asarray(z)), dtype=float)
    pairs = 0.5 * (a + b)
    out = plain(pairs)
    out["rho"] = float(np.corrcoef(a, b)[0, 1])
    return out


def even_odd_parts(f):
    """The even and odd parts f_e(z) = (f(z) + f(-z))/2 and f_o(z) = (f(z) - f(-z))/2."""
    return (lambda z: 0.5 * (f(z) + f(-z))), (lambda z: 0.5 * (f(z) - f(-z)))


def antithetic_correlation(f, kinks=()):
    """Exact correlation of f(Z) and f(-Z), Z ~ N(0, 1), by quadrature (f scalar -> scalar)."""
    ks = list(kinks) + [-k for k in kinks]
    m = gauss_expect(f, ks)
    var = gauss_expect(lambda u: f(u) ** 2, ks) - m * m
    cross = gauss_expect(lambda u: f(u) * f(-u), ks) - m * m
    return cross / var


def disjoint_antithetic(mean, var):
    """The disjoint-payoff formulas: if f(Z) f(-Z) = 0 almost surely, then

    rho = -mean^2 / var, the standard-error ratio of n antithetic pairs to n plain draws
    is sqrt((1 + rho) / 2), and the efficiency at twice the cost per pair is 1 / (1 + rho).
    """
    rho = -mean * mean / var
    return dict(rho=rho, se_ratio=np.sqrt(0.5 * (1.0 + rho)), efficiency=1.0 / (1.0 + rho))


# ------------------------------------------------------------------ moment matching
def moment_match(z, kind="log", s=None):
    """Affine adjustment z' = alpha + beta z of a sample of standard normals (last axis).

    kind="log": z' = (z - mean) / std (divisor n), so that the sample mean of z' is 0 and
        its sample variance is 1; then ln S_T' has exactly the log-mean and log-variance of
        the model.
    kind="exp": beta = 1 / std as above, and alpha chosen so that the sample mean of
        exp(s z') equals E[exp(s Z)] = exp(s^2 / 2), with s = sigma sqrt(T). Then the
        sample mean of S_T' equals the forward exactly ("exponential mean"), and the
        log-variance is matched as before.
    """
    z = np.asarray(z, dtype=float)
    beta = 1.0 / z.std(axis=-1, keepdims=True)
    if kind == "log":
        return (z - z.mean(axis=-1, keepdims=True)) * beta
    if kind == "exp":
        if s is None or s <= 0:
            raise ValueError("kind='exp' needs s = sigma sqrt(T) > 0")
        w = beta * z
        # log of the sample mean of exp(s w), computed stably
        mx = (s * w).max(axis=-1, keepdims=True)
        log_mean = mx + np.log(np.mean(np.exp(s * w - mx), axis=-1, keepdims=True))
        alpha = 0.5 * s - log_mean / s
        return alpha + w
    raise ValueError("kind must be 'log' or 'exp'")


def moment_matched(f, z, n_batches=32, kind="log", s=None):
    """Moment-matched estimator of E[f(Z)] with batch standard errors.

    The draws ``z`` (1-D, length divisible by ``n_batches``) are split into ``n_batches``
    batches; each batch is moment matched separately, so the batch means are i.i.d. and
    their spread gives a valid standard error. Also returns ``naive_se``, the CLT formula
    applied to the dependent adjusted draws as if they were independent (not valid).
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size % n_batches:
        raise ValueError("z must be 1-D with a length divisible by n_batches")
    zb = moment_match(z.reshape(n_batches, -1), kind=kind, s=s)
    vals = np.asarray(f(zb), dtype=float)
    batch_means = vals.mean(axis=1)
    out = plain(batch_means)
    out["naive_se"] = float(vals.std(ddof=1) / np.sqrt(vals.size))
    out["n"] = vals.size
    return out


# ------------------------------------------------------------- common random numbers
def crn_difference(f1, f2, z, z_other=None):
    """Estimate E[f1(Z)] - E[f2(Z)].

    With ``z_other`` None, both terms use the same draws (common random numbers);
    otherwise f2 is evaluated on the independent draws ``z_other``.
    """
    a = np.asarray(f1(z), dtype=float)
    if z_other is None:
        return plain(a - np.asarray(f2(z), dtype=float))
    b = np.asarray(f2(z_other), dtype=float)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return dict(mean=float(a.mean() - b.mean()), se=float(se), n=a.size)


# ------------------------------------------------------------------ control variates
def control_variate(y, x, mu_x, beta="estimated"):
    """Control-variate estimator of E[Y] from samples (y_i, x_i) and the known mean of X.

    beta: a number (fixed coefficient, e.g. 1 as in the course), "estimated" (the in-sample
    least-squares coefficient, slightly biased), or "split" (the coefficient estimated on
    each half of the sample is applied to the other half, which removes that bias).
    Returns mean, se, the coefficient used, and the sample correlation rho of X and Y.
    """
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    rho = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    if beta == "split":
        h = y.size // 2
        b1 = _ls_beta(y[:h], x[:h])            # fitted on the first half
        b2 = _ls_beta(y[h:], x[h:])            # fitted on the second half
        adj = np.concatenate([y[:h] - b2 * (x[:h] - mu_x), y[h:] - b1 * (x[h:] - mu_x)])
        out = plain(adj)
        out.update(beta=0.5 * (b1 + b2), rho=rho)
        return out
    b = _ls_beta(y, x) if beta == "estimated" else float(beta)
    adj = y - b * (x - mu_x)
    out = plain(adj)
    if beta == "estimated":                    # one coefficient fitted: n - 2 degrees of freedom
        out["se"] = float(np.sqrt(np.sum((adj - adj.mean()) ** 2) / (y.size - 2) / y.size))
    out.update(beta=b, rho=rho)
    return out


def _ls_beta(y, x):
    """Least-squares slope; 0 by convention when all the x are equal (zero denominator)."""
    xc = x - x.mean()
    sxx = float(np.dot(xc, xc))
    return float(np.dot(xc, y - y.mean()) / sxx) if sxx > 0 else 0.0


def regression_control_variates(y, X, mu_X):
    """Multiple control variates as least-squares regression.

    Regress y_i on (1, X_i - mu_X) by least squares; the estimate of E[Y] is the fitted
    intercept, i.e. the regression evaluated at X = mu_X. ``X`` has shape (n, q).
    Returns mean, se (residual variance with n - q - 1 degrees of freedom), the
    coefficients ``beta`` (length q) and the sample coefficient of determination ``R2``.
    """
    y = np.asarray(y, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    n, q = X.shape
    D = np.column_stack([np.ones(n), X - np.asarray(mu_X, dtype=float)])
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ coef
    s2 = resid @ resid / (n - q - 1)
    # standard error of the intercept: s^2 (D^T D)^{-1}_{00}
    G = np.linalg.inv(D.T @ D)
    R2 = 1.0 - resid @ resid / np.sum((y - y.mean()) ** 2)
    return dict(mean=float(coef[0]), se=float(np.sqrt(s2 * G[0, 0])), beta=coef[1:], R2=float(R2), n=n)


# ------------------------------------------------------------- the square-root option
def sqrt_call(S, K, T, r, y, sigma):
    """Price of the square-root call (sqrt(S_T) - sqrt(K))^+ by the "pre-wash" map.

    sqrt(S_T) is lognormal with volatility sigma/2 and risk-neutral drift
    r - y_eff with y_eff = (r + y + sigma^2 / 4) / 2, so the price is a Black--Scholes call
    on sqrt(S) with strike sqrt(K), yield y_eff and volatility sigma / 2.
    """
    return call(np.sqrt(S), np.sqrt(K), T, r, 0.5 * (r + y + 0.25 * sigma**2), 0.5 * sigma)


def sqrt_option_samples(z, S, K, T, r, y, sigma):
    """Discounted payoffs (Y, X) of the square-root call and of its control.

    The control is the vanilla call scaled by the slope of sqrt at the strike,
    X = (S_T - K)^+ / (2 sqrt K): the tangent of the payoff at the kink.
    """
    ST = terminal_price(z, S, T, r, y, sigma)
    disc = np.exp(-r * T)
    Y = disc * np.maximum(np.sqrt(ST) - np.sqrt(K), 0.0)
    X = disc * np.maximum(ST - K, 0.0) / (2.0 * np.sqrt(K))
    return Y, X


def sqrt_control_mean(S, K, T, r, y, sigma):
    """Known mean of the control in ``sqrt_option_samples``: call / (2 sqrt K)."""
    return float(call(S, K, T, r, y, sigma) / (2.0 * np.sqrt(K)))


# ------------------------------------------------------------- exact asymptotic factors
def regression_factor(f, controls, kinks=()):
    """Population 1 - R^2 of f(Z) regressed on the controls g_k(Z), Z ~ N(0, 1), by quadrature.

    This is the variance factor of the optimal (multiple) control variate estimator.
    Returns (factor, beta*).
    """
    ks = list(kinks) + [-k for k in kinks]
    q = len(controls)
    mf = gauss_expect(f, ks)
    vf = gauss_expect(lambda u: f(u) ** 2, ks) - mf * mf
    mc = [gauss_expect(g, ks) for g in controls]
    Sxx = np.array([[gauss_expect(lambda u: controls[i](u) * controls[j](u), ks) - mc[i] * mc[j]
                     for j in range(q)] for i in range(q)])
    Sxy = np.array([gauss_expect(lambda u: controls[i](u) * f(u), ks) - mc[i] * mf for i in range(q)])
    beta = np.linalg.solve(Sxx, Sxy)
    return float(1.0 - Sxy @ beta / vf), beta


def mm_asymptotic_factor(f, kinks=(), kind="log", s=None):
    """Asymptotic variance factor n Var(moment-matched estimator) / Var f(Z), by quadrature.

    Linearising the matched estimator (see the notes) gives the influence function
      log: psi(z) = f(z) - mu - a1 z - a2 (z^2 - 1) / 2,
      exp: psi(z) = f(z) - mu - (a2 - s a1) (z^2 - 1) / 2 - (a1 / s) (exp(s z - s^2/2) - 1),
    with a1 = E[Z f(Z)] and a2 = E[(Z^2 - 1) f(Z)]; the factor is Var psi(Z) / Var f(Z).
    For kind="log" this equals the regression factor of f on (Z, Z^2).
    """
    ks = list(kinks) + [-k for k in kinks]
    mu = gauss_expect(f, ks)
    var = gauss_expect(lambda u: f(u) ** 2, ks) - mu * mu
    a1 = gauss_expect(lambda u: u * f(u), ks)
    a2 = gauss_expect(lambda u: (u * u - 1.0) * f(u), ks)
    if kind == "log":
        psi = lambda u: f(u) - mu - a1 * u - 0.5 * a2 * (u * u - 1.0)
    else:
        psi = lambda u: (f(u) - mu - 0.5 * (a2 - s * a1) * (u * u - 1.0)
                         - (a1 / s) * (np.exp(s * u - 0.5 * s * s) - 1.0))
    return gauss_expect(lambda u: psi(u) ** 2, ks) / var


def mm_bias_constant(f, kinks=()):
    """Leading constant c in E[moment-matched estimator] - E f(Z) = c / n + o(1/n) (kind="log").

    A second-order expansion gives c = -E[f(Z) He_4(Z)] / 4 with He_4(z) = z^4 - 6 z^2 + 3.
    """
    ks = list(kinks) + [-k for k in kinks]
    return -0.25 * gauss_expect(lambda u: f(u) * (u**4 - 6 * u * u + 3.0), ks)


def cv_bias_constant(f, g, kinks=()):
    """Leading constant c in E[in-sample control-variate estimator] - E f(Z) = c / n + o(1/n).

    c = -E[(X - mu_X)^2 eps] / Var X, with X = g(Z) and eps = Y - mu_Y - beta* (X - mu_X).
    """
    ks = list(kinks) + [-k for k in kinks]
    my, mx = gauss_expect(f, ks), gauss_expect(g, ks)
    vx = gauss_expect(lambda u: g(u) ** 2, ks) - mx * mx
    b = (gauss_expect(lambda u: f(u) * g(u), ks) - mx * my) / vx
    eps = lambda u: f(u) - my - b * (g(u) - mx)
    return -gauss_expect(lambda u: (g(u) - mx) ** 2 * eps(u), ks) / vx


def scalar_payoffs(params=None):
    """Scalar discounted payoffs of the test case, as functions of z, with the kink z*.

    Returns dict with "stock" (e^{-rT} S_T), "call", and "z_star". Used with the
    quadrature helpers above.
    """
    p = dict(TEST_CASE if params is None else params)
    S, K, T, r, y, sig = (p[k] for k in ("S", "K", "T", "r", "y", "sigma"))
    disc = np.exp(-r * T)
    return dict(stock=lambda u: disc * float(terminal_price(u, S, T, r, y, sig)),
                call=lambda u: disc * max(float(terminal_price(u, S, T, r, y, sig)) - K, 0.0),
                z_star=float(strike_to_z(K, S, T, r, y, sig)))


def theory_table():
    """Exact (asymptotic) standard-error ratios to plain Monte Carlo at equal n, by quadrature.

    Keys: (target, method). Antithetic ratios compare n pairs with n plain draws.
    """
    p = scalar_payoffs()
    st, c, zs = p["stock"], p["call"], p["z_star"]
    s = TEST_CASE["sigma"] * np.sqrt(TEST_CASE["T"])
    out = {}
    out["stock", "antithetic"] = np.sqrt(0.5 * (1.0 - np.exp(-s * s)))
    out["stock", "mm-log"] = np.sqrt(mm_asymptotic_factor(st, kind="log"))
    out["stock", "mm-exp"] = 0.0
    out["stock", "cv-stock"] = 0.0
    out["call", "antithetic"] = np.sqrt(0.5 * (1.0 + antithetic_correlation(c, [zs])))
    out["call", "mm-log"] = np.sqrt(mm_asymptotic_factor(c, [zs], "log"))
    out["call", "mm-exp"] = np.sqrt(mm_asymptotic_factor(c, [zs], "exp", s))
    out["call", "cv-stock"] = np.sqrt(regression_factor(c, [st], [zs])[0])
    ce = lambda u: 0.5 * (c(u) + c(-u))
    se = lambda u: 0.5 * (st(u) + st(-u))
    fe = regression_factor(ce, [se], [zs])[0]
    mf = gauss_expect(c, [zs, -zs])
    vf = gauss_expect(lambda u: c(u) ** 2, [zs, -zs]) - mf * mf
    ve = gauss_expect(lambda u: ce(u) ** 2, [zs, -zs]) - mf * mf
    out["call", "anti+cv"] = np.sqrt(ve / vf * fe)
    d = DEVIOUS
    dd = np.exp(-d["r"] * d["T"])
    zd = float(strike_to_z(d["K"], d["S"], d["T"], d["r"], d["y"], d["sigma"]))
    STd = lambda u: float(terminal_price(u, d["S"], d["T"], d["r"], d["y"], d["sigma"]))
    Y = lambda u: dd * max(np.sqrt(STd(u)) - np.sqrt(d["K"]), 0.0)
    X = lambda u: dd * max(STd(u) - d["K"], 0.0) / (2.0 * np.sqrt(d["K"]))
    my, mx = gauss_expect(Y, [zd]), gauss_expect(X, [zd])
    vy = gauss_expect(lambda u: Y(u) ** 2, [zd]) - my * my
    vx = gauss_expect(lambda u: X(u) ** 2, [zd]) - mx * mx
    cxy = gauss_expect(lambda u: X(u) * Y(u), [zd]) - mx * my
    out["sqrt", "rho"] = cxy / np.sqrt(vx * vy)
    out["sqrt", "beta*"] = cxy / vx
    out["sqrt", "cv-beta1"] = np.sqrt((vy + vx - 2 * cxy) / vy)
    out["sqrt", "cv-beta*"] = np.sqrt(1.0 - out["sqrt", "rho"] ** 2)
    out["sqrt", "cv-multiple"] = np.sqrt(regression_factor(Y, [X, lambda u: dd * STd(u)], [zd])[0])
    return out


# ------------------------------------------------------------- the laboratory
def laboratory(n=2**20, seed=1, n_batches=256):
    """Run every method of the chapter on three targets with the same n normal draws.

    Targets: "stock" (e^{-rT} S_T, exact value S e^{-yT}) and "call" for the test case, and
    "sqrt" (the square-root call with the course's devious inputs, exact value 7.36429).
    Antithetic methods use the n draws and their negatives (n pairs); moment matching
    uses ``n_batches`` separately matched batches. Returns target -> method -> estimator.
    """
    p = TEST_CASE
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    S, K, T, r, y, sig = (p[k] for k in ("S", "K", "T", "r", "y", "sigma"))
    disc, s = np.exp(-r * T), sig * np.sqrt(T)
    mu_stock = S * np.exp(-y * T)                     # e^{-rT} E[S_T]

    def stock(u):
        return disc * terminal_price(u, S, T, r, y, sig)

    def callpay(u):
        return disc * np.maximum(terminal_price(u, S, T, r, y, sig) - K, 0.0)

    out = {}
    for name, f, exact in (("stock", stock, mu_stock), ("call", callpay, float(call(S, K, T, r, y, sig)))):
        res = {"exact": exact}
        res["plain"] = plain(f(z))
        res["antithetic"] = antithetic(f, z)
        res["mm-log"] = moment_matched(f, z, n_batches, "log")
        res["mm-exp"] = moment_matched(f, z, n_batches, "exp", s)
        res["cv-stock"] = control_variate(f(z), stock(z), mu_stock)
        ya, xa = 0.5 * (f(z) + f(-z)), 0.5 * (stock(z) + stock(-z))
        res["anti+cv"] = control_variate(ya, xa, mu_stock)
        out[name] = res

    d = DEVIOUS
    Y, X = sqrt_option_samples(z, **d)
    mux = sqrt_control_mean(**d)
    dd = np.exp(-d["r"] * d["T"])
    STd = terminal_price(z, d["S"], d["T"], d["r"], d["y"], d["sigma"])
    res = {"exact": float(sqrt_call(**d))}
    res["plain"] = plain(Y)
    res["cv-beta1"] = control_variate(Y, X, mux, beta=1.0)
    res["cv-beta*"] = control_variate(Y, X, mux)
    res["cv-split"] = control_variate(Y, X, mux, beta="split")
    res["cv-multiple"] = regression_control_variates(
        Y, np.column_stack([X, dd * STd]), [mux, d["S"] * np.exp(-d["y"] * d["T"])])
    out["sqrt"] = res
    return out


# ------------------------------------------------------------- hedges as controls
def delta_hedge_control(z, S, K, T, r, y, sigma):
    """Discounted call payoffs Y and the delta-hedge martingale control M along exact GBM paths.

    ``z`` has shape (n, m): m standard normal increments per path on the uniform grid
    t_k = k T / m. With the discounted dividend-reinvested price S~_t = e^{-(r-y)t} S_t,
        M = sum_k e^{-y t_k} Delta(t_k, S_{t_k}) (S~_{t_{k+1}} - S~_{t_k}),
    where Delta = e^{-y (T - t)} N(d_+) is the Black--Scholes delta of the call. E[M] = 0,
    and Y - M tends to the price in mean square as m grows.
    """
    from .bs import greeks
    z = np.atleast_2d(np.asarray(z, dtype=float))
    n, m = z.shape
    dt = T / m
    nu = r - y - 0.5 * sigma**2
    logS = np.log(S) + np.cumsum(nu * dt + sigma * np.sqrt(dt) * z, axis=1)
    St = np.concatenate([np.full((n, 1), float(S)), np.exp(logS)], axis=1)
    t = np.arange(m + 1) * dt
    Sdisc = np.exp(-(r - y) * t) * St
    delta = greeks(St[:, :-1], K, T - t[:-1], r, y, sigma, "call")["delta"]
    M = np.sum(np.exp(-y * t[:-1]) * delta * np.diff(Sdisc, axis=1), axis=1)
    Y = np.exp(-r * T) * np.maximum(St[:, -1] - K, 0.0)
    return Y, M
