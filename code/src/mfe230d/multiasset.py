"""FX and multi-asset options (Chapter 17).

Conventions follow the notes. Under the (domestic) risk-neutral measure Q, d assets follow

    dS_i / S_i = (r - y_i) dt + sigma_i dW_i,     d<W_i, W_j>_t = corr_ij dt,

so that (log S_{1,T}, ..., log S_{d,T}) is Gaussian with means log F_i - Sigma_ii / 2 and
covariances Sigma_ij = corr_ij sigma_i sigma_j T, where F_i = S_i e^{(r - y_i) T} is the
forward. A foreign currency is an asset whose yield is the foreign interest rate: X is the
domestic price of one unit of foreign currency (USD per GBP for "cable").

Contents
  * currencies: Garman--Kohlhagen prices and hedges, currency put--call symmetry;
  * changes of numeraire: drifts under Q^N, Monte Carlo under the share measure;
  * exchange options: Margrabe's formula, the cross-rate volatility triangle;
  * correlated simulation (Cholesky or principal components, from ``rng``) and quantos;
  * basket options: geometric (Gentle) basket, moment matching (Levy), Monte Carlo with
    control variates (through ``variance``), and a reference price by integrating out the
    geometric direction exactly.

Black--Scholes closed forms come from ``bs.py``; every method of the chapter is implemented
here. Samplers take a ``seed`` (anything ``numpy.random.default_rng`` accepts) or the normal
draws ``Z`` themselves, so that every number in the notes can be reproduced.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import ndtr

from . import bs
from .rng import cholesky, pca_factor
from .variance import control_variate, plain, regression_control_variates

# The course's currency inputs (Lecture 13, data of 22 July 2022; one-year rates and ATM vols).
# X_GBP is the USD price of one pound, X_EUR the USD price of one euro.
FX = dict(r_usd=0.03, r_gbp=0.025, r_eur=0.01, X_gbp=1.20, X_eur=1.02, vol_gbp=0.11, vol_eur=0.10)

# Three-asset test case for baskets: the test case for each asset, volatilities 20/25/30%,
# and the correlation matrix of the three-asset example of Chapter 5.
CORR3 = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.6], [0.3, 0.6, 1.0]])
BASKET = dict(S=np.array([100.0, 100.0, 100.0]), w=np.full(3, 1.0 / 3.0), K=100.0, T=1.0, r=0.01,
              y=np.array([0.02, 0.02, 0.02]), sigma=np.array([0.20, 0.25, 0.30]), corr=CORR3)


# ------------------------------------------------------------------------- lognormal calls
def black(F, K, v, D=1.0, kind="call"):
    """D E[(L - K)^+] (or the put) for a lognormal L with mean F and log-variance v.

    Equals D (F N(d+) - K N(d-)), d+- = (log(F/K) +- v/2) / sqrt(v), for v > 0 and K > 0;
    D (F - K)^+ if v = 0; and D (F - K) (call) or 0 (put) if K <= 0. Scalars only.
    """
    F, K, v = float(F), float(K), float(v)
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if K <= 0:
        return D * (F - K) if kind == "call" else 0.0
    if v <= 0:
        return D * max(F - K, 0.0) if kind == "call" else D * max(K - F, 0.0)
    f = bs.call if kind == "call" else bs.put
    return D * float(f(F, K, 1.0, 0.0, 0.0, np.sqrt(v)))


# ------------------------------------------------------------------------------ currencies
def fx_forward(X, T, rd, rf):
    """Forward exchange rate F = X e^{(r_d - r_f) T} (domestic per foreign)."""
    return bs.forward(X, T, rd, rf)


def garman_kohlhagen(X, K, T, rd, rf, sigma, kind="call"):
    """Domestic price of a call (put) on one unit of foreign currency struck at K.

    This is the Black--Scholes formula with the foreign rate r_f as the yield. Divide by
    X for the price in foreign currency.
    """
    f = bs.call if kind == "call" else bs.put
    return f(X, K, T, rd, rf, sigma)


def gk_hedge(X, K, T, rd, rf, sigma, kind="call"):
    """Units of foreign and of domestic currency in the replicating portfolio at time 0.

    foreign = dV/dX = phi e^{-r_f T} N(phi d+), domestic = V - X dV/dX = -phi K e^{-r_d T} N(phi d-),
    so that V = X * foreign + domestic.
    """
    phi = 1.0 if kind == "call" else -1.0
    dp, dm = bs.d_pm(X, K, T, rd, rf, sigma)
    return dict(foreign=phi * np.exp(-rf * T) * ndtr(phi * dp),
                domestic=-phi * K * np.exp(-rd * T) * ndtr(phi * dm))


def currency_symmetry(X, K, T, rd, rf, sigma):
    """Both sides of currency put--call symmetry in the Garman--Kohlhagen model.

    Returns (domestic price of the call on one unit of foreign currency struck at K,
    X K times the foreign price of a put on one unit of domestic currency struck at 1/K).
    The foreign investor sees 1/X as a lognormal asset with rate r_f and yield r_d.
    """
    return (garman_kohlhagen(X, K, T, rd, rf, sigma, "call"),
            X * K * garman_kohlhagen(1.0 / X, 1.0 / K, T, rf, rd, sigma, "put"))


# ------------------------------------------------------------------- changes of numeraire
def numeraire_drift(mu, sigma_S, theta):
    """Drift of dS/S under Q^N when it is mu under Q: mu + sigma_S . theta.

    sigma_S and theta are the (vector) volatilities of S and of N/B with respect to the same
    d-dimensional Brownian motion.
    """
    return mu + float(np.dot(np.atleast_1d(sigma_S), np.atleast_1d(theta)))


def share_measure_terminal(z, S, T, r, y, sigma):
    """S_T under the share measure Q^S: log-drift r - y + sigma^2 / 2 instead of r - y - sigma^2 / 2."""
    return S * np.exp((r - y + 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.asarray(z, dtype=float))


def call_samples_share_measure(z, S, K, T, r, y, sigma):
    """Samples whose Q^S-mean is the call price: S e^{-yT} (1 - K / S_T)^+ with S_T drawn under Q^S."""
    ST = share_measure_terminal(z, S, T, r, y, sigma)
    return S * np.exp(-y * T) * np.maximum(1.0 - K / ST, 0.0)


# ---------------------------------------------------------------------- correlated assets
def log_covariance(sigma, corr, T):
    """Sigma_ij = corr_ij sigma_i sigma_j T, the covariance of the log terminal prices."""
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    return np.outer(sigma, sigma) * np.asarray(corr, dtype=float) * T


def factor(corr, method="cholesky"):
    """A with A A^T = corr: Cholesky (positive definite corr) or principal components (semidefinite)."""
    corr = np.asarray(corr, dtype=float)
    if method == "cholesky":
        return cholesky(corr)
    if method == "pca":
        return pca_factor(corr)[0]
    raise ValueError("method must be 'cholesky' or 'pca'")


def terminal_prices(S, T, r, y, sigma, corr, n=None, seed=None, method="cholesky", Z=None, drift=None):
    """Exact draws of (S_{1,T}, ..., S_{d,T}) under Q, one row per draw.

    log S_{i,T} = log S_i + (mu_i - sigma_i^2 / 2) T + sigma_i sqrt(T) (A Z)_i with A A^T = corr,
    where mu_i = r - y_i unless ``drift`` gives the vector mu directly. Pass ``Z`` of shape
    (n, d) to reuse normal draws.
    """
    S, y, sigma = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (S, y, sigma))
    d = S.size
    A = factor(corr, method)
    if Z is None:
        Z = np.random.default_rng(seed).standard_normal((n, A.shape[1]))
    mu = (r - y) if drift is None else np.atleast_1d(np.asarray(drift, dtype=float))
    X = np.asarray(Z, dtype=float) @ A.T
    return S * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * X.reshape(-1, d))


# ----------------------------------------------------------------------- exchange options
def exchange_vol(s1, s2, rho):
    """Volatility of S1/S2: sqrt(s1^2 - 2 rho s1 s2 + s2^2), in [|s1 - s2|, s1 + s2]."""
    return np.sqrt(np.maximum(s1**2 - 2.0 * rho * s1 * s2 + s2**2, 0.0))


def margrabe(S1, S2, T, y1, y2, s1, s2, rho):
    """Price of (S1_T - S2_T)^+ (Margrabe), with dividend yields y1, y2.

    For sigma = exchange_vol > 0 it is the Black--Scholes call with spot S1, strike S2, rate y2,
    yield y1 and volatility sigma; the interest rate does not enter. For sigma = 0 it is
    (S1 e^{-y1 T} - S2 e^{-y2 T})^+.
    """
    v = float(exchange_vol(s1, s2, rho))
    if v == 0.0:
        return max(S1 * np.exp(-y1 * T) - S2 * np.exp(-y2 * T), 0.0)
    return float(bs.call(S1, S2, T, y2, y1, v))


def margrabe_hedge(S1, S2, T, y1, y2, s1, s2, rho):
    """Units of asset 1 and of asset 2 in the replicating portfolio: e^{-y1 T} N(d+), -e^{-y2 T} N(d-)."""
    v = float(exchange_vol(s1, s2, rho))
    if v == 0.0:
        itm = float(S1 * np.exp(-y1 * T) > S2 * np.exp(-y2 * T))
        return dict(asset1=itm * np.exp(-y1 * T), asset2=-itm * np.exp(-y2 * T))
    dp, dm = bs.d_pm(S1, S2, T, y2, y1, v)
    return dict(asset1=float(np.exp(-y1 * T) * ndtr(dp)), asset2=float(-np.exp(-y2 * T) * ndtr(dm)))


def cross_vol_range(s1, s2):
    """The volatilities of a cross rate consistent with correlations in [-1, 1]."""
    return abs(s1 - s2), s1 + s2


def implied_correlation(s1, s2, s12):
    """Correlation of the two rates against the common currency implied by the cross volatility."""
    return (s1**2 + s2**2 - s12**2) / (2.0 * s1 * s2)


def cross_atmf_call(T=1.0, s12=None, rho=None, fx=FX):
    """USD value of the ATMF GBP/EUR call paying (1 GBP - F_cross EUR)^+, by Margrabe.

    Give either the cross volatility s12 or the correlation rho of GBP/USD and EUR/USD.
    Returns (value in USD, cross forward in EUR per GBP, cross volatility).
    """
    X1, X2 = fx["X_gbp"], fx["X_eur"]
    s1, s2 = fx["vol_gbp"], fx["vol_eur"]
    F12 = (X1 / X2) * np.exp((fx["r_eur"] - fx["r_gbp"]) * T)
    if s12 is None:
        s12 = float(exchange_vol(s1, s2, rho))
    else:
        rho = implied_correlation(s1, s2, s12)
    value = margrabe(X1, F12 * X2, T, fx["r_gbp"], fx["r_eur"], s1, s2, rho)
    return value, F12, s12


# --------------------------------------------------------------------------------- quantos
def quanto_yield(y, rd, rf, sS, sX, rho):
    """Yield y_q = y + r_d - r_f + rho sS sX that turns a Black--Scholes routine into a quanto pricer."""
    return y + rd - rf + rho * sS * sX


def quanto_forward(S, T, rf, y, sS, sX, rho):
    """E^{Q_d}[S^f_T] = S e^{(r_f - y - rho sS sX) T}: the quanto forward."""
    return S * np.exp((rf - y - rho * sS * sX) * T)


def quanto_option(S, K, T, rd, rf, y, sS, sX, rho, Xbar=1.0, kind="call"):
    """Domestic price of Xbar (S^f_T - K)^+ (or the put): foreign stock, fixed exchange rate Xbar."""
    f = bs.call if kind == "call" else bs.put
    return Xbar * f(S, K, T, rd, quanto_yield(y, rd, rf, sS, sX, rho), sS)


def composite_option(S, X, K, T, rd, y, sS, sX, rho, kind="call"):
    """Domestic price of (X_T S^f_T - K)^+: foreign stock struck in domestic currency."""
    f = bs.call if kind == "call" else bs.put
    return f(X * S, K, T, rd, y, np.sqrt(sS**2 + 2.0 * rho * sS * sX + sX**2))


def foreign_option(S, X, K, T, rf, y, sS, kind="call"):
    """Domestic price of X_T (S^f_T - K)^+: a foreign option, converted at the spot rate."""
    f = bs.call if kind == "call" else bs.put
    return X * f(S, K, T, rf, y, sS)


def fx_equity_terminal(S, X, T, rd, rf, y, sS, sX, rho, n=None, seed=None, Z=None):
    """Exact draws of (S^f_T, X_T) under the domestic measure Q_d.

    S^f has log-drift r_f - y - rho sS sX - sS^2/2 and X has r_d - r_f - sX^2/2; the two
    drivers have correlation rho (principal-components factor, so rho = +-1 is allowed).
    """
    out = terminal_prices([S, X], T, 0.0, [0.0, 0.0], [sS, sX], [[1.0, rho], [rho, 1.0]], n=n, seed=seed,
                          method="pca", Z=Z, drift=[rf - y - rho * sS * sX, rd - rf])
    return out[:, 0], out[:, 1]


# --------------------------------------------------------------------------------- baskets
def _basket_inputs(S, w, T, r, y, sigma, corr):
    S, w, y, sigma = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (S, w, y, sigma))
    F = S * np.exp((r - y) * T)
    return S, w, y, sigma, F, log_covariance(sigma, corr, T)


def basket_moments(S, w, T, r, y, sigma, corr):
    """First two moments of B_T = sum_i w_i S_{i,T}: (sum w_i F_i, sum_ij w_i w_j F_i F_j e^{Sigma_ij})."""
    _, w, _, _, F, Sig = _basket_inputs(S, w, T, r, y, sigma, corr)
    wF = w * F
    return float(wF.sum()), float(wF @ np.exp(Sig) @ wF)


def basket_lognormal(S, w, K, T, r, y, sigma, corr, kind="call"):
    """Two-moment lognormal approximation (Levy): Black with mean E B and log-variance log(E B^2 / (E B)^2).

    Raises ValueError when E B <= 0, where no lognormal law has the mean of the basket.
    """
    m1, m2 = basket_moments(S, w, T, r, y, sigma, corr)
    if m1 <= 0:
        raise ValueError("the basket has a nonpositive mean: no lognormal law matches it")
    return black(m1, K, np.log(m2 / m1**2), np.exp(-r * T), kind)


def geometric_basket_law(S, w, T, r, y, sigma, corr):
    """The geometric (Gentle) basket G = Phi prod_i (S_{i,T} / F_i)^{a_i}, a_i = w_i F_i / Phi.

    Requires w_i > 0. Returns Phi = sum w_i F_i, the weights a, E log G, its variance v_G,
    E G and the covariances c_i = Cov(log S_{i,T}, log G) = (Sigma a)_i.
    """
    _, w, _, _, F, Sig = _basket_inputs(S, w, T, r, y, sigma, corr)
    if np.any(w <= 0):
        raise ValueError("the geometric basket needs positive weights")
    Phi = float(w @ F)
    a = w * F / Phi
    mean = np.log(Phi) - 0.5 * float(a @ np.diag(Sig))
    vG = float(a @ Sig @ a)
    return dict(Phi=Phi, a=a, mean=mean, var=vG, EG=float(np.exp(mean + 0.5 * vG)), c=Sig @ a, F=F, Sigma=Sig)


def geometric_basket_option(S, w, K, T, r, y, sigma, corr, kind="call"):
    """Exact price of (G - K)^+ (or the put) for the geometric basket: a lower bound for the basket call."""
    g = geometric_basket_law(S, w, T, r, y, sigma, corr)
    return black(g["EG"], K, g["var"], np.exp(-r * T), kind)


def gentle_call(S, w, K, T, r, y, sigma, corr):
    """Gentle's approximation: the geometric call with the strike shifted by E B - E G."""
    g = geometric_basket_law(S, w, T, r, y, sigma, corr)
    return black(g["EG"], K - (g["Phi"] - g["EG"]), g["var"], np.exp(-r * T))


def curran_lower_bound(S, w, K, T, r, y, sigma, corr):
    """Lower bound D E[(E[B | G] - K)^+] for the basket call (conditioning on the geometric basket).

    Given log G = x, E[B | G] = m(x) = sum_i w_i F_i exp(c_i (x - mu_G) / v_G - c_i^2 / (2 v_G)); when
    every c_i >= 0, m is nondecreasing and the bound is L(x*) with m(x*) = K, where
    L(x) = D (sum_i w_i F_i N((mu_G + c_i - x) / sqrt(v_G)) - K N((mu_G - x) / sqrt(v_G))).
    Requires positive weights, K > 0, v_G > 0 and c_i >= 0 (raises ValueError otherwise).
    """
    g = geometric_basket_law(S, w, T, r, y, sigma, corr)
    c, v, mu = g["c"], g["var"], g["mean"]
    if K <= 0 or v <= 0 or np.any(c < 0):
        raise ValueError("needs K > 0, v_G > 0 and nonnegative covariances with log G")
    wF = np.atleast_1d(np.asarray(w, dtype=float)) * g["F"]
    m = lambda x: float(wF @ np.exp(c * (x - mu) / v - c**2 / (2.0 * v))) - K
    lo, hi = mu - 40.0 * np.sqrt(v), mu + 40.0 * np.sqrt(v)
    if m(lo) >= 0:
        xs = -np.inf
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if m(mid) > 0 else (mid, hi)
        xs = 0.5 * (lo + hi)
    sd = np.sqrt(v)
    return float(np.exp(-r * T) * (wF @ ndtr((mu + c - xs) / sd) - K * ndtr((mu - xs) / sd)))


def basket_cv_bound(S, w, T, r, y, sigma, corr):
    """D^2 E[(B - G)^2]: a bound on Var(Y - X^g) for the geometric control, independent of K."""
    g = geometric_basket_law(S, w, T, r, y, sigma, corr)
    _, m2 = basket_moments(S, w, T, r, y, sigma, corr)
    wF = np.atleast_1d(np.asarray(w, dtype=float)) * g["F"]
    EBG = g["EG"] * float(wF @ np.exp(g["c"]))
    EG2 = g["EG"] ** 2 * np.exp(g["var"])
    return float(np.exp(-2 * r * T) * (m2 - 2.0 * EBG + EG2))


def basket_samples(S, w, K, T, r, y, sigma, corr, n=None, seed=None, Z=None, method="cholesky"):
    """Discounted samples of the basket call Y = D (B - K)^+ and of three controls.

    geometric: D (G - K)^+; forward: D B (mean D sum w_i F_i); strip: D sum_i w_i (S_i - K)^+,
    an upper bound for Y when the weights are nonnegative and sum to one.
    """
    S_, w_, y_, sigma_, F, _ = _basket_inputs(S, w, T, r, y, sigma, corr)
    ST = terminal_prices(S_, T, r, y_, sigma_, corr, n=n, seed=seed, method=method, Z=Z)
    D = np.exp(-r * T)
    B = ST @ w_
    out = dict(Y=D * np.maximum(B - K, 0.0), forward=D * B,
               strip=D * np.maximum(ST - K, 0.0) @ w_, B=B, ST=ST)
    if np.all(w_ > 0):
        a = w_ * F / (w_ @ F)
        G = (w_ @ F) * np.exp(np.log(ST / F) @ a)
        out.update(G=G, geometric=D * np.maximum(G - K, 0.0))
    return out


def basket_control_means(S, w, K, T, r, y, sigma, corr):
    """Exact means of the controls of ``basket_samples``."""
    S_, w_, y_, sigma_, F, _ = _basket_inputs(S, w, T, r, y, sigma, corr)
    D = np.exp(-r * T)
    out = dict(forward=D * float(w_ @ F),
               strip=float(w_ @ bs.call(S_, K, T, r, y_, sigma_)))
    if np.all(w_ > 0):
        out["geometric"] = geometric_basket_option(S, w, K, T, r, y, sigma, corr)
    return out


def basket_mc(S, w, K, T, r, y, sigma, corr, n=2**16, seed=None, Z=None, controls=()):
    """Monte Carlo price of the basket call: plain, one control (in-sample coefficient), or several.

    ``controls`` is a tuple of names from ("geometric", "forward", "strip"). One control uses
    ``variance.control_variate``; several use the regression estimator
    ``variance.regression_control_variates``. Returns mean, se and the fitted coefficients.
    """
    s = basket_samples(S, w, K, T, r, y, sigma, corr, n=n, seed=seed, Z=Z)
    if not controls:
        return plain(s["Y"])
    mu = basket_control_means(S, w, K, T, r, y, sigma, corr)
    if len(controls) == 1:
        c = controls[0]
        return control_variate(s["Y"], s[c], mu[c])
    X = np.column_stack([s[c] for c in controls])
    return regression_control_variates(s["Y"], X, [mu[c] for c in controls])


def _excess_given_rest(alpha, b, K, L=12.0, iters=90):
    """E[(sum_i alpha_i e^{b_i xi} - K)^+] for xi ~ N(0,1), one value per row of alpha (alpha > 0).

    The sum is convex in xi, so it exceeds K outside an interval [xi1, xi2] (possibly empty),
    found by bisection on [-L, L]; then E[e^{b xi} 1(xi < x)] = e^{b^2/2} N(x - b) gives the value.
    Mass beyond |xi| > L (about 1e-33 for L = 12) is ignored.
    """
    alpha = np.atleast_2d(alpha)
    n = alpha.shape[0]
    Bf = lambda x: np.sum(alpha * np.exp(np.outer(x, b)), axis=1)
    dB = lambda x: np.sum(alpha * b * np.exp(np.outer(x, b)), axis=1)
    lo, hi = np.full(n, -L), np.full(n, L)
    # minimiser of the convex function on [-L, L]
    left, right = dB(lo) >= 0, dB(hi) <= 0
    a_, c_ = lo.copy(), hi.copy()
    for _ in range(iters):
        m = 0.5 * (a_ + c_)
        pos = dB(m) > 0
        c_ = np.where(pos, m, c_)
        a_ = np.where(pos, a_, m)
    xm = np.where(left, lo, np.where(right, hi, 0.5 * (a_ + c_)))
    below = Bf(xm) < K
    # left root on [-L, xm] (B decreasing there), right root on [xm, L] (B increasing)
    x1 = np.full(n, -np.inf)
    x2 = np.full(n, np.inf)
    has1 = below & (Bf(lo) > K)
    has2 = below & (Bf(hi) > K)
    a_, c_ = lo.copy(), xm.copy()
    for _ in range(iters):
        m = 0.5 * (a_ + c_)
        up = Bf(m) > K
        a_ = np.where(up, m, a_)
        c_ = np.where(up, c_, m)
    x1 = np.where(has1, 0.5 * (a_ + c_), x1)
    a_, c_ = xm.copy(), hi.copy()
    for _ in range(iters):
        m = 0.5 * (a_ + c_)
        up = Bf(m) > K
        c_ = np.where(up, m, c_)
        a_ = np.where(up, a_, m)
    x2 = np.where(has2, 0.5 * (a_ + c_), x2)
    # no interval where B <= K: the whole line
    x1 = np.where(below, x1, -np.inf)
    x2 = np.where(below, x2, -np.inf)
    eb = np.exp(0.5 * b**2)
    lin = alpha * eb * (ndtr(x1[:, None] - b) + ndtr(b - x2[:, None]))
    return lin.sum(axis=1) - K * (ndtr(x1) + ndtr(-x2))


def basket_reference(S, w, K, T, r, y, sigma, corr, nodes=64):
    """Basket call by integrating out the direction of log G exactly and the rest by Gauss--Hermite.

    Write the Gaussian vector of log prices as m + L Z with L L^T = Sigma, and Z = e xi + Q Z'
    with e the unit vector along L^T a (the direction of log G) and Q an orthonormal basis of its
    complement. Given Z', the basket is sum_i alpha_i e^{b_i xi}, b = Sigma a / sqrt(v_G), and its
    call is ``_excess_given_rest``; the d-1 remaining dimensions are integrated with a product
    Gauss--Hermite rule with ``nodes`` points per dimension. Positive weights only; d <= 4.
    """
    g = geometric_basket_law(S, w, T, r, y, sigma, corr)
    Sig, a, F = g["Sigma"], g["a"], g["F"]
    w_ = np.atleast_1d(np.asarray(w, dtype=float))
    d = F.size
    if d > 4:
        raise ValueError("product quadrature is meant for d <= 4")
    lam, V = np.linalg.eigh(Sig)
    L = V * np.sqrt(np.clip(lam, 0.0, None))          # L L^T = Sigma, also when singular
    m = np.log(F) - 0.5 * np.diag(Sig)
    D = np.exp(-r * T)
    if g["var"] <= 0:
        return black(g["EG"], K, 0.0, D)
    u = L.T @ a
    e = u / np.linalg.norm(u)
    b = L @ e
    if d == 1:
        return black(F[0] * w_[0], K, Sig[0, 0], D)
    Q = np.linalg.svd(e[None, :])[2][1:].T                # d x (d-1), orthonormal, Q^T e = 0
    M = L @ Q
    x, wq = hermegauss(nodes)
    wq = wq / np.sqrt(2.0 * np.pi)
    grids = np.meshgrid(*([x] * (d - 1)), indexing="ij")
    Zp = np.column_stack([gr.ravel() for gr in grids])
    W = np.prod(np.meshgrid(*([wq] * (d - 1)), indexing="ij"), axis=0).ravel()
    alpha = w_ * np.exp(m + Zp @ M.T)
    return float(D * W @ _excess_given_rest(alpha, b, K))
