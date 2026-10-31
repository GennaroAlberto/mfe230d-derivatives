"""Hedging in practice (Chapter 18).

Discrete delta hedging and its error, hedging at the wrong volatility (implied against
realised), and Leland's volatility adjustment for proportional transaction costs.

Conventions follow the notes. Under the real-world measure the stock has total expected
return ``mu`` (Chapter 1): dS/S = (mu - y) dt + sigma dW; the risk-neutral measure is the
case mu = r. The hedger *sells* the claim, receives a premium, and rebalances at the
dates t_j = j h, h = T / m, to the Black--Scholes delta computed with a hedging
volatility. Dividends are reinvested in the stock at once, so delta_j shares bought at
t_j have become delta_j e^{y h} shares at t_{j+1} (the convention of Chapter 8 and of the
course's Homework 5). With the discounted dividend-reinvested price
S~_t = e^{-(r-y) t} S_t, the discounted terminal wealth is then exactly

    premium + sum_j e^{-y t_j} delta_j (S~_{t_{j+1}} - S~_{t_j}) - discounted costs.

Transaction costs: ``k`` is the ROUND-TRIP proportional cost of Leland (1985), so a trade
of q shares at price S costs (k / 2) |q| S. The course's one-way cost k' is k / 2. As in
Homework 5, only the rebalancing trades at t_1, ..., t_{m-1} are charged by default (no
cost for setting up the hedge at t_0 or for settling at T).

Every quantity that the notes quote is computed here: the closed-form gamma moments,
the leading constant of the error variance, the exact variance of the discrete hedge
under Q (by one-dimensional quadrature), the expected P&L of hedging at the implied
volatility, and the Monte Carlo laboratory itself. ``bs.py`` supplies the Black--Scholes
closed forms; paths come from ``paths.gbm_paths``.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import ndtr

from . import bs
from .paths import gbm_paths, time_grid
from .variance import call_payoff_moments

KINDS = ("call", "put", "digital")


# --------------------------------------------------------------------- claims and Greeks
def payoff(kind, s, K):
    """Payoff of a call (s - K)^+, a put (K - s)^+ or a cash-or-nothing call 1{s > K}."""
    s = np.asarray(s, dtype=float)
    if kind == "call":
        return np.maximum(s - K, 0.0)
    if kind == "put":
        return np.maximum(K - s, 0.0)
    if kind == "digital":
        return (s > K).astype(float)
    raise ValueError(f"kind must be one of {KINDS}")


def price_delta_gamma(kind, s, K, tau, r, y, sigma):
    """Black--Scholes price, delta and gamma at time to maturity ``tau`` > 0 (vectorised).

    Calls and puts use ``bs.greeks``; the cash-or-nothing call pays 1 if S_T > K and has
    price e^{-r tau} N(d_-), delta e^{-r tau} N'(d_-) / (s sigma sqrt tau) and gamma
    -e^{-r tau} N'(d_-) d_+ / (s^2 sigma^2 tau).
    """
    s = np.asarray(s, dtype=float)
    if kind in ("call", "put"):
        g = bs.greeks(s, K, tau, r, y, sigma, kind)
        f = bs.call if kind == "call" else bs.put
        return f(s, K, tau, r, y, sigma), g["delta"], g["gamma"]
    if kind == "digital":
        dp, dm = bs.d_pm(s, K, tau, r, y, sigma)
        w = sigma * np.sqrt(tau)
        disc = np.exp(-r * np.asarray(tau, dtype=float))
        pdf = bs.norm_pdf(dm)
        return disc * ndtr(dm), disc * pdf / (s * w), -disc * pdf * dp / (s * s * w * w)
    raise ValueError(f"kind must be one of {KINDS}")


def delta_only(kind, s, K, tau, r, y, sigma):
    """The Black--Scholes delta alone (one normal CDF per point): the hot loop of the hedges."""
    s = np.asarray(s, dtype=float)
    tau = np.asarray(tau, dtype=float)
    w = sigma * np.sqrt(tau)
    dp = (np.log(s / K) + (r - y) * tau) / w + 0.5 * w
    if kind == "call":
        return np.exp(-y * tau) * ndtr(dp)
    if kind == "put":
        return -np.exp(-y * tau) * ndtr(-dp)
    return price_delta_gamma(kind, s, K, tau, r, y, sigma)[1]


def leland_vol(sigma, k, h):
    """Leland's hedging volatility sigma sqrt(1 + sqrt(2/pi) k / (sigma sqrt h)), k round trip."""
    return sigma * np.sqrt(1.0 + np.sqrt(2.0 / np.pi) * k / (sigma * np.sqrt(h)))


# ------------------------------------------------------------------- the hedging laboratory
def simulate_stock(S, T, m, n, mu, y, sigma, seed=None, Z=None):
    """Exact GBM paths with total return ``mu`` on the uniform grid t_j = j T / m.

    Returns (t, paths) with paths of shape (n, m + 1). ``gbm_paths`` with r = mu gives
    dS/S = (mu - y) dt + sigma dW, the real-world dynamics of the notes.
    """
    t = time_grid(T, m)
    return t, gbm_paths(S, t, r=mu, y=y, sigma=sigma, n_paths=n, seed=seed, Z=Z)


def hedge_paths(paths, t, K, r, y, sigma_hedge, kind="call", premium=None, k=0.0,
                charge_initial=False, charge_final=False):
    """Seller's discrete delta hedge along given stock paths (shape (n, m + 1), grid ``t``).

    The seller receives ``premium`` (default: the Black--Scholes price at ``sigma_hedge``),
    holds delta_j = Delta(t_j, S_{t_j}; sigma_hedge) shares from t_j, reinvests dividends in
    the stock, finances at r, and pays (k / 2) |q| S for every trade of q shares at the
    rebalancing dates t_1..t_{m-1} (and at t_0 / T if ``charge_initial`` / ``charge_final``;
    the final trade delivers the payoff's delta, 1{S_T > K} for a call, -1{S_T < K} for a put).

    Returns a dict of arrays over paths: ``error`` = X_T - payoff (time-T money, the
    seller's hedging error), ``disc_error`` = e^{-rT} error, ``costs`` (time-T value of the
    costs paid), ``gains`` (discounted trading gains), ``ST`` and ``premium``.
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    t = np.asarray(t, dtype=float)
    T = t[-1]
    h = np.diff(t)
    S0 = paths[0, 0]
    if premium is None:
        premium = float(price_delta_gamma(kind, S0, K, T, r, y, sigma_hedge)[0])
    St = paths[:, :-1]
    tau = T - t[:-1]
    delta = delta_only(kind, St, K, tau, r, y, sigma_hedge)
    Stil = np.exp(-(r - y) * t) * paths
    gains = np.sum(np.exp(-y * t[:-1]) * delta * np.diff(Stil, axis=1), axis=1)
    disc_costs = np.zeros(paths.shape[0])
    if k > 0.0:
        held = delta[:, :-1] * np.exp(y * h[:-1])          # shares just before t_1..t_{m-1}
        trades = delta[:, 1:] - held
        disc_costs += 0.5 * k * np.sum(np.exp(-r * t[1:-1]) * np.abs(trades) * paths[:, 1:-1], axis=1)
        if charge_initial:
            disc_costs += 0.5 * k * np.abs(delta[:, 0]) * paths[:, 0]
        if charge_final:
            target = {"call": (paths[:, -1] > K).astype(float),
                      "put": -(paths[:, -1] < K).astype(float),
                      "digital": np.zeros(paths.shape[0])}[kind]
            last = delta[:, -1] * np.exp(y * h[-1])
            disc_costs += 0.5 * k * np.exp(-r * T) * np.abs(target - last) * paths[:, -1]
    disc_error = premium + gains - disc_costs - np.exp(-r * T) * payoff(kind, paths[:, -1], K)
    return dict(error=np.exp(r * T) * disc_error, disc_error=disc_error,
                costs=np.exp(r * T) * disc_costs, gains=gains, ST=paths[:, -1], premium=premium)


def hedge_experiment(S, K, T, r, y, sigma, m, n, mu=None, sigma_true=None, sigma_hedge=None,
                     kind="call", k=0.0, premium=None, seed=None, Z=None, chunk=10_000, **kw):
    """Simulate the stock (drift ``mu``, default r; volatility ``sigma_true``, default sigma)
    and run ``hedge_paths`` with ``sigma_hedge`` (default sigma). ``m = 0`` means no hedge:
    the premium is kept in the bank. Returns the dict of ``hedge_paths``.

    To keep the memory bounded, paths are simulated and hedged in blocks of ``chunk`` rows:
    block i uses the next ``chunk`` x max(m, 1) normals of ``default_rng(seed)`` (or the rows
    of ``Z`` if given), so the result for a given seed does not depend on the machine."""
    mu = r if mu is None else mu
    sigma_true = sigma if sigma_true is None else sigma_true
    sigma_hedge = sigma if sigma_hedge is None else sigma_hedge
    if premium is None:
        premium = float(price_delta_gamma(kind, S, K, T, r, y, sigma_hedge)[0])
    rng = np.random.default_rng(seed) if Z is None else None
    steps = max(int(m), 1)
    t = time_grid(T, steps)
    parts = []
    for start in range(0, n, chunk):
        nc = min(chunk, n - start)
        Zc = rng.standard_normal((nc, steps)) if Z is None else np.asarray(Z, dtype=float)[start:start + nc]
        P = gbm_paths(S, t, r=mu, y=y, sigma=sigma_true, Z=Zc)
        if m == 0:
            err = np.exp(r * T) * premium - payoff(kind, P[:, -1], K)
            parts.append(dict(error=err, disc_error=np.exp(-r * T) * err, costs=np.zeros(nc),
                              gains=np.zeros(nc), ST=P[:, -1]))
        else:
            out = hedge_paths(P, t, K, r, y, sigma_hedge, kind=kind, premium=premium, k=k, **kw)
            out.pop("premium")
            parts.append(out)
    res = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    res["premium"] = premium
    return res


def hedge_stream(S, K, T, r, y, sigma, m, n, mu=None, sigma_true=None, sigma_hedge=None,
                 kind="call", k=0.0, premium=None, seed=None):
    """The experiment of ``hedge_experiment`` stepped through time, with memory O(n) instead
    of O(n m): for many rebalancing dates (m in the thousands). Costs are charged at
    t_1..t_{m-1} only. The normals are drawn date by date, so the paths differ from those of
    ``hedge_experiment`` with the same seed. Returns dict(error, costs, ST) in time-T money."""
    mu = r if mu is None else mu
    sigma_true = sigma if sigma_true is None else sigma_true
    sigma_hedge = sigma if sigma_hedge is None else sigma_hedge
    rng = np.random.default_rng(seed)
    h = T / m
    if premium is None:
        premium = float(price_delta_gamma(kind, S, K, T, r, y, sigma_hedge)[0])
    s = np.full(n, float(S))
    stil = s.copy()
    d = np.full(n, float(price_delta_gamma(kind, S, K, T, r, y, sigma_hedge)[1]))
    gains = np.zeros(n)
    costs = np.zeros(n)
    drift, vol = (mu - y - 0.5 * sigma_true**2) * h, sigma_true * np.sqrt(h)
    for j in range(m):
        tj = j * h
        s = s * np.exp(drift + vol * rng.standard_normal(n))
        stil_new = np.exp(-(r - y) * (tj + h)) * s
        gains += np.exp(-y * tj) * d * (stil_new - stil)
        stil = stil_new
        if j < m - 1:
            d_new = delta_only(kind, s, K, T - tj - h, r, y, sigma_hedge)
            if k > 0.0:
                costs += 0.5 * k * np.exp(-r * (tj + h)) * np.abs(d_new - d * np.exp(y * h)) * s
            d = d_new
    err = np.exp(r * T) * (premium + gains - costs) - payoff(kind, s, K)
    return dict(error=err, costs=np.exp(r * T) * costs, ST=s, premium=premium)


def summarize(err):
    """Mean, its standard error, standard deviation, mean absolute error and RMS of samples."""
    err = np.asarray(err, dtype=float)
    sd = err.std(ddof=1)
    return dict(mean=err.mean(), se=sd / np.sqrt(err.size), sd=sd, mae=np.abs(err).mean(),
                rms=np.sqrt(np.mean(err * err)))


# ------------------------------------------------- Proposition hd-discrete-error: the constant
def gamma_moment(t, S, K, T, r, y, sigma, mu=None, kind="call"):
    """psi(t) = E[e^{-2rt} S_t^4 Gamma(t, S_t)^2] in closed form, for 0 <= t < T (vectorised).

    S_t is lognormal with total return ``mu`` (default r, the risk-neutral case) and
    volatility sigma; Gamma is the Black--Scholes gamma of the call (= that of the put).
    With ln S_t ~ N(m, v) and S^4 Gamma^2 = e^{2X} e^{-2 y tau} N'(d_+)^2 / (sigma^2 tau),
    the Gaussian integral gives the formula below; psi(t) ~ c (T - t)^{-1/2} as t -> T.
    """
    if kind not in ("call", "put"):
        raise ValueError("closed form only for calls and puts")
    mu = r if mu is None else mu
    t = np.asarray(t, dtype=float)
    tau = T - t
    mean = np.log(S) + (mu - y - 0.5 * sigma**2) * t
    v = sigma**2 * t
    c = np.log(K) - (r - y + 0.5 * sigma**2) * tau
    a = 0.5 * sigma**2 * tau                   # half the variance w^2 = sigma^2 tau
    s2 = v * a / (v + a)
    xb = (c * v + mean * a) / (v + a)
    w = sigma * np.sqrt(tau)
    return (np.exp(-2 * r * t - 2 * y * tau) / (2.0 * np.sqrt(np.pi) * w)
            * np.exp(-(c - mean) ** 2 / (2 * (a + v))) / np.sqrt(2 * np.pi * (a + v))
            * np.exp(2 * xb + 2 * s2))


def gamma_integral(S, K, T, r, y, sigma, mu=None, kind="call"):
    """I = int_0^T psi(t) dt, by quadrature after the substitution T - t = u^2."""
    f = lambda u: 2.0 * u * gamma_moment(T - u * u, S, K, T, r, y, sigma, mu, kind)
    return integrate.quad(f, 0.0, np.sqrt(T), epsabs=1e-12, epsrel=1e-12, limit=400)[0]


def asymptotic_sd(S, K, T, r, y, sigma, m, mu=None, kind="call", time_T=True):
    """Leading-order standard deviation sqrt(sigma^4 h I / 2) of the hedging error, h = T / m.

    Discounted if ``time_T`` is False; in time-T money (times e^{rT}) otherwise.
    """
    I = gamma_integral(S, K, T, r, y, sigma, mu, kind)
    sd = np.sqrt(0.5 * sigma**4 * (T / np.asarray(m, dtype=float)) * I)
    return np.exp(r * T) * sd if time_T else sd


# ---------------------------------------------- exact variance of the discrete hedge under Q
def exact_error_variance(S, K, T, r, y, sigma, m, n_nodes=10001, width=9.0):
    """E_Q[e_m^2] for the call: the variance of the discounted error of the m-date hedge.

    e_m = V_0 + M_m - Y with Y = e^{-rT}(S_T - K)^+ and M_m the discounted hedge gains. Since
    E_Q M_m = 0 and E_Q Y = V_0,
        E e_m^2 = Var Y + E M_m^2 - 2 E[Y M_m],
        E M_m^2 = sum_j e^{-2 y t_j} E[Delta_j^2 S~_j^2] (e^{sigma^2 h} - 1),
        E[Y M_m] = sum_j e^{-(r + y) t_j} E[Delta_j S~_j (V(t_j, S_j e^{sigma^2 h}) - V(t_j, S_j))],
    the last by a change of measure on one step. Each expectation over S_{t_j} is a smooth
    one-dimensional Gaussian integral, computed with the trapezoid rule on ``n_nodes``
    points of [-width, width]. Returns (E e_m^2, Var Y).
    """
    h = T / m
    nu = r - y - 0.5 * sigma**2
    _, var = call_payoff_moments(S, K, T, r, y, sigma)
    var_y = np.exp(-2 * r * T) * var
    z = np.linspace(-width, width, n_nodes)
    wts = bs.norm_pdf(z) * (z[1] - z[0])
    wts[0] *= 0.5
    wts[-1] *= 0.5
    em2 = emy = 0.0
    for j in range(m):
        tj = j * h
        if j == 0:
            Sj, w = np.array([float(S)]), np.array([1.0])
        else:
            Sj, w = S * np.exp(nu * tj + sigma * np.sqrt(tj) * z), wts
        tau = T - tj
        V, D, _ = price_delta_gamma("call", Sj, K, tau, r, y, sigma)
        Vup = bs.call(Sj * np.exp(sigma**2 * h), K, tau, r, y, sigma)
        Stil = np.exp(-(r - y) * tj) * Sj
        em2 += np.exp(-2 * y * tj) * np.sum(w * D * D * Stil * Stil) * np.expm1(sigma**2 * h)
        emy += np.exp(-(r + y) * tj) * np.sum(w * D * Stil * (Vup - V))
    return var_y + em2 - 2 * emy, var_y


# ---------------------------------------------- Proposition hd-wrong-vol: implied vs realised
def gamma_s2_moment(t, S, K, T, r, y, sigma_h, sigma_r, mu):
    """E[S_t^2 Gamma_h(t, S_t)] for a call or put, Gamma_h the Black--Scholes gamma at the
    hedging volatility sigma_h, S lognormal with total return mu and volatility sigma_r.

    With ln S_t ~ N(m, v), c = ln K - (r - y + sigma_h^2/2) tau and w^2 = sigma_h^2 tau:
        E[S_t^2 Gamma_h] = e^{-y tau} e^{x + s^2/2} exp(-(c - m)^2 / (2 (w^2 + v))) / sqrt(2 pi (w^2 + v)),
    x = (c v + m w^2) / (v + w^2), s^2 = v w^2 / (v + w^2). Valid on 0 <= t <= T.
    """
    t = np.asarray(t, dtype=float)
    tau = T - t
    mean = np.log(S) + (mu - y - 0.5 * sigma_r**2) * t
    v = sigma_r**2 * t
    w2 = sigma_h**2 * tau
    c = np.log(K) - (r - y + 0.5 * sigma_h**2) * tau
    tot = v + w2
    x = (c * v + mean * w2) / tot
    s2 = v * w2 / tot
    return np.exp(-y * tau + x + 0.5 * s2 - (c - mean) ** 2 / (2 * tot)) / np.sqrt(2 * np.pi * tot)


def expected_vol_pnl(S, K, T, r, y, sigma_h, sigma_r, mu):
    """E[1/2 int_0^T e^{-rt} Gamma_h S_t^2 (sigma_r^2 - sigma_h^2) dt]: the expected discounted
    P&L of a LONG call (or put) bought at the sigma_h price and hedged continuously at sigma_h,
    when the stock has realised volatility sigma_r and total return mu."""
    f = lambda t: np.exp(-r * t) * gamma_s2_moment(t, S, K, T, r, y, sigma_h, sigma_r, mu)
    return 0.5 * (sigma_r**2 - sigma_h**2) * integrate.quad(f, 0.0, T, epsabs=1e-12, epsrel=1e-12, limit=400)[0]


def vol_pnl_paths(paths, t, K, r, y, sigma_h, sigma_r, kind="call"):
    """Riemann sums of 1/2 int e^{-rt} Gamma_h S^2 (sigma_r^2 - sigma_h^2) dt along paths
    (left endpoints, the last interval excluded): the long position's continuous P&L."""
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    t = np.asarray(t, dtype=float)
    T = t[-1]
    G = price_delta_gamma(kind, paths[:, :-1], K, T - t[:-1], r, y, sigma_h)[2]
    integrand = np.exp(-r * t[:-1]) * G * paths[:, :-1] ** 2
    return 0.5 * (sigma_r**2 - sigma_h**2) * np.sum(integrand * np.diff(t), axis=1)


def mark_to_market(paths, t, K, r, y, sigma_h, sigma_mark, premium, kind="call"):
    """Discounted mark-to-market P&L of a LONG claim bought at ``premium``, short
    Delta(sigma_h) shares rebalanced at the grid dates, marked at ``sigma_mark``:
        e^{-r t_j} V(t_j, S_j; sigma_mark) - premium - sum_{i<j} e^{-y t_i} delta_i (S~_{i+1} - S~_i).
    Returns an array of shape (n, m + 1); the last column uses the payoff."""
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    t = np.asarray(t, dtype=float)
    T = t[-1]
    delta = price_delta_gamma(kind, paths[:, :-1], K, T - t[:-1], r, y, sigma_h)[1]
    Stil = np.exp(-(r - y) * t) * paths
    gains = np.concatenate([np.zeros((paths.shape[0], 1)),
                            np.cumsum(np.exp(-y * t[:-1]) * delta * np.diff(Stil, axis=1), axis=1)], axis=1)
    value = np.empty_like(paths)
    value[:, :-1] = price_delta_gamma(kind, paths[:, :-1], K, T - t[:-1], r, y, sigma_mark)[0]
    value[:, -1] = payoff(kind, paths[:, -1], K)
    return np.exp(-r * t) * value - premium - gains


# --------------------------------------------------------------- Theorem hd-leland: costs
def expected_cost_rate(S, K, T, r, y, sigma, mu=None):
    """C = (sigma / 2) sqrt(2/pi) int_0^T e^{-rt} E[S_t^2 Gamma(t, S_t)] dt for the call.

    To leading order the expected discounted transaction costs of the m-date hedge with
    round-trip cost k are k C / sqrt(h) (one-period lemma of the notes, summed)."""
    mu = r if mu is None else mu
    f = lambda t: np.exp(-r * t) * gamma_s2_moment(t, S, K, T, r, y, sigma, sigma, mu)
    return 0.5 * sigma * np.sqrt(2 / np.pi) * integrate.quad(f, 0.0, T, epsabs=1e-12, epsrel=1e-12, limit=400)[0]


def one_period_cost(s, K, T, t, h, r, y, sigma, sigma_hedge=None, mu=None, n_nodes=20001, width=9.0):
    """E[(1/2) S_{t+h} |Delta(t+h, S_{t+h}) - e^{y h} Delta(t, s)| | S_t = s] by quadrature.

    The expected cost of the rebalance at t + h (t + h < T) per unit of round-trip cost k,
    for a call hedged at ``sigma_hedge`` (default sigma) while the stock has volatility
    sigma and total return ``mu`` (default r). Its leading term as h -> 0 is
    (1/2) s^2 Gamma(t, s) sigma sqrt(2 h / pi) (``one_period_cost_leading``)."""
    mu = r if mu is None else mu
    sigma_hedge = sigma if sigma_hedge is None else sigma_hedge
    z = np.linspace(-width, width, n_nodes)
    w = bs.norm_pdf(z) * (z[1] - z[0])
    S1 = s * np.exp((mu - y - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * z)
    D0 = price_delta_gamma("call", s, K, T - t, r, y, sigma_hedge)[1]
    D1 = price_delta_gamma("call", S1, K, T - t - h, r, y, sigma_hedge)[1]
    return 0.5 * float(np.sum(w * S1 * np.abs(D1 - np.exp(y * h) * D0)))


def one_period_cost_leading(s, K, T, t, h, r, y, sigma, sigma_hedge=None):
    """(1/2) s^2 Gamma_h(t, s) sigma sqrt(2 h / pi): the leading term of ``one_period_cost``."""
    sigma_hedge = sigma if sigma_hedge is None else sigma_hedge
    G = price_delta_gamma("call", s, K, T - t, r, y, sigma_hedge)[2]
    return 0.5 * s * s * G * sigma * np.sqrt(2 * h / np.pi)
