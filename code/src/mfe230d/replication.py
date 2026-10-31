"""Static replication, power payoffs, homogeneity and the Kolmogorov equations (Chapter 4).

Static replication (Carr--Madan):

* ``bs_market`` collects Black--Scholes prices of the bond, the prepaid forward, calls, puts
  and digitals into a dict that plays the role of a market;
* ``carr_madan_price`` prices a payoff from its decomposition: f(kappa) bonds, f'(kappa+)
  forwards struck at kappa, puts below kappa and calls above it with weights f''(K) dK,
  options at the kinks, and digitals at the jumps; the strike integrals are computed by a
  composite Gauss--Legendre rule in log-strike (``log_strike_rule``);
* ``grid_replication`` / ``grid_payoff`` / ``grid_price`` build the finite strip: the
  piecewise-linear interpolant of a payoff at listed strikes, which is exactly a portfolio of
  bonds, shares and calls at those strikes;
* ``sqrt_call`` and ``sqrt_call_digital`` are the payoffs used in the notes.

Power payoffs ("pre-wash"):

* ``power_params`` maps (p, r, y, sigma) to the yield and volatility of S^p,
  ``power_call`` / ``power_put`` price (S_T^p - K)^+ and (K - S_T^p)^+ with the Black--Scholes
  formula of ``bs`` and those inputs, ``power_greeks`` converts the Greeks back ("post-wash");
* ``leveraged_params`` / ``leveraged_call`` price S_T^p (S_T - K)^+ by the change of measure
  with density proportional to S_T^p;
* ``discounted_expectation`` computes e^{-rT} E[g(S_T)] by quadrature, an independent check;
* ``put_call_symmetry`` returns both sides of C(K) = (K/F) P(F^2/K).

Homogeneity: ``euler_residual`` (S dV/dS + K dV/dK - V by central differences),
``bs_second_order`` (spot, cross and strike gammas), ``bachelier_call`` (a model whose prices
are not homogeneous in (S, K)).

Kolmogorov equations: ``fd_grid``, ``backward_solve`` and ``forward_solve`` solve the backward
equation (one payoff, all spots) and the forward equation (one spot, all payoffs) for
dX = b dt + a dW with discount rate r on a uniform grid, with the theta scheme and
Rannacher start-up steps. The forward matrices are the transposes of the backward ones, so
the two give the same price up to rounding (discrete adjointness). ``skew_local_vol`` is the
bounded local volatility of the notes and ``local_vol_terminal`` simulates log S_T by Euler.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.linalg import solve_banded
from scipy.special import ndtr

from . import bs


# ---------------------------------------------------------------------------------------
# A Black--Scholes "market" and the Carr--Madan formula
# ---------------------------------------------------------------------------------------

def bs_market(S, T, r, y, sigma) -> dict:
    """Prices of the instruments used in static replication, in the standard model.

    Keys: ``bond`` (e^{-rT}), ``prepaid`` (S e^{-yT}), ``forward`` (S e^{(r-y)T}), and
    vectorised functions of the strike ``call``, ``put``, ``digital`` (pays 1 if S_T > K).
    """
    return dict(
        bond=float(np.exp(-r * T)),
        prepaid=float(S * np.exp(-y * T)),
        forward=float(bs.forward(S, T, r, y)),
        call=lambda K: bs.call(S, K, T, r, y, sigma),
        put=lambda K: bs.put(S, K, T, r, y, sigma),
        digital=lambda K: bs.digital_call(S, K, T, r, y, sigma),
    )


def log_strike_rule(a: float, b: float, n_panels: int = 64, order: int = 12):
    """Nodes K_i and weights w_i with sum w_i g(K_i) ~ int_a^b g(K) dK (0 < a < b).

    Composite Gauss--Legendre rule with ``n_panels`` equal panels in u = log K, each with
    ``order`` nodes; the Jacobian dK = K du is included in the weights.
    """
    if not 0 < a < b:
        raise ValueError("need 0 < a < b")
    z, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(np.log(a), np.log(b), n_panels + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    u = (mid[:, None] + half[:, None] * z[None, :]).ravel()
    wu = (half[:, None] * w[None, :]).ravel()
    K = np.exp(u)
    return K, wu * K


def carr_madan_price(kappa, f_kappa, df_kappa, market, d2f=None, kinks=(), digitals=(),
                     k_lo=None, k_hi=None, n_panels: int = 64, order: int = 12) -> float:
    """Price of a payoff from its static replication with split point kappa > 0.

    price = bond f(kappa) + f'(kappa+) (prepaid - bond kappa)
            + int_{k_lo}^{kappa} f''(K) P(K) dK + int_{kappa}^{k_hi} f''(K) C(K) dK
            + sum_{kinks (k, w)} w * (P(k) if k <= kappa else C(k))
            + sum_{digitals (a, J)} J * D(a),
    where f''(K) dK is the absolutely continuous part ``d2f`` (a vectorised function, or None),
    a kink (k, w) is an atom of f'' of mass w = f'(k+) - f'(k-), and a digital (a, J) is a
    jump of size J at a, priced with ``market["digital"]`` (pays 1 if S_T > a).
    ``k_lo`` and ``k_hi`` truncate the strike integrals; by default they are F e^{-14 s}
    and F e^{14 s} for s = 1 (pass them explicitly for small volatilities). The integrals are
    split at kappa, at the kinks and at the digitals, then computed with ``log_strike_rule``.
    """
    F = market["forward"]
    k_lo = F * np.exp(-14.0) if k_lo is None else k_lo
    k_hi = F * np.exp(14.0) if k_hi is None else k_hi
    price = market["bond"] * f_kappa + df_kappa * (market["prepaid"] - market["bond"] * kappa)
    if d2f is not None:
        pts = [k_lo, kappa, k_hi] + [k for k, _ in kinks] + [a for a, _ in digitals]
        pts = np.unique([p for p in pts if k_lo <= p <= k_hi])
        for a, b in zip(pts[:-1], pts[1:]):
            K, w = log_strike_rule(a, b, n_panels, order)
            opt = market["put"](K) if b <= kappa else market["call"](K)
            price += float(np.sum(w * d2f(K) * opt))
    for k, wk in kinks:
        price += wk * float(market["put"](k) if k <= kappa else market["call"](k))
    for a, J in digitals:
        price += J * float(market["digital"](a))
    return float(price)


# ---------------------------------------------------------------------------------------
# Finitely many strikes
# ---------------------------------------------------------------------------------------

def grid_replication(f_values, strikes) -> dict:
    """The portfolio whose payoff is the piecewise-linear interpolant of a payoff at the strikes.

    With nodes k_0 < ... < k_m, values f_i and slopes s_i on [k_i, k_{i+1}], the interpolant
    (extended linearly beyond k_0 and k_m) is
        f_h(x) = f_0 - s_0 k_0 + s_0 x + sum_{i=1}^{m-1} (s_i - s_{i-1}) (x - k_i)^+ :
    ``bonds`` = f_0 - s_0 k_0 bonds, ``shares`` = s_0 shares, and ``weights`` calls at the
    interior ``strikes``.
    """
    k = np.asarray(strikes, dtype=float)
    fv = np.asarray(f_values, dtype=float)
    if k.ndim != 1 or k.size < 2 or np.any(np.diff(k) <= 0):
        raise ValueError("strikes must be increasing, at least two")
    s = np.diff(fv) / np.diff(k)
    return dict(bonds=fv[0] - s[0] * k[0], shares=s[0], strikes=k[1:-1], weights=np.diff(s))


def grid_payoff(rep: dict, x):
    """Payoff of the portfolio returned by ``grid_replication`` at terminal prices x."""
    x = np.asarray(x, dtype=float)
    calls = np.maximum(x[..., None] - rep["strikes"], 0.0) @ rep["weights"]
    return rep["bonds"] + rep["shares"] * x + calls


def grid_price(rep: dict, market: dict) -> float:
    """Price of the portfolio returned by ``grid_replication``."""
    return float(rep["bonds"] * market["bond"] + rep["shares"] * market["prepaid"]
                 + np.sum(rep["weights"] * market["call"](rep["strikes"])))


def sqrt_call(x, K=100.0):
    """The square-root call payoff (sqrt(x) - sqrt(K))^+."""
    return np.maximum(np.sqrt(np.maximum(x, 0.0)) - np.sqrt(K), 0.0)


def sqrt_call_digital(x, K=100.0, H=120.0, J=1.0):
    """Square-root call plus a digital bonus: (sqrt(x) - sqrt(K))^+ + J 1{x >= H}."""
    return sqrt_call(x, K) + J * (np.asarray(x) >= H)


# ---------------------------------------------------------------------------------------
# Power payoffs: the pre-wash map
# ---------------------------------------------------------------------------------------

def power_params(p, r, y, sigma):
    """(y_p, sigma_p) such that S^p is a standard-model stock with yield y_p and volatility sigma_p.

    y_p = r - p (r - y) - p (p - 1) sigma^2 / 2 and sigma_p = |p| sigma, so that
    E^Q[S_T^p] = S_0^p e^{(r - y_p) T}.
    """
    return r - p * (r - y) - 0.5 * p * (p - 1.0) * sigma**2, abs(p) * sigma


def power_call(S, K, T, r, y, sigma, p):
    """Price of (S_T^p - K)^+ in the standard model: bs.call(S^p, K, T, r, y_p, |p| sigma).

    For p = 0 the payoff is the constant (1 - K)^+ and the price is e^{-rT} (1 - K)^+.
    """
    if p == 0:
        return np.exp(-r * T) * np.maximum(1.0 - np.asarray(K, dtype=float), 0.0)
    yp, sp = power_params(p, r, y, sigma)
    return bs.call(np.asarray(S, dtype=float) ** p, K, T, r, yp, sp)


def power_put(S, K, T, r, y, sigma, p):
    """Price of (K - S_T^p)^+ in the standard model (p = 0: e^{-rT} (K - 1)^+)."""
    if p == 0:
        return np.exp(-r * T) * np.maximum(np.asarray(K, dtype=float) - 1.0, 0.0)
    yp, sp = power_params(p, r, y, sigma)
    return bs.put(np.asarray(S, dtype=float) ** p, K, T, r, yp, sp)


def power_greeks(S, K, T, r, y, sigma, p, kind="call") -> dict:
    """Greeks of (S_T^p - K)^+ (or the put) with respect to the original inputs ("post-wash").

    With V(S, sigma, r, y) = BS(S^p, K, T, r, y_p, |p| sigma) and primes for the Greeks of the
    Black--Scholes formula at the washed inputs:
        delta = p S^{p-1} delta',  gamma = p (p-1) S^{p-2} delta' + p^2 S^{2p-2} gamma',
        vega = |p| vega' - p (p-1) sigma rho_y',  rho_r = rho_r' + (1 - p) rho_y',
        rho_y = p rho_y',  theta = theta'.   (p != 0)
    """
    if p == 0:
        raise ValueError("p = 0: the payoff is constant")
    yp, sp = power_params(p, r, y, sigma)
    S = np.asarray(S, dtype=float)
    g = bs.greeks(S**p, K, T, r, yp, sp, kind)
    return dict(
        delta=p * S ** (p - 1) * g["delta"],
        gamma=p * (p - 1) * S ** (p - 2) * g["delta"] + p**2 * S ** (2 * p - 2) * g["gamma"],
        vega=abs(p) * g["vega"] - p * (p - 1) * sigma * g["rho_y"],
        rho_r=g["rho_r"] + (1 - p) * g["rho_y"],
        rho_y=p * g["rho_y"],
        theta=g["theta"],
    )


def leveraged_params(p, r, y, sigma):
    """(r', y') such that e^{-rT} E[S_T^p g(S_T)] = S_0^p * (price of g with inputs r', y', sigma).

    r' = y_p (the yield of S^p) and y' = y - p (r - y + (p + 1) sigma^2 / 2).
    """
    yp, _ = power_params(p, r, y, sigma)
    return yp, y - p * (r - y + 0.5 * (p + 1.0) * sigma**2)


def leveraged_call(S, K, T, r, y, sigma, p):
    """Price of S_T^p (S_T - K)^+: S^p * bs.call(S, K, T, r', y', sigma)."""
    rr, yy = leveraged_params(p, r, y, sigma)
    S = np.asarray(S, dtype=float)
    return S**p * bs.call(S, K, T, rr, yy, sigma)


def discounted_expectation(g, S, T, r, y, sigma, breaks=(), zmax: float = 12.0) -> float:
    """e^{-rT} E^Q[g(S_T)] in the standard model, by adaptive quadrature in z (an independent check).

    S_T = S exp(nu T + sigma sqrt(T) z), z standard normal; ``breaks`` are values of S_T where
    g has a kink or a jump, passed to the integrator as break points.
    """
    nu = r - y - 0.5 * sigma**2
    sd = sigma * np.sqrt(T)
    zb = sorted((np.log(b / S) - nu * T) / sd for b in breaks if b > 0)
    zb = [z for z in zb if -zmax < z < zmax]
    h = lambda z: float(g(S * np.exp(nu * T + sd * z))) * float(bs.norm_pdf(z))
    edges = [-zmax] + zb + [zmax]
    total = sum(integrate.quad(h, a, b, limit=200, epsabs=1e-13, epsrel=1e-12)[0]
                for a, b in zip(edges[:-1], edges[1:]))
    return float(np.exp(-r * T) * total)


def put_call_symmetry(S, K, T, r, y, sigma):
    """Both sides of put--call symmetry: (C(K), (K/F) P(F^2/K)) with F = S e^{(r-y)T}."""
    F = bs.forward(S, T, r, y)
    return bs.call(S, K, T, r, y, sigma), K / F * bs.put(S, F**2 / K, T, r, y, sigma)


# ---------------------------------------------------------------------------------------
# Homogeneity
# ---------------------------------------------------------------------------------------

def euler_residual(V, S, K, rel: float = 1e-4):
    """S dV/dS + K dV/dK - V for a price function V(S, K), by central differences."""
    S, K = float(S), float(K)
    dS = (V(S * (1 + rel), K) - V(S * (1 - rel), K)) / (2 * rel)
    dK = (V(S, K * (1 + rel)) - V(S, K * (1 - rel))) / (2 * rel)
    return dS + dK - V(S, K)


def bs_second_order(S, K, T, r, y, sigma) -> dict:
    """Spot, cross and strike gammas of the Black--Scholes call (and put).

    gamma_SS = e^{-yT} N'(d_+) / (S w), gamma_SK = -e^{-yT} N'(d_+) / (K w),
    gamma_KK = e^{-rT} N'(d_-) / (K w), with w = sigma sqrt(T).
    """
    dp, dm = bs.d_pm(S, K, T, r, y, sigma)
    w = sigma * np.sqrt(T)
    a = np.exp(-y * T) * bs.norm_pdf(dp) / w
    return dict(SS=a / S, SK=-a / K, KK=np.exp(-r * T) * bs.norm_pdf(dm) / (K * w))


def bachelier_call(S, K, T, r, y, sigma_n):
    """Call in the normal (Bachelier) model for the forward: dF_t = sigma_n dW_t, F_0 = S e^{(r-y)T}.

    C = e^{-rT} [ (F - K) N(d) + sigma_n sqrt(T) N'(d) ],  d = (F - K) / (sigma_n sqrt(T)).
    sigma_n is in currency units, so C is homogeneous of degree one in (S, K, sigma_n), not in (S, K).
    """
    F = bs.forward(S, T, r, y)
    sd = sigma_n * np.sqrt(T)
    d = (F - K) / sd
    return np.exp(-r * T) * ((F - K) * ndtr(d) + sd * bs.norm_pdf(d))


# ---------------------------------------------------------------------------------------
# Kolmogorov equations on a grid: backward (all spots) and forward (all payoffs)
# ---------------------------------------------------------------------------------------

def skew_local_vol(S, sigma0: float = 0.25, skew: float = 0.10, width: float = 0.5, S_ref: float = 100.0):
    """Bounded smooth local volatility sigma0 - skew * tanh(log(S / S_ref) / width).

    It lies in (sigma0 - skew, sigma0 + skew) and decreases with S: an equity-like skew.
    """
    return sigma0 - skew * np.tanh(np.log(np.asarray(S, dtype=float) / S_ref) / width)


def fd_grid(x0: float, half_width: float, n: int = 801):
    """Uniform grid of n points on [x0 - half_width, x0 + half_width]; x0 is the middle node."""
    if n % 2 == 0:
        raise ValueError("n must be odd so that x0 is a grid node")
    return np.linspace(x0 - half_width, x0 + half_width, n)


def _coeffs(c, t, x):
    v = c(t, x) if callable(c) else c
    return np.broadcast_to(np.asarray(v, dtype=float), x.shape).copy()


def _bands(x, t, b, a, r):
    """Sub-, main and super-diagonals of L = b d/dx + (a^2/2) d^2/dx^2 - r on the interior nodes."""
    h = x[1] - x[0]
    xi = x[1:-1]
    bb, aa, rr = _coeffs(b, t, xi), _coeffs(a, t, xi) ** 2, _coeffs(r, t, xi)
    lo = 0.5 * aa / h**2 - 0.5 * bb / h
    up = 0.5 * aa / h**2 + 0.5 * bb / h
    di = -aa / h**2 - rr
    return lo, di, up


def theta_schedule(n_t: int, n_rannacher: int = 2):
    """theta = 1 (implicit Euler) on the first and last ``n_rannacher`` steps, 1/2 otherwise."""
    th = np.full(n_t, 0.5)
    th[:n_rannacher] = 1.0
    th[n_t - n_rannacher:] = 1.0
    return th


def backward_solve(x, payoff, T: float, b, a, r=0.0, n_t: int = 400, n_rannacher: int = 2):
    """Values at time 0, on the grid x, of the claim paying payoff(x) at T: the backward equation.

    Solves d_t u + b d_x u + (a^2/2) d_xx u - r u = 0, u(T) = payoff, with u = 0 at the two ends
    of the grid (killing), by the theta scheme on n_t equal steps. b, a, r are numbers or
    functions of (t, x). Returns u(0, x) (zero at the end nodes).
    """
    x = np.asarray(x, dtype=float)
    dt = T / n_t
    th = theta_schedule(n_t, n_rannacher)
    u = np.asarray(payoff(x) if callable(payoff) else payoff, dtype=float)[1:-1].copy()
    for n in range(n_t - 1, -1, -1):        # step from t_{n+1} to t_n
        lo, di, up = _bands(x, (n + 0.5) * dt, b, a, r)
        c1, c0 = th[n] * dt, (1 - th[n]) * dt
        # rhs = B u with B = I + c0 L: (L u)_i = lo_i u_{i-1} + di_i u_i + up_i u_{i+1}
        rhs = u + c0 * (di * u + np.r_[0.0, lo[1:] * u[:-1]] + np.r_[up[:-1] * u[1:], 0.0])
        ab = np.zeros((3, u.size))           # banded form of A = I - c1 L
        ab[0, 1:] = -c1 * up[:-1]
        ab[1] = 1 - c1 * di
        ab[2, :-1] = -c1 * lo[1:]
        u = solve_banded((1, 1), ab, rhs)
    return np.r_[0.0, u, 0.0]


def forward_solve(x, i0: int, T: float, b, a, r=0.0, n_t: int = 400, n_rannacher: int = 2):
    """Discounted transition density G(0, x[i0]; T, .) on the grid x: the forward equation.

    Solves d_T G = (1/2) d_xixi (a^2 G) - d_xi (b G) - r G with G(0) = delta at x[i0] (the vector
    1/h at node i0), using the transposes of the matrices of ``backward_solve``. Hence, for every
    payoff, h * sum(G * payoff) equals the backward value at x[i0] up to rounding.
    """
    x = np.asarray(x, dtype=float)
    h = x[1] - x[0]
    dt = T / n_t
    th = theta_schedule(n_t, n_rannacher)
    p = np.zeros(x.size - 2)
    p[i0 - 1] = 1.0 / h
    for n in range(n_t):                     # step from t_n to t_{n+1}
        lo, di, up = _bands(x, (n + 0.5) * dt, b, a, r)
        c1, c0 = th[n] * dt, (1 - th[n]) * dt
        ab = np.zeros((3, p.size))           # banded form of A^T, A = I - c1 L
        ab[0, 1:] = -c1 * lo[1:]
        ab[1] = 1 - c1 * di
        ab[2, :-1] = -c1 * up[:-1]
        q = solve_banded((1, 1), ab, p)
        # p <- B^T q with B = I + c0 L: (L^T q)_i = up_{i-1} q_{i-1} + di_i q_i + lo_{i+1} q_{i+1}
        p = q + c0 * (di * q + np.r_[0.0, up[:-1] * q[:-1]] + np.r_[lo[1:] * q[1:], 0.0])
    return np.r_[0.0, p, 0.0]


def call_euler_terms(x, i0: int, K: float, T: float, b, a, r=0.0, n_t: int = 400) -> dict:
    """The three terms of Euler's identity for the call struck at K (a grid node), spot S = exp(x[i0]).

    The price and the spot derivative come from one backward solve (all spots), the strike
    derivative -e^{-rT} Q(S_T > K) from one forward solve (all strikes), with the trapezoid weight
    1/2 at the node K. Returns dict(C, S_dS, K_dK, residual = S_dS + K_dK - C).
    """
    x = np.asarray(x, dtype=float)
    h = x[1] - x[0]
    k = int(np.argmin(np.abs(x - np.log(K))))
    if not np.isclose(x[k], np.log(K)):
        raise ValueError("log K must be a grid node")
    pay = np.maximum(np.exp(x) - K, 0.0)
    u = backward_solve(x, pay, T, b, a, r, n_t)
    G = forward_solve(x, i0, T, b, a, r, n_t)
    S = np.exp(x[i0])
    dS = (u[i0 + 1] - u[i0 - 1]) / (np.exp(x[i0 + 1]) - np.exp(x[i0 - 1]))
    digital = h * (np.sum(G[k + 1:]) + 0.5 * G[k])
    out = dict(C=float(u[i0]), S_dS=float(S * dS), K_dK=float(-K * digital))
    out["residual"] = out["S_dS"] + out["K_dK"] - out["C"]
    return out


def local_vol_terminal(S0, T, r, y, sigma_fn, n_steps: int = 250, n_paths: int = 100_000, seed=None):
    """Samples of S_T for dS = (r - y) S dt + sigma(S) S dW, by the Euler scheme for log S."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    X = np.full(n_paths, np.log(S0))
    for _ in range(n_steps):
        s = sigma_fn(np.exp(X))
        X += (r - y - 0.5 * s * s) * dt + s * np.sqrt(dt) * rng.standard_normal(n_paths)
    return np.exp(X)
