"""Trees and finite differences (Chapter 20).

Conventions follow the notes. The stock is the Black--Scholes stock with yield y under the
risk-neutral measure, nu = r - y - sigma^2/2 is the log-drift, and prices are functions of
the time to maturity tau and of the log price x = log S:

    d_tau u = (sigma^2/2) d_xx u + nu d_x u - r u,     u(0, x) = g(e^x).

Contents:
  * binomial trees: Cox--Ross--Rubinstein (CRR), Jarrow--Rudd (JR) and Leisen--Reimer (LR),
    European prices from the terminal binomial law (O(n) work) and backward induction for
    European, Bermudan and American exercise, the smoothing of Broadie and Detemple (the
    Black--Scholes price one step before maturity) and Richardson extrapolation (BBSR);
    the position of the strike in the CRR lattice and the oscillation coefficient;
  * the trinomial tree of Kamrad and Ritchken in log price, with stretch parameter lam;
  * the theta-scheme on a uniform grid in log price (explicit, implicit, Crank--Nicolson),
    with the Rannacher start, Dirichlet far-field conditions, Greeks, Bermudan or American
    exercise (projected SOR, Brennan--Schwartz, penalty) and discretely or continuously
    monitored knock-out barriers; a Crank--Nicolson grid in the price itself;
  * von Neumann amplification factors, the one-step Crank--Nicolson kernel, and the exit
    probability that bounds the far-field error.

Nothing here is random. The American references of Chapter 14 (``american.binomial`` and
``american.bermudan_grid``) are trees and grids of the same family; the tests compare them.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ndtr
from scipy.stats import binom

from . import bs

# ---------------------------------------------------------------------------------------
# Payoffs
# ---------------------------------------------------------------------------------------


def payoff(kind: str, K: float):
    """g(s) for kind in {"call", "put", "straddle", "digital"} (digital: 1{s > K})."""
    if kind == "call":
        return lambda s: np.maximum(np.asarray(s, dtype=float) - K, 0.0)
    if kind == "put":
        return lambda s: np.maximum(K - np.asarray(s, dtype=float), 0.0)
    if kind == "straddle":
        return lambda s: np.abs(np.asarray(s, dtype=float) - K)
    if kind == "digital":
        return lambda s: (np.asarray(s, dtype=float) > K).astype(float)
    raise ValueError(f"unknown payoff {kind!r}")


def _european(kind, S, K, T, r, y, sigma):
    """Black--Scholes price of a call, put, straddle or digital call (T may be 0)."""
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    if np.all(T <= 0):
        return payoff(kind, K)(S)
    if kind == "call":
        return bs.call(S, K, T, r, y, sigma)
    if kind == "put":
        return bs.put(S, K, T, r, y, sigma)
    if kind == "straddle":
        return bs.call(S, K, T, r, y, sigma) + bs.put(S, K, T, r, y, sigma)
    if kind == "digital":
        return bs.digital_call(S, K, T, r, y, sigma)
    raise ValueError(f"no closed form for {kind!r}")


# ---------------------------------------------------------------------------------------
# Binomial trees
# ---------------------------------------------------------------------------------------


def peizer_pratt(z, n: int):
    """Peizer--Pratt inversion (method 2): the p with P(Bin(n, p) >= (n+1)/2) ~ N(z), n odd."""
    z = np.asarray(z, dtype=float)
    t = (z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2 * (n + 1.0 / 6.0)
    return 0.5 + np.sign(z) * 0.5 * np.sqrt(1.0 - np.exp(-t))


def tree_parameters(T, r, y, sigma, n: int, method: str = "crr", S=None, K=None) -> dict:
    """Up and down factors u, d, risk-neutral probability q, step dt and discount factor.

    method: "crr" (u = e^{sigma sqrt dt}, d = 1/u), "jr" (Jarrow--Rudd: u, d = e^{nu dt +- sigma
    sqrt dt}) or "lr" (Leisen--Reimer, n odd, needs S and K: q = h^{-1}(d_-), q' = h^{-1}(d_+)
    with the Peizer--Pratt inversion h^{-1}, u = e^{(r-y) dt} q'/q, d = (e^{(r-y) dt} - q u)/(1 - q)).
    q = (e^{(r-y) dt} - d)/(u - d) is returned even when it is outside (0, 1); ``valid`` says.
    """
    dt = T / n
    growth = np.exp((r - y) * dt)
    if method == "crr":
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
    elif method == "jr":
        nu = r - y - 0.5 * sigma**2
        u = np.exp(nu * dt + sigma * np.sqrt(dt))
        d = np.exp(nu * dt - sigma * np.sqrt(dt))
    elif method == "lr":
        if n % 2 == 0:
            raise ValueError("the Leisen--Reimer tree needs an odd number of steps")
        dp, dm = bs.d_pm(S, K, T, r, y, sigma)
        p, pd = float(peizer_pratt(dm, n)), float(peizer_pratt(dp, n))
        u = growth * pd / p
        d = (growth - p * u) / (1.0 - p)
    else:
        raise ValueError(f"unknown tree {method!r}")
    q = (growth - d) / (u - d)
    return dict(u=float(u), d=float(d), q=float(q), dt=dt, disc=float(np.exp(-r * dt)),
                valid=bool(0.0 < q < 1.0))


def crr_min_steps(T, r, y, sigma) -> int:
    """Smallest n with q in (0, 1) for the CRR tree: n > (r - y)^2 T / sigma^2."""
    return int(np.floor((r - y) ** 2 * T / sigma**2)) + 1


def binomial_european(S, K, T, r, y, sigma, n: int, kind: str = "call", method: str = "crr",
                      smooth: bool = False) -> float:
    """European price from the binomial law of the number of up moves (O(n) work).

    smooth=True is the Broadie--Detemple "binomial Black--Scholes" tree: the values one step
    before maturity are the Black--Scholes prices with maturity dt, so the sum runs over the
    n - 1 step law.
    """
    p = tree_parameters(T, r, y, sigma, n, method, S, K)
    if not p["valid"]:
        raise ValueError("risk-neutral probability outside (0, 1): take more steps")
    u, d, q, dt = p["u"], p["d"], p["q"], p["dt"]
    m = n - 1 if smooth else n
    j = np.arange(m + 1)
    nodes = S * np.exp(j * np.log(u) + (m - j) * np.log(d))
    values = _european(kind, nodes, K, dt, r, y, sigma) if smooth else payoff(kind, K)(nodes)
    return float(np.exp(-r * m * dt) * np.sum(binom.pmf(j, m, q) * values))


def binomial(S, K, T, r, y, sigma, n: int, kind: str = "put", exercise="american",
             method: str = "crr", smooth: bool = False, greeks: bool = False, history: bool = False):
    """Backward induction in a binomial tree with n steps.

    exercise: "european", "american" (every step) or an integer m (Bermudan: dates k T / m,
    k = 1..m, and t_0 = 0; n must be a multiple of m). smooth: Black--Scholes values one step
    before maturity, maximised with the payoff where exercise is allowed (Broadie--Detemple).
    Returns the price, or with greeks=True a dict with price, delta and gamma from the nodes of
    steps 1 and 2. history=True adds to the dict the times t_i = i dt and the exercise boundary
    at each step (the largest exercised node for puts, the smallest for calls; nan if none).
    """
    g = payoff(kind, K)
    if exercise == "european":
        every = 0
    elif exercise == "american":
        every = 1
    else:
        m = int(exercise)
        if n % m:
            raise ValueError("n must be a multiple of the number of exercise dates")
        every = n // m
    p = tree_parameters(T, r, y, sigma, n, method, S, K)
    if not p["valid"]:
        raise ValueError("risk-neutral probability outside (0, 1): take more steps")
    lu, ld, q, disc, dt = np.log(p["u"]), np.log(p["d"]), p["q"], p["disc"], p["dt"]

    def nodes(i):                                   # j = 0..i up moves
        j = np.arange(i + 1)
        return S * np.exp(j * lu + (i - j) * ld)

    if smooth:
        i = n - 1
        V = _european(kind, nodes(i), K, dt, r, y, sigma)
        if every and i % every == 0:
            V = np.maximum(V, g(nodes(i)))
        start = n - 2
    else:
        V = g(nodes(n))
        start = n - 1
    keep = {}
    bnd = np.full(n + 1, np.nan)
    for i in range(start, -1, -1):
        V = disc * (q * V[1:] + (1.0 - q) * V[:-1])
        if every and i % every == 0:
            gi = g(nodes(i))
            if history:
                ex = np.flatnonzero((gi >= V) & (gi > 0))
                if ex.size:
                    bnd[i] = nodes(i)[ex.max() if kind == "put" else ex.min()]
            V = np.maximum(V, gi)
        if i <= 2:
            keep[i] = V.copy()
    if not (greeks or history):
        return float(V[0])
    s1, s2 = nodes(1), nodes(2)
    v1, v2 = keep[1], keep[2]
    delta = (v1[1] - v1[0]) / (s1[1] - s1[0])
    gamma = ((v2[2] - v2[1]) / (s2[2] - s2[1]) - (v2[1] - v2[0]) / (s2[1] - s2[0])) / (0.5 * (s2[2] - s2[0]))
    out = dict(price=float(V[0]), delta=float(delta), gamma=float(gamma))
    if history:
        out.update(t=dt * np.arange(n + 1), boundary=bnd)
    return out


def bbsr(S, K, T, r, y, sigma, n: int, kind: str = "put", exercise="american") -> float:
    """Broadie--Detemple BBSR: 2 V(n) - V(n/2) with the smoothed CRR tree (n even)."""
    if n % 2:
        raise ValueError("n must be even")
    a = binomial(S, K, T, r, y, sigma, n, kind, exercise, smooth=True)
    b = binomial(S, K, T, r, y, sigma, n // 2, kind, exercise, smooth=True)
    return 2.0 * a - b


def crr_strike_offset(S, K, T, sigma, n: int) -> float:
    """theta_n in [0, 1): fractional position of log K between the terminal CRR nodes.

    The nodes are log S + (2j - n) delta, delta = sigma sqrt(T/n), with spacing 2 delta;
    theta_n = frac((log(K/S) + n delta) / (2 delta)). theta_n = 0: K is a node.
    """
    delta = sigma * np.sqrt(T / n)
    eta = (np.log(K / S) + n * delta) / (2.0 * delta)
    return float(eta - np.floor(eta + 1e-12)) if abs(eta - round(eta)) > 1e-9 else 0.0


def crr_oscillation_coefficient(S, K, T, r, y, sigma) -> float:
    """B = 2 sigma sqrt(T) K e^{-rT} N'(d_-): n (C_n - C_BS) = A + B theta_n (1 - theta_n) + o(1)."""
    _, dm = bs.d_pm(S, K, T, r, y, sigma)
    return float(2.0 * sigma * np.sqrt(T) * K * np.exp(-r * T) * bs.norm_pdf(dm))


# ---------------------------------------------------------------------------------------
# Trinomial tree (Kamrad--Ritchken)
# ---------------------------------------------------------------------------------------


def trinomial_probabilities(T, r, y, sigma, n: int, lam: float = np.sqrt(1.5)) -> dict:
    """Moves +h, 0, -h of log S with h = lam sigma sqrt(dt) and probabilities

        p_+- = 1/(2 lam^2) +- nu sqrt(dt)/(2 lam sigma),   p_0 = 1 - 1/lam^2,

    which give the mean nu dt and the second moment sigma^2 dt exactly. ``valid`` says whether
    all three are in [0, 1] (lam >= 1 and h |nu| <= sigma^2).
    """
    dt = T / n
    nu = r - y - 0.5 * sigma**2
    h = lam * sigma * np.sqrt(dt)
    pu = 0.5 / lam**2 + nu * np.sqrt(dt) / (2.0 * lam * sigma)
    pd = 0.5 / lam**2 - nu * np.sqrt(dt) / (2.0 * lam * sigma)
    pm = 1.0 - 1.0 / lam**2
    return dict(pu=pu, pm=pm, pd=pd, h=h, dt=dt, disc=np.exp(-r * dt),
                valid=bool(min(pu, pm, pd) >= 0.0))


def trinomial(S, K, T, r, y, sigma, n: int, kind: str = "put", exercise="european",
              lam: float = np.sqrt(1.5), discount: str = "exact") -> float:
    """Backward induction in the Kamrad--Ritchken trinomial tree.

    discount="exact" multiplies the expectation by e^{-r dt}; discount="scheme" subtracts
    r dt V at the middle node instead, V <- p_+ V_+ + p_0 V_0 + p_- V_- - r dt V_0, which is exactly
    the explicit finite-difference scheme with k = dt and the same h (the two differ by O(dt^2)
    per step).
    exercise: "european", "american" or an integer m (Bermudan, n a multiple of m).
    """
    g = payoff(kind, K)
    p = trinomial_probabilities(T, r, y, sigma, n, lam)
    if discount not in ("exact", "scheme"):
        raise ValueError("discount must be 'exact' or 'scheme'")
    every = 0 if exercise == "european" else 1 if exercise == "american" else n // int(exercise)
    x0, h = np.log(S), p["h"]
    V = g(np.exp(x0 + h * np.arange(-n, n + 1)))
    for i in range(n - 1, -1, -1):
        if discount == "exact":
            V = p["disc"] * (p["pu"] * V[2:] + p["pm"] * V[1:-1] + p["pd"] * V[:-2])
        else:
            V = p["pu"] * V[2:] + (p["pm"] - r * p["dt"]) * V[1:-1] + p["pd"] * V[:-2]
        if every and i % every == 0:
            V = np.maximum(V, g(np.exp(x0 + h * np.arange(-i, i + 1))))
    return float(V[0])


# ---------------------------------------------------------------------------------------
# Finite differences in log price
# ---------------------------------------------------------------------------------------


def coefficients(h, r, y, sigma):
    """(p_-, p_0, p_+) of the central-difference operator L_h = a D2 + b D1 - c, a = sigma^2/2, b = nu, c = r."""
    a, b = 0.5 * sigma**2, r - y - 0.5 * sigma**2
    return a / h**2 - 0.5 * b / h, -2.0 * a / h**2 - r, a / h**2 + 0.5 * b / h


def log_grid(S, T, sigma, J: int, width: float = 6.0, K=None, barrier=None):
    """Uniform grid x_j = log S + (j - J/2) h, j = 0..J (J even), half-width ~ width sigma sqrt(T).

    If K is given, h is shrunk slightly so that log(K/S) is a multiple of h (K is a node). If a
    barrier B is given, h is chosen so that log(B/S) lies midway between two nodes.
    """
    if J % 2:
        raise ValueError("J must be even so that S is the middle node")
    h = 2.0 * width * sigma * np.sqrt(T) / J
    if barrier is not None:
        d = abs(np.log(barrier / S))
        h = d / (np.ceil(d / h - 0.5) + 0.5)
    elif K is not None and not np.isclose(K, S):
        d = abs(np.log(K / S))
        h = d / np.ceil(d / h)
    return np.log(S) + h * (np.arange(J + 1) - J // 2), h


def _boundary_values(kind, exercise, bc, K, r, y, sigma, x_lo, x_hi):
    """Functions tau -> (left value, right value) for the far-field Dirichlet conditions."""
    s_lo, s_hi = np.exp(x_lo), np.exp(x_hi)
    g = payoff(kind, K)
    if callable(bc):
        return lambda tau: bc(tau, s_lo, s_hi)
    if bc == "exact":
        return lambda tau: (float(_european(kind, s_lo, K, tau, r, y, sigma)),
                            float(_european(kind, s_hi, K, tau, r, y, sigma)))
    if bc == "payoff":
        return lambda tau: (float(g(s_lo)), float(g(s_hi)))
    if bc != "asymptotic":
        raise ValueError(f"unknown boundary condition {bc!r}")
    american = exercise != "european"

    def fwd(s, tau):                                   # zero-volatility value of the payoff
        if kind == "call":
            v = max(s * np.exp(-y * tau) - K * np.exp(-r * tau), 0.0)
        elif kind == "put":
            v = max(K * np.exp(-r * tau) - s * np.exp(-y * tau), 0.0)
        elif kind == "straddle":
            v = abs(s * np.exp(-y * tau) - K * np.exp(-r * tau))
        else:
            v = float(s * np.exp(-y * tau) > K * np.exp(-r * tau)) * np.exp(-r * tau)
        return max(v, float(g(s))) if american else v

    return lambda tau: (fwd(s_lo, tau), fwd(s_hi, tau))


def _schedule(N: int, theta: float, rannacher: int, resets=()):
    """List of (theta, dt fraction) sub-steps, and the indices (in N) where each step ends.

    rannacher = m replaces the first m theta-steps after the start and after each reset date by
    2m implicit half-steps (Giles and Carter 2006 use m = 2).
    """
    steps = []
    restart = {0} | {int(n) for n in resets}
    count = 0
    for n in range(N):
        if n in restart:
            count = rannacher
        if count > 0 and theta < 1.0:
            steps.append((n, [(1.0, 0.5), (1.0, 0.5)]))
            count -= 1
        else:
            steps.append((n, [(theta, 1.0)]))
    return steps


def _bands(theta, k, pm, p0, pp, size):
    """Banded storage of B = I - theta k A for solve_banded((1, 1), ...), A = tridiag(pm, p0, pp)."""
    ab = np.zeros((3, size))
    ab[0, 1:] = -theta * k * pp
    ab[1, :] = 1.0 - theta * k * p0
    ab[2, :-1] = -theta * k * pm
    return ab


# ---------------------------------------------------------------- linear complementarity


def lcp_residual(ab, q, g, z) -> float:
    """max of (g - z)^+, (q - Bz)^+ and |min(z - g, Bz - q)|: zero iff z solves the LCP."""
    w = ab[1] * z - q
    w[:-1] += ab[0, 1:] * z[1:]
    w[1:] += ab[2, :-1] * z[:-1]
    return float(max(np.max(np.maximum(g - z, 0.0)), np.max(np.maximum(-w, 0.0)),
                     np.max(np.abs(np.minimum(z - g, w)))))


def psor(ab, q, g, z0, omega: float = 1.2, tol: float = 1e-12, maxit: int = 100_000):
    """Projected SOR for z >= g, Bz >= q, (z - g).(Bz - q) = 0, in red--black order.

    ab is B in banded form (sub-diagonal ab[2, :-1], diagonal ab[1], super-diagonal ab[0, 1:]).
    Returns (z, number of sweeps). Stops when a sweep changes no entry by more than
    tol * max(1, |z|_inf).
    """
    lo = np.r_[0.0, ab[2, :-1]]                     # lo[j] multiplies z[j-1]
    di = ab[1]
    up = np.r_[ab[0, 1:], 0.0]                      # up[j] multiplies z[j+1]
    z = np.maximum(np.asarray(z0, dtype=float).copy(), g)
    for it in range(1, maxit + 1):
        change = 0.0
        for parity in (0, 1):
            left = np.r_[0.0, z[:-1]]
            right = np.r_[z[1:], 0.0]
            gs = (q - lo * left - up * right) / di
            idx = slice(parity, None, 2)
            new = np.maximum(g[idx], z[idx] + omega * (gs[idx] - z[idx]))
            change = max(change, float(np.max(np.abs(new - z[idx]))))
            z[idx] = new
        if change <= tol * max(1.0, float(np.max(np.abs(z)))):
            return z, it
    raise RuntimeError("PSOR did not converge")


def brennan_schwartz(ab, q, g, side: str = "low"):
    """Brennan--Schwartz: elimination away from the exercise region, projected substitution.

    side="low": exercise region at the bottom of the grid (puts): eliminate the
    super-diagonal from the top, then z_j = max(g_j, (q'_j - l_j z_{j-1}) / d'_j) upwards.
    side="high": the mirror image (calls). The result solves the LCP when the solution has a
    single exercise region on that side (``lcp_residual`` checks it).
    """
    lo = ab[2, :-1].copy()                           # lo[j-1] multiplies z[j-1] in row j
    di = ab[1].copy()
    up = ab[0, 1:].copy()                            # up[j] multiplies z[j+1] in row j
    q = np.asarray(q, dtype=float).copy()
    g = np.asarray(g, dtype=float)
    if side == "high":
        lo, up = up[::-1].copy(), lo[::-1].copy()
        di, q, g = di[::-1].copy(), q[::-1].copy(), g[::-1]
    n = q.size
    dp = di.copy()
    for j in range(n - 2, -1, -1):                   # eliminate the super-diagonal from the top
        m = up[j] / dp[j + 1]
        dp[j] -= m * lo[j]
        q[j] -= m * q[j + 1]
    z = np.empty(n)
    z[0] = max(g[0], q[0] / dp[0])
    zl, lol, ql, dpl, gl = z.tolist(), lo.tolist(), q.tolist(), dp.tolist(), g.tolist()
    prev = zl[0]
    for j in range(1, n):
        v = (ql[j] - lol[j - 1] * prev) / dpl[j]
        prev = v if v > gl[j] else gl[j]
        zl[j] = prev
    z = np.array(zl)
    return z[::-1].copy() if side == "high" else z


def penalty(ab, q, g, z0, rho: float = 1e8, maxit: int = 100):
    """Penalty method of Forsyth and Vetzal: solve Bz = q + rho (g - z)^+ by policy iteration.

    Each iteration solves (B + rho P_k) z = q + rho P_k g with P_k = diag(1{z_k < g}); it stops
    when the set {z < g} repeats. In exact arithmetic the iteration terminates (Forsyth and
    Vetzal 2002, Theorem 6.1); in floating point a node where z = g to round-off can flip in and
    out of the set, and on such a cycle we return the iterate with the smallest violation
    max (g - z)^+. Returns (z, iterations).
    """
    z = np.asarray(z0, dtype=float).copy()
    active = z < g
    seen = {}
    for it in range(1, maxit + 1):
        a = ab.copy()
        a[1] += rho * active
        z = solve_banded((1, 1), a, q + rho * active * g)
        new = z < g
        if np.array_equal(new, active):
            return z, it
        key = new.tobytes()
        if key in seen:                                   # a round-off cycle
            cands = list(seen.values()) + [z]
            best = min(cands, key=lambda v: float(np.max(np.maximum(g - v, 0.0))))
            return best, it
        seen[active.tobytes()] = z
        active = new
    raise RuntimeError("penalty iteration did not terminate")


# ---------------------------------------------------------------- the theta-scheme


def fd_solve(S, K, T, r, y, sigma, J: int = 400, N: int = 200, theta: float = 0.5,
             kind: str = "call", exercise="european", rannacher: int = 0, width: float = 6.0,
             bc="asymptotic", solver: str = "psor", omega: float = 1.2, tol: float = 1e-12,
             rho: float = 1e8, barrier=None, monitoring=None, align_strike: bool = True,
             history: bool = False) -> dict:
    """The theta-scheme for the pricing equation on a uniform grid in log price.

    Grid: ``log_grid`` (J intervals, S the middle node, K a node if align_strike); N equal time
    steps of k = T/N; theta = 0 explicit, 1 implicit, 1/2 Crank--Nicolson; rannacher = m
    replaces m Crank--Nicolson steps after the start (and after each monitoring date) by 2m
    implicit half-steps. bc: "asymptotic" (zero-volatility values, maximised with the payoff
    for American exercise), "exact" (Black--Scholes values), "payoff", or a function
    (tau, s_lo, s_hi) -> (left, right).

    exercise: "european", "american" (a linear complementarity problem at every step, solved by
    ``solver`` in {"psor", "bs" (Brennan--Schwartz), "penalty"}; explicit steps take the
    maximum with the payoff) or an integer m (Bermudan: the maximum is taken at the dates
    k T / m; N a multiple of m).

    barrier: B for a down-and-out (B < S) or up-and-out (B > S) knock-out with zero rebate.
    monitoring=None: continuous; the grid then ends at B, where the value is 0, S is a node,
    and the far end lies about 2 width sigma sqrt(T) away. monitoring=m: the dates j T / m,
    j = 1..m (N a multiple of m), at which the values beyond B are set to zero; log B lies
    midway between two nodes.

    Returns a dict with price, delta, gamma at S (central differences), the grid (x, s), the
    values V at tau = T, h, k, lam = (sigma^2/2) k / h^2, the total and the largest number of
    LCP iterations per step, the largest LCP residual (PSOR and Brennan--Schwartz), and with
    history=True the exercise boundary after each step (largest exercised node for puts,
    smallest for calls; nan if none) with the times to maturity ``tau``.
    """
    g = payoff(kind, K)
    k = T / N
    if barrier is not None and monitoring is None:
        h0 = 2.0 * width * sigma * np.sqrt(T) / J
        dist = abs(np.log(S / barrier))
        iS = max(1, int(round(dist / h0)))
        h = dist / iS
        if barrier < S:
            x = np.log(barrier) + h * np.arange(J + 1)
        else:
            x = np.log(barrier) - h * np.arange(J, -1, -1)
            iS = J - iS
    else:
        x, h = log_grid(S, T, sigma, J, width, K if align_strike else None,
                        barrier if monitoring is not None else None)
        iS = J // 2
    s = np.exp(x)
    pm, p0, pp = coefficients(h, r, y, sigma)
    american = exercise == "american"
    if exercise in ("european", "american"):
        ex_steps = set()
    else:
        m = int(exercise)
        if N % m:
            raise ValueError("N must be a multiple of the number of exercise dates")
        ex_steps = {N // m * i for i in range(1, m + 1)}
    alive = np.ones(J + 1)
    mon_steps = set()
    if barrier is not None and monitoring is not None:
        mm = int(monitoring)
        if N % mm:
            raise ValueError("N must be a multiple of the number of monitoring dates")
        mon_steps = {N // mm * i for i in range(1, mm)}
        alive = (s > barrier if barrier < S else s < barrier).astype(float)
    far = _boundary_values(kind, "european" if barrier is not None else exercise, bc, K, r, y, sigma,
                           x[0], x[-1])
    if barrier is None:
        bvals = far
    elif barrier < S:
        bvals = lambda tau: (0.0, far(tau)[1])
    else:
        bvals = lambda tau: (far(tau)[0], 0.0)
    V = g(s) * alive
    gi = g(s[1:-1])
    total_it, max_it, max_res = 0, 0, 0.0
    hist = []
    tau = 0.0
    cache = {}
    side = "low" if kind == "put" else "high"
    for n, subs in _schedule(N, theta, rannacher, resets=sorted(mon_steps)):
        for th, frac in subs:
            kk = frac * k
            lv1, rv1 = bvals(tau + kk)
            rhs = V[1:-1] + (1 - th) * kk * (pm * V[:-2] + p0 * V[1:-1] + pp * V[2:])
            tau += kk
            if th == 0.0:
                Vi = np.maximum(rhs, gi) if american else rhs
            else:
                rhs[0] += th * kk * pm * lv1
                rhs[-1] += th * kk * pp * rv1
                key = (th, kk)
                if key not in cache:
                    cache[key] = _bands(th, kk, pm, p0, pp, J - 1)
                ab = cache[key]
                if american:
                    if solver == "psor":
                        Vi, it = psor(ab, rhs, gi, V[1:-1], omega, tol)
                    elif solver == "bs":
                        Vi, it = brennan_schwartz(ab, rhs, gi, side), 1
                    elif solver == "penalty":
                        Vi, it = penalty(ab, rhs, gi, V[1:-1], rho)
                    else:
                        raise ValueError(f"unknown solver {solver!r}")
                    total_it += it
                    max_it = max(max_it, it)
                    max_res = max(max_res, lcp_residual(ab, rhs, gi, Vi))
                else:
                    Vi = solve_banded((1, 1), ab, rhs)
            V = np.r_[lv1, Vi, rv1]
        if (n + 1) in ex_steps:
            V = np.maximum(V, g(s))
        if (n + 1) in mon_steps:
            V = V * alive
        if history:
            exd = np.flatnonzero((V[1:-1] <= gi + 1e-12) & (gi > 0)) + 1
            hist.append(np.nan if exd.size == 0 else float(s[exd.max()] if kind == "put" else s[exd.min()]))
    ux = (V[iS + 1] - V[iS - 1]) / (2 * h)
    uxx = (V[iS + 1] - 2 * V[iS] + V[iS - 1]) / h**2
    out = dict(price=float(V[iS]), delta=float(ux / S), gamma=float((uxx - ux) / S**2), x=x, s=s, V=V,
               h=h, k=k, lam=0.5 * sigma**2 * k / h**2, iterations=total_it, max_iterations=max_it,
               residual=max_res, index=iS)
    if history:
        out["boundary"] = np.array(hist)
        out["tau"] = k * np.arange(1, N + 1)
    return out


def fd_greeks(out: dict) -> dict:
    """Delta and gamma on the whole grid (interior nodes) from a result of ``fd_solve``."""
    V, s, h = out["V"], out["s"], out["h"]
    ux = (V[2:] - V[:-2]) / (2 * h)
    uxx = (V[2:] - 2 * V[1:-1] + V[:-2]) / h**2
    si = s[1:-1]
    return dict(s=si, delta=ux / si, gamma=(uxx - ux) / si**2)


def fd_price(*args, **kwargs) -> float:
    """``fd_solve(...)["price"]``."""
    return fd_solve(*args, **kwargs)["price"]


# ---------------------------------------------------------------- a grid in the price itself


def s_grid_call(K, T, r, y, sigma, S_max: float, M: int = 600, N: int = 600, bc: str = "asymptotic",
                rannacher: int = 2) -> dict:
    """European call by Crank--Nicolson on S_j = j S_max / M, j = 0..M (S = 0 needs no condition).

    At S = 0 the equation reduces to d_tau V = -r V, so V = 0 for a call. At S_max the Dirichlet
    value is S_max e^{-y tau} - K e^{-r tau} ("asymptotic"), S_max - K ("payoff") or the exact
    Black--Scholes value ("exact"). Returns the grid and the values at tau = T.
    """
    dS = S_max / M
    j = np.arange(1, M)
    a = 0.5 * sigma**2 * j**2
    b = 0.5 * (r - y) * j
    lo, di, up = a - b, -2 * a - r, a + b                   # per unit time
    k = T / N

    def top(tau):
        if bc == "asymptotic":
            return S_max * np.exp(-y * tau) - K * np.exp(-r * tau)
        if bc == "payoff":
            return S_max - K
        if bc == "exact":
            return float(_european("call", S_max, K, tau, r, y, sigma))
        raise ValueError(bc)

    S = dS * np.arange(M + 1)
    V = np.maximum(S - K, 0.0)
    tau = 0.0
    for th, kk in [(1.0, 0.5 * k)] * (2 * rannacher) + [(0.5, k)] * (N - rannacher):
        rhs = V[1:-1] + (1 - th) * kk * (lo * V[:-2] + di * V[1:-1] + up * V[2:])
        rv = top(tau + kk)
        rhs[-1] += th * kk * up[-1] * rv
        ab = np.zeros((3, M - 1))
        ab[0, 1:] = -th * kk * up[:-1]
        ab[1] = 1 - th * kk * di
        ab[2, :-1] = -th * kk * lo[1:]
        V = np.r_[0.0, solve_banded((1, 1), ab, rhs), rv]
        tau += kk
    return dict(S=S, V=V)


# ---------------------------------------------------------------------------------------
# Stability, positivity and the far field
# ---------------------------------------------------------------------------------------


def amplification(xi_h, lam, theta, sigma=0.2, r=0.0, y=0.0, h=None):
    """von Neumann factor g(xi) of the theta-scheme, as a function of xi h, for lam = (sigma^2/2) k / h^2.

    With h=None the drift and discount terms are dropped (the heat equation); otherwise
    k = 2 lam h^2 / sigma^2 and the symbol is -(2 sigma^2/h^2) sin^2(xi h/2) - r + i (nu/h) sin(xi h).
    """
    xi_h = np.asarray(xi_h, dtype=float)
    if h is None:
        kl = -4.0 * lam * np.sin(0.5 * xi_h) ** 2 + 0j
    else:
        k = 2.0 * lam * h**2 / sigma**2
        nu = r - y - 0.5 * sigma**2
        kl = k * (-(2 * sigma**2 / h**2) * np.sin(0.5 * xi_h) ** 2 - r + 1j * nu / h * np.sin(xi_h))
    return (1.0 + (1.0 - theta) * kl) / (1.0 - theta * kl)


def cn_kernel(lam: float, half: int = 60, theta: float = 0.5):
    """One theta-step of the heat equation applied to a unit spike: the column of G = B^{-1} C.

    Nodes -half..half, zero Dirichlet values beyond. On the infinite grid the centre weight of
    Crank--Nicolson is 2/sqrt(1 + 2 lam) - 1, negative iff lam > 3/2.
    """
    size = 2 * half + 1
    ab = np.zeros((3, size))
    ab[0, 1:] = -theta * lam
    ab[1] = 1 + 2 * theta * lam
    ab[2, :-1] = -theta * lam
    e = np.zeros(size)
    e[half] = 1.0
    rhs = (1 - 2 * (1 - theta) * lam) * e
    rhs[half - 1] += (1 - theta) * lam
    rhs[half + 1] += (1 - theta) * lam
    return np.arange(-half, half + 1), solve_banded((1, 1), ab, rhs)


def exit_probability(x, x_lo, x_hi, tau, r, y, sigma) -> float:
    """Upper bound on Q(X leaves (x_lo, x_hi) before tau), X_s = x + nu s + sigma W_s.

    The sum of the exact one-sided probabilities (reflection principle with drift):
    Q(max_{s<=tau} (nu s + sigma W_s) >= m) = N((-m + nu tau)/(sigma sqrt tau))
    + e^{2 nu m / sigma^2} N((-m - nu tau)/(sigma sqrt tau)), and its mirror image.
    """
    nu = r - y - 0.5 * sigma**2
    st = sigma * np.sqrt(tau)

    def up(m, mu):
        return ndtr((-m + mu * tau) / st) + np.exp(2 * mu * m / sigma**2) * ndtr((-m - mu * tau) / st)

    total = 0.0
    if np.isfinite(x_hi):
        total += up(x_hi - x, nu)
    if np.isfinite(x_lo):
        total += up(x - x_lo, -nu)
    return float(min(total, 1.0))


def exit_bound(d_lo, d_hi, tau, r, y, sigma) -> float:
    """The Gaussian bound exp(-(d_hi - nu^+ tau)^2/(2 sigma^2 tau)) + exp(-(d_lo - nu^- tau)^2/(2 sigma^2 tau))."""
    nu = r - y - 0.5 * sigma**2
    out = 0.0
    for d, drift in ((d_hi, max(nu, 0.0)), (d_lo, max(-nu, 0.0))):
        if np.isfinite(d):
            m = d - drift * tau
            out += np.exp(-m * m / (2 * sigma**2 * tau)) if m > 0 else 1.0
    return float(min(out, 1.0))
