"""Multilevel Monte Carlo (Chapter 12).

Levels l = 0, 1, ..., L of a time discretisation with M^l uniform steps (h_l = T M^-l,
M = 2 by default). The level-l sample is the coupled difference

    Y_0 = P_0,    Y_l = P_l^f - P_{l-1}^c   (l >= 1),

of the payoff computed on a fine path with M^l steps and on a coarse path with M^(l-1)
steps, both driven by the same Brownian path: each coarse increment is the sum of M
consecutive fine increments (``paths.coarsen``). The coarse path therefore has exactly the
law of a level-(l-1) path, which is what makes the telescoping sum unbiased for E[P_L].

Costs are counted in time steps, not seconds: one sample of Y_0 costs C_0 = 1 step and one
sample of Y_l costs C_l = M^l + M^(l-1) (fine and coarse path). This makes every number
reproducible; wall-clock costs of vectorised code are not proportional to the number of
steps for small arrays.

Contents
--------
* Level samplers: ``GBMLevels`` (vectorised Euler or Milstein for geometric Brownian
  motion, European or path-dependent payoffs) and ``SDELevels`` (any scalar SDE, through
  ``paths.euler`` / ``paths.milstein``). A sampler is called as ``sampler(l, n, rng)`` and
  returns the arrays ``(Pf, Pc)`` of fine and coarse payoffs (``Pc = 0`` at level 0).
* ``level_statistics``: means and variances of Y_l and P_l, level by level.
* ``optimal_samples``, ``mlmc_fixed``: the estimator of prop:ml-telescoping for given levels.
* ``mlmc``: the adaptive algorithm (Giles 2008, Section 5; Giles 2015, Section 3.1).
* ``complexity_*``: the cost bounds of thm:ml-complexity and the budget comparison with the
  Duffie-Glynn allocation of Chapter 7.
* ``gbm_level_exact``: exact mean and variance of Y_l for the payoff S_T (Euler and
  Milstein), from Gaussian moments.
* ``single_term``, ``coupled_sum``: the randomised unbiased estimators of Rhee and Glynn.

Statements of the notes are cited by their LaTeX labels (e.g. thm:ml-complexity).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from . import bs
from .paths import coarsen

TEST = bs.TEST_CASE       # S = K = 100, T = 1, r = 1%, y = 2%, sigma = 20%


# ---------------------------------------------------------------------------------------
# Costs and payoffs
# ---------------------------------------------------------------------------------------


def level_cost(l: int, M: int = 2) -> float:
    """Cost of one sample of Y_l in time steps: 1 at level 0, M^l + M^(l-1) above."""
    return 1.0 if l == 0 else float(M**l + M ** (l - 1))


def call_payoff(K=100.0, discount=1.0):
    """x -> discount * (x - K)^+."""
    return lambda x: discount * np.maximum(x - K, 0.0)


def digital_payoff(K=100.0, discount=1.0):
    """x -> discount * 1{x > K}."""
    return lambda x: discount * (x > K).astype(float)


def asian_call_payoff(K=100.0, discount=1.0):
    """Path functional discount * (A - K)^+ with A the trapezoidal time average of the path.

    ``S`` has shape (n, m + 1) on a uniform grid of [0, T]; the trapezoid average
    (S_0/2 + S_1 + ... + S_{m-1} + S_m/2)/m approximates (1/T) int_0^T S_t dt. Fine and
    coarse paths each use their own grid, so the level-(l-1) value is the same function of
    the path whether it is computed as a fine or as a coarse path (telescoping condition).
    """
    def f(S):
        m = S.shape[1] - 1
        A = (S[:, 1:-1].sum(axis=1) + 0.5 * (S[:, 0] + S[:, -1])) / m
        return discount * np.maximum(A - K, 0.0)
    f.path = True
    return f


# ---------------------------------------------------------------------------------------
# Level samplers
# ---------------------------------------------------------------------------------------


class GBMLevels:
    """Coupled fine/coarse samples of a payoff of dS = mu S dt + sigma S dW, mu = r - y.

    ``scheme`` is "euler" (factor 1 + mu h + sigma dW per step) or "milstein" (plus
    sigma^2 (dW^2 - h)/2). ``payoff`` maps terminal values to payoffs, or, if it has the
    attribute ``path = True`` (e.g. ``asian_call_payoff``), whole paths of shape
    (n, m + 1) to payoffs. ``h0`` steps at level 0 are T / M^0 = T unless ``n0_steps`` > 1.
    Calling ``sampler(l, n, rng)`` returns (Pf, Pc), two arrays of length n.
    """

    def __init__(self, payoff=None, S0=100.0, r=0.01, y=0.02, sigma=0.2, T=1.0,
                 scheme="euler", M=2, n0_steps=1, chunk=2**21):
        if scheme not in ("euler", "milstein"):
            raise ValueError("scheme must be 'euler' or 'milstein'")
        self.payoff = payoff if payoff is not None else call_payoff(100.0, np.exp(-r * T))
        self.S0, self.mu, self.sigma, self.T = float(S0), float(r - y), float(sigma), float(T)
        self.scheme, self.M, self.n0, self.chunk = scheme, int(M), int(n0_steps), int(chunk)
        self.path = bool(getattr(self.payoff, "path", False))

    def steps(self, l: int) -> int:
        return self.n0 * self.M**l

    def cost(self, l: int) -> float:
        """Steps per sample of Y_l (fine plus coarse)."""
        return float(self.steps(l) + (self.steps(l - 1) if l > 0 else 0))

    def terminal_pair(self, l: int, n: int, rng):
        """Terminal values (S_f, S_c) of n coupled fine/coarse paths (S_c = nan at level 0),
        so that several European payoffs can be evaluated on the same paths."""
        m = self.steps(l)
        h = self.T / m
        Sf, Sc = np.empty(n), np.full(n, np.nan)
        rows = max(1, self.chunk // m)
        for i in range(0, n, rows):
            k = min(rows, n - i)
            dW = np.sqrt(h) * rng.standard_normal((k, m))
            Sf[i:i + k] = self.S0 * np.prod(self._factors(dW, h), axis=1)
            if l > 0:
                Sc[i:i + k] = self.S0 * np.prod(self._factors(coarsen(dW, self.M), self.M * h), axis=1)
        return Sf, Sc

    def _factors(self, dW, h):
        f = 1.0 + self.mu * h + self.sigma * dW
        if self.scheme == "milstein":
            f += 0.5 * self.sigma**2 * (dW * dW - h)
        return f

    def _evaluate(self, dW, h):
        f = self._factors(dW, h)
        if self.path:
            S = np.empty((dW.shape[0], dW.shape[1] + 1))
            S[:, 0] = self.S0
            np.cumprod(f, axis=1, out=S[:, 1:])
            S[:, 1:] *= self.S0
            return self.payoff(S)
        return self.payoff(self.S0 * np.prod(f, axis=1))

    def __call__(self, l: int, n: int, rng):
        m = self.steps(l)
        h = self.T / m
        Pf, Pc = np.empty(n), np.zeros(n)
        rows = max(1, self.chunk // m)
        for i in range(0, n, rows):
            k = min(rows, n - i)
            dW = np.sqrt(h) * rng.standard_normal((k, m))
            Pf[i:i + k] = self._evaluate(dW, h)
            if l > 0:
                Pc[i:i + k] = self._evaluate(coarsen(dW, self.M), self.M * h)
        return Pf, Pc


class SDELevels:
    """Coupled samples for a scalar SDE dX = a(t, X) dt + b(t, X) dW, by ``paths.euler`` or
    ``paths.milstein`` (pass ``b_x`` for Milstein). ``payoff`` acts on X_T, or on the whole
    path if it has ``path = True``. Slower than ``GBMLevels`` (a Python loop over steps)."""

    def __init__(self, a, b, x0, T, payoff, scheme="euler", b_x=None, M=2):
        from . import paths as _paths
        self._paths = _paths
        if scheme == "milstein" and b_x is None:
            raise ValueError("Milstein needs b_x")
        self.a, self.b, self.b_x, self.x0, self.T = a, b, b_x, x0, float(T)
        self.payoff, self.scheme, self.M = payoff, scheme, int(M)
        self.path = bool(getattr(payoff, "path", False))

    def cost(self, l):
        return level_cost(l, self.M)

    def _run(self, dW):
        m = dW.shape[1]
        t = np.linspace(0.0, self.T, m + 1)
        if self.scheme == "euler":
            X = self._paths.euler(self.a, self.b, self.x0, t, dW=dW)
        else:
            X = self._paths.milstein(self.a, self.b, self.b_x, self.x0, t, dW=dW)
        return self.payoff(X) if self.path else self.payoff(X[:, -1])

    def __call__(self, l, n, rng):
        m = self.M**l
        dW = np.sqrt(self.T / m) * rng.standard_normal((n, m))
        Pf = self._run(dW)
        Pc = self._run(coarsen(dW, self.M)) if l > 0 else np.zeros(n)
        return Pf, Pc


# ---------------------------------------------------------------------------------------
# Level statistics and the estimator for given levels
# ---------------------------------------------------------------------------------------


def _cost(sampler, l):
    return sampler.cost(l) if hasattr(sampler, "cost") else level_cost(l)


def level_statistics(sampler, levels, n, seed=None) -> dict:
    """Sample means and variances of Y_l = P_l^f - P_{l-1}^c and of P_l^f, level by level.

    ``n`` is an int or one sample size per level. Level l uses its own generator
    ``default_rng([seed, l])``, so that any single level can be recomputed. Returns arrays
    ``mean`` (E Y_l), ``var`` (V_l), ``mean_P``, ``var_P`` (Var P_l), ``kurt`` (kurtosis
    of Y_l, which measures how reliable the variance estimate is), ``cost`` (C_l), ``n``.
    """
    levels = list(levels)
    ns = np.broadcast_to(np.asarray(n), (len(levels),)).astype(int)
    out = {k: np.zeros(len(levels)) for k in ("mean", "var", "mean_P", "var_P", "kurt", "cost")}
    for i, (l, k) in enumerate(zip(levels, ns)):
        rng = np.random.default_rng(None if seed is None else [seed, l])
        Pf, Pc = sampler(l, int(k), rng)
        Y = Pf - Pc
        out["mean"][i], out["var"][i] = Y.mean(), Y.var(ddof=1)
        out["mean_P"][i], out["var_P"][i] = Pf.mean(), Pf.var(ddof=1)
        c = Y - Y.mean()
        v2 = np.mean(c**2)
        out["kurt"][i] = np.mean(c**4) / v2**2 if v2 > 0 else np.nan
        out["cost"][i] = _cost(sampler, l)
    out["levels"], out["n"] = np.array(levels), ns.copy()
    return out


def optimal_samples(V, C, eps, theta=0.5) -> np.ndarray:
    """Integer sample sizes N_l = max(1, ceil(sqrt(V_l/C_l) sum_k sqrt(V_k C_k) / ((1-theta) eps^2))).

    For these N_l the variance sum V_l/N_l is at most (1 - theta) eps^2, and the cost
    sum N_l C_l is at most (sum sqrt(V_l C_l))^2 / ((1-theta) eps^2) + sum C_l
    (prop:ml-telescoping and the proof of thm:ml-complexity). Levels with V_l = 0 get one sample.
    """
    V, C = np.maximum(np.asarray(V, dtype=float), 0.0), np.asarray(C, dtype=float)
    s = np.sum(np.sqrt(V * C))
    x = np.sqrt(V / C) * s / ((1.0 - theta) * eps**2)
    return np.maximum(1, np.ceil(x)).astype(np.int64)


def mlmc_fixed(sampler, N, seed=None) -> dict:
    """Multilevel estimator sum_l mean(Y_l) with N[l] independent samples on level l.

    Returns ``estimate``, the plug-in variance sum V_l/N_l and standard error, the level
    means and variances, and the cost sum N_l C_l in steps.
    """
    rng = np.random.default_rng(seed)
    means, var = np.zeros(len(N)), np.zeros(len(N))
    cost = 0.0
    for l, n in enumerate(N):
        Pf, Pc = sampler(l, int(n), rng)
        Y = Pf - Pc
        means[l] = Y.mean()
        var[l] = Y.var(ddof=1) if n > 1 else 0.0
        cost += n * _cost(sampler, l)
    v = float(np.sum(var / np.asarray(N, dtype=float)))
    return dict(estimate=float(means.sum()), variance=v, se=np.sqrt(v), means=means, var=var,
                cost=cost, N=np.asarray(N))


# ---------------------------------------------------------------------------------------
# The adaptive algorithm
# ---------------------------------------------------------------------------------------


def _fit_rate(levels, values, floor=0.0):
    """Least-squares slope of -log2 |values| against the level (levels >= 1)."""
    lv, vv = np.asarray(levels, dtype=float), np.abs(np.asarray(values, dtype=float))
    ok = (lv >= 1) & (vv > 0)
    if ok.sum() < 2:
        return None
    return max(floor, -np.polyfit(lv[ok], np.log2(vv[ok]), 1)[0])


def mlmc(sampler, eps, alpha=None, beta=None, N0=1000, L0=2, Lmax=14, theta=0.5,
         seed=None, history=False) -> dict:
    """Adaptive multilevel Monte Carlo for a root-mean-square accuracy eps.

    Giles (2008), Section 5, as reorganised in Giles (2015), Section 3.1:
      1. start with levels 0..L0 and N0 samples on each;
      2. estimate V_l from the samples and set N_l by ``optimal_samples`` (variance at most
         (1 - theta) eps^2); draw the missing samples;
      3. when no level needs more samples, estimate the remaining bias by
         max(|m_L|, |m_{L-1}|/2^alpha, |m_{L-2}|/4^alpha)/(2^alpha - 1), m_l the sample mean
         of Y_l (a heuristic, exact only if E[P_l - P] = c 2^(-alpha l)); if it exceeds
         sqrt(theta) eps, add level L+1, with V_{L+1} = V_L / 2^beta as initial guess.
    alpha and beta are estimated by regression on the levels l >= 1 when not given
    (floored at 1/2). Variance estimates on levels l >= 2 are not allowed to fall below
    half the value extrapolated from the level below, which guards against a sample that
    happens to contain no large differences. Returns the estimate, its standard error,
    the final L, N_l, level means and variances, the cost in steps, and whether the bias
    test was passed before Lmax (``converged``).
    """
    rng = np.random.default_rng(seed)
    L = L0
    S1, S2, N = [np.zeros(L + 1) for _ in range(3)]
    C = np.array([_cost(sampler, l) for l in range(L + 1)])
    dN = np.full(L + 1, int(N0), dtype=np.int64)
    log = []
    converged = False
    while dN.sum() > 0:
        for l in range(L + 1):
            if dN[l] > 0:
                Pf, Pc = sampler(l, int(dN[l]), rng)
                Y = Pf - Pc
                S1[l] += Y.sum(); S2[l] += (Y * Y).sum(); N[l] += dN[l]
        m = S1 / N
        V = np.maximum(S2 / N - m**2, 0.0) * N / np.maximum(N - 1, 1)
        a = alpha if alpha is not None else (_fit_rate(range(L + 1), m, 0.5) or 1.0)
        b = beta if beta is not None else (_fit_rate(range(L + 1), V, 0.5) or 1.0)
        for l in range(2, L + 1):
            V[l] = max(V[l], 0.5 * V[l - 1] / 2.0**b)
        Ns = optimal_samples(V, C, eps, theta)
        dN = np.maximum(0, Ns - N.astype(np.int64))
        if history:
            log.append(dict(L=L, N=N.copy(), V=V.copy(), m=m.copy()))
        if np.all(dN <= 0.01 * N):
            tail = [abs(m[L])] + [abs(m[L - j]) / 2.0 ** (j * a) for j in (1, 2) if L - j >= 1]
            rem = max(tail) / (2.0**a - 1.0)
            if rem <= np.sqrt(theta) * eps:
                converged = True
                break
            if L == Lmax:
                break
            L += 1
            S1, S2, N = (np.append(x, 0.0) for x in (S1, S2, N))
            C = np.append(C, _cost(sampler, L))
            V = np.append(V, V[-1] / 2.0**b)
            Ns = optimal_samples(V, C, eps, theta)
            dN = np.maximum(0, Ns - N.astype(np.int64))
            dN[L] = max(dN[L], 1)
    m = S1 / N
    V = np.maximum(S2 / N - m**2, 0.0) * N / np.maximum(N - 1, 1)
    var = float(np.sum(V / N))
    out = dict(estimate=float(m.sum()), variance=var, se=np.sqrt(var), L=L, N=N.astype(np.int64),
               means=m, var=V, cost=float(np.sum(N * C)), converged=converged, alpha=a, beta=b)
    if history:
        out["history"] = log
    return out


def standard_mc_cost(var_P_L, L, eps, theta=0.5, M=2, n0_steps=1):
    """Cost in steps of plain Monte Carlo on the finest level L with variance (1-theta) eps^2:
    ceil(var(P_L) / ((1 - theta) eps^2)) paths of n0 M^L steps."""
    return np.ceil(var_P_L / ((1.0 - theta) * eps**2)) * n0_steps * M**L


# ---------------------------------------------------------------------------------------
# Complexity: thm:ml-complexity and the comparison with the Duffie-Glynn allocation
# ---------------------------------------------------------------------------------------


def complexity_levels(eps, c1, alpha, theta=0.5):
    """Finest level L = max(0, ceil(log2(c1 / (sqrt(theta) eps)) / alpha)) of the proof of
    thm:ml-complexity: the smallest L with c1 2^(-alpha L) <= sqrt(theta) eps."""
    x = np.log2(c1 / (np.sqrt(theta) * eps)) / alpha
    return int(max(0, np.ceil(x - 1e-12)))


def complexity_bound(eps, alpha, beta, gamma, c1=1.0, c2=1.0, c3=1.0, theta=0.5):
    """The explicit cost bound of the proof of thm:ml-complexity (with the integer ceilings):
    (sum_{l<=L} sqrt(c2 c3 2^((gamma-beta) l)))^2 / ((1-theta) eps^2) + sum_{l<=L} c3 2^(gamma l)."""
    L = complexity_levels(eps, c1, alpha, theta)
    l = np.arange(L + 1)
    s = np.sum(np.sqrt(c2 * c3 * 2.0 ** ((gamma - beta) * l)))
    return float(s**2 / ((1.0 - theta) * eps**2) + np.sum(c3 * 2.0 ** (gamma * l))), L


def rmse_at_budget(V, C, bias, budget) -> dict:
    """Best root-mean-square error of a multilevel estimator with a given budget.

    For each finest level L, the variance-minimising continuous allocation
    N_l = budget sqrt(V_l/C_l) / sum_k sqrt(V_k C_k) (prop:ml-telescoping(iii)) gives the
    variance (sum_{l<=L} sqrt(V_l C_l))^2 / budget; ``bias[L]`` is E[P_L] - E[P]. The
    returned L minimises bias^2 + variance; N are the (continuous) sample sizes.
    """
    V, C, bias = (np.asarray(x, dtype=float) for x in (V, C, bias))
    best = None
    for L in range(len(V)):
        s = np.sum(np.sqrt(V[:L + 1] * C[:L + 1]))
        mse = bias[L] ** 2 + s**2 / budget
        if best is None or mse < best["mse"]:
            N = budget * np.sqrt(V[:L + 1] / C[:L + 1]) / s
            best = dict(L=L, mse=mse, rmse=np.sqrt(mse), N=N, bias=bias[L], sd=s / np.sqrt(budget))
    return best


# ---------------------------------------------------------------------------------------
# Exact level moments for GBM and the payoff S_T
# ---------------------------------------------------------------------------------------


def _pair_moments(scheme, mu, sigma, h, M=2):
    """Exact Gaussian moments over one coarse step (M fine steps): with F the product of
    the M fine factors, G the coarse factor and Delta = F - G, returns E G, E G^2,
    E Delta, e = E[Delta G], q = E[Delta^2] and E F^2, by Gauss-Hermite quadrature (exact:
    the integrands are polynomials). Delta is formed node by node, so the small moments
    e and q carry no cancellation error."""
    x, w = hermegauss(8)
    w = w / np.sqrt(2.0 * np.pi)

    def fac(d, step):
        f = 1.0 + mu * step + sigma * d
        if scheme == "milstein":
            f = f + 0.5 * sigma**2 * (d * d - step)
        return f

    grids = np.meshgrid(*([x] * M), indexing="ij")
    weight = np.ones_like(grids[0])
    for wg in np.meshgrid(*([w] * M), indexing="ij"):
        weight = weight * wg
    d = [np.sqrt(h) * g for g in grids]
    F = np.ones_like(grids[0])
    for di in d:
        F = F * fac(di, h)
    G = fac(sum(d), M * h)
    Dl = F - G
    E = lambda z: float(np.sum(weight * z))  # noqa: E731
    return dict(g=E(G), gg=E(G * G), d=E(Dl), e=E(Dl * G), q=E(Dl * Dl), ff=E(F * F))


def _powers_combination(v, e, q, n):
    """(v + 2e + q)^n - 2 (v + e)^n + v^n, computed without cancellation."""
    return v**n * (np.expm1(n * np.log1p((2.0 * e + q) / v)) - 2.0 * np.expm1(n * np.log1p(e / v)))


def gbm_level_exact(l, scheme="euler", S0=100.0, mu=-0.01, sigma=0.2, T=1.0, discount=1.0, M=2):
    """Exact moments of the level-l sample for the payoff discount * S_T of GBM.

    The fine path has M^l steps, the coarse path M^(l-1) steps on the summed increments.
    The n = M^(l-1) coarse steps contribute independent factors, so every moment is a power
    of a one-step Gaussian moment: E[Y_l^2] = D^2 S0^2 ((v+2e+q)^n - 2(v+e)^n + v^n) with
    v = E G^2, e = E[(F-G) G], q = E[(F-G)^2] (``_pair_moments``). Returns ``mean_f`` = E P_l,
    ``mean`` = E Y_l, ``var`` = V_l = Var Y_l, ``var_P`` = Var P_l, ``bias`` = E P_l - E P.
    """
    mexact = discount * S0 * np.exp(mu * T)
    m = M**l
    h = T / m
    if l == 0:
        a = 1.0 + mu * T
        second = a * a + sigma**2 * T + (0.5 * sigma**4 * T * T if scheme == "milstein" else 0.0)
        mean_f = discount * S0 * a
        var_f = discount**2 * S0**2 * second - mean_f**2
        return dict(mean_f=mean_f, mean=mean_f, var=var_f, var_P=var_f, bias=mean_f - mexact)
    p = _pair_moments(scheme, mu, sigma, h, M)
    n = m // M
    DS = discount * S0
    g = 1.0 + M * mu * h                                                 # E G exactly
    dmean = (1.0 + mu * h) ** M - g                                      # E F - E G exactly
    Ec = DS * np.exp(n * np.log1p(M * mu * h))
    EY = Ec * np.expm1(n * np.log1p(dmean / g))
    EY2 = DS**2 * _powers_combination(p["gg"], p["e"], p["q"], n)
    Ef = Ec + EY
    return dict(mean_f=Ef, mean=EY, var=EY2 - EY**2, var_P=DS**2 * p["ff"] ** n - Ef**2,
                bias=Ef - mexact)


def euler_gbm_level_variance(l, S0=100.0, mu=-0.01, sigma=0.2, T=1.0, discount=1.0):
    """Closed form of V_l for Euler, payoff D S_T with D = discount, M = 2 (exa:ml-gbm-exact):

        V_l = D^2 S0^2 [ u^n - 2 w^n + v^n - (a^(2n) - A^n)^2 ],
        u = (a^2 + s^2 h)^2,  w = a^2 A + 2 a s^2 h,  v = A^2 + 2 s^2 h,
    with h = T 2^-l, n = 2^(l-1) coarse steps, a = 1 + mu h, A = 1 + 2 mu h, s = sigma.
    Evaluated through u - 2w + v = s^4 h^2 + 2 mu^2 s^2 h^3 + mu^4 h^4 and
    w - v = (A mu^2 + 2 mu s^2) h^2, which avoids the cancellation of the three powers.
    """
    if l == 0:
        return discount**2 * S0**2 * sigma**2 * T
    h, n = T / 2**l, 2 ** (l - 1)
    A, s2 = 1.0 + 2.0 * mu * h, sigma**2
    v = A * A + 2.0 * s2 * h
    e = (A * mu**2 + 2.0 * mu * s2) * h * h
    q = s2 * s2 * h * h + 2.0 * mu**2 * s2 * h**3 + mu**4 * h**4
    dmean = A**n * np.expm1(n * np.log1p(mu**2 * h * h / A))             # a^(2n) - A^n
    return discount**2 * S0**2 * (_powers_combination(v, e, q, n) - dmean**2)


# ---------------------------------------------------------------------------------------
# Randomised unbiased estimators (Rhee and Glynn)
# ---------------------------------------------------------------------------------------


def geometric_level_probabilities(r, n_max):
    """p_n = P(N = n) = (1 - 2^-r) 2^(-r n), n = 0..n_max, and tails P(N >= n) = 2^(-r n)."""
    n = np.arange(n_max + 1)
    return (1.0 - 2.0**-r) * 2.0 ** (-r * n), 2.0 ** (-r * n)


def single_term(sampler, n, r, seed=None, max_level=None) -> dict:
    """Rhee-Glynn single-term estimator with geometric randomisation P(J = k) ∝ 2^(-r k).

    Draws J_1, ..., J_n independently, then for each level k the samples Y_k needed, and
    returns the samples Z_i = Y_{J_i} / P(J = J_i), their mean, standard error, the levels
    drawn and the mean cost per sample (in steps). Unbiased for E P = lim E P_l when the
    second moment sum_k E[Y_k^2]/p_k is finite (prop:ml-unbiased). With ``max_level`` = M,
    J is replaced by min(J, M), whose law puts the tail mass 2^(-r M) on M: the estimator is
    then unbiased for E P_M only, but its cost per sample is bounded (for demonstrations).
    """
    rng = np.random.default_rng(seed)
    q = 1.0 - 2.0**-r
    levels = rng.geometric(q, size=n) - 1          # P(levels = k) = q (1 - q)^k = q 2^(-r k)
    if max_level is not None:
        levels = np.minimum(levels, int(max_level))
    Z = np.empty(n)
    cost = 0.0
    for k in np.unique(levels):
        idx = np.flatnonzero(levels == k)
        Pf, Pc = sampler(int(k), idx.size, rng)
        pk = 2.0 ** (-r * k) if (max_level is not None and k == max_level) else q * 2.0 ** (-r * k)
        Z[idx] = (Pf - Pc) / pk
        cost += idx.size * _cost(sampler, int(k))
    return dict(mean=float(Z.mean()), se=float(Z.std(ddof=1) / np.sqrt(n)), Z=Z, levels=levels,
                mean_cost=cost / n)


def coupled_sum(sampler, n, r, seed=None, max_level=None) -> dict:
    """Rhee-Glynn coupled-sum estimator Z = sum_{k=0}^J Y_k / P(J >= k), with P(J >= k) = 2^(-r k).

    All the differences Y_k = P_k - P_{k-1} of one sample are computed on ONE Brownian path
    with M^J steps (coarsened level by level), so they are dependent. Requires a sampler
    with the attributes of ``GBMLevels`` (terminal-value payoffs). Returns the samples,
    their mean and standard error, and the mean cost (sum_{k<=J} M^k steps per sample).
    With ``max_level`` = M, J is replaced by min(J, M): the estimator is then unbiased for
    E P_M only, but its cost per sample is bounded (for demonstrations).
    """
    if not isinstance(sampler, GBMLevels) or sampler.path:
        raise ValueError("coupled_sum needs a GBMLevels sampler with a terminal payoff")
    rng = np.random.default_rng(seed)
    q = 1.0 - 2.0**-r
    levels = rng.geometric(q, size=n) - 1
    if max_level is not None:
        levels = np.minimum(levels, int(max_level))
    Z = np.zeros(n)
    cost = 0.0
    for kmax in np.unique(levels):
        idx = np.flatnonzero(levels == kmax)
        m = sampler.steps(int(kmax))
        rows = max(1, sampler.chunk // m)
        for start in range(0, idx.size, rows):
            sub = idx[start:start + rows]
            d = np.sqrt(sampler.T / m) * rng.standard_normal((sub.size, m))
            vals = []                        # P_kmax, P_{kmax-1}, ..., P_0 on this path
            for k in range(int(kmax), -1, -1):
                vals.append(sampler._evaluate(d, sampler.T / d.shape[1]))
                if k > 0:
                    d = coarsen(d, sampler.M)
            vals = vals[::-1]                # vals[k] = P_k
            acc = np.zeros(sub.size)
            for k in range(int(kmax) + 1):
                acc += (vals[k] - (vals[k - 1] if k > 0 else 0.0)) / 2.0 ** (-r * k)
            Z[sub] = acc
        cost += idx.size * sum(sampler.steps(k) for k in range(int(kmax) + 1))
    return dict(mean=float(Z.mean()), se=float(Z.std(ddof=1) / np.sqrt(n)), Z=Z, levels=levels,
                mean_cost=cost / n)


def single_term_work(second_moments, costs, r, mean=0.0, tail_rate=None):
    """Variance and expected cost of the single-term estimator for P(N = n) = (1-2^-r) 2^(-r n).

    ``second_moments[n]`` = E[Y_n^2] and ``costs[n]`` = C_n for n = 0..n_max. If
    ``tail_rate = (beta, gamma)`` is given, the sums are continued beyond n_max with
    E[Y_n^2] = E[Y_{n_max}^2] 2^(-beta (n - n_max)) and C_n = C_{n_max} 2^(gamma (n - n_max))
    (geometric tails, summed in closed form; infinite when they diverge). Returns
    (variance, expected cost, their product).
    """
    m, C = np.asarray(second_moments, dtype=float), np.asarray(costs, dtype=float)
    p, _ = geometric_level_probabilities(r, m.size - 1)
    second = np.sum(m / p)
    cost = np.sum(p * C)
    if tail_rate is not None:
        b, g = tail_rate
        rv, rc = 2.0 ** (r - b), 2.0 ** (g - r)     # ratios of consecutive tail terms
        second += np.inf if rv >= 1 else m[-1] / p[-1] * rv / (1 - rv)
        cost += np.inf if rc >= 1 else p[-1] * C[-1] * rc / (1 - rc)
    var = second - mean**2
    return var, cost, var * cost
