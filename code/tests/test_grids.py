"""Checks of the Chapter 20 code against the results proved in the notes and the numbers they quote.

Everything here is deterministic. The whole file runs in about fifteen seconds.
"""
import numpy as np
import pytest
from scipy.linalg import solve_banded

from mfe230d import american, bs
from mfe230d import grids as G

P = bs.TEST_CASE
S, K, T, r, y, SIG = (P[k] for k in ("S", "K", "T", "r", "y", "sigma"))
CALL = float(bs.call(S, K, T, r, y, SIG))
PUT = float(bs.put(S, K, T, r, y, SIG))


# ------------------------------------------------------------------ binomial trees
def test_tree_validity_conditions():
    # CRR: q in (0, 1) iff n > (r - y)^2 T / sigma^2
    for sig, rr, yy in ((0.011, 0.05, 0.0), (0.0023, 0.0, 0.01), (0.2, 0.01, 0.02)):
        n0 = G.crr_min_steps(T, rr, yy, sig)
        assert G.tree_parameters(T, rr, yy, sig, n0)["valid"]
        if n0 > 1:
            assert not G.tree_parameters(T, rr, yy, sig, n0 - 1)["valid"]
    # r = y: q = 1/(1 + u) for every n
    p = G.tree_parameters(T, 0.03, 0.03, 0.2, 7)
    assert p["q"] == pytest.approx(1 / (1 + p["u"]), rel=1e-14)
    # Jarrow--Rudd: valid iff sigma sqrt(dt) < 2, whatever r - y
    assert G.tree_parameters(1.0, 0.5, 0.0, 0.01, 1, "jr")["valid"]
    assert not G.tree_parameters(1.0, 0.0, 0.0, 2.5, 1, "jr")["valid"]
    # CRR expansion q = 1/2 + nu sqrt(dt) / (2 sigma) + O(dt)
    nu = r - y - SIG**2 / 2
    for n in (100, 10_000):
        q = G.tree_parameters(T, r, y, SIG, n)["q"]
        assert abs(q - 0.5 - nu * np.sqrt(T / n) / (2 * SIG)) < 0.5 * T / n


def test_terminal_law_equals_backward_induction():
    for method, n in (("crr", 60), ("jr", 60), ("lr", 61)):
        for kind in ("call", "put"):
            a = G.binomial_european(S, 95.0, T, r, y, SIG, n, kind, method)
            b = G.binomial(S, 95.0, T, r, y, SIG, n, kind, "european", method)
            assert a == pytest.approx(b, rel=1e-12)
    a = G.binomial_european(S, K, T, r, y, SIG, 80, "call", smooth=True)
    b = G.binomial(S, K, T, r, y, SIG, 80, "call", "european", smooth=True)
    assert a == pytest.approx(b, rel=1e-12)


def test_crr_converges_with_parity_and_oscillation():
    # the numbers of the notes: n (C_n - C) at n = 1000, 1001 and the coefficient B
    assert 1000 * (G.binomial_european(S, K, T, r, y, SIG, 1000) - CALL) == pytest.approx(-1.96392, abs=1e-5)
    assert 1001 * (G.binomial_european(S, K, T, r, y, SIG, 1001) - CALL) == pytest.approx(1.94162, abs=1e-5)
    assert G.crr_strike_offset(S, K, T, SIG, 1000) == 0.0
    assert G.crr_strike_offset(S, K, T, SIG, 1001) == pytest.approx(0.5)
    B = G.crr_oscillation_coefficient(S, K, T, r, y, SIG)
    assert B == pytest.approx(15.6222, abs=1e-4)
    # n (C_n - C) = A + B theta (1 - theta) + o(1): fit A and B on n = 1000..1400, K = 110
    KK = 110.0
    c = float(bs.call(S, KK, T, r, y, SIG))
    ns = np.arange(1000, 1401)
    e = np.array([n * (G.binomial_european(S, KK, T, r, y, SIG, int(n)) - c) for n in ns])
    th = np.array([G.crr_strike_offset(S, KK, T, SIG, int(n)) for n in ns])
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(th), th * (1 - th)]), e, rcond=None)
    assert coef[1] == pytest.approx(G.crr_oscillation_coefficient(S, KK, T, r, y, SIG), rel=2e-3)
    assert G.crr_oscillation_coefficient(S, KK, T, r, y, SIG) == pytest.approx(14.2815, abs=1e-4)


def test_smoothing_richardson_and_leisen_reimer():
    # BBS: parity effect 0.65 vs 0.627 per n; BBSR along multiples of 4 is second order
    e400 = 400 * (G.binomial_european(S, K, T, r, y, SIG, 400, smooth=True) - CALL)
    e401 = 401 * (G.binomial_european(S, K, T, r, y, SIG, 401, smooth=True) - CALL)
    assert e400 == pytest.approx(0.6498, abs=1e-4) and e401 == pytest.approx(0.6266, abs=1e-4)
    errs = [abs(G.bbsr(S, K, T, r, y, SIG, n, "call", "european") - CALL) for n in (100, 200, 400, 800)]
    assert np.all(np.array(errs[:-1]) / np.array(errs[1:]) > 3.8)
    # Leisen--Reimer: n^2 (C_n - C) -> -0.398 (order two)
    for n, v in ((201, -0.3955), (801, -0.3978)):
        assert n * n * (G.binomial_european(S, K, T, r, y, SIG, n, method="lr") - CALL) == pytest.approx(v, abs=5e-4)
    with pytest.raises(ValueError):
        G.tree_parameters(T, r, y, SIG, 100, "lr", S, K)


def test_trees_agree_with_chapter_14():
    # same CRR algorithm as american.binomial, and the BBSR reference
    for kind in ("put", "call"):
        assert G.binomial(S, K, T, r, y, SIG, 300, kind) == pytest.approx(
            american.binomial(S, K, T, r, y, SIG, 300, kind), rel=1e-11)
        assert G.bbsr(S, K, T, r, y, SIG, 400, kind) == pytest.approx(
            american.american_binomial(S, K, T, r, y, SIG, 400, kind), rel=1e-11)
    # the converged test-case values of Chapter 14
    assert G.bbsr(S, K, T, r, y, SIG, 2000, "put") == pytest.approx(8.349408, abs=2e-6)
    assert G.binomial(S, K, T, r, y, SIG, 2001, "put", method="lr") == pytest.approx(8.349408, abs=5e-6)
    assert G.bbsr(S, K, T, r, y, SIG, 2000, "call") == pytest.approx(7.4597, abs=1e-4)


# ------------------------------------------------------------------ trinomial trees
def test_trinomial_moments_positivity_and_kp2():
    for lam in (1.0, np.sqrt(1.5), np.sqrt(3.0)):
        p = G.trinomial_probabilities(T, r, y, SIG, 50, lam)
        nu = r - y - SIG**2 / 2
        assert p["pu"] + p["pm"] + p["pd"] == pytest.approx(1.0)
        assert p["h"] * (p["pu"] - p["pd"]) == pytest.approx(nu * p["dt"], rel=1e-12)
        assert p["h"] ** 2 * (p["pu"] + p["pd"]) == pytest.approx(SIG**2 * p["dt"], rel=1e-12)
        assert p["valid"]
    assert not G.trinomial_probabilities(T, r, y, SIG, 50, 0.9)["valid"]
    # course key point: nu = 0 and lam = sqrt(3) give (1/6, 2/3, 1/6) and the fourth moment 3 sigma^4 dt^2
    p = G.trinomial_probabilities(1.0, 0.02, 0.0, 0.2, 10, np.sqrt(3.0))
    assert (p["pu"], p["pm"], p["pd"]) == pytest.approx((1 / 6, 2 / 3, 1 / 6), rel=1e-12)
    assert p["h"] ** 4 * (p["pu"] + p["pd"]) == pytest.approx(3 * (0.2**2 * p["dt"]) ** 2, rel=1e-12)


def test_trinomial_is_the_explicit_scheme():
    n = 120
    for lam in (1.0, np.sqrt(1.5), np.sqrt(3.0)):
        tri = G.trinomial(S, K, T, r, y, SIG, n, "call", lam=lam, discount="scheme")
        h = lam * SIG * np.sqrt(T / n)
        J = 2 * n + 4
        fd = G.fd_solve(S, K, T, r, y, SIG, J, n, theta=0.0, width=J * h / (2 * SIG * np.sqrt(T)))
        assert fd["lam"] == pytest.approx(1 / (2 * lam**2))
        assert tri == pytest.approx(fd["price"], abs=1e-11)
    # first-order convergence of the KR tree (the numbers of the notes) and the explosion for lam < 1
    for lam, v in ((1.0, -1.97), (np.sqrt(1.5), -0.53), (np.sqrt(3.0), -2.06)):
        assert 400 * (G.trinomial(S, K, T, r, y, SIG, 400, "call", lam=lam) - CALL) == pytest.approx(v, abs=0.01)
        assert 1600 * (G.trinomial(S, K, T, r, y, SIG, 1600, "call", lam=lam) - CALL) == pytest.approx(v, abs=0.01)
    assert abs(G.trinomial(S, K, T, r, y, SIG, 200, "call", lam=0.9)) > 1e20


# ------------------------------------------------------------------ stability and positivity
def test_von_neumann_factors():
    xi = np.linspace(-np.pi, np.pi, 2001)
    for theta in (0.5, 0.75, 1.0):
        for lam in (0.1, 1.0, 10.0, 1e3):
            g = G.amplification(xi, lam, theta, sigma=SIG, r=r, y=y, h=0.01)
            assert np.max(np.abs(g)) <= 1 + 1e-14
    # explicit: |g(pi/h)| = |1 - 4 lam - rk| exceeds 1 iff lam > 1/2 (heat part)
    assert abs(G.amplification(np.pi, 0.5, 0.0)) == pytest.approx(1.0)
    assert abs(G.amplification(np.pi, 0.55, 0.0)) == pytest.approx(1.2)
    # theta < 1/2: stable iff (1 - 2 theta) lam <= 1/2
    assert np.max(np.abs(G.amplification(xi, 0.99, 0.25))) <= 1 + 1e-14
    assert np.max(np.abs(G.amplification(xi, 1.01, 0.25))) > 1


def test_crank_nicolson_positivity_threshold():
    for lam in (0.5, 1.0, 1.5, 2.0, 4.0, 89.0):
        m, w = G.cn_kernel(lam, half=400)
        centre = 2 / np.sqrt(1 + 2 * lam) - 1
        assert w[400] == pytest.approx(centre, abs=1e-12)
        assert np.all(np.delete(w, 400) >= -1e-15)
    assert G.cn_kernel(1.5, 400)[1][400] == pytest.approx(0.0, abs=1e-12)
    assert G.cn_kernel(4.0, 400)[1][400] == pytest.approx(-1 / 3, abs=1e-12)
    # theta-scheme with h |nu| <= sigma^2 and (1 - theta) k (sigma^2/h^2 + r) <= 1: G >= 0, row sums <= 1
    x, h = G.log_grid(S, T, SIG, 60)
    pm, p0, pp = G.coefficients(h, r, y, SIG)
    for theta, k in ((0.5, h * h / SIG**2), (1.0, 0.5), (0.0, h * h / (SIG**2 + r * h * h))):
        A = np.diag(np.full(59, p0)) + np.diag(np.full(58, pm), -1) + np.diag(np.full(58, pp), 1)
        Gm = np.linalg.solve(np.eye(59) - theta * k * A, np.eye(59) + (1 - theta) * k * A)
        assert Gm.min() >= -1e-14 and Gm.sum(axis=1).max() <= 1 + 1e-12


# ------------------------------------------------------------------ finite differences
def test_orders_of_the_schemes():
    def err(J, theta, N=None, rann=0):
        h = 12 * SIG / J
        if N is None:
            N = int(np.ceil(T / (0.4 * h * h / (0.5 * SIG**2))))
        return G.fd_price(S, K, T, r, y, SIG, J, N, theta=theta, rannacher=rann) - CALL
    ex = [err(J, 0.0) for J in (200, 400)]
    assert ex[0] / ex[1] == pytest.approx(4.0, rel=0.02)
    assert ex[1] == pytest.approx(-1.75e-4, abs=1e-6)
    cnr = [err(J, 0.5, int(round(SIG * J / (12 * SIG))), 2) for J in (800, 1600)]
    assert cnr[0] / cnr[1] == pytest.approx(4.0, rel=0.05)
    # plain CN with k = h / sigma: first order, and the gamma at the strike diverges
    cn = G.fd_solve(S, K, T, r, y, SIG, 1600, 133, theta=0.5)
    assert cn["price"] - CALL == pytest.approx(2.0e-3, abs=1e-4)
    assert cn["gamma"] == pytest.approx(0.0195 - 0.307, abs=0.01)
    cr = G.fd_solve(S, K, T, r, y, SIG, 1600, 133, theta=0.5, rannacher=2)
    gbs = float(bs.greeks(S, K, T, r, y, SIG)["gamma"])
    assert abs(cr["gamma"] - gbs) < 1e-6 and abs(cr["price"] - CALL) < 2e-4


def test_gamma_oscillation_of_figure():
    a = G.fd_greeks(G.fd_solve(S, K, T, r, y, SIG, 800, 25, theta=0.5))
    b = G.fd_greeks(G.fd_solve(S, K, T, r, y, SIG, 800, 25, theta=0.5, rannacher=2))
    sel = (a["s"] > 85) & (a["s"] < 115)
    ge = bs.greeks(a["s"][sel], K, T, r, y, SIG)["gamma"]
    assert a["gamma"][sel].min() == pytest.approx(-1.49, abs=0.01)
    assert np.max(np.abs(b["gamma"][sel] - ge)) < 1e-5


# ------------------------------------------------------------------ far field
def test_far_field_bound_and_course_example():
    Kc, Tc, rc, sc, Smax = 40.0, 1.0, 0.07, 0.6, 120.0
    assert float(bs.call(Smax, Kc, Tc, rc, 0.0, sc)) == pytest.approx(83.082, abs=1e-3)
    assert Smax - Kc * np.exp(-rc * Tc) == pytest.approx(82.704, abs=1e-3)
    taus = np.linspace(1e-6, Tc, 400)
    exact = G.s_grid_call(Kc, Tc, rc, 0.0, sc, Smax, M=1200, N=600, bc="exact")
    for bc, eps in (("payoff", np.max(np.abs(bs.call(Smax, Kc, taus, rc, 0.0, sc) - (Smax - Kc)))),
                    ("asymptotic", np.max(np.abs(bs.call(Smax, Kc, taus, rc, 0.0, sc)
                                                 - (Smax - Kc * np.exp(-rc * taus)))))):
        o = G.s_grid_call(Kc, Tc, rc, 0.0, sc, Smax, M=1200, N=600, bc=bc)
        for S0 in (20.0, 40.0, 60.0, 80.0):
            i = int(round(S0 / 0.1))
            q = G.exit_probability(np.log(S0), -np.inf, np.log(Smax), Tc, rc, 0.0, sc)
            assert abs(o["V"][i] - exact["V"][i]) <= eps * q
            assert q <= G.exit_bound(np.inf, np.log(Smax / S0), Tc, rc, 0.0, sc)
    o = G.s_grid_call(Kc, Tc, rc, 0.0, sc, Smax, M=1200, N=600, bc="payoff")
    assert o["V"][400] - exact["V"][400] == pytest.approx(-0.036, abs=1e-3)
    assert abs(exact["V"][400] - float(bs.call(40.0, Kc, Tc, rc, 0.0, sc))) < 1e-4


def test_exit_probability_is_the_reflection_formula():
    # one barrier, no drift: 2 N(-m / (sigma sqrt tau))
    p = G.exit_probability(0.0, -np.inf, 0.3, 1.0, 0.02, 0.0, 0.2)
    assert p == pytest.approx(2 * bs.norm_cdf(-0.3 / 0.2), rel=1e-12)
    # 2 N(-z) <= exp(-z^2/2)
    z = np.linspace(0, 8, 400)
    assert np.all(2 * bs.norm_cdf(-z) <= np.exp(-z * z / 2) + 1e-15)


# ------------------------------------------------------------------ American options
def _lcp(theta=0.5, J=200, k=0.01):
    x, h = G.log_grid(S, T, SIG, J)
    pm, p0, pp = G.coefficients(h, r, y, SIG)
    ab = G._bands(theta, k, pm, p0, pp, J - 1)
    g = np.maximum(K - np.exp(x[1:-1]), 0.0)
    q = g + 0.3 * np.sin(np.linspace(0, 9, J - 1))               # an arbitrary right side
    return ab, q, g


def test_lcp_solvers_agree_and_minimality():
    ab, q, g = _lcp()
    zb = G.brennan_schwartz(ab, q, g, "low")
    zp, _ = G.psor(ab, q, g, g, omega=1.3, tol=1e-14)
    zq, _ = G.penalty(ab, q, g, g, rho=1e10)
    assert G.lcp_residual(ab, q, g, zp) < 1e-11
    assert np.max(np.abs(zp - zq)) < 1e-8
    # projected Gauss--Seidel from g increases monotonically to the solution
    z, prev = g.copy(), g.copy()
    lo, di, up = np.r_[0.0, ab[2, :-1]], ab[1], np.r_[ab[0, 1:], 0.0]
    for _ in range(50):
        for j in range(z.size):
            left = z[j - 1] if j else 0.0
            right = z[j + 1] if j < z.size - 1 else 0.0
            z[j] = max(g[j], (q[j] - lo[j] * left - up[j] * right) / di[j])
        assert np.all(z >= prev - 1e-14)
        prev = z.copy()
    assert np.all(z <= zp + 1e-12)
    # the solution is the smallest supersolution: here s = B^{-1} max(q, Bg) has s >= g and Bs >= q
    def mul(v):
        w = ab[1] * v
        w[:-1] += ab[0, 1:] * v[1:]
        w[1:] += ab[2, :-1] * v[:-1]
        return w
    s = solve_banded((1, 1), ab, np.maximum(q, mul(g)))
    assert np.all(s >= g - 1e-12) and np.all(mul(s) >= q - 1e-12) and np.all(zp <= s + 1e-12)
    # Brennan--Schwartz on a genuine put step (exercise region at the bottom) is exact
    q2 = mul(g) + 0.01 * (np.exp(np.linspace(-3, 3, g.size)) > 1.0)       # Bg - q2 >= 0 low, < 0 high
    q2 = solve_banded((1, 1), ab, q2)
    q2 = mul(np.maximum(q2, g))
    z2b = G.brennan_schwartz(ab, q2, g, "low")
    z2p, _ = G.psor(ab, q2, g, g, omega=1.3, tol=1e-14)
    assert G.lcp_residual(ab, q2, g, z2b) < 1e-11 and np.max(np.abs(z2b - z2p)) < 1e-10
    if G.lcp_residual(ab, q, g, zb) < 1e-11:
        assert np.max(np.abs(zb - zp)) < 1e-10


def test_penalty_bracket():
    ab, q, g = _lcp(theta=1.0)
    zs, _ = G.psor(ab, q, g, g, omega=1.2, tol=1e-14)
    Bg = ab[1] * g
    Bg[:-1] += ab[0, 1:] * g[1:]
    Bg[1:] += ab[2, :-1] * g[:-1]
    for rho in (1e2, 1e4, 1e6):
        z, _ = G.penalty(ab, q, g, g, rho=rho)
        assert np.all(z <= zs + 1e-10)
        assert np.all(zs <= z + np.max(np.maximum(Bg - q, 0.0)) / rho + 1e-10)


def test_american_test_case_and_solvers():
    prices = {}
    for solver in ("psor", "bs", "penalty"):
        a = G.fd_solve(S, K, T, r, y, SIG, 800, 400, 0.5, "put", "american", rannacher=2, solver=solver)
        assert a["residual"] < 1e-8
        prices[solver] = a["price"]
    assert max(prices.values()) - min(prices.values()) < 1e-7
    e = G.fd_price(S, K, T, r, y, SIG, 800, 400, 0.5, "put", rannacher=2)
    assert prices["bs"] - e == pytest.approx(2.0e-6, abs=1e-7)              # early-exercise premium
    assert prices["bs"] - e + PUT == pytest.approx(8.349408, abs=1e-6)
    ac = G.fd_solve(S, K, T, r, y, SIG, 1600, 800, 0.5, "call", "american", rannacher=2, solver="penalty")
    ec = G.fd_price(S, K, T, r, y, SIG, 1600, 800, 0.5, "call", rannacher=2)
    assert ac["price"] - ec + CALL == pytest.approx(7.4597, abs=5e-5)
    bc = G.fd_price(S, K, T, r, y, SIG, 1600, 800, 0.5, "call", 50, rannacher=2)
    assert bc - ec + CALL == pytest.approx(7.4567, abs=5e-5)


def test_longstaff_schwartz_rows():
    for S0, sig, TT, am in ((36, 0.2, 1, 4.487), (44, 0.4, 2, 5.647)):
        a = G.fd_price(S0, 40.0, TT, 0.06, 0.0, sig, 1600, 800, 0.5, "put", "american", rannacher=2,
                       solver="penalty")
        assert a == pytest.approx(am, abs=5e-4)
        m = int(50 * TT)
        b = G.fd_price(S0, 40.0, TT, 0.06, 0.0, sig, 1600, 800, 0.5, "put", m, rannacher=2)
        assert b == pytest.approx(american.bermudan_reference(S0, 40.0, TT, 0.06, 0.0, sig, m, "put", h=2e-3),
                                  abs=1e-4)
    a = G.fd_price(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 3200, 1600, 0.5, "put", "american", rannacher=2,
                   solver="penalty")
    assert a == pytest.approx(4.486662, abs=2e-6)


def test_straddle_breaks_brennan_schwartz():
    o = G.fd_solve(S, K, T, 0.05, 0.05, SIG, 400, 200, 0.5, "straddle", "american", rannacher=2, solver="bs")
    p = G.fd_solve(S, K, T, 0.05, 0.05, SIG, 400, 200, 0.5, "straddle", "american", rannacher=2, solver="psor")
    assert o["residual"] > 1e-4 and p["residual"] < 1e-8


# ------------------------------------------------------------------ barriers
def test_barriers_against_chapter_16():
    d = G.fd_price(S, K, T, r, y, SIG, 1600, 960, 0.5, "call", barrier=95.0, monitoring=12, rannacher=2)
    assert d == pytest.approx(5.329790, abs=2e-4)
    c = G.fd_price(S, K, T, r, y, SIG, 1600, 960, 0.5, "call", barrier=95.0, rannacher=2)
    assert c == pytest.approx(3.75412, abs=2e-5)


# ------------------------------------------------------------------ further numbers quoted in the notes
def test_leisen_reimer_centres_the_strike_and_american_order():
    for KK in (80.0, 100.0, 130.0):
        for n in (11, 101, 1001):
            p = G.tree_parameters(T, r, y, SIG, n, "lr", S, KK)
            j = (n - 1) // 2
            assert S * p["u"] ** j * p["d"] ** (n - j) < KK < S * p["u"] ** (j + 1) * p["d"] ** (n - j - 1)
    ref = 4.486676                     # LS put S0 = 36, sigma = 0.2, T = 1 (grid and BBSR agree to 2e-5)
    assert G.binomial(36, 40, 1, 0.06, 0, 0.2, 101, "put", method="lr") - ref == pytest.approx(-4.8e-3, abs=1e-4)
    assert G.binomial(36, 40, 1, 0.06, 0, 0.2, 1601, "put", method="lr") - ref == pytest.approx(-2.9e-4, abs=2e-5)


def test_crank_nicolson_errors_quoted_in_the_rannacher_remark():
    for J, e_cn, e_cr in ((800, 3.77e-3, -4.04e-4), (3200, 9.95e-4, -2.53e-5)):
        N = int(round(SIG * J / (12 * SIG)))
        assert G.fd_price(S, K, T, r, y, SIG, J, N, theta=0.5) - CALL == pytest.approx(e_cn, rel=0.02)
        assert G.fd_price(S, K, T, r, y, SIG, J, N, theta=0.5, rannacher=2) - CALL == pytest.approx(e_cr, rel=0.02)
    T2 = 0.05
    g2 = float(bs.greeks(S, K, T2, r, y, SIG)["gamma"])
    assert g2 == pytest.approx(0.0891, abs=1e-4)
    assert G.fd_solve(S, K, T2, r, y, SIG, 800, 20, theta=0.5, width=10)["gamma"] == pytest.approx(2.93, abs=0.01)
    assert G.fd_solve(S, K, T2, r, y, SIG, 800, 20, theta=0.5, width=10, rannacher=2)["gamma"] == pytest.approx(0.0892, abs=1e-4)


def test_table_values_and_bermudan_grid():
    # penalty and Brennan--Schwartz on the grid of the table (J = 3200, N = 1600)
    a = G.fd_price(S, K, T, r, y, SIG, 3200, 1600, 0.5, "put", "american", rannacher=2, solver="penalty")
    e = G.fd_price(S, K, T, r, y, SIG, 3200, 1600, 0.5, "put", rannacher=2)
    assert a == pytest.approx(8.349393, abs=2e-6) and a - e + PUT == pytest.approx(8.349408, abs=1e-6)
    b = G.fd_price(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 3200, 1600, 0.5, "put", 50, rannacher=2)
    assert b == pytest.approx(4.477807, abs=1e-6)
    assert american.bermudan_reference(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 50, "put", h=1e-3) == pytest.approx(4.477810, abs=1e-6)
    o = G.fd_solve(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 1600, 800, 0.5, "put", "american", rannacher=2, solver="bs")
    assert o["residual"] < 1e-12
    assert G.trinomial(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 2000, "put", "american") == pytest.approx(4.486642, abs=1e-6)
    assert G.binomial(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 4001, "put", method="lr") == pytest.approx(4.486560, abs=1e-6)


def test_referee_numbers_bbsr_solvers_and_finite_grid_threshold():
    # BBSR at the money: first order for n = 2 mod 4; off the money: first order, BBS still oscillates
    assert 102 * (G.bbsr(S, K, T, r, y, SIG, 102, "call", "european") - CALL) == pytest.approx(0.061, abs=1e-3)
    assert 1602 * (G.bbsr(S, K, T, r, y, SIG, 1602, "call", "european") - CALL) == pytest.approx(0.047, abs=1e-3)
    c110 = float(bs.call(S, 110.0, T, r, y, SIG))
    for n, v in ((400, 5.8), (800, -1.6), (1600, -28.0)):
        assert n * n * (G.bbsr(S, 110.0, T, r, y, SIG, n, "call", "european") - c110) == pytest.approx(v, abs=0.1)
    e = [n * (G.binomial_european(S, 110.0, T, r, y, SIG, n, smooth=True) - c110) for n in range(400, 1601)]
    assert min(e) == pytest.approx(0.467, abs=1e-3) and max(e) == pytest.approx(0.488, abs=1e-3)
    # B for K = 110 fitted on n = 8000..8400
    ns = np.arange(8000, 8401)
    err = np.array([n * (G.binomial_european(S, 110.0, T, r, y, SIG, int(n)) - c110) for n in ns])
    th = np.array([G.crr_strike_offset(S, 110.0, T, SIG, int(n)) for n in ns])
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(th), th * (1 - th)]), err, rcond=None)
    assert coef[1] == pytest.approx(14.2800, abs=1e-3)
    # penalty and PSOR work per step (test-case put, N = J / 2, two extra half-steps)
    for J, mean_psor, max_psor in ((400, 15.0, 16), (800, 34.4, 36)):
        p = G.fd_solve(S, K, T, r, y, SIG, J, J // 2, 0.5, "put", "american", rannacher=2, solver="penalty")
        assert p["iterations"] / (J // 2 + 2) < 1.12 and p["max_iterations"] <= 5
        q = G.fd_solve(S, K, T, r, y, SIG, J, J // 2, 0.5, "put", "american", rannacher=2, solver="psor")
        assert q["iterations"] / (J // 2 + 2) == pytest.approx(mean_psor, abs=0.2) and q["max_iterations"] == max_psor
    # Crank--Nicolson on a finite grid with zero boundary values: positive iff lam <= 4 - 2 sqrt(2)
    lam_star = 4 - 2 * np.sqrt(2)
    for J in (11, 51):
        for lam, sign in ((lam_star - 1e-6, 1), (lam_star + 1e-6, -1)):
            ab = np.zeros((3, J - 1))
            ab[0, 1:], ab[1], ab[2, :-1] = -lam / 2, 1 + lam, -lam / 2
            diag = 2 * np.diag(solve_banded((1, 1), ab, np.eye(J - 1))) - 1
            assert np.sign(diag.min()) == sign
    # the straddle residuals quoted in the notes (r = y = 5%)
    lo = G.fd_solve(S, K, T, 0.05, 0.05, SIG, 800, 400, 0.5, "straddle", "american", rannacher=2, solver="bs")
    assert lo["residual"] == pytest.approx(5.0e-3, abs=2e-4)
