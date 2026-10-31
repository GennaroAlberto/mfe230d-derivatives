"""Checks of the Chapter 4 code against the results proved in the notes and the numbers quoted there."""
import numpy as np
import pytest

from mfe230d import bs
from mfe230d.replication import (
    bachelier_call,
    backward_solve,
    bs_market,
    bs_second_order,
    call_euler_terms,
    carr_madan_price,
    discounted_expectation,
    euler_residual,
    fd_grid,
    forward_solve,
    grid_payoff,
    grid_price,
    grid_replication,
    leveraged_call,
    leveraged_params,
    local_vol_terminal,
    log_strike_rule,
    power_call,
    power_greeks,
    power_params,
    power_put,
    put_call_symmetry,
    skew_local_vol,
    sqrt_call,
    sqrt_call_digital,
)
from mfe230d.standard import green

TC = bs.TEST_CASE
S, K, T, R, Y, SIG = TC["S"], TC["K"], TC["T"], TC["r"], TC["y"], TC["sigma"]
COURSE = dict(S=1e4, K=1e4, T=1.0, r=0.01, y=-0.01, sigma=0.40)   # the course's "devious" inputs


# ---------------------------------------------------------------------------------------
# Carr--Madan
# ---------------------------------------------------------------------------------------

def test_log_strike_rule_integrates_exactly():
    k, w = log_strike_rule(50.0, 200.0, n_panels=8, order=10)
    assert np.sum(w) == pytest.approx(150.0, rel=1e-13)
    assert np.sum(w / k) == pytest.approx(np.log(4.0), rel=1e-13)


def test_spanning_identity_pointwise():
    # Theorem sr-carr-madan (i) for f = -log, split at kappa: the strip integral, done by quadrature.
    kappa = 100.0
    for x in (40.0, 99.0, 100.0, 180.0):
        lo, hi = sorted((x, kappa))
        k, w = log_strike_rule(lo, hi, 16, 12) if lo < hi else (np.array([kappa]), np.array([0.0]))
        opt = np.maximum(k - x, 0) if x < kappa else np.maximum(x - k, 0)
        rhs = -np.log(kappa) - (x - kappa) / kappa + np.sum(w * opt / k**2)
        assert rhs == pytest.approx(-np.log(x), abs=1e-12)


def test_sqrt_call_strip_course_inputs():
    # Example sr-sqrt: tangent call 8.43615, strip 1.07186, price 7.36429.
    c = COURSE
    m = bs_market(c["S"], c["T"], c["r"], c["y"], c["sigma"])
    Kc = c["K"]
    tangent = bs.call(c["S"], Kc, c["T"], c["r"], c["y"], c["sigma"]) / (2 * np.sqrt(Kc))
    price = carr_madan_price(Kc, 0.0, 1 / (2 * np.sqrt(Kc)), m,
                             d2f=lambda k: np.where(k > Kc, -0.25 * k**-1.5, 0.0),
                             kinks=[(Kc, 1 / (2 * np.sqrt(Kc)))], k_lo=Kc * np.exp(-8), k_hi=Kc * np.exp(8))
    assert tangent == pytest.approx(8.43615, abs=5e-6)
    assert tangent - price == pytest.approx(1.07186, abs=5e-6)
    assert price == pytest.approx(7.36429, abs=5e-6)
    assert price == pytest.approx(discounted_expectation(lambda s: sqrt_call(s, Kc), c["S"], c["T"], c["r"],
                                                         c["y"], c["sigma"], breaks=[Kc]), abs=1e-10)


def test_split_point_does_not_matter_and_jumps_are_digitals():
    # Payoff with a kink at 100 and a jump at 120, priced with three split points.
    m = bs_market(S, T, R, Y, SIG)
    exact = power_call(S, 10.0, T, R, Y, SIG, 0.5) + bs.digital_call(S, 120.0, T, R, Y, SIG)
    assert exact == pytest.approx(0.488321, abs=5e-7)
    d2f = lambda k: np.where(k > 100, -0.25 * k**-1.5, 0.0)
    f = lambda k: sqrt_call(k, 100.0)
    df = lambda k: np.where(k >= 100, 0.5 / np.sqrt(k), 0.0)   # right derivative
    for kappa in (80.0, 100.0, 150.0):
        # the kink at 100 has mass f'(100+) - f'(100-) = 0.05 (a put if 100 <= kappa, a call otherwise)
        p = carr_madan_price(kappa, float(f(kappa)), float(df(kappa)), m, d2f=d2f, kinks=[(100.0, 0.05)],
                             digitals=[(120.0, 1.0)], k_lo=5.0, k_hi=2000.0)
        assert p == pytest.approx(exact, abs=1e-9)


def test_convex_payoff_price_is_at_least_bond_times_value_at_forward():
    # Jensen through Carr--Madan: f convex, split at F, all strip weights >= 0.
    m = bs_market(S, T, R, Y, SIG)
    F = m["forward"]
    p = carr_madan_price(F, F**2, 2 * F, m, d2f=lambda k: 2.0 + 0 * k, k_lo=1e-3, k_hi=2000.0)
    second_moment = S**2 * np.exp((2 * (R - Y) + SIG**2) * T) * np.exp(-R * T)
    assert p == pytest.approx(second_moment, rel=1e-9)
    assert p >= m["bond"] * F**2


def test_grid_replication_is_the_interpolant():
    k = np.linspace(80.0, 140.0, 13)
    rep = grid_replication(np.sqrt(k), k)
    assert np.allclose(grid_payoff(rep, k), np.sqrt(k), atol=1e-12)
    xm = 0.5 * (k[1:] + k[:-1])
    assert np.allclose(grid_payoff(rep, xm), 0.5 * (np.sqrt(k[1:]) + np.sqrt(k[:-1])), atol=1e-12)
    # and its price is the expectation of the interpolant
    m = bs_market(S, T, R, Y, SIG)
    direct = discounted_expectation(lambda s: float(grid_payoff(rep, s)), S, T, R, Y, SIG, breaks=list(k))
    assert grid_price(rep, m) == pytest.approx(direct, abs=1e-9)


def test_strike_grid_error_orders():
    # Figure sr-replication: first order with the jump interpolated, second order with the digital held.
    m = bs_market(S, T, R, Y, SIG)
    exact = power_call(S, 10.0, T, R, Y, SIG, 0.5) + bs.digital_call(S, 120.0, T, R, Y, SIG)
    G120 = green(120.0, S, T, R, Y, SIG)
    e1, e2 = [], []
    hs = np.array([5.0, 2.5, 1.25])
    for h in hs:
        k = 100 + h * np.arange(-1, int(round(300 / h)) + 1)
        e1.append(grid_price(grid_replication(sqrt_call_digital(k), k), m) - exact)
        e2.append(grid_price(grid_replication(sqrt_call(k), k), m) + bs.digital_call(S, 120, T, R, Y, SIG) - exact)
    e1, e2 = np.array(e1), np.array(e2)
    # interpolation is linear, so e1 - e2 is the error of the call spread replacing the digital, which is
    # exactly J h int_0^1 (1 - u) G(a - h u) du (eq:sr-jump-error) and tends to h G(a) / 2
    u, w = np.polynomial.legendre.leggauss(20)
    u, w = 0.5 * (u + 1), 0.5 * w
    jump = np.array([h * np.sum(w * (1 - u) * green(120.0 - h * u, S, T, R, Y, SIG)) for h in hs])
    assert np.allclose(e1 - e2, jump, rtol=1e-9)
    ratio = e1 / (0.5 * hs * G120)
    assert np.all(np.diff(ratio) < 0) and ratio[-1] == pytest.approx(1.0, abs=0.025)
    assert np.all(e2 < 0)                       # concave payoff: the chord lies below
    assert e2[0] / e2[1] == pytest.approx(4.0, rel=0.02)
    assert e2[1] / e2[2] == pytest.approx(4.0, rel=0.02)


# ---------------------------------------------------------------------------------------
# Power payoffs
# ---------------------------------------------------------------------------------------

def test_power_params_course_sqrt_map():
    yp, sp = power_params(0.5, COURSE["r"], COURSE["y"], COURSE["sigma"])
    assert yp == pytest.approx(0.02, abs=1e-15) and sp == pytest.approx(0.2)
    assert yp == pytest.approx(0.5 * (COURSE["r"] + COURSE["y"] + COURSE["sigma"] ** 2 / 4))
    price = power_call(COURSE["S"], np.sqrt(COURSE["K"]), 1.0, COURSE["r"], COURSE["y"], COURSE["sigma"], 0.5)
    assert price == pytest.approx(7.36429, abs=5e-6)
    assert price == pytest.approx(bs.call(S, K, T, R, Y, SIG), abs=1e-12)


@pytest.mark.parametrize("p", [-2.0, -1.0, -0.5, 0.5, 1.5, 2.0])
def test_power_call_and_put_against_quadrature(p):
    Kp = 1.1 * 100.0**p
    kink = Kp ** (1 / p)
    c = power_call(S, Kp, T, R, Y, SIG, p)
    q = power_put(S, Kp, T, R, Y, SIG, p)
    assert c == pytest.approx(discounted_expectation(lambda s: max(s**p - Kp, 0.0), S, T, R, Y, SIG, [kink]),
                              rel=1e-9, abs=1e-14)
    assert q == pytest.approx(discounted_expectation(lambda s: max(Kp - s**p, 0.0), S, T, R, Y, SIG, [kink]),
                              rel=1e-9, abs=1e-14)


def test_power_edges():
    assert power_params(1.0, R, Y, SIG) == (pytest.approx(Y), pytest.approx(SIG))
    assert power_call(S, 0.7, T, R, Y, SIG, 0.0) == pytest.approx(np.exp(-R * T) * 0.3)
    assert power_put(S, 0.7, T, R, Y, SIG, 0.0) == pytest.approx(0.0)
    # K = 0: the power forward S^p e^{-y_p T}
    for p in (-1.0, 0.5, 2.0):
        yp, _ = power_params(p, R, Y, SIG)
        assert power_call(S, 0.0 + 1e-300, T, R, Y, SIG, p) == pytest.approx(S**p * np.exp(-yp * T), rel=1e-12)
    # leveraged with p = 0 is the plain call
    assert leveraged_params(0.0, R, Y, SIG) == (pytest.approx(R), pytest.approx(Y))


def test_power_greeks_post_wash_against_bumps():
    p, Kp = 0.5, 10.5
    g = power_greeks(S, Kp, T, R, Y, SIG, p)
    V = lambda **kw: power_call(kw.get("S", S), Kp, T, kw.get("r", R), kw.get("y", Y), kw.get("sigma", SIG), p)
    h = 1e-4
    assert g["delta"] == pytest.approx((V(S=S + h) - V(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((V(S=S + 1e-2) - 2 * V() + V(S=S - 1e-2)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((V(sigma=SIG + h) - V(sigma=SIG - h)) / (2 * h), rel=1e-6)
    assert g["rho_r"] == pytest.approx((V(r=R + h) - V(r=R - h)) / (2 * h), rel=1e-6)
    assert g["rho_y"] == pytest.approx((V(y=Y + h) - V(y=Y - h)) / (2 * h), rel=1e-6)


@pytest.mark.parametrize("p", [-1.0, 0.5, 2.0])
def test_leveraged_call_against_quadrature(p):
    c = leveraged_call(S, 95.0, T, R, Y, SIG, p)
    q = discounted_expectation(lambda s: s**p * max(s - 95.0, 0.0), S, T, R, Y, SIG, [95.0])
    assert c == pytest.approx(q, rel=1e-9)


def test_course_power_payoffs():
    # Exercise sr-power-payoffs: the asymptotic call 5.7213 (course) and the payoff 2 (S - sqrt(K S))^+.
    asym = 100.0 * leveraged_call(100.0, 100.0, 1.0, 0.0075, 0.02, 0.2, -1.0)
    assert asym == pytest.approx(5.72127, abs=5e-6)
    assert leveraged_params(-1.0, 0.0075, 0.02, 0.2) == (pytest.approx(2 * 0.0075 - 0.02 - 0.04), pytest.approx(0.0075))
    dp, _ = bs.d_pm(S, K, T, R, Y, SIG)
    x0 = (np.log(S / K) + (R - Y) * T) / (SIG * np.sqrt(T))
    sst = 2 * (S * np.exp(-Y * T) * bs.norm_cdf(dp)
               - np.sqrt(K * S) * np.exp(-(0.5 * (R + Y) + SIG**2 / 8) * T) * bs.norm_cdf(x0))
    assert sst == pytest.approx(7.81760, abs=5e-6)
    assert sst == pytest.approx(discounted_expectation(lambda s: 2 * max(s - np.sqrt(K * s), 0.0), S, T, R, Y, SIG,
                                                       [K]), abs=1e-9)


def test_put_call_symmetry():
    for k in (60.0, 99.0, 140.0):
        c, p = put_call_symmetry(S, k, T, R, Y, SIG)
        assert c == pytest.approx(p, rel=1e-12)


def test_barrier_static_hedge_numbers():
    # Exercise sr-barrier: with r = y, the down-and-in call is K/H puts struck at H^2/K.
    Sb, Kb, H, r, y, s = 100.0, 100.0, 90.0, 0.02, 0.02, 0.2
    assert Kb / H * bs.put(Sb, H * H / Kb, 1.0, r, y, s) == pytest.approx(1.46853, abs=5e-6)
    cost = bs.call(H, Kb, 0.5, 0.05, 0.0, s) - Kb / H * bs.put(H, H * H / Kb, 0.5, 0.05, 0.0, s)
    assert cost == pytest.approx(1.07302, abs=5e-6)


# ---------------------------------------------------------------------------------------
# Homogeneity
# ---------------------------------------------------------------------------------------

def test_euler_identity_and_numbers_test_case():
    g = bs.greeks(S, K, T, R, Y, SIG)
    dK = -bs.digital_call(S, K, T, R, Y, SIG)
    assert S * g["delta"] == pytest.approx(50.9643, abs=5e-5)
    assert K * dK == pytest.approx(-43.6000, abs=5e-5)
    assert S * g["delta"] + K * dK == pytest.approx(bs.call(S, K, T, R, Y, SIG), abs=1e-12)
    for s_, k_ in ((80.0, 120.0), (130.0, 70.0)):
        V = lambda a, b: bs.call(a, b, T, R, Y, SIG)
        assert abs(euler_residual(V, s_, k_)) < 1e-6


def test_gamma_sum_rules_and_density():
    for s_, k_ in ((100.0, 100.0), (90.0, 115.0)):
        so = bs_second_order(s_, k_, T, R, Y, SIG)
        assert s_ * so["SS"] + k_ * so["SK"] == pytest.approx(0.0, abs=1e-15)
        assert s_ * so["SK"] + k_ * so["KK"] == pytest.approx(0.0, abs=1e-15)
        assert s_**2 * so["SS"] == pytest.approx(k_**2 * so["KK"], rel=1e-12)
        assert so["KK"] == pytest.approx(green(k_, s_, T, R, Y, SIG), rel=1e-12)      # Breeden--Litzenberger
        assert so["SS"] == pytest.approx(bs.greeks(s_, k_, T, R, Y, SIG)["gamma"], rel=1e-12)
    assert bs_second_order(S, K, T, R, Y, SIG)["KK"] == pytest.approx(0.0195277, abs=5e-8)


def test_conservation_of_vega_and_theta():
    dp, dm = bs.d_pm(S, K, T, R, Y, SIG)
    vanna_S = -S * np.exp(-Y * T) * bs.norm_pdf(dp) * dm / SIG
    vanna_K = K * np.exp(-R * T) * bs.norm_pdf(dm) * dp / SIG
    assert vanna_S == pytest.approx(29.2916, abs=5e-5)
    assert vanna_K == pytest.approx(9.7639, abs=5e-5)
    assert vanna_S + vanna_K == pytest.approx(bs.greeks(S, K, T, R, Y, SIG)["vega"], rel=1e-12)
    assert bs.greeks(S, K, T, R, Y, SIG)["vega"] == pytest.approx(39.0554, abs=5e-5)
    e = 1e-5
    delta = lambda TT: bs.greeks(S, K, TT, R, Y, SIG)["delta"]
    dkc = lambda TT: -bs.digital_call(S, K, TT, R, Y, SIG)
    charm_S = -S * (delta(T + e) - delta(T - e)) / (2 * e)        # calendar time: d/dt = -d/dT
    charm_K = -K * (dkc(T + e) - dkc(T - e)) / (2 * e)
    assert charm_S == pytest.approx(0.04290, abs=5e-6)
    assert charm_K == pytest.approx(-3.36516, abs=5e-6)
    assert charm_S + charm_K == pytest.approx(bs.greeks(S, K, T, R, Y, SIG)["theta"], abs=1e-7)
    assert bs.greeks(S, K, T, R, Y, SIG)["theta"] == pytest.approx(-3.32226, abs=5e-6)


def test_bachelier_is_not_homogeneous_in_spot_and_strike():
    V = lambda a, b: bachelier_call(a, b, 1.0, 0.0, 0.0, 20.0)
    assert V(100.0, 100.0) == pytest.approx(7.97885, abs=5e-6)
    assert euler_residual(V, 100.0, 100.0) == pytest.approx(-V(100.0, 100.0), abs=1e-6)
    # homogeneous once the normal volatility is scaled too
    assert bachelier_call(250.0, 230.0, 1.0, 0.01, 0.02, 50.0) == pytest.approx(
        2.5 * bachelier_call(100.0, 92.0, 1.0, 0.01, 0.02, 20.0), rel=1e-12)


# ---------------------------------------------------------------------------------------
# Kolmogorov equations
# ---------------------------------------------------------------------------------------

def _gbm_grid(n=801):
    x = fd_grid(np.log(S), 8 * SIG * np.sqrt(T), n)
    return x, n // 2, R - Y - 0.5 * SIG**2


def test_forward_backward_duality_and_test_case_price():
    x, i0, nu = _gbm_grid()
    h = x[1] - x[0]
    pay = np.maximum(np.exp(x) - K, 0.0)
    G = forward_solve(x, i0, T, nu, SIG, R, n_t=400)
    u = backward_solve(x, pay, T, nu, SIG, R, n_t=400)
    cf = h * np.sum(G * pay)
    assert cf == pytest.approx(u[i0], abs=1e-11)                  # exact discrete adjointness
    assert cf == pytest.approx(7.36386, abs=5e-6)
    assert cf - bs.call(S, K, T, R, Y, SIG) == pytest.approx(-4.3e-4, abs=5e-6)
    assert h * np.sum(G) * np.exp(R * T) == pytest.approx(1.0, abs=1e-7)   # mass of the density


def test_forward_solve_is_second_order():
    errs = []
    for n, nt in ((401, 200), (801, 400), (1601, 800)):
        x, i0, nu = _gbm_grid(n)
        h = x[1] - x[0]
        G = forward_solve(x, i0, T, nu, SIG, R, n_t=nt)
        errs.append(h * np.sum(G * np.maximum(np.exp(x) - K, 0.0)) - bs.call(S, K, T, R, Y, SIG))
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.03)
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.03)


def test_forward_density_matches_green_function():
    x, i0, nu = _gbm_grid(1601)
    G = forward_solve(x, i0, T, nu, SIG, R, n_t=800)
    xi = np.exp(x)
    mid = (xi > 60) & (xi < 160)
    exact = green(xi[mid], S, T, R, Y, SIG) * xi[mid]        # density in log S
    assert np.max(np.abs(G[mid] - exact)) < 2e-4 * np.max(exact)


def _lv():
    b = lambda t, z: R - Y - 0.5 * skew_local_vol(np.exp(z)) ** 2
    a = lambda t, z: skew_local_vol(np.exp(z))
    return b, a


def test_local_vol_forward_solve_against_monte_carlo():
    # Figure sr-kolmogorov: one forward solve against Euler paths (seed 4).
    b, a = _lv()
    x = fd_grid(np.log(S), 8 * 0.35, 801)
    h = x[1] - x[0]
    G = forward_solve(x, 400, T, b, a, R, n_t=400)
    ST = local_vol_terminal(S, T, R, Y, skew_local_vol, n_steps=100, n_paths=100_000, seed=4)
    for k in (80.0, 100.0, 120.0):
        pay = np.exp(-R * T) * np.maximum(ST - k, 0.0)
        c = h * np.sum(G * np.maximum(np.exp(x) - k, 0.0))
        assert abs(c - pay.mean()) < 3.5 * pay.std() / np.sqrt(pay.size) + 0.01


def test_local_vol_breaks_euler_identity():
    # Remark sr-homogeneous: C = 9.297 but S dC/dS + K dC/dK = 1.815.
    b, a = _lv()
    x = fd_grid(np.log(S), 8 * 0.35, 1601)
    t = call_euler_terms(x, 800, K, T, b, a, R, n_t=800)
    assert t["C"] == pytest.approx(9.297, abs=1e-3)
    assert t["S_dS"] + t["K_dK"] == pytest.approx(1.815, abs=2e-3)
    # the same machinery in the standard model satisfies Euler's identity
    x2, i0, nu = _gbm_grid(1601)
    t2 = call_euler_terms(x2, i0, K, T, nu, SIG, R, n_t=800)
    assert abs(t2["residual"]) < 2e-3


def test_local_vol_scaling_with_reference_level():
    # Homogeneity in all money-valued inputs: scaling S, K and the level S_ref of the local vol.
    def price(S0, Kc, S_ref):
        sig = lambda s: skew_local_vol(s, S_ref=S_ref)
        b = lambda t, z: R - Y - 0.5 * sig(np.exp(z)) ** 2
        a = lambda t, z: sig(np.exp(z))
        x = fd_grid(np.log(S0), 8 * 0.35, 801)
        h = x[1] - x[0]
        G = forward_solve(x, 400, T, b, a, R, n_t=200)
        return h * np.sum(G * np.maximum(np.exp(x) - Kc, 0.0))
    assert price(200.0, 220.0, 200.0) == pytest.approx(2.0 * price(100.0, 110.0, 100.0), rel=1e-10)
