"""Checks of the Chapter 8 code against the results proved in the notes and the numbers they quote."""
import numpy as np
import pytest

from mfe230d.bs import TEST_CASE, call, put
from mfe230d.variance import (
    DEVIOUS,
    antithetic,
    antithetic_correlation,
    call_payoff_moments,
    control_variate,
    crn_difference,
    cv_bias_constant,
    delta_hedge_control,
    disjoint_antithetic,
    even_odd_parts,
    gauss_expect,
    laboratory,
    mm_asymptotic_factor,
    mm_bias_constant,
    moment_match,
    moment_matched,
    partial_moment,
    plain,
    regression_control_variates,
    regression_factor,
    scalar_payoffs,
    sqrt_call,
    sqrt_control_mean,
    sqrt_option_samples,
    strike_to_z,
    terminal_price,
    theory_table,
)

P = TEST_CASE
S, K, T, r, y, SIG = (P[k] for k in ("S", "K", "T", "r", "y", "sigma"))
DISC = np.exp(-r * T)


# ------------------------------------------------------------------ the model and quadrature
def test_partial_moments_and_call_moments_match_quadrature():
    zs = strike_to_z(K, S, T, r, y, SIG)
    st = lambda u: float(terminal_price(u, S, T, r, y, SIG))
    for p in (0, 1, 2, -0.5):
        quad = gauss_expect(lambda u: st(u) ** p * (u > zs), [zs])
        assert partial_moment(p, K, S, T, r, y, SIG) == pytest.approx(quad, rel=1e-9)
    m, v = call_payoff_moments(**P)
    assert DISC * m == pytest.approx(7.36429, abs=5e-6)            # the course's test case
    assert DISC * m == pytest.approx(float(call(**P)), rel=1e-12)
    assert v == pytest.approx(161.198, abs=1e-3)                    # Var of (S_T - K)^+
    assert DISC**2 * v == pytest.approx(158.006, abs=1e-3)          # discounted payoff


# ------------------------------------------------------------------ antithetic variates
def test_antithetic_variance_formula_and_even_odd_split():
    p = scalar_payoffs()
    c, zs = p["call"], p["z_star"]
    fe, fo = even_odd_parts(c)
    ks = [zs, -zs]
    mu = gauss_expect(c, ks)
    var = gauss_expect(lambda u: c(u) ** 2, ks) - mu * mu
    ve = gauss_expect(lambda u: fe(u) ** 2, ks) - mu * mu
    vo = gauss_expect(lambda u: fo(u) ** 2, ks)
    rho = antithetic_correlation(c, [zs])
    assert ve + vo == pytest.approx(var, rel=1e-9)                  # Var f = Var f_e + Var f_o
    assert rho == pytest.approx((ve - vo) / (ve + vo), rel=1e-9)
    assert ve == pytest.approx(var * (1 + rho) / 2, rel=1e-9)       # Var of a pair average
    # simulation: n pairs
    z = np.random.default_rng(0).standard_normal(200_000)
    out = antithetic(lambda u: DISC * np.maximum(terminal_price(u, S, T, r, y, SIG) - K, 0), z)
    assert out["se"] == pytest.approx(np.sqrt(ve / z.size), rel=0.02)
    assert out["rho"] == pytest.approx(rho, abs=0.01)


def test_stock_antithetic_numbers():
    s = SIG * np.sqrt(T)
    st = scalar_payoffs()["stock"]
    rho = antithetic_correlation(st)
    assert rho == pytest.approx(-np.exp(-s * s), rel=1e-9)
    assert rho == pytest.approx(-0.9608, abs=5e-5)
    assert np.sqrt((1 + rho) / 2) == pytest.approx(0.1400, abs=5e-5)    # course: about 14%
    assert 1 / (1 + rho) == pytest.approx(25.5, abs=0.05)               # course: about 25


def test_disjoint_formulas_for_the_test_case_call():
    p = scalar_payoffs()
    m, v = call_payoff_moments(**P)
    assert S * np.exp((r - y - SIG**2 / 2) * T) == pytest.approx(97.04, abs=0.005)  # median < K
    d = disjoint_antithetic(m, v)
    assert d["rho"] == pytest.approx(antithetic_correlation(p["call"], [p["z_star"]]), rel=1e-8)
    assert d["rho"] == pytest.approx(-0.3432, abs=5e-5)
    assert d["se_ratio"] == pytest.approx(0.5730, abs=5e-5)              # course: about 57%
    assert d["efficiency"] == pytest.approx(1.523, abs=5e-4)            # course: about 1.5
    # below the median the supports overlap and the formula fails
    m2, v2 = call_payoff_moments(S, 90.0, T, r, y, SIG)
    rho90 = antithetic_correlation(
        lambda u: max(float(terminal_price(u, S, T, r, y, SIG)) - 90.0, 0.0), [strike_to_z(90.0, S, T, r, y, SIG)])
    assert rho90 > disjoint_antithetic(m2, v2)["rho"] + 0.01


def test_digital_disjoint_correlation():
    from scipy.special import ndtr
    zs = strike_to_z(110.0, S, T, r, y, SIG)
    pr = ndtr(-zs)                                  # Q(S_T > 110) < 1/2: disjoint
    rho = antithetic_correlation(lambda u: float(u > zs), [zs])
    assert rho == pytest.approx(-pr / (1 - pr), rel=1e-7)


def test_monotone_functions_give_nonpositive_correlation():
    # Random monotone functions of two independent normals (nondecreasing in z1,
    # nonincreasing in z2): the antithetic correlation is never positive.
    rng = np.random.default_rng(11)
    z = rng.standard_normal((400_000, 2))
    for k in range(5):
        a, b, c = rng.uniform(0.2, 2, 3)
        f = lambda u: np.tanh(a * u[:, 0] - b * u[:, 1] ** 3) + np.maximum(c * u[:, 0], 0) - np.exp(u[:, 1])
        out = antithetic(f, z)
        assert out["rho"] < 0.003


# ------------------------------------------------------------------ moment matching
def test_moment_matching_matches_moments_exactly():
    z = np.random.default_rng(1).standard_normal((3, 1000))
    zl = moment_match(z, "log")
    assert np.allclose(zl.mean(axis=1), 0, atol=1e-14)
    assert np.allclose((zl**2).mean(axis=1), 1, atol=1e-12)
    s = SIG * np.sqrt(T)
    ze = moment_match(z, "exp", s)
    assert np.allclose(ze.var(axis=1), 1, atol=1e-12)
    ST = terminal_price(ze, S, T, r, y, SIG)
    assert np.allclose(ST.mean(axis=1), S * np.exp((r - y) * T), rtol=1e-13)   # forward repriced
    # put-call parity holds exactly in the matched sample
    C = DISC * np.maximum(ST - K, 0).mean(axis=1)
    Pu = DISC * np.maximum(K - ST, 0).mean(axis=1)
    assert np.allclose(C - Pu, S * np.exp(-y * T) - K * DISC, atol=1e-11)


def test_moment_matching_bias_is_order_one_over_n():
    p = scalar_payoffs()
    c, zs = p["call"], p["z_star"]
    const = mm_bias_constant(c, [zs])
    assert const == pytest.approx(1.75, abs=0.005)
    # z^4: exact bias -6/(n+1) for the divisor-n matching; constant -6
    assert mm_bias_constant(lambda u: u**4) == pytest.approx(-6.0, rel=1e-9)
    n, R = 16, 200_000
    z = np.random.default_rng(7).standard_normal((R, n))
    zz = moment_match(z, "log")
    assert (zz**4).mean() == pytest.approx(3 - 6 / (n + 1), abs=0.01)
    est = (DISC * np.maximum(terminal_price(zz, S, T, r, y, SIG) - K, 0)).mean(axis=1)
    bias, se = est.mean() - float(call(**P)), est.std() / np.sqrt(R)
    assert abs(n * bias - 1.875) < 4 * n * se                       # 1.875 measured; tends to 1.75


def test_moment_matching_asymptotic_factors():
    p = scalar_payoffs()
    c, st, zs = p["call"], p["stock"], p["z_star"]
    s = SIG * np.sqrt(T)
    assert np.sqrt(mm_asymptotic_factor(st)) == pytest.approx(0.01625, abs=5e-6)   # course: about 1.6%
    assert np.sqrt(mm_asymptotic_factor(c, [zs])) == pytest.approx(0.1883, abs=5e-5)  # course: about 20%
    assert np.sqrt(mm_asymptotic_factor(c, [zs], "exp", s)) == pytest.approx(0.1823, abs=5e-5)
    # log matching = optimal regression on Z and Z^2
    f, _ = regression_factor(c, [lambda u: u, lambda u: u * u], [zs])
    assert mm_asymptotic_factor(c, [zs]) == pytest.approx(f, rel=1e-8)
    # the simulated variance agrees
    z = np.random.default_rng(5).standard_normal((4000, 1024))
    est = (DISC * np.maximum(terminal_price(moment_match(z, "log"), S, T, r, y, SIG) - K, 0)).mean(axis=1)
    sd = np.sqrt(158.006 / 1024)
    assert est.std() / sd == pytest.approx(0.1883, rel=0.05)


def test_batches_give_valid_errors_and_naive_errors_do_not():
    z = np.random.default_rng(2).standard_normal(2**16)
    f = lambda u: DISC * terminal_price(u, S, T, r, y, SIG)
    out = moment_matched(f, z, n_batches=64, kind="log")
    assert out["naive_se"] > 30 * out["se"]          # the naive CLT error ignores the matching


# ------------------------------------------------------------------ common random numbers
def test_crn_vega_difference():
    a, b = 0.20, 0.21
    z = np.random.default_rng(4).standard_normal(100_000)
    fa = lambda u: DISC * np.maximum(terminal_price(u, S, T, r, y, a) - K, 0)
    fb = lambda u: DISC * np.maximum(terminal_price(u, S, T, r, y, b) - K, 0)
    exact = float(call(S, K, T, r, y, b) - call(S, K, T, r, y, a))
    assert exact == pytest.approx(0.39048, abs=5e-6)
    same = crn_difference(fb, fa, z)
    indep = crn_difference(fb, fa, z, np.random.default_rng(5).standard_normal(100_000))
    assert same["se"] / indep["se"] == pytest.approx(0.0400, abs=0.002)
    assert abs(same["mean"] - exact) < 4 * same["se"]


def test_crn_hurts_for_opposite_monotonicity():
    # Call minus put: CRN variance Var C + Var P + 2 E[C] E[P], independent Var C + Var P.
    m, v = call_payoff_moments(**P)
    mp = DISC * float(put(**P)) / DISC
    pv = gauss_expect(lambda u: DISC**2 * max(K - float(terminal_price(u, S, T, r, y, SIG)), 0) ** 2,
                      [strike_to_z(K, S, T, r, y, SIG)]) - float(put(**P)) ** 2
    cv = DISC**2 * v
    crn_var = cv + pv + 2 * float(call(**P)) * float(put(**P))
    assert crn_var == pytest.approx(392.11, abs=0.01)
    assert cv + pv == pytest.approx(269.13, abs=0.01)
    z = np.random.default_rng(6).standard_normal(200_000)
    fc = lambda u: DISC * np.maximum(terminal_price(u, S, T, r, y, SIG) - K, 0)
    fp = lambda u: DISC * np.maximum(K - terminal_price(u, S, T, r, y, SIG), 0)
    assert crn_difference(fc, fp, z)["se"] ** 2 * z.size == pytest.approx(crn_var, rel=0.02)


# ------------------------------------------------------------------ control variates
def test_optimal_coefficient_and_variance_factor():
    p = scalar_payoffs()
    fac, beta = regression_factor(p["call"], [p["stock"]], [p["z_star"]])
    assert beta[0] == pytest.approx(0.5598, abs=5e-5)
    assert np.sqrt(1 - fac) == pytest.approx(0.8818, abs=5e-5)       # rho
    assert np.sqrt(fac) == pytest.approx(0.4716, abs=5e-5)
    # variance of Y - beta (X - mu) is a parabola in beta with minimum 1 - rho^2 at beta*
    z = np.random.default_rng(8).standard_normal(400_000)
    Y = DISC * np.maximum(terminal_price(z, S, T, r, y, SIG) - K, 0)
    X = DISC * terminal_price(z, S, T, r, y, SIG)
    for b in (0.0, beta[0], 2 * beta[0]):
        out = control_variate(Y, X, S * np.exp(-y * T), beta=b)
        expected = fac if b == beta[0] else 1.0
        assert (out["se"] / plain(Y)["se"]) ** 2 == pytest.approx(expected, rel=0.02)


def test_square_root_option_numbers():
    d = DEVIOUS
    assert float(sqrt_call(**d)) == pytest.approx(7.36429, abs=5e-6)   # pre-wash to the test case
    assert sqrt_control_mean(**d) == pytest.approx(8.43615, abs=5e-6)  # course: 8.43615
    th = theory_table()
    assert th["sqrt", "rho"] == pytest.approx(0.9960, abs=5e-5)
    assert th["sqrt", "beta*"] == pytest.approx(0.8154, abs=5e-5)
    assert th["sqrt", "cv-beta1"] == pytest.approx(0.2424, abs=5e-5)   # error reduction ~75.8%
    assert th["sqrt", "cv-beta*"] == pytest.approx(0.0889, abs=5e-5)
    assert th["sqrt", "cv-multiple"] == pytest.approx(0.0815, abs=5e-5)
    # the course's equal-variance reading of the beta = 1 result: 2 (1 - rho) = 0.25^2
    assert 1 - 0.25**2 / 2 == pytest.approx(0.969, abs=5e-4)


def test_estimated_coefficient_bias_and_splitting():
    p = scalar_payoffs()
    const = cv_bias_constant(p["call"], p["stock"], [p["z_star"]])
    assert const == pytest.approx(-8.18, abs=0.005)
    n, R = 16, 100_000
    z = np.random.default_rng(9).standard_normal((R, n))
    Y = DISC * np.maximum(terminal_price(z, S, T, r, y, SIG) - K, 0)
    X = DISC * terminal_price(z, S, T, r, y, SIG)
    mux = S * np.exp(-y * T)

    def beta(yy, xx):
        xc = xx - xx.mean(1, keepdims=True)
        return (xc * (yy - yy.mean(1, keepdims=True))).sum(1) / (xc * xc).sum(1)

    est = Y.mean(1) - beta(Y, X) * (X.mean(1) - mux)
    h = n // 2
    b1, b2 = beta(Y[:, :h], X[:, :h]), beta(Y[:, h:], X[:, h:])
    split = 0.5 * (Y[:, :h].mean(1) - b2 * (X[:, :h].mean(1) - mux) + Y[:, h:].mean(1) - b1 * (X[:, h:].mean(1) - mux))
    price = float(call(**P))
    se_e, se_s = est.std() / np.sqrt(R), split.std() / np.sqrt(R)
    assert n * (est.mean() - price) == pytest.approx(-8.74, abs=max(0.3, 4 * n * se_e))
    assert abs(split.mean() - price) < 4 * se_s


def test_unbiased_when_regression_is_linear():
    # (X, Y) bivariate normal: the in-sample estimator is exactly unbiased, and its variance is
    # (1 - rho^2) Var Y / n times (n - 2)/(n - 3) (Lavenberg--Welch, one control).
    rng = np.random.default_rng(10)
    n, R, rho = 8, 400_000, 0.9
    X = rng.standard_normal((R, n))
    Y = 1.0 + rho * X + np.sqrt(1 - rho**2) * rng.standard_normal((R, n))
    xc = X - X.mean(1, keepdims=True)
    b = (xc * (Y - Y.mean(1, keepdims=True))).sum(1) / (xc * xc).sum(1)
    est = Y.mean(1) - b * X.mean(1)
    assert abs(est.mean() - 1.0) < 4 * est.std() / np.sqrt(R)
    assert est.var() == pytest.approx((1 - rho**2) / n * (n - 2) / (n - 3), rel=0.01)


def test_multiple_controls_regression():
    rng = np.random.default_rng(12)
    z = rng.standard_normal(50_000)
    d = DEVIOUS
    Y, X1 = sqrt_option_samples(z, **d)
    X2 = np.exp(-d["r"] * d["T"]) * terminal_price(z, d["S"], d["T"], d["r"], d["y"], d["sigma"])
    mus = [sqrt_control_mean(**d), d["S"] * np.exp(-d["y"] * d["T"])]
    out = regression_control_variates(Y, np.column_stack([X1, X2]), mus)
    # the intercept equals Ybar - beta^T (Xbar - mu)
    Xs = np.column_stack([X1, X2])
    assert out["mean"] == pytest.approx(Y.mean() - out["beta"] @ (Xs.mean(0) - mus), rel=1e-12)
    single = control_variate(Y, X1, mus[0])
    assert out["R2"] >= single["rho"] ** 2 - 1e-12                  # adding a control never hurts
    assert abs(out["mean"] - 7.36429) < 4 * out["se"]


def test_delta_hedge_control():
    # m = 1: the stock control with beta = N(d_+) = 0.520, exact ratio 0.476 (quoted 0.48).
    p = scalar_payoffs()
    c, st, zs = p["call"], p["stock"], p["z_star"]
    from mfe230d.bs import d_pm
    from scipy.special import ndtr
    b = float(ndtr(d_pm(S, K, T, r, y, SIG)[0]))
    my, mx = gauss_expect(c, [zs]), gauss_expect(st)
    vy = gauss_expect(lambda u: c(u) ** 2, [zs]) - my * my
    vx = gauss_expect(lambda u: st(u) ** 2) - mx * mx
    cxy = gauss_expect(lambda u: c(u) * st(u), [zs]) - mx * my
    assert b == pytest.approx(0.520, abs=5e-4)
    assert np.sqrt((vy - 2 * b * cxy + b * b * vx) / vy) == pytest.approx(0.476, abs=5e-4)
    # Simulated ratios about 0.48, 0.15 and 0.075 for m = 1, 12, 52 dates (20,000 paths, seed 3).
    for m, ratio in ((1, 0.48), (12, 0.15), (52, 0.075)):
        z = np.random.default_rng(3).standard_normal((20_000, m))
        Y, M = delta_hedge_control(z, **P)
        assert abs(M.mean()) < 4 * M.std() / np.sqrt(M.size)
        out = control_variate(Y, M, 0.0, beta=1.0)
        assert out["se"] / plain(Y)["se"] == pytest.approx(ratio, abs=0.01)
        assert abs(out["mean"] - 7.36429) < 4 * out["se"]


def test_zero_denominator_convention_and_atom():
    from scipy.special import ndtr
    # The tangent call vanishes with probability N(0.15) = 0.56 (devious inputs).
    d = DEVIOUS
    assert float(ndtr(strike_to_z(d["K"], d["S"], d["T"], d["r"], d["y"], d["sigma"]))) == pytest.approx(0.56, abs=0.005)
    out = control_variate(np.array([1.0, 2.0, 4.0]), np.zeros(3), 0.5)
    assert out["beta"] == 0.0 and out["mean"] == pytest.approx(7.0 / 3.0)


def test_mm_bias_constants_for_monomials():
    # -E[f He_4]/4 is -6 for z^4 and -90 for z^6.
    assert mm_bias_constant(lambda u: u**6) == pytest.approx(-90.0, rel=1e-9)
    n, R = 32, 100_000
    z = np.random.default_rng(13).standard_normal((R, n))
    m6 = (moment_match(z, "log") ** 6).mean()
    assert n * (m6 - 15.0) == pytest.approx(-90.0, abs=12.0)


def test_other_numbers_quoted_in_the_control_section():
    p = scalar_payoffs()
    c, st, zs = p["call"], p["stock"], p["z_star"]
    # ratio estimator: beta_h = E Y / mu_X
    assert float(call(**P)) / (S * np.exp(-y * T)) == pytest.approx(0.075, abs=5e-4)
    # combination: correlation of the even parts, and of the raw payoffs
    ce = lambda u: 0.5 * (c(u) + c(-u))
    se = lambda u: 0.5 * (st(u) + st(-u))
    fe, _ = regression_factor(ce, [se], [zs])
    fr, _ = regression_factor(c, [st], [zs])
    assert np.sqrt(1 - fe) == pytest.approx(0.9645, abs=5e-5)
    assert np.sqrt(1 - fr) == pytest.approx(0.882, abs=5e-4)
    assert 0.573 * 0.472 == pytest.approx(0.270, abs=5e-4)
    # the control Z is useless after pairing: its even part is zero, Cov(f_e, Z) = 0
    assert gauss_expect(lambda u: ce(u) * u, [zs, -zs]) == pytest.approx(0.0, abs=1e-9)
    # square-root call: R^2 with one and two controls
    th = theory_table()
    assert th["sqrt", "rho"] ** 2 == pytest.approx(0.9921, abs=5e-5)
    assert 1 - th["sqrt", "cv-multiple"] ** 2 == pytest.approx(0.9934, abs=5e-5)
    assert (th["sqrt", "cv-beta1"] / th["sqrt", "cv-beta*"]) ** 2 == pytest.approx(7.4, abs=0.05)
    # moment-matching batches: bias 1.75 * 256 / 2^20 against the standard error 2.2e-3
    assert 1.75 * 256 / 2**20 == pytest.approx(4e-4, abs=0.3e-4)
    assert laboratory()["call"]["mm-log"]["se"] == pytest.approx(2.2e-3, abs=0.05e-3)
    # efficiencies in the table: 1 / (2 ratio^2)
    eff = {("stock", "antithetic"): 25.5, ("call", "antithetic"): 1.5, ("call", "mm-log"): 14,
           ("call", "mm-exp"): 15, ("call", "cv-stock"): 2.2, ("call", "anti+cv"): 22,
           ("sqrt", "cv-beta1"): 8.5, ("sqrt", "cv-beta*"): 63, ("sqrt", "cv-multiple"): 75}
    for key, val in eff.items():
        e = 1 / (2 * float(th[key]) ** 2)
        digits = 1 if val < 30 and val != int(val) else 0
        assert round(e, digits) == pytest.approx(val), key
    assert 1 / (2 * float(th["stock", "mm-log"]) ** 2) == pytest.approx(1.9e3, abs=50)


def test_crn_variances_per_draw():
    a, b = 0.20, 0.21
    fa = lambda u: DISC * max(float(terminal_price(u, S, T, r, y, a)) - K, 0.0)
    fb = lambda u: DISC * max(float(terminal_price(u, S, T, r, y, b)) - K, 0.0)
    ks = [strike_to_z(K, S, T, r, y, a), strike_to_z(K, S, T, r, y, b)]
    ma, mb = gauss_expect(fa, ks), gauss_expect(fb, ks)
    va = gauss_expect(lambda u: fa(u) ** 2, ks) - ma * ma
    vb = gauss_expect(lambda u: fb(u) ** 2, ks) - mb * mb
    cab = gauss_expect(lambda u: fa(u) * fb(u), ks) - ma * mb
    assert va + vb == pytest.approx(334.85, abs=0.005)
    assert va + vb - 2 * cab == pytest.approx(0.535, abs=5e-4)
    assert cab / np.sqrt(va * vb) == pytest.approx(0.99999, abs=5e-6)


# ------------------------------------------------------------------ the laboratory (Table in the notes)
def test_laboratory_numbers_quoted_in_the_notes():
    th = theory_table()
    lab = laboratory()                 # n = 2^20, seed 1, 256 batches
    ratio = lambda t, k: lab[t][k]["se"] / lab[t]["plain"]["se"]
    quoted_lab = {("stock", "antithetic"): 0.140, ("stock", "mm-log"): 0.017,
                  ("call", "antithetic"): 0.574, ("call", "mm-log"): 0.180, ("call", "mm-exp"): 0.174,
                  ("call", "cv-stock"): 0.472, ("call", "anti+cv"): 0.152,
                  ("sqrt", "cv-beta1"): 0.242, ("sqrt", "cv-beta*"): 0.089, ("sqrt", "cv-multiple"): 0.082}
    for key, val in quoted_lab.items():
        assert round(ratio(*key), 3) == pytest.approx(val, abs=1e-9), key
    quoted_theory = {("stock", "antithetic"): 0.140, ("stock", "mm-log"): 0.016,
                     ("call", "antithetic"): 0.573, ("call", "mm-log"): 0.188, ("call", "mm-exp"): 0.182,
                     ("call", "cv-stock"): 0.472, ("call", "anti+cv"): 0.151,
                     ("sqrt", "cv-beta1"): 0.242, ("sqrt", "cv-beta*"): 0.089, ("sqrt", "cv-multiple"): 0.081}
    for key, val in quoted_theory.items():
        assert round(float(th[key]), 3) == pytest.approx(val, abs=1e-9), key
    assert lab["stock"]["mm-exp"]["se"] < 1e-9 and lab["stock"]["cv-stock"]["se"] < 1e-9
    assert lab["sqrt"]["cv-beta*"]["beta"] == pytest.approx(0.815, abs=0.001)
