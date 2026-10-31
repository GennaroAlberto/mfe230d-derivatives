"""Checks of the Chapter 6 code against closed forms and against every number the notes quote."""
import numpy as np
import pytest
from scipy.stats import t as student_t

from mfe230d import bs
from mfe230d import montecarlo as mc

TC = bs.TEST_CASE
S, K, T, r, y, SIG = (TC[k] for k in ("S", "K", "T", "r", "y", "sigma"))
C_EXACT = float(bs.call(**TC))
F = float(bs.forward(S, T, r, y))


def call_payoff(s):
    return np.maximum(s - K, 0.0)


# --- the estimator, standard errors, batches ------------------------------------------

def test_mc_estimate_formulas():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    e = mc.mc_estimate(x)
    assert e["mean"] == pytest.approx(3.5)
    assert e["sd"] == pytest.approx(np.sqrt(np.sum((x - 3.5) ** 2) / 3))
    assert e["se"] == pytest.approx(e["sd"] / 2)
    assert e["hi"] - e["mean"] == pytest.approx(1.959964 * e["se"], rel=1e-6)
    et = mc.mc_estimate(x, student=True)
    assert et["hi"] - et["mean"] == pytest.approx(student_t.ppf(0.975, 3) * e["se"])


def test_batch_mean_is_overall_mean_and_one_sample_batches_are_plain():
    x = np.random.default_rng(1).standard_normal(4096)
    b = mc.batch_estimate(x, m=64)
    assert b["mean"] == pytest.approx(x.mean(), abs=1e-14)
    full = mc.batch_estimate(x, m=4096)          # batches of one sample
    assert full["se"] == pytest.approx(mc.mc_estimate(x)["se"], rel=1e-12)
    with pytest.raises(ValueError):
        mc.batch_estimate(x[:1000], m=64)


def test_nested_estimates_agree_below_m():
    x = np.random.default_rng(2).standard_normal(2**12)
    est = mc.nested_estimates(x, [2, 16, 1024, 4096], m=1024)
    assert np.allclose(est["se"][:3], est["se_batch"][:3], rtol=1e-12)
    assert est["mean"][-1] == pytest.approx(x.mean())


def test_batch_se_unbiased_for_iid_samples():
    # E[V_m] = sigma^2 / n (Proposition: batch means (ii)); average over many runs.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4000, 256))
    V = mc.batch_estimate(x, m=16)["se"] ** 2
    assert V.mean() == pytest.approx(1 / 256, rel=0.03)


# --- the bare-bones Monte Carlo and the numbers of the example ---------------------------

def test_bbmc_example_numbers():
    out = mc.bs_call_bbmc(**TC, n=2**20, seed=2024)
    C_hat, se = out["price"]
    assert C_hat == pytest.approx(7.3786, abs=5e-5)
    assert se == pytest.approx(0.0123, abs=5e-5)
    assert (C_hat - C_EXACT) / se == pytest.approx(1.17, abs=0.005)
    m, s = out["ST"]
    assert m == pytest.approx(99.0263, abs=5e-5)
    assert s == pytest.approx(0.0196, abs=5e-5)
    assert F == pytest.approx(99.0050, abs=5e-5)
    disc = np.exp(-r * T)
    assert out["payoff"][0] * disc == pytest.approx(C_hat)
    batch = mc.batch_estimate(disc * out["samples"]["CT"], m=1024)
    assert batch["se"] == pytest.approx(se, rel=0.05)


def test_bbmc_is_unbiased_within_clt_error():
    for seed in (10, 11, 12):
        C_hat, se = mc.bs_call_bbmc(**TC, n=2**16, seed=seed)["price"]
        assert abs(C_hat - C_EXACT) < 4 * se


def test_exact_payoff_moments_by_quadrature():
    mean, var = mc.payoff_moments_quadrature(call_payoff, S, T, r, y, SIG)
    assert mean == pytest.approx(C_EXACT, abs=1e-6)
    assert np.sqrt(var) == pytest.approx(12.570, abs=5e-4)      # sigma_C quoted in the notes
    assert np.sqrt(var) / 1024 == pytest.approx(0.0123, abs=5e-5)
    n_cent = (1.96 * np.sqrt(var) / 0.01) ** 2
    assert n_cent == pytest.approx(6.1e6, rel=0.01)
    _, varS = mc.payoff_moments_quadrature(lambda s: s * np.exp(r * T), S, T, r, y, SIG)
    assert np.sqrt(varS) == pytest.approx(F * np.sqrt(np.exp(SIG**2 * T) - 1), rel=1e-8)


def test_shape_ratios_quoted_in_notes():
    _, _, be_call, kappa_call = mc.payoff_moments_quadrature(call_payoff, S, T, r, y, SIG, shape=True)
    _, _, be_S, _ = mc.payoff_moments_quadrature(lambda s: s, S, T, r, y, SIG, shape=True)
    assert be_S == pytest.approx(1.70, abs=0.005)
    assert be_call == pytest.approx(2.48, abs=0.005)
    assert mc.bernoulli_skewness_ratio(0.01) == pytest.approx(9.85, abs=0.005)
    assert kappa_call == pytest.approx(8.95, abs=0.005)
    assert np.sqrt((kappa_call - 1) / 4) == pytest.approx(1.41, abs=0.005)


def test_quantiles_quoted_in_notes():
    assert mc.normal_quantile(0.95) == pytest.approx(1.9600, abs=5e-5)
    assert mc.t_quantile(0.95, 999) == pytest.approx(1.9623, abs=5e-5)
    assert mc.t_quantile(0.95, 1023) == pytest.approx(1.9623, abs=5e-5)
    assert mc.t_quantile(0.95, 9) == pytest.approx(2.262, abs=5e-4)
    assert 1 - 2 * student_t.sf(1.96, 9) == pytest.approx(0.918, abs=5e-4)
    assert 0.5 * np.sqrt(2 / 1023) == pytest.approx(0.022, abs=1e-3)   # relative noise of the batch SE


# --- coverage and heavy tails (figure 6.2) ---------------------------------------------

def _samplers():
    k_dig = F * np.exp(-0.5 * SIG**2 * T + SIG * np.sqrt(T) * mc.normal_quantile(0.98))
    disc = np.exp(-r * T)

    def call(rng, shape):
        return disc * call_payoff(mc.terminal_prices(S, T, r, y, SIG, rng.standard_normal(shape)))

    def digital(rng, shape):
        return (mc.terminal_prices(S, T, r, y, SIG, rng.standard_normal(shape)) > k_dig).astype(float)

    return k_dig, call, digital


def test_digital_strike_and_probability():
    k_dig, _, _ = _samplers()
    assert k_dig == pytest.approx(154.54, abs=0.005)
    assert np.exp(r * T) * bs.digital_call(S, k_dig, T, r, y, SIG) == pytest.approx(0.01, abs=1e-12)


def test_coverage_numbers_of_figure():
    # Same seeds as code/figures/ch06_montecarlo.py: seed = 100 j + i, n = 2^(3+i).
    _, call, digital = _samplers()
    assert 0.80 <= mc.coverage(call, 8, 2000, C_EXACT, seed=0) <= 0.84      # "about 82%" in the notes
    assert mc.coverage(call, 128, 2000, C_EXACT, seed=4) > 0.94
    assert mc.coverage(digital, 32, 2000, 0.01, seed=102) < 0.30
    par = lambda rng, shape: mc.pareto(1.5, shape, rng=rng)
    assert mc.coverage(par, 8, 2000, 3.0, seed=200) == pytest.approx(0.55, abs=0.005)
    assert mc.coverage(par, 2**15, 2000, 3.0, seed=212) == pytest.approx(0.80, abs=0.01)


def test_pareto_sampler():
    x = mc.pareto(3.0, 400_000, seed=5)
    assert x.min() >= 1.0
    assert np.mean(x > 2.0) == pytest.approx(2.0**-3, abs=0.002)
    assert x.mean() == pytest.approx(mc.pareto_mean(3.0), abs=0.01)
    assert x.var() == pytest.approx(0.75, rel=0.1)                   # sigma^2 = 3/4 for alpha = 3


def test_sample_sd_grows_with_infinite_variance():
    x = mc.pareto(1.5, 2**20, seed=11)
    s_small, s_large = x[:2**10].std(ddof=1), x.std(ddof=1)
    assert s_large > 3 * s_small


def test_berry_esseen_bound():
    assert mc.berry_esseen_bound(1.0, 100) == pytest.approx(0.04748)
    # for a rare event the bound depends on np only: about 0.4748 / sqrt(np)
    b = mc.berry_esseen_bound(mc.bernoulli_skewness_ratio(1e-4), 1e6)
    assert b == pytest.approx(0.4748 / np.sqrt(100), rel=1e-3)


# --- efficiency, MSE, delta method and ratio estimator -----------------------------------

def test_relative_efficiency_examples():
    assert mc.relative_efficiency(1.0, 1.0, 0.25**2, 2.0) == pytest.approx(8.0)
    assert mc.relative_efficiency(1.0, 1.0, 0.8**2, 2.0) == pytest.approx(0.78, abs=0.005)
    assert mc.efficiency(2.0, 0.5) == pytest.approx(1.0)


def test_optimal_allocation():
    beta, gamma = 1.0, 1.0
    res = mc.optimal_allocation(1e6, c_bias=0.5, beta=beta, var=4.0, c_cost=2.0, gamma=gamma)
    assert res["bias2_share"] == pytest.approx(gamma / (2 * beta + gamma))
    mse = lambda h: 0.25 * h ** (2 * beta) + 4.0 * 2.0 * h ** (-gamma) / 1e6
    assert mse(res["h"]) < min(mse(1.01 * res["h"]), mse(0.99 * res["h"]))
    res2 = mc.optimal_allocation(8e6, c_bias=0.5, beta=beta, var=4.0, c_cost=2.0, gamma=gamma)
    assert res2["mse"] / res["mse"] == pytest.approx(8.0 ** (-2 * beta / (2 * beta + gamma)))


def test_delta_method_for_the_standard_deviation():
    # Example in the notes: sqrt(n)(s_n - sigma) -> N(0, (mu4 - sigma^4) / (4 sigma^2)).
    # Exponential(1): sigma = 1, mu4 = 9, so the limit variance is 2.
    rng = np.random.default_rng(6)
    n = 400
    s = rng.exponential(size=(20_000, n)).std(axis=1, ddof=1)
    # after centring (s_n is shift invariant): Var X = 1, Cov(X, X^2) = mu3 = 2, Var X^2 = mu4 - 1 = 8
    se_pred = mc.delta_method([0.0, 0.5], [[1.0, 2.0], [2.0, 8.0]], n)
    assert se_pred == pytest.approx(np.sqrt(2.0 / n))
    assert s.std() == pytest.approx(np.sqrt(2.0 / n), rel=0.05)


def test_ratio_estimator_bias_and_se():
    # X ~ U(1, 2), Y = X^2: theta = 14/9, Var X = 1/12, Cov(X, Y) = 1/4, E X = 3/2.
    theta, varx, cov, m = 14 / 9, 1 / 12, 0.25, 1.5
    n = 4
    bias_pred = (theta * varx - cov) / (n * m**2)
    rng = np.random.default_rng(7)
    X = 1.0 + rng.random((2_000_000, n))
    R = (X**2).mean(axis=1) / X.mean(axis=1)
    err = R.std() / np.sqrt(R.size)
    assert abs(R.mean() - theta - bias_pred) < 4 * err
    assert abs(bias_pred) > 10 * err                                       # the bias is resolved
    one = mc.ratio_estimate(X[0] ** 2, X[0])
    assert one["mean"] == pytest.approx((X[0] ** 2).mean() / X[0].mean())
    big = 1.0 + rng.random(200_000)
    est = mc.ratio_estimate(big**2, big)
    var_d = np.var(big**2 - theta * big)
    assert est["se"] == pytest.approx(np.sqrt(var_d / big.size) / m, rel=0.02)
    assert est["bias"] == pytest.approx((theta * varx - cov) / (big.size * m**2), rel=0.05)


# --- quadrature against Monte Carlo -------------------------------------------------------

def sine(x):
    return 0.5 * np.pi * np.sin(np.pi * x)


def test_one_dimensional_rules():
    lin = lambda x: 3.0 * x + 1.0
    for rule in (mc.midpoint_1d, mc.trapezoid_1d):
        assert rule(lin, 7) == pytest.approx(2.5, abs=1e-14)
    for k in (2, 8, 64):
        assert mc.riemann_1d(sine, k, "left") == pytest.approx(mc.trapezoid_1d(sine, k), abs=1e-13)
        assert mc.riemann_1d(sine, k, "right") == pytest.approx(mc.trapezoid_1d(sine, k), abs=1e-13)
        x = np.pi / (2 * k)
        assert mc.midpoint_1d(sine, k) == pytest.approx(x / np.sin(x), abs=1e-13)


def test_product_midpoint_bound_and_rate():
    for d, k in ((2, 16), (4, 6), (8, 3)):
        err = abs(mc.product_midpoint(mc.sine_product, d, k) - 1.0)
        h = 1.0 / k
        bound = h**2 / 24 * d * np.pi**2 * (np.pi / 2) ** d     # sup |d_jj f| = pi^2 (pi/2)^d
        assert err <= bound
    err = abs(mc.product_midpoint(mc.sine_product, 2, 512) - 1.0)
    assert err == pytest.approx(2 * np.pi**2 / (24 * 512**2), rel=1e-4)


def test_product_midpoint_chunking():
    f = lambda x: np.exp(x.sum(axis=1))
    a = mc.product_midpoint(f, 3, 9)
    b = mc.product_midpoint(f, 3, 9, chunk=10)
    assert a == pytest.approx(b, rel=1e-13)
    half = 0.5 / 9                                            # exact midpoint rule for exp: (h/2)/sinh(h/2) per axis
    assert a == pytest.approx(((np.e - 1) * half / np.sinh(half)) ** 3, rel=1e-12)


def test_dimension_figure_claim():
    # d = 8: Monte Carlo with 10^4 points beats the product rule with 6^8 = 1.7e6 points.
    quad = abs(mc.product_midpoint(mc.sine_product, 8, 6) - 1.0)
    rmse = np.sqrt(((np.pi**2 / 8) ** 8 - 1) / 1e4)
    assert 6**8 == pytest.approx(1.7e6, rel=0.02)
    assert rmse < quad / 4
    # d = 4: equal rates; Monte Carlo constant smaller
    assert np.sqrt((np.pi**2 / 8) ** 4 - 1) < 4 * np.pi**2 / 24


def test_mc_integrate_variance():
    d = 3
    var = (np.pi**2 / 8) ** d - 1
    e = mc.mc_integrate(mc.sine_product, d, 200_000, seed=8)
    assert e["sd"] ** 2 == pytest.approx(var, rel=0.03)
    assert abs(e["mean"] - 1.0) < 4 * e["se"]
