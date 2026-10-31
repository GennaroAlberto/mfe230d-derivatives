"""Checks of the Chapter 9 code against the results proved in the notes and the numbers quoted there."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import poisson

from mfe230d import importance as im
from mfe230d.bs import TEST_CASE, call, digital_call
from mfe230d.montecarlo import mc_estimate
from mfe230d.variance import strike_to_z

P = TEST_CASE
S, K, T, r, y, SIG = (P[k] for k in ("S", "K", "T", "r", "y", "sigma"))
SD = SIG * np.sqrt(T)


# ---------------------------------------------------------------- likelihood ratio, zero variance
def test_weight_has_mean_one_and_unbiased_estimator():
    # Proposition is-likelihood-ratio (i): E~[L] = 1, E~[f L] = E_Q[f].
    x = 0.3 + 0.8 * np.random.default_rng(0).standard_normal(400_000)
    L = np.exp(im.gaussian_log_lr(x, 0.0, 1.0, 0.3, 0.8))
    assert L.mean() == pytest.approx(1.0, abs=0.01)
    Y, _, _ = im.sample_terminal_is("call", S, K, T, r, y, SIG, 0.26, 1.0, 400_000, seed=1)
    e = mc_estimate(Y)
    assert abs(e["mean"] - call(**P)) < 4 * e["se"]


def test_zero_variance_for_the_forward():
    # alpha = sigma^2 (the share measure): S_T L is the forward, on every sample.
    _, ST, L = im.sample_terminal_is("call", S, K, T, r, y, SIG, SIG**2, 1.0, 1000, seed=2)
    assert np.allclose(ST * L, S * np.exp((r - y) * T), rtol=1e-12)


# ---------------------------------------------------------------- Gaussian shift and scale
@pytest.mark.parametrize("c,beta", [(0.0, 1.0), (0.7, 1.0), (0.5, 0.8), (-1.0, 1.5), (0.3, 0.72)])
def test_weight_second_moment_closed_form(c, beta):
    # E~[L^2] = int phi(z)^2 / q~(z) dz against quadrature (Proposition is-gaussian (i)).
    log_integrand = lambda z: -z * z - np.log(2 * np.pi) + 0.5 * ((z - c) / beta) ** 2 + np.log(beta * np.sqrt(2 * np.pi))  # noqa: E731
    val = quad(lambda z: np.exp(log_integrand(z)), -60, 60, limit=400, points=[0.0, c])[0]
    assert im.weight_second_moment(c, beta) == pytest.approx(val, rel=1e-8)
    if beta == 1.0:
        assert im.weight_second_moment(c, 1.0) == pytest.approx(np.exp(c * c), rel=1e-12)


def test_infinite_variance_at_and_below_the_edge():
    assert im.weight_second_moment(0.3, 1 / np.sqrt(2)) == np.inf
    assert im.weight_second_moment(0.0, 0.6) == np.inf
    assert im.is_moments("call", S, K, T, r, y, SIG, 0.27, 0.6)[1] == np.inf


@pytest.mark.parametrize("kind,alpha,beta", [("call", 0.26, 1.0), ("call", 0.3, 0.8), ("call", -0.1, 1.3),
                                             ("digital", 0.4, 1.0), ("digital", 0.2, 0.9)])
def test_mirror_formula_against_quadrature(kind, alpha, beta):
    # E~[(e^{-rT} f L)^2] = e^{-2rT} int f^2 q^2 / q~ dx (Proposition is-gaussian (ii)).
    m = (r - y - 0.5 * SIG**2) * T
    f = (lambda x: np.maximum(S * np.exp(x) - K, 0.0)) if kind == "call" else (lambda x: float(S * np.exp(x) > K))
    q = lambda x, mu, sd: np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))  # noqa: E731
    k = np.log(K / S)
    val = quad(lambda x: f(x) ** 2 * q(x, m, SD) ** 2 / q(x, m + alpha * T, beta * SD), k, k + 6, limit=400)[0]
    price, m2 = im.is_moments(kind, S, K, T, r, y, SIG, alpha, beta)
    assert m2 == pytest.approx(np.exp(-2 * r * T) * val, rel=1e-7)
    ref = call(**P) if kind == "call" else digital_call(**P)
    assert price == pytest.approx(ref, rel=1e-12)


def test_second_moment_by_simulation():
    Y, _, _ = im.sample_terminal_is("call", S, K, T, r, y, SIG, 0.3, 0.8, 400_000, seed=3)
    assert np.mean(Y**2) == pytest.approx(im.is_moments("call", S, K, T, r, y, SIG, 0.3, 0.8)[1], rel=0.02)


def test_course_variance_ratios():
    # Numbers quoted in Section is-gaussian (lecture 8&9 contour plots).
    for strike, a_star, ratio in ((90, 0.2063, 9.032), (100, 0.2581, 10.583), (110, 0.3144, 14.293)):
        a = im.optimal_alpha("call", S, strike, T, r, y, SIG)
        assert a == pytest.approx(a_star, abs=1e-4)
        assert im.variance_ratio("call", S, strike, T, r, y, SIG, a) == pytest.approx(ratio, abs=2e-3)
    for beta, a_star, ratio in ((1 / np.sqrt(2) + 1e-9, 0.2702, 34.61), (0.75, 0.2667, 30.91)):
        a = im.optimal_alpha("call", S, K, T, r, y, SIG, beta=beta, bracket=(0, 1))
        assert a == pytest.approx(a_star, abs=1e-4)
        assert im.variance_ratio("call", S, K, T, r, y, SIG, a, beta) == pytest.approx(ratio, abs=5e-3)
    # the variance ratio increases as beta decreases to the edge
    vr = [im.variance_ratio("call", S, K, T, r, y, SIG, 0.27, b) for b in (1.0, 0.9, 0.8, 0.75, 0.72, 0.71)]
    assert np.all(np.diff(vr) > 0)


def test_optimal_shift_tends_to_sigma_squared_for_small_strikes():
    assert im.optimal_alpha("call", S, 1.0, T, r, y, SIG) == pytest.approx(0.0404, abs=1e-4)
    assert im.optimal_alpha("call", S, 20.0, T, r, y, SIG) == pytest.approx(0.0501, abs=1e-4)


def test_quoted_standard_deviations():
    for beta, sd in ((1.0, 3.884), (0.75, 2.266)):
        p, m2 = im.is_moments("call", S, K, T, r, y, SIG, 0.27, beta)
        assert np.sqrt(m2 - p * p) == pytest.approx(sd, abs=1e-3)
    p, m2 = im.is_moments("call", S, K, T, r, y, SIG)
    assert np.sqrt(m2 - p * p) == pytest.approx(12.570, abs=1e-3)


# ---------------------------------------------------------------- tilting
def test_poisson_tilting_example():
    lam, n_terms, a = 2.0, 10, 4.0
    theta, I = im.poisson_rate_function(lam, a)
    assert (theta, I) == pytest.approx((np.log(2), 4 * np.log(2) - 2), rel=1e-12)
    exact = poisson.sf(n_terms * a - 1, n_terms * lam)
    assert exact == pytest.approx(5.320e-5, rel=1e-3)
    assert np.exp(-n_terms * I) == pytest.approx(4.413e-4, rel=1e-3)
    smp, cgf = im.poisson_tilt(lam)
    Y = im.tilted_sum_estimator(smp, cgf, theta, n_terms, n_terms * a, 10**5, seed=7)
    e = mc_estimate(Y)
    assert Y.max() <= np.exp(-n_terms * I) * (1 + 1e-12)       # the estimator is bounded by e^{-nI}
    assert abs(e["mean"] - exact) < 4 * e["se"]
    assert e["sd"] / exact == pytest.approx(2.17, abs=0.05)   # relative error per sample, seed 7
    assert np.sqrt((1 - exact) / exact) == pytest.approx(137.1, abs=0.1)


# ---------------------------------------------------------------- rare events
@pytest.mark.parametrize("d", [0.5, 1.0, 2.0, 4.0, 8.0, 20.0])
def test_mills_bounds_and_shift_to_the_boundary(d):
    p = np.exp(im.log_tail(d))
    phi = np.exp(-0.5 * d * d) / np.sqrt(2 * np.pi)
    assert d / (1 + d * d) * phi <= p <= phi / d
    rel = im.tail_shift_relative_moment(d, d)
    assert 2 * np.sqrt(2 * np.pi) * d**3 / (1 + 4 * d * d) <= rel <= np.sqrt(2 * np.pi) * (1 + d * d) ** 2 / (2 * d**3)
    assert rel <= np.sqrt(2 * np.pi) * (1 + d * d) / d          # the GHS bound e^{H(mu)}/p
    assert im.tail_exponential_relative_moment(d) <= (1 + d * d) / d**2


def test_shift_relative_moment_asymptotics():
    d = 60.0
    assert im.tail_shift_relative_moment(d, d) / (np.sqrt(np.pi / 2) * d) == pytest.approx(1.0, abs=0.01)


def test_exponential_tail_sampler():
    d = 3.0
    Y = im.tail_exponential_is(d, 200_000, seed=4)
    p = np.exp(im.log_tail(d))
    assert Y.mean() == pytest.approx(p, rel=0.005)
    assert np.mean(Y**2) / p**2 == pytest.approx(im.tail_exponential_relative_moment(d), rel=0.01)


def test_rare_event_table():
    rows = {150: (2.177, 1.587, 2.379, 1.563, 2.531, 1.577, 0.234, 8.18, 2.581, 1.010, 2.731, 0.998, 2.844, 1.005, 11.43),
            200: (3.616, 2.018, 3.747, 2.007, 3.861, 2.015, 0.119, 81.71, 3.880, 1.312, 3.995, 1.305, 4.092, 1.310, 115.26)}
    for strike, v in rows.items():
        d = float(strike_to_z(strike, S, T, r, y, SIG))
        rel = lambda kind, c: np.sqrt(im.is_moments(kind, S, strike, T, r, y, SIG, c * SD / T)[1]  # noqa: E731
                                      / im.is_moments(kind, S, strike, T, r, y, SIG)[0] ** 2 - 1)
        od = im.optimal_alpha("digital", S, strike, T, r, y, SIG) * T / SD
        oc = im.optimal_alpha("call", S, strike, T, r, y, SIG) * T / SD
        gc, cd, cc = im.ghs_shift_call(d, SD), im.ce_shift_digital(d), im.ce_shift_call(d, SD)
        got = (d, rel("digital", d), od, rel("digital", od), cd, rel("digital", cd),
               np.sqrt(im.tail_exponential_relative_moment(d) - 1), rel("digital", 0),
               gc, rel("call", gc), oc, rel("call", oc), cc, rel("call", cc), rel("call", 0))
        assert got == pytest.approx(v, abs=6e-3)


def test_ce_shifts_against_quadrature():
    for d in (0.5, 2.0, 4.0):
        f = lambda z: np.expm1(SD * (z - d)) * np.exp(-0.5 * z * z)  # noqa: E731
        ref = quad(lambda z: z * f(z), d, d + 40)[0] / quad(f, d, d + 40)[0]
        assert im.ce_shift_call(d, SD) == pytest.approx(ref, rel=1e-9)
        assert im.ce_shift_digital(d) > d                      # CE overshoots the boundary point
        g = im.ghs_shift_call(d, SD)                           # grad log G(g) = g
        assert g * np.expm1(SD * (g - d)) == pytest.approx(SD * np.exp(SD * (g - d)), rel=1e-10)


# ---------------------------------------------------------------- path space
def test_piecewise_constant_girsanov_weight_is_the_discrete_weight():
    # Theorem is-girsanov (iii): the continuous-time density equals exp(-mu.Z + |mu|^2/2).
    rng = np.random.default_rng(5)
    m, k, T_ = 8, 64, 1.0
    dt = T_ / m
    mu = rng.normal(size=m)
    dB = rng.normal(scale=np.sqrt(dt / k), size=(1000, m, k))       # fine increments of B
    theta = mu / np.sqrt(dt)                                         # theta on each coarse interval
    stoch = np.einsum("j,njk->n", theta, dB)                         # int theta dB
    L_cont = np.exp(-stoch - 0.5 * np.sum(theta**2) * T_ / m)
    Z = dB.sum(axis=2) / np.sqrt(dt) + mu                            # increments of W~ = B + int theta
    assert np.allclose(L_cont, im.girsanov_weight(Z, mu), rtol=1e-10)


def test_weight_second_moment_is_exp_energy():
    mu = np.array([0.4, 0.3, 0.2, 0.1])
    Z = np.random.default_rng(6).standard_normal((400_000, 4)) + mu
    L = im.girsanov_weight(Z, mu)
    assert L.mean() == pytest.approx(1.0, abs=0.01)
    assert np.mean(L**2) == pytest.approx(np.exp(mu @ mu), rel=0.02)


def test_ghs_zero_variance_for_log_linear_payoff():
    v, kappa = np.array([0.5, -0.2, 0.3]), 0.1
    F = lambda z: (kappa + v @ z, v)  # noqa: E731
    mu = im.ghs_drift(F, np.zeros(3))
    assert np.allclose(mu, v, atol=1e-8)
    Z = np.random.default_rng(7).standard_normal((10, 3)) + mu
    Y = np.exp(kappa + Z @ v) * im.girsanov_weight(Z, mu)
    assert np.allclose(Y, np.exp(kappa + 0.5 * v @ v), rtol=1e-10)


def test_ghs_bound_for_the_log_concave_call():
    # Proposition is-optimal-drift (iii): Y <= e^{H(mu)} surely for G(z) = (e^{s(z-d)} - 1)^+.
    d = 2.5
    mu = im.ghs_shift_call(d, SD)
    H = np.log(np.expm1(SD * (mu - d))) - 0.5 * mu * mu
    z = mu + np.random.default_rng(8).standard_normal(200_000)
    Y = np.maximum(np.expm1(SD * (z - d)), 0.0) * np.exp(-mu * z + 0.5 * mu * mu)
    assert Y.max() <= np.exp(H) * (1 + 1e-12)


def test_geometric_asian_closed_form():
    Y, C, _ = im.asian_is_samples(S, 100.0, T, r, y, SIG, 16, 0.0, 400_000, seed=9)
    e = mc_estimate(C)
    assert abs(e["mean"] - im.geometric_asian_call(S, 100.0, T, r, y, SIG, 16)) < 4 * e["se"]


def test_asian_laboratory_numbers():
    lab = im.asian_laboratory()
    mu1 = [row["mu"][0] for row in lab]
    norms = [np.linalg.norm(row["mu"]) for row in lab]
    assert mu1 == pytest.approx([0.448, 0.840, 1.236, 1.583], abs=1e-3)
    assert norms == pytest.approx([1.103, 2.110, 3.167, 4.129], abs=1e-3)
    for row in lab:                     # all four estimators agree within their errors
        ref, se = row["is_cv"]["mean"], row["is_cv"]["se"]
        for key in ("plain", "plain_cv", "is"):
            assert abs(row[key]["mean"] - ref) < 4 * np.hypot(row[key]["se"], se)
        assert np.all(np.diff(row["mu"]) < 0)          # the drift decreases along the path
    vr = {key: [row["plain_var"] / (row[key]["se"] ** 2 * row["n"]) for row in lab] for key in ("is", "is_cv")}
    vr["plain_cv"] = [row["plain_var"] / row["cv_var"] for row in lab]
    # our implementation (default_rng streams spawned from SeedSequence(2026), n = 2^17): Table tab:is-lab
    expected = {"plain_cv": [1458, 220, 34.3, 9.24], "is": [9.17, 42.3, 618, 1.894e4], "is_cv": [2560, 1621, 4359, 4.702e4]}
    for key in expected:
        assert vr[key] == pytest.approx(expected[key], rel=2e-3)
    prices = [row["is_cv"]["mean"] for row in lab]
    assert prices == pytest.approx([4.4963, 0.36113, 0.012755, 2.7333e-4], rel=1e-4)
    se = [[np.sqrt(row["plain_var"] / row["n"]), np.sqrt(row["cv_var"] / row["n"]), row["is"]["se"], row["is_cv"]["se"]]
          for row in lab]
    assert np.array(se) == pytest.approx(np.array([[2.04e-2, 5.34e-4, 6.74e-3, 4.03e-4], [5.75e-3, 3.87e-4, 8.84e-4, 1.43e-4],
                                                   [1.01e-3, 1.73e-4, 4.07e-5, 1.53e-5], [1.42e-4, 4.67e-5, 1.03e-6, 6.55e-7]]),
                                         rel=5e-3)
    # the in-sample control fit on the thin plain sample at K = 160 rests on few paying paths
    Y, _, _ = im.asian_is_samples(S, 160.0, T, r, y, SIG, 16, 0.0, 2**17, seed=np.random.SeedSequence(2026).spawn(8)[6])
    assert np.count_nonzero(Y) == 9


# ---------------------------------------------------------------- cross-entropy, ESS, controls
def test_cross_entropy_converges_to_conditional_mean_for_a_digital():
    d = 3.0
    hist, levels = im.cross_entropy_drift(lambda Z: Z[:, 0], d, 1, n=20_000, n_iter=6, kind="digital", seed=10)
    assert levels[-1] == d
    assert hist[-1][0] == pytest.approx(im.ce_shift_digital(d), abs=0.02)


def test_effective_sample_size():
    assert im.effective_sample_size(np.ones(50)) == pytest.approx(50.0)
    w = np.random.default_rng(11).exponential(size=1000)
    assert 1.0 <= im.effective_sample_size(w) <= 1000.0
    c = 0.8
    x = c + np.random.default_rng(12).standard_normal(400_000)
    L = np.exp(im.gaussian_log_lr(x, 0.0, 1.0, c, 1.0))
    assert im.effective_sample_size(L) / L.size == pytest.approx(np.exp(-c * c), rel=0.03)


def test_self_normalized_is_a_control_variate_estimator():
    Y, _, L = im.sample_terminal_is("call", S, 120.0, T, r, y, SIG, 0.3, 1.0, 5000, seed=13)
    sn = im.self_normalized(Y, L)
    assert np.mean(Y - sn * (L - 1.0)) == pytest.approx(sn, rel=1e-12)


def test_linear_control_after_the_shift_gains_little():
    # Section is-cv: exact 1 - rho^2 between f L and S_T L under the shift to the strike.
    m, disc, F = (r - y - 0.5 * SIG**2) * T, np.exp(-r * T), S * np.exp((r - y) * T)
    for strike, c, rho2 in ((150.0, 0.0, 0.1020), (200.0, 0.0, 0.002775), (100.0, 0.0, 0.7776),
                            (150.0, None, 0.003099), (200.0, None, 7.60e-7)):
        d = float(strike_to_z(strike, S, T, r, y, SIG))
        a = (im.ghs_shift_call(d, SD) if c is None else c) * SD
        q = lambda x, mu: np.exp(-0.5 * ((x - mu) / SD) ** 2) / (SD * np.sqrt(2 * np.pi))  # noqa: E731
        w = lambda x: q(x, m) ** 2 / q(x, m + a)  # noqa: E731
        f = lambda x: disc * max(S * np.exp(x) - strike, 0.0)  # noqa: E731
        lo, hi, k = min(m, m + a) - 12 * SD, max(m, m + a) + 12 * SD, np.log(strike / S)
        integ = lambda h, l=lo: quad(lambda x: h(x) * w(x), l, hi, limit=800, epsabs=0, epsrel=1e-11,  # noqa: E731
                                     points=[k] if l < k else None)[0]
        p = im.is_moments("call", S, strike, T, r, y, SIG)[0]
        vf = integ(lambda x: f(x) ** 2, k) - p * p
        vg = integ(lambda x: (S * np.exp(x)) ** 2) - F * F
        cov = integ(lambda x: f(x) * S * np.exp(x), k) - p * F
        assert cov * cov / (vf * vg) == pytest.approx(rho2, rel=3e-3)


def test_heuristic_optimal_shift_expansion():
    # Remark is-share: c* = d + 3/(2d) + o(1/d) for the call, checked numerically.
    for strike, tol in ((1000.0, 0.01), (3000.0, 0.002)):
        d = float(strike_to_z(strike, S, T, r, y, SIG))
        c = im.optimal_alpha("call", S, strike, T, r, y, SIG, bracket=(0, 10)) * T / SD
        assert (c - d) * d == pytest.approx(1.5, abs=tol)


def test_cross_entropy_asian_levels_and_overshoot():
    # Section is-ce: the target level is reached at iteration 2 or 3 and CE overshoots GHS.
    score = lambda Z: im.asian_dates_prices(Z, S, T, r, y, SIG).mean(axis=1)  # noqa: E731
    for strike, first in ((120.0, 1), (140.0, 1), (160.0, 2)):
        hist, levels = im.cross_entropy_drift(score, strike, 16, n=10_000, n_iter=8, seed=int(strike))
        assert levels.index(strike) == first
        mu = im.asian_ghs_drift(S, strike, T, r, y, SIG, 16)
        assert np.linalg.norm(mu) < np.linalg.norm(hist[-1]) < 1.15 * np.linalg.norm(mu)
    ratios = []
    for strike in (100.0, 120.0, 140.0, 160.0):
        hist, _ = im.cross_entropy_drift(score, strike, 16, n=10_000, n_iter=8, seed=int(strike))
        ratios.append(np.linalg.norm(hist[-1]) / np.linalg.norm(im.asian_ghs_drift(S, strike, T, r, y, SIG, 16)))
    assert ratios == pytest.approx([1.23, 1.13, 1.07, 1.05], abs=5e-3)
