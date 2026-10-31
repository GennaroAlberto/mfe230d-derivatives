"""Tests for mfe230d.asian (Chapter 15): closed forms, bounds, approximations, Monte Carlo,
and every number quoted in the notes."""
import math

import numpy as np
import pytest

from mfe230d import asian as A
from mfe230d import bs, greeks, importance, paths, qmc, stratified

P = dict(r=0.01, y=0.02, sigma=0.2)
t12 = A.TEST_DATES
REF = 4.5612893072          # m = 12 test case, recursion converged to 1e-10


# ----------------------------------------------------------------- the geometric average
def test_geometric_matches_other_chapters():
    g = A.geometric_price(100, 100, t12, **P)
    assert g == pytest.approx(4.40286, abs=5e-6)
    assert g == pytest.approx(stratified.geometric_asian_call(100, 100, 1, 0.01, 0.02, 0.2, 12), abs=1e-11)
    assert g == pytest.approx(qmc.geometric_asian_call(paths.time_grid(1, 12), 100, 100, 0.01, 0.02, 0.2), abs=1e-11)
    for m, K, T in [(16, 120.0, 1.0), (4, 90.0, 2.0)]:
        t = np.arange(1, m + 1) * T / m
        ref = importance.geometric_asian_call(100, K, T, 0.01, 0.02, 0.2, m)
        assert A.geometric_price(100, K, t, **P) == pytest.approx(ref, abs=1e-11)


def test_geometric_greeks_match_chapter13_and_bumps():
    g = A.geometric_greeks(100, 100, t12, **P)
    ref = greeks.geometric_asian_greeks(100, 100, 1, 0.01, 0.02, 0.2, 12)
    assert g["delta"] == pytest.approx(0.486882, abs=5e-7) and g["delta"] == pytest.approx(ref["delta"], abs=1e-12)
    assert g["vega"] == pytest.approx(22.4010, abs=5e-5) and g["vega"] == pytest.approx(ref["vega"], abs=1e-9)
    h = 1e-3
    f = lambda S: A.geometric_price(S, 100, t12, **P)
    assert g["gamma"] == pytest.approx((f(100 + h) - 2 * f(100) + f(100 - h)) / h**2, rel=1e-5)
    # vega for a piecewise-constant curve: parallel shift of every value
    sig = paths.PiecewiseConstant([0.5], [0.2, 0.1])
    gs = A.geometric_greeks(100, 100, t12, r=0.01, y=0.0, sigma=sig)
    bump = lambda e: A.geometric_price(100, 100, t12, r=0.01, y=0.0, sigma=paths.PiecewiseConstant([0.5], [0.2 + e, 0.1 + e]))
    assert gs["vega"] == pytest.approx((bump(1e-5) - bump(-1e-5)) / 2e-5, rel=1e-6)


def test_geometric_variance_tail_sums_and_closed_form():
    sig = paths.PiecewiseConstant([0.3, 0.7], [0.25, 0.15, 0.3])
    t = np.array([0.0, 0.2, 0.45, 0.6, 1.1])
    w = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    mu, v = A.geometric_law(100, t, 0.03, 0.01, sig, w)
    L = A.fixing_law(100, t, 0.03, 0.01, sig)
    wn = w / w.sum()
    assert v == pytest.approx(wn @ L["cov"] @ wn, rel=1e-13)
    assert mu == pytest.approx(wn @ L["mean"], rel=1e-13)
    # equal weights, constant parameters: (m+1)(2m+1)/(6m^2)
    for m in (1, 5, 12):
        _, v = A.geometric_law(100, np.arange(1, m + 1) / m, **P)
        assert v == pytest.approx(0.04 * (m + 1) * (2 * m + 1) / (6 * m * m), rel=1e-13)


def test_geometric_term_structure_by_simulation():
    sig = paths.PiecewiseConstant([0.5], [0.2, 0.1])
    r = lambda u: 0.02 * u
    exact = A.geometric_price(100, 100, t12, r=r, y=0.0, sigma=sig)
    assert exact == pytest.approx(4.677322, abs=5e-7)
    smp = A.payoff_samples(100, 100, t12, r=r, y=0.0, sigma=sig, n_paths=2**17, seed=5)
    est = smp["geometric"]
    assert abs(est.mean() - exact) < 4 * est.std() / math.sqrt(est.size)


# ----------------------------------------------------------------- degenerate cases
def test_one_fixing_is_the_european_call():
    c = bs.call(100, 100, 1, 0.01, 0.02, 0.2)
    for f in (A.geometric_price, A.lognormal_price, A.shifted_lognormal_price, A.recursion_price, A.strip_bound):
        assert f(100, 100, [1.0], **P) == pytest.approx(c, abs=1e-10)
    assert A.rqmc_reference(100, 100, [1.0], **P)["mean"] == pytest.approx(c, abs=1e-12)
    # payment after the fixing: discount from T
    assert A.geometric_price(100, 100, [1.0], **P, T=1.5) == pytest.approx(c * math.exp(-0.005), abs=1e-10)


def test_zero_volatility_and_nonpositive_strike():
    t = np.array([0.0, 0.25, 0.5, 1.0])
    L = A.fixing_law(100, t, 0.03, 0.01, 0.0)
    D = math.exp(-0.03)
    for K in (90.0, 100.0, 110.0):
        intrinsic = D * max(L["F"].mean() - K, 0.0)
        assert A.recursion_price(100, K, t, 0.03, 0.01, 0.0) == pytest.approx(intrinsic, abs=1e-12)
        assert A.geometric_price(100, K, t, 0.03, 0.01, 0.0) == pytest.approx(
            D * max(math.exp(np.log(L["F"]).mean()) - K, 0.0), abs=1e-12)
    # K <= 0: the call is a forward on the average; the put is worthless
    fwd = math.exp(-0.01) * (A.fixing_law(100, t12, **P)["F"].mean() + 5.0)
    assert A.recursion_price(100, -5.0, t12, **P) == pytest.approx(fwd, rel=1e-9)
    assert A.lognormal_price(100, -5.0, t12, **P) == pytest.approx(fwd, rel=1e-12)
    assert A.geometric_price(100, -5.0, t12, **P, kind="put") == 0.0


def test_put_call_parity_and_seasoned_strike():
    for K in (90.0, 110.0):
        c = A.geometric_price(100, K, t12, **P)
        p = A.geometric_price(100, K, t12, **P, kind="put")
        mu, v = A.geometric_law(100, t12, **P)
        assert c - p == pytest.approx(math.exp(-0.01) * (math.exp(mu + v / 2) - K), abs=1e-11)
    # a fixing at t = 0 is already known: strike shift on the other fixings
    t = np.arange(0, 13) / 12
    W, Kp = A.seasoned_strike(100.0, [100.0], [1 / 13], np.full(12, 1 / 13))
    lhs = A.recursion_price(100, 100, t, **P)
    rhs = W * A.recursion_price(100, Kp, t12, **P)
    assert lhs == pytest.approx(rhs, abs=1e-8)
    W, Kp = A.seasoned_strike(50.0, [130.0, 120.0], [0.25, 0.25], [0.25, 0.25])
    assert (W, Kp) == (0.5, pytest.approx(-25.0))


# ----------------------------------------------------------------- moments
def test_moments_against_direct_sums():
    sig = paths.PiecewiseConstant([0.3], [0.3, 0.15])
    t = np.array([0.0, 0.2, 0.5, 0.9])
    w = np.array([0.1, 0.4, 0.2, 0.3])
    mom = A.arithmetic_moments(100, t, 0.02, 0.01, sig, w, order=3)
    L = A.fixing_law(100, t, 0.02, 0.01, sig)
    a, E = w * L["F"], np.exp(L["cov"])
    assert mom[1] == pytest.approx(a.sum(), rel=1e-13)
    assert mom[2] == pytest.approx(a @ E @ a, rel=1e-13)
    assert mom[3] == pytest.approx(np.einsum("i,j,k,ij,ik,jk->", a, a, a, E, E, E), rel=1e-12)


def test_shifted_lognormal_fit_recovers_parameters():
    tau, a, b2 = -3.0, 0.7, 0.09
    m = math.exp(a + b2 / 2)
    e = math.exp(b2)
    mu2 = m * m * (e - 1)
    mu3 = (e + 2) * math.sqrt(e - 1) * mu2**1.5
    t_, mY, b2_ = A.shifted_lognormal_fit(tau + m, mu2, mu3)
    assert (t_, mY, b2_) == pytest.approx((tau, m, b2), rel=1e-11)


def test_lognormal_density_derivatives():
    a, b2, x, h = 4.6, 0.04, 97.0, 1e-3
    l = lambda u: A._lognormal_density_derivatives(u, a, b2)[0]
    _, l1, l2 = A._lognormal_density_derivatives(x, a, b2)
    assert l1 == pytest.approx((l(x + h) - l(x - h)) / (2 * h), rel=1e-6)
    assert l2 == pytest.approx((l(x + h) - 2 * l(x) + l(x - h)) / h**2, rel=1e-5)


# ----------------------------------------------------------------- the reference price
def test_recursion_reference_and_convergence():
    a = A.recursion_price(100, 100, t12, **P)
    b = A.recursion_price(100, 100, t12, **P, N=8001, nq=96)
    assert a == pytest.approx(REF, abs=2e-10) and b == pytest.approx(REF, abs=2e-10)


def test_recursion_reproduces_published_benchmarks():
    # Cerny and Kyriakou (2011), Table 4: S0 = 100, T = 1, sigma = 0.17801, r = 0.0367, y = 0,
    # 13 fixings t = 0, 1/12, ..., 1 (the initial price is included in the average).
    t = np.arange(13) / 12
    for K, ref in [(90.0, 11.9049157), (100.0, 4.8819616), (110.0, 1.3630380)]:
        assert A.recursion_price(100, K, t, r=0.0367, y=0.0, sigma=0.17801) == pytest.approx(ref, abs=1e-7)


def test_conditional_rqmc_agrees_with_recursion():
    rq = A.rqmc_reference(100, 100, t12, **P, n=2**12, R=8, seed=3)
    assert abs(rq["mean"] - REF) < 4 * rq["se"] and rq["se"] < 2e-5
    sig = paths.PiecewiseConstant([0.5], [0.2, 0.1])
    r = lambda u: 0.02 * u
    ref_ts = A.recursion_price(100, 100, t12, r=r, y=0.0, sigma=sig)
    assert ref_ts == pytest.approx(4.780529, abs=5e-7)
    rq = A.rqmc_reference(100, 100, t12, r=r, y=0.0, sigma=sig, n=2**12, R=8, seed=4)
    assert abs(rq["mean"] - ref_ts) < 4 * rq["se"]
    flat = A.recursion_price(100, 100, t12, r=r, y=0.0, sigma=float(paths.term_vol(sig, 1.0)))
    assert flat == pytest.approx(4.027961, abs=5e-7)


def test_conditional_samples_are_unbiased():
    rng = np.random.default_rng(8)
    t = np.array([0.0, 0.3, 0.6, 1.0])                      # includes a fixing at 0
    c = A.conditional_samples(100, 95, t, rng.standard_normal((2**15, 3)), **P)
    plain = A.payoff_samples(100, 95, t, **P, n_paths=2**15, seed=9)["arithmetic"]
    se = math.sqrt(c.var() / c.size + plain.var() / plain.size)
    assert abs(c.mean() - plain.mean()) < 4 * se
    assert c.std() < 0.05 * plain.std()


def test_chapter9_m16_values_within_their_errors():
    t = np.arange(1, 17) / 16
    for K, v, se in [(100, 4.4963, 4.0e-4), (120, 0.36113, 1.4e-4), (140, 0.012755, 1.5e-5), (160, 2.7333e-4, 6.6e-7)]:
        assert abs(A.recursion_price(100, K, t, **P, N=2001, nq=48) - v) < 2 * se


# ----------------------------------------------------------------- bounds and approximations
@pytest.mark.parametrize("K,sigma", [(80.0, 0.2), (100.0, 0.2), (120.0, 0.2), (100.0, 0.5)])
def test_bound_ordering(K, sigma):
    Q = dict(r=0.01, y=0.02, sigma=sigma)
    ref = A.recursion_price(100, K, t12, **Q, N=2001, nq=48)
    first = A.curran_first_term(100, K, t12, **Q)
    lb = A.conditioning_lower_bound(100, K, t12, **Q)
    assert A.geometric_price(100, K, t12, **Q) <= first <= lb <= ref
    assert ref <= A.conditional_upper_bound(100, K, t12, **Q) and ref <= A.strip_bound(100, K, t12, **Q)
    x = A.curran_threshold(100, K, t12, **Q)
    for dx in (-0.05, 0.05):
        assert A.threshold_bound(100, K, t12, x + dx, **Q) <= lb


def test_approximation_values_quoted_in_the_notes():
    q = lambda f, **kw: f(100, 100, t12, **P, **kw)
    assert q(A.lognormal_price) == pytest.approx(4.568041, abs=5e-7)
    assert q(A.shifted_lognormal_price) == pytest.approx(4.562922, abs=5e-7)
    assert q(A.edgeworth_price) == pytest.approx(4.561478, abs=5e-7)
    assert q(A.curran_first_term) == pytest.approx(4.560107, abs=5e-7)
    assert q(A.conditioning_lower_bound) == pytest.approx(4.560841, abs=5e-7)
    assert q(A.curran_refined) == pytest.approx(4.561290, abs=5e-7)
    assert q(A.conditional_upper_bound) == pytest.approx(4.62117, abs=5e-6)
    assert q(A.strip_bound) == pytest.approx(5.272144, abs=5e-7)
    # the table of the notes: reference and errors in units of 1e-4 (sigma = 20%, then 40% at K = 100)
    table = [(0.2, 80, 19.414851, (164.6, -10.0, 4.8, -16.7, -4.6, -0.03)),
             (0.2, 90, 10.679692, (315.8, -1.5, -98.4, -17.2, -6.5, 0.03)),
             (0.2, 100, 4.561289, (67.5, 16.3, 1.9, -11.8, -4.5, 0.01)),
             (0.2, 110, 1.492200, (-216.1, 8.4, 66.1, -14.4, -5.3, -0.02)),
             (0.2, 120, 0.383703, (-210.0, -3.4, 17.3, -16.1, -5.2, -0.01)),
             (0.4, 100, 9.354588, (660.3, 109.0, 86.3, -94.4, -36.0, 0.16))]
    fs = (A.lognormal_price, A.shifted_lognormal_price, A.edgeworth_price, A.curran_first_term,
          A.conditioning_lower_bound, A.curran_refined)
    for sig, K, ref, errs in table:
        Q = dict(r=0.01, y=0.02, sigma=sig)
        r_ = A.recursion_price(100, K, t12, **Q, N=2001, nq=48)
        assert r_ == pytest.approx(ref, abs=5e-7)
        got = [1e4 * (f(100, K, t12, **Q) - r_) for f in fs]
        assert got[:5] == pytest.approx(errs[:5], abs=0.051)
        assert got[5] == pytest.approx(errs[5], abs=0.0051)


def test_skewness_positive_and_moment_matching_not_a_bound():
    m1, mu2, mu3, _ = A._central(A.arithmetic_moments(100, t12, **P))
    assert mu3 > 0
    e90 = A.lognormal_price(100, 90, t12, **P) - A.recursion_price(100, 90, t12, **P, N=2001, nq=48)
    e110 = A.lognormal_price(100, 110, t12, **P) - A.recursion_price(100, 110, t12, **P, N=2001, nq=48)
    assert e90 > 0 > e110


# ----------------------------------------------------------------- control variates
def test_control_variate_laboratory_numbers():
    lab = A.cv_laboratory(100, 100, t12, **P, n=2**18, seed=2026)
    g, tr = lab["geometric"], lab["triple"]
    assert abs(g["mean"] - REF) < 4 * g["se"] and abs(tr["mean"] - REF) < 4 * tr["se"]
    assert abs(lab["plain"]["mean"] - REF) < 4 * lab["plain"]["se"]
    assert g["rho"] == pytest.approx(0.99966, abs=3e-5)
    assert tr["R2"] == pytest.approx(0.99962, abs=3e-5)
    assert tr["beta"] == pytest.approx([0.9285, 0.0108, 0.0857], abs=2e-3)


def test_control_variate_full_run_pinned():
    lab = A.cv_laboratory(100, 100, t12, **P, n=2**20, seed=2026)
    assert lab["sd"] == pytest.approx(7.4981, abs=1e-4)
    assert lab["geometric"]["mean"] == pytest.approx(4.56146, abs=1e-5)
    assert lab["geometric"]["se"] == pytest.approx(1.91e-4, abs=1e-6)
    assert lab["triple"]["mean"] == pytest.approx(4.56137, abs=1e-5)
    assert lab["triple"]["se"] == pytest.approx(1.43e-4, abs=1e-6)
    assert lab["triple"]["R2"] == pytest.approx(0.999618, abs=1e-6)


def test_exact_correlation_and_dispersion_bound():
    pm = A.geometric_pair_moments(100, 100, t12, **P, n=2**12, R=8)
    assert pm["rho"] == pytest.approx(0.999656, abs=2e-6)
    assert pm["beta"] == pytest.approx(1.02940, abs=2e-5)
    assert pm["varY"] == pytest.approx(56.040, abs=2e-3)
    db = A.dispersion_bound(100, t12, **P)
    assert db == pytest.approx(0.19708, abs=1e-5)
    smp = A.payoff_samples(100, 100, t12, **P, n_paths=2**16, seed=12)
    d = smp["arithmetic"] - smp["geometric"]
    assert d.min() >= 0
    assert np.mean(d**2) < db
    # E[S_j G] closed form used by the bound
    g = A.geometric_conditioning(100, t12, **P)
    Pth = A.fixing_paths(100, t12, **P, n_paths=2**16, seed=13)
    G = np.exp(np.log(Pth).mean(axis=1))
    est = (Pth[:, 5] * G)
    exact = g["F"][5] * math.exp(g["mu_G"] + g["v_G"] / 2 + g["c"][5])
    assert abs(est.mean() - exact) < 4 * est.std() / math.sqrt(est.size)


def test_reference_run_quoted_in_the_notes():
    rq = A.rqmc_reference(100, 100, t12, **P, n=2**14, R=32, seed=99)
    assert rq["mean"] == pytest.approx(4.5612897, abs=5e-8)
    assert rq["se"] == pytest.approx(2.6e-6, abs=5e-8)
    # exact standard deviation per draw of the conditional estimator: RQMC on its second moment
    m1, m2 = [], []
    for s in np.random.SeedSequence(5).spawn(8):
        c = A.conditional_samples(100, 100, t12, qmc.to_normals(qmc.rqmc_points(2**14, 11, "owen", s)), **P)
        m1.append(c.mean())
        m2.append(np.mean(c * c))
    assert math.sqrt(np.mean(m2) - np.mean(m1) ** 2) == pytest.approx(0.1082, abs=4e-4)


def test_edgeworth_density_moments_and_sign():
    x = np.linspace(1.0, 600.0, 300001)
    h = x[1] - x[0]
    for sig, positive in ((0.2, True), (0.4, False)):
        f = A.edgeworth_density(x, 100, t12, 0.01, 0.02, sig)
        m1, mu2, mu3, mu4 = A._central(A.arithmetic_moments(100, t12, 0.01, 0.02, sig))
        mean = h * np.sum(x * f)
        assert h * np.sum(f) == pytest.approx(1.0, abs=1e-9) and mean == pytest.approx(m1, rel=1e-9)
        assert h * np.sum((x - mean) ** 3 * f) == pytest.approx(mu3, rel=1e-6)
        assert h * np.sum((x - mean) ** 4 * f) == pytest.approx(mu4, rel=1e-6)
        if positive:
            assert f.min() >= 0.0
        else:
            assert f.min() == pytest.approx(-1.31e-3, abs=1e-5) and abs(x[f.argmin()] - 52) < 0.5
        for K in (90.0, 110.0):            # the price is the integral of the payoff against f
            num = math.exp(-0.01) * h * np.sum(np.maximum(x - K, 0.0) * f)
            assert num == pytest.approx(A.edgeworth_price(100, K, t12, 0.01, 0.02, sig), abs=1e-7)
    # at sigma = 20% the corrected function stays above 0.81 times the lognormal density
    m1, mu2, _, _ = A._central(A.arithmetic_moments(100, t12, **P))
    b2 = math.log1p(mu2 / m1**2)
    z = (np.log(x) - math.log(m1) + b2 / 2) / math.sqrt(b2)
    l0 = np.exp(-z * z / 2) / (x * math.sqrt(2 * math.pi * b2))
    keep = l0 > 1e-12
    assert (A.edgeworth_density(x, 100, t12, **P)[keep] / l0[keep]).min() == pytest.approx(0.81, abs=0.005)


def test_numbers_of_the_referee_round():
    # an Asian call dearer than the European call at a low strike when y > r (introduction)
    assert A.recursion_price(100, 60, t12, **P, N=2001, nq=48) == pytest.approx(39.06759, abs=1e-5)
    assert bs.call(100, 60, 1, 0.01, 0.02, 0.2) == pytest.approx(38.64695, abs=1e-5)
    # the strike-free bound relative to Var Y, against 1 - rho^2 (Section 15.3)
    db = A.dispersion_bound(100, t12, **P)
    for K, ratio, one_minus in [(100, 3.5e-3, 6.9e-4), (120, 0.042, 4.3e-3), (140, 1.26, 0.027), (160, 59, 0.097)]:
        pm = A.geometric_pair_moments(100, K, t12, **P, n=2**12, R=8)
        assert db / pm["varY"] == pytest.approx(ratio, rel=0.02)
        assert 1 - pm["rho"] ** 2 == pytest.approx(one_minus, rel=0.03)
    # maxima of the errors over the strikes of Figure 15.2
    Ks = np.arange(60.0, 160.1, 2.5)
    for sig, first, three in ((0.2, 0.0019, 0.0017), (0.4, 0.0147, 0.0130)):
        ref = [A.recursion_price(100, K, t12, 0.01, 0.02, sig, N=1201, nq=32) for K in Ks]
        e1 = max(abs(A.curran_first_term(100, K, t12, 0.01, 0.02, sig) - c) for K, c in zip(Ks, ref))
        e3 = max(abs(A.shifted_lognormal_price(100, K, t12, 0.01, 0.02, sig) - c) for K, c in zip(Ks, ref))
        assert (e1, e3) == pytest.approx((first, three), abs=6e-5)
    # Edgeworth against two moments at K = 80, sigma = 40%
    Q = dict(r=0.01, y=0.02, sigma=0.4)
    ref = A.recursion_price(100, 80, t12, **Q)
    assert A.edgeworth_price(100, 80, t12, **Q) - ref == pytest.approx(-0.238, abs=5e-4)
    assert A.lognormal_price(100, 80, t12, **Q) - ref == pytest.approx(0.141, abs=5e-4)
    # the seasoned option of the exercises, one fixing left
    W, Kp = A.seasoned_strike(100.0, np.full(11, 100.0), np.full(11, 1 / 12), [1 / 12])
    assert Kp == pytest.approx(100.0)
    assert W * bs.call(105, Kp, 1 / 12, 0.01, 0.02, 0.2) == pytest.approx(0.465083, abs=1e-6)
