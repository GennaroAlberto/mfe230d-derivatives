"""Checks of the Chapter 16 code against closed forms, quadrature, the course's table and the
numbers quoted in the notes."""
import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr, zeta
from scipy.stats import multivariate_normal

from mfe230d import bs
from mfe230d.barriers import (
    BETA_BGK,
    barrier_mc_samples,
    barrier_price,
    bgk_price,
    bridge_extreme,
    corridor_price,
    crossing_probability,
    discrete_barrier_cv_study,
    discrete_barrier_price,
    discrete_corridor_price,
    double_barrier_density,
    double_barrier_price,
    double_barrier_sine_density,
    double_barrier_terms,
    double_barrier_truncation_bound,
    exp_max_moment,
    expected_max,
    extreme_tail,
    knockout_density,
    log_drift,
    lookback_fixed,
    lookback_floating,
    max_call_tail,
    spitzer_expected_max,
)

TC = dict(T=1.0, r=0.01, y=0.02, sigma=0.2)
S0 = K = 100.0
NU = log_drift(0.01, 0.02, 0.2)

# The course's table of discretely monitored down-and-out calls (Lecture 12): rows B = 85..100,
# columns m = 1, 2, 3, 4, 6, 12 and the continuous price.
COURSE_TABLE = {
    85: [7.36429, 7.31639, 7.28662, 7.26633, 7.23745, 7.18475, 6.95077],
    86: [7.36429, 7.30014, 7.25860, 7.23013, 7.19060, 7.12136, 6.82176],
    87: [7.36429, 7.27957, 7.22280, 7.18359, 7.13011, 7.04018, 6.66215],
    88: [7.36429, 7.25390, 7.17782, 7.12484, 7.05336, 6.93765, 6.46739],
    89: [7.36429, 7.22230, 7.12224, 7.05195, 6.95766, 6.80998, 6.23285],
    90: [7.36429, 7.18392, 7.05463, 6.96303, 6.84042, 6.65328, 5.95396],
    91: [7.36429, 7.13787, 6.97361, 6.85638, 6.69927, 6.46380, 5.62633],
    92: [7.36429, 7.08331, 6.87796, 6.73048, 6.53230, 6.23828, 5.24589],
    93: [7.36429, 7.01942, 6.76662, 6.58421, 6.33818, 5.97442, 4.80899],
    94: [7.36429, 6.94544, 6.63878, 6.41684, 6.11637, 5.67128, 4.31255],
    95: [7.36429, 6.86072, 6.49393, 6.22816, 5.86723, 5.32979, 3.75412],
    96: [7.36429, 6.76471, 6.33190, 6.01851, 5.59208, 4.95304, 3.13200],
    97: [7.36429, 6.65702, 6.15286, 5.78881, 5.29326, 4.54642, 2.44525],
    98: [7.36429, 6.53741, 5.95739, 5.54055, 4.97402, 4.11750, 1.69375],
    99: [7.36429, 6.40580, 5.74641, 5.27574, 4.63845, 3.67565, 0.87818],
    100: [7.36429, 6.26232, 5.52122, 4.99687, 4.29126, 3.23135, 0.00000],
}


def test_bgk_constant():
    assert BETA_BGK == pytest.approx(-zeta(0.5) / np.sqrt(2 * np.pi))
    assert round(BETA_BGK, 5) == 0.58260


# ------------------------------------------------------------------ reflection principle


def test_extreme_tail_against_density_and_at_zero():
    for b in (-0.3, -0.05, 0.05, 0.3):
        lo, hi = (b, 4.0) if b < 0 else (-4.0, b)
        surv = integrate.quad(lambda x: float(knockout_density(x, b, 1.0, NU, 0.2)), lo, hi, limit=200)[0]
        assert float(extreme_tail(b, 1.0, NU, 0.2)) == pytest.approx(1.0 - surv, abs=1e-10)
    assert float(extreme_tail(0.0, 1.0, NU, 0.2)) == pytest.approx(1.0)
    assert float(extreme_tail(np.log(0.95), 1.0, NU, 0.2)) == pytest.approx(0.82717, abs=5e-6)
    assert float(extreme_tail(-1e-9, 1.0, NU, 0.2)) == pytest.approx(1.0, abs=1e-6)


def test_knockout_density_vanishes_on_barrier_and_matches_simulation():
    b = np.log(0.9)
    assert float(knockout_density(b + 1e-12, b, 1.0, NU, 0.2)) == pytest.approx(0.0, abs=1e-9)
    # exact simulation: X_T, then the minimum of the bridge from 0 to X_T
    rng = np.random.default_rng(5)
    n = 200_000
    XT = NU + 0.2 * rng.standard_normal(n)
    mins = bridge_extreme(0.0, XT, 0.2, 1.0, 1.0 - rng.random(n), "min")
    alive = mins > b
    p_alive = 1.0 - float(extreme_tail(b, 1.0, NU, 0.2))
    assert abs(alive.mean() - p_alive) < 4 * np.sqrt(p_alive * (1 - p_alive) / n)
    sel = XT[alive]
    frac = np.mean(sel < 0.0)
    exact = integrate.quad(lambda x: float(knockout_density(x, b, 1.0, NU, 0.2)), b, 0.0)[0] / p_alive
    assert abs(frac - exact) < 4 * np.sqrt(exact * (1 - exact) / sel.size)


# ------------------------------------------------------------------ images


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("barrier,B", [("down-out", 90.0), ("up-out", 115.0)])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_images_against_integrated_knockout_density(kind, barrier, B, strike):
    b = np.log(B / S0)

    def f(x):
        s = S0 * np.exp(x)
        pay = max(s - strike, 0.0) if kind == "call" else max(strike - s, 0.0)
        return np.exp(-0.01) * pay * float(knockout_density(x, b, 1.0, NU, 0.2))

    lo, hi = (b, 4.0) if b < 0 else (-4.0, b)
    k = np.log(strike / S0)
    val = integrate.quad(f, lo, hi, points=[k] if lo < k < hi else None, limit=200)[0]
    assert float(barrier_price(S0, strike, B, **TC, kind=kind, barrier=barrier)) == pytest.approx(val, abs=1e-9)


def test_parity_rebates_and_degenerate_cases():
    for kind in ("call", "put"):
        for direction, B in (("down", 92.0), ("up", 110.0)):
            out = float(barrier_price(S0, K, B, **TC, kind=kind, barrier=f"{direction}-out", rebate=3.0))
            inn = float(barrier_price(S0, K, B, **TC, kind=kind, barrier=f"{direction}-in", rebate=3.0))
            van = float(bs.call(S0, K, **TC) if kind == "call" else bs.put(S0, K, **TC))
            assert out + inn == pytest.approx(van + 3.0 * np.exp(-0.01))
    # spot on or beyond the barrier: knocked out at once (continuous monitoring)
    for S in (95.0, 90.0):
        assert float(barrier_price(S, K, 95.0, **TC, rebate=2.0)) == pytest.approx(2.0 * np.exp(-0.01))
        assert float(barrier_price(S, K, 95.0, **TC, barrier="down-in")) == pytest.approx(float(bs.call(S, K, **TC)))
    # K >= B: up-and-out call is worthless, up-and-in call is the vanilla
    assert float(barrier_price(S0, 120.0, 115.0, **TC, barrier="up-out")) == 0.0
    # Merton's formula for K >= B, y = 0
    lam = 2 * 0.01 / 0.04 - 1
    mert = bs.call(S0, K, 1.0, 0.01, 0.0, 0.2) - (95 / S0) ** lam * bs.call(95**2 / S0, K, 1.0, 0.01, 0.0, 0.2)
    assert float(barrier_price(S0, K, 95.0, 1.0, 0.01, 0.0, 0.2)) == pytest.approx(float(mert))


def test_corridor_table_entries():
    x = np.array([80.0, 100.0, 130.0])
    for kind in ("call", "put"):
        for lo, hi in ((90.0, np.inf), (0.0, 110.0), (85.0, 115.0)):
            for strike in (80.0, 100.0, 120.0):
                def f(z, s):
                    st = s * np.exp(z)
                    pay = max(st - strike, 0.0) if kind == "call" else max(strike - st, 0.0)
                    return np.exp(-0.01) * pay * (lo < st < hi) * float(np.exp(-0.5 * ((z - NU) / 0.2) ** 2) / (0.2 * np.sqrt(2 * np.pi)))
                for s, v in zip(x, corridor_price(x, strike, lo, hi, **TC, kind=kind)):
                    pts = [np.log(c / s) for c in (lo, hi, strike) if 0 < c < np.inf]
                    ref = integrate.quad(f, -3, 3, args=(s,), points=pts, limit=200)[0]
                    assert v == pytest.approx(ref, abs=1e-8)


def test_quoted_continuous_prices():
    assert float(barrier_price(S0, K, 95.0, **TC)) == pytest.approx(3.75412, abs=5e-6)
    assert float(barrier_price(S0, K, 95.0, **TC, barrier="down-in")) == pytest.approx(3.61017, abs=5e-6)
    assert float(barrier_price(S0, K, 115.0, **TC, barrier="up-out")) == pytest.approx(0.44998, abs=5e-6)


# ------------------------------------------------------------------ double barriers


def test_double_barrier_series_bound_and_prices():
    l, u = np.log(0.8), np.log(1.2)
    xs = np.linspace(l, u, 801)[1:-1]
    ref = double_barrier_density(xs, l, u, 1.0, NU, 0.2, n_terms=60)
    err1 = np.max(np.abs(double_barrier_density(xs, l, u, 1.0, NU, 0.2, n_terms=1) - ref))
    assert 3.5e-6 < err1 < 3.7e-6
    assert double_barrier_truncation_bound(1, l, u, 1.0, NU, 0.2) == pytest.approx(2.9e-3, rel=0.02)
    assert np.max(np.abs(double_barrier_density(xs, l, u, 1.0, NU, 0.2, n_terms=2) - ref)) < 1e-14
    for T in (0.25, 1.0, 4.0):
        r_ = double_barrier_density(xs, l, u, T, NU, 0.2, n_terms=60)
        for n in range(4):
            e = np.max(np.abs(double_barrier_density(xs, l, u, T, NU, 0.2, n_terms=n) - r_))
            assert e <= double_barrier_truncation_bound(n, l, u, T, NU, 0.2) + 1e-15
        assert np.allclose(double_barrier_sine_density(xs, l, u, T, NU, 0.2, 300), r_, atol=1e-12)
    xx = np.linspace(l, u, 4001)
    surv = np.trapezoid(double_barrier_density(xx, l, u, 1.0, NU, 0.2), xx)
    assert surv == pytest.approx(0.38057, abs=5e-5)
    price = double_barrier_price(S0, K, 80, 120, **TC)
    assert price == pytest.approx(1.00574, abs=5e-6)
    quad = integrate.quad(lambda x: np.exp(-0.01) * max(S0 * np.exp(x) - K, 0)
                          * float(double_barrier_density(x, l, u, 1.0, NU, 0.2)), 0.0, u)[0]
    assert price == pytest.approx(quad, abs=1e-10)
    assert double_barrier_terms(u - l, 1.0, 0.2) == 2
    # one barrier far away: the single-barrier price
    assert double_barrier_price(S0, K, 90, 1e6, **TC) == pytest.approx(float(barrier_price(S0, K, 90, **TC)), abs=1e-9)


# ------------------------------------------------------------------ lookbacks


def test_lookbacks_quoted_and_by_quadrature():
    lp = lookback_floating(S0, **TC, kind="put")
    lc = lookback_floating(S0, **TC, kind="call")
    assert lp == pytest.approx(17.23062, abs=5e-6)
    assert lc == pytest.approx(14.27527, abs=5e-6)
    assert lookback_fixed(S0, 100.0, **TC) == pytest.approx(16.24550, abs=5e-6)
    assert lookback_fixed(S0, 100.0, **TC, kind="put") == pytest.approx(15.26039, abs=5e-6)
    tail = lambda z: float(extreme_tail(z, 1.0, NU, 0.2))  # noqa: E731
    em = 1 + integrate.quad(lambda z: np.exp(z) * tail(z), 0, 6)[0]
    assert lp == pytest.approx(S0 * np.exp(-0.01) * em - S0 * np.exp(-0.02), abs=1e-9)
    emin = 1 - integrate.quad(lambda z: np.exp(z) * tail(z), -6, 0)[0]
    assert lc == pytest.approx(S0 * np.exp(-0.02) - S0 * np.exp(-0.01) * emin, abs=1e-9)
    assert lookback_fixed(S0, 110.0, **TC) == pytest.approx(
        S0 * np.exp(-0.01) * integrate.quad(lambda z: np.exp(z) * tail(z), np.log(1.1), 6)[0], abs=1e-9)
    assert lookback_fixed(S0, 90.0, **TC, kind="put") == pytest.approx(
        S0 * np.exp(-0.01) * integrate.quad(lambda z: np.exp(z) * tail(z), -6, np.log(0.9))[0], abs=1e-9)
    # the explicit formula of Proposition bar-lookback (ii)
    r, y, s = 0.01, 0.02, 0.2
    dp = (r - y + s * s / 2) / s
    dm = dp - s
    a = s * s / (2 * (r - y))
    formula = S0 * (np.exp(-y) * ndtr(dp) * (1 + a) - np.exp(-y) + np.exp(-r) * ndtr(-dm) * (1 - a))
    assert lp == pytest.approx(formula, abs=1e-10)
    # y = r: continuous limit
    mid = lookback_floating(S0, 1.0, 0.02, 0.02, 0.2, "put")
    side = [lookback_floating(S0, 1.0, 0.02, 0.02 + e, 0.2, "put") for e in (-1e-6, 1e-6)]
    assert mid == pytest.approx(0.5 * sum(side), abs=1e-8)
    # prior maximum
    em2 = 1.1 + integrate.quad(lambda z: np.exp(z) * tail(z), np.log(1.1), 6)[0]
    assert lookback_floating(S0, **TC, kind="put", extreme=110.0) == pytest.approx(
        S0 * np.exp(-0.01) * em2 - S0 * np.exp(-0.02), abs=1e-9)
    assert max_call_tail(0.0, 1.0, NU, 0.2) == pytest.approx(exp_max_moment(1.0, 0.0, 1.0, NU, 0.2) - 1.0)


# ------------------------------------------------------------------ discrete monitoring


def test_course_table_reproduced():
    for B, row in COURSE_TABLE.items():
        for m, v in zip((1, 2, 3, 4, 6, 12), row[:6]):
            assert discrete_barrier_price(S0, K, B, **TC, m=m) == pytest.approx(v, abs=6e-6)
        assert float(barrier_price(S0, K, B, **TC)) == pytest.approx(row[6], abs=6e-6)


def test_discrete_quadrature_converged_and_bivariate_check():
    v = discrete_barrier_price(S0, K, 95, **TC, m=12)
    assert v == pytest.approx(5.329789841605, abs=1e-11)
    assert discrete_barrier_price(S0, K, 95, **TC, m=12, panels_per_sd=2.0) == pytest.approx(v, abs=1e-11)
    assert discrete_barrier_price(S0, K, 95, **TC, m=12, order=16, width=10.0) == pytest.approx(v, abs=1e-11)
    # m = 2 by the bivariate normal, for B = 95 and B = S0 = 100
    rho = np.sqrt(0.5)
    mvn = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    for B in (95.0, 100.0):
        dm = NU / 0.2
        em = (np.log(S0 / B) + NU * 0.5) / (0.2 * np.sqrt(0.5))
        val = (S0 * np.exp(-0.02) * mvn.cdf([dm + 0.2, em + 0.2 * np.sqrt(0.5)])
               - K * np.exp(-0.01) * mvn.cdf([dm, em]))
        assert discrete_barrier_price(S0, K, B, **TC, m=2) == pytest.approx(val, abs=1e-7)
    # in-out parity for discrete monitoring
    din = discrete_barrier_price(S0, K, 95, **TC, m=12, barrier="down-in")
    assert din + v == pytest.approx(float(bs.call(S0, K, **TC)))


def test_discrete_against_monte_carlo():
    for kind, bar, B, k in (("call", "up-out", 115.0, 100.0), ("put", "down-out", 90.0, 100.0)):
        q = discrete_barrier_price(S0, k, B, **TC, m=12, kind=kind, barrier=bar)
        x = barrier_mc_samples(S0, k, B, **TC, m=12, n=200_000, seed=3, kind=kind, barrier=bar)
        assert abs(x.mean() - q) < 4 * x.std() / np.sqrt(x.size)


def test_discrete_table_and_bgk_numbers():
    table = {  # (barrier, B): exact and BGK for m = 4, 12, 52, 252
        ("down-out", 95.0): ([6.22816, 5.32979, 4.57782, 4.14873], [6.06446, 5.30339, 4.57963, 4.14875]),
        ("down-out", 99.0): ([5.27574, 3.67565, 2.26089, 1.48813], [4.61914, 3.30925, 2.14258, 1.47501]),
        ("up-out", 115.0): ([1.11608, 0.83710, 0.63269, 0.53166], [1.35690, 0.90999, 0.64809, 0.53468]),
    }
    for (bar, B), (exact, corrected) in table.items():
        for m, e, c in zip((4, 12, 52, 252), exact, corrected):
            assert discrete_barrier_price(S0, K, B, **TC, m=m, barrier=bar) == pytest.approx(e, abs=5e-6)
            assert float(bgk_price(S0, K, B, **TC, m=m, barrier=bar)) == pytest.approx(c, abs=5e-6)
    # sign change of the BGK error for B = 95 between m = 16 and m = 24
    err = [float(bgk_price(S0, K, 95, **TC, m=m)) - discrete_barrier_price(S0, K, 95, **TC, m=m) for m in (16, 24)]
    assert err[0] < 0 < err[1]
    # the shift moves the barrier away from the spot
    assert float(bgk_price(S0, K, 95, **TC, m=12)) == pytest.approx(
        float(barrier_price(S0, K, 95 * np.exp(-BETA_BGK * 0.2 / np.sqrt(12)), **TC)))


def test_expected_maximum_bias():
    em = expected_max(1.0, NU, 0.2)
    assert em == pytest.approx(integrate.quad(lambda z: float(extreme_tail(z, 1.0, NU, 0.2)), 0, 6)[0], abs=1e-10)
    ratio = (em - spitzer_expected_max(4000, 1.0, NU, 0.2)) / (0.2 * np.sqrt(1 / 4000))
    assert abs(ratio - BETA_BGK) < 0.01


# ------------------------------------------------------------------ the bridge


def test_crossing_probability_and_extreme_sampling():
    # crossing probability = image ratio, independent of the drift
    b, z, h = 0.1, -0.05, 0.5
    for nu in (-0.2, 0.0, 0.3):
        lam = 2 * nu / 0.04
        p = lambda x: np.exp(-0.5 * ((x - nu * h) / (0.2 * np.sqrt(h))) ** 2)  # noqa: E731
        assert float(crossing_probability(0.0, z, b, 0.2, h)) == pytest.approx(np.exp(lam * b) * p(z - 2 * b) / p(z))
    assert float(crossing_probability(0.0, 0.2, 0.1, 0.2, h)) == 1.0
    assert float(crossing_probability(0.0, -0.05, -0.1, 0.2, h, "down")) == pytest.approx(
        np.exp(-2 * 0.1 * 0.05 / (0.04 * h)))
    # sampled maxima have the conditional law
    rng = np.random.default_rng(11)
    M = bridge_extreme(0.0, z, 0.2, h, 1.0 - rng.random(100_000), "max")
    for level in (0.02, 0.1, 0.25):
        exact = float(crossing_probability(0.0, z, level, 0.2, h))
        assert abs(np.mean(M >= level) - exact) < 4 * np.sqrt(exact * (1 - exact) / M.size)
    assert np.all(M >= 0.0)


def test_bridge_estimators_unbiased_and_quoted_sds():
    exact = float(barrier_price(S0, K, 95.0, **TC))
    sds = {}
    for m in (1, 12):
        for mon in ("bridge", "bridge-sample"):
            x = barrier_mc_samples(S0, K, 95.0, **TC, m=m, n=200_000, seed=7, monitor=mon)
            assert abs(x.mean() - exact) < 4 * x.std() / np.sqrt(x.size)
            sds[(m, mon)] = x.std()
    assert round(sds[(1, "bridge")], 2) == 8.06
    assert round(sds[(12, "bridge")], 2) == 9.88
    assert round(sds[(1, "bridge-sample")], 1) == 10.6
    assert round(sds[(12, "bridge-sample")], 1) == 10.6
    # grid monitoring is unbiased for the discrete contract, not the continuous one
    g = barrier_mc_samples(S0, K, 95.0, **TC, m=12, n=200_000, seed=7, monitor="grid")
    assert abs(g.mean() - 5.32979) < 4 * g.std() / np.sqrt(g.size)


def test_course_mc_study_numbers():
    R = discrete_barrier_cv_study(n=1_000_000, seed=2026)
    assert R["exact_discrete"] == pytest.approx(5.32979, abs=5e-6)
    v0 = R["plain"]["se"] ** 2
    quoted = {  # estimate, s.e., VRF in the notes' table
        "plain": (5.33642, 0.01191, 1.0),
        "antithetic": (5.33197, 0.00752, 2.51),
        "cv_cont_sampled": (5.33882, 0.00637, 3.49),
        "cv_cont_prob": (5.33621, 0.00491, 5.89),
        "cv_vanilla": (5.32051, 0.00582, 4.19),
        "cv_both": (5.32770, 0.00464, 6.59),
        "cv_both_prob": (5.32900, 0.00399, 8.91),
    }
    for key, (est, se, vrf) in quoted.items():
        assert R[key]["mean"] == pytest.approx(est, abs=5e-6)
        assert R[key]["se"] == pytest.approx(se, abs=5e-6)
        assert round(v0 / R[key]["se"] ** 2, 2) == vrf
        assert abs(R[key]["mean"] - R["exact_discrete"]) < 1.6 * R[key]["se"]
    assert round(R["antithetic"]["rho"], 3) == -0.201
    assert round(R["cv_cont_sampled"]["beta"], 3) == 0.947
    assert round(R["cv_cont_prob"]["beta"], 3) == 1.101
    assert round(R["cv_vanilla"]["beta"], 3) == 0.825


def test_discrete_double_barrier_matches_monte_carlo():
    q = discrete_corridor_price(S0, K, 80.0, 120.0, **TC, m=12)
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((200_000, 12))
    X = np.cumsum(NU / 12 + 0.2 * np.sqrt(1 / 12) * Z, axis=1)
    pay = np.exp(-0.01) * np.maximum(S0 * np.exp(X[:, -1]) - K, 0) * np.all(
        (X > np.log(0.8)) & (X < np.log(1.2)), axis=1)
    assert abs(pay.mean() - q) < 4 * pay.std() / np.sqrt(pay.size)


def test_images_stable_for_small_volatility():
    # up-and-out call, S0 = K = 100, B = 103, r = 5%, y = 0: the image weight (B/S0)^lam has
    # lam of order 1/sigma^2, so the image term must be computed in log space
    def by_quadrature(sig):
        nu, b, sd = 0.05 - 0.5 * sig**2, np.log(1.03), sig
        f = lambda x: (np.exp(-0.05) * max(100 * np.exp(x) - 100, 0.0)  # noqa: E731
                       * np.exp(-0.5 * ((x - nu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
                       * -np.expm1(-2 * b * (b - x) / sig**2))
        return integrate.quad(f, 0.0, b, limit=500, epsabs=1e-14)[0]
    v = float(barrier_price(100.0, 100.0, 103.0, 1.0, 0.05, 0.0, 0.01, barrier="up-out"))
    assert v == pytest.approx(0.034767, abs=5e-7)
    assert v == pytest.approx(by_quadrature(0.01), abs=1e-12)
    v = float(barrier_price(100.0, 100.0, 103.0, 1.0, 0.05, 0.0, 0.001, barrier="up-out"))
    assert np.isfinite(v) and 0.0 <= v < 1e-12          # the forward path reaches 103: knocked out
    # down barrier never reached by the increasing forward path: the discounted intrinsic value
    v = float(barrier_price(100.0, 90.0, 95.0, 1.0, 0.05, 0.0, 0.001))
    assert v == pytest.approx(100.0 - 90.0 * np.exp(-0.05), abs=1e-9)
    # Exercise 16.1(c): the limit sigma -> 0 on the test case with B = 95
    assert float(barrier_price(100.0, 90.0, 95.0, **dict(TC, sigma=0.001))) == pytest.approx(
        np.exp(-0.01) * (100 * np.exp(-0.01) - 90), abs=1e-6)
    # double barrier with the same weights
    assert double_barrier_price(100.0, 100.0, 98.0, 103.0, 1.0, 0.05, 0.0, 0.01) == pytest.approx(
        0.0347674, abs=1e-6)
