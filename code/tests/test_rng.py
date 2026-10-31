"""Checks of the Chapter 5 code against the results proved in the notes, and of every number
the chapter quotes."""
import math

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtr, ndtri

from mfe230d import rng as R


# ---------------------------------------------------------------- linear congruential generators

def test_course_lcg_sequence_and_full_period():
    x = R.lcg_sequence(17, **R.COURSE_LCG, x0=0)
    assert list(map(int, x)) == [0, 5, 2, 7, 4, 9, 6, 11, 8, 13, 10, 15, 12, 1, 14, 3, 0]
    assert R.hull_dobell(16, 9, 5) and R.lcg_period(16, 9, 5, 0) == 16
    # x_{i+2} = x_i + 2 (mod 16), since a^2 = 81 = 1 (mod 16)
    assert np.all((x[2:].astype(int) - x[:-2].astype(int)) % 16 == 2)


def test_hull_dobell_against_brute_force():
    for m in range(1, 21):
        for a in range(m):
            for c in range(m):
                full = R.lcg_period(m, a, c, 0) == m      # full period from one seed = from all
                assert full == R.hull_dobell(m, a, c), (m, a, c)


def test_multiplicative_periods():
    p = 2**31 - 1                                       # MINSTD: 16807 is a primitive root
    assert all(pow(16807, (p - 1) // q, p) != 1 for q in (2, 3, 7, 11, 31, 151, 331))
    assert 2 * 3**2 * 7 * 11 * 31 * 151 * 331 == p - 1
    A29, _ = R.lcg_jump(2**31, 65539, 0, 2**29)         # RANDU: order of a is 2^29
    A28, _ = R.lcg_jump(2**31, 65539, 0, 2**28)
    assert A29 == 1 and A28 != 1
    assert R.hull_dobell(2**32, 1664525, 1013904223)    # "LCG 2^32": full period 2^32
    assert R.hull_dobell(2**64, R.PCG_MULT, R.PCG32(1, 54).inc)   # PCG32's state: period 2^64


def test_jump_ahead_and_vectorised_sequence():
    for (m, a, c, x0) in [(2**64, R.PCG_MULT, 12345, 777), (2**31 - 1, 16807, 0, 1),
                          (2**32, 1664525, 1013904223, 5), (2**31, 65539, 0, 1)]:
        seq = R.lcg_sequence(3000, m, a, c, x0, block=256)
        x = x0
        for i in range(3000):
            assert int(seq[i]) == x
            x = (a * x + c) % m
        A, C = R.lcg_jump(m, a, c, 2999)
        assert (A * x0 + C) % m == int(seq[-1])


def test_pcg32_reference_outputs():
    # pcg32 with initstate 42, initseq 54, as in O'Neill's reference demo program
    out = R.PCG32(42, 54).random_raw(6)
    assert [int(v) for v in out] == [0xa15c02b7, 0x7b47f409, 0xba1d3330,
                                     0x83d2f293, 0xbfa4784b, 0xcbed606e]
    g1, g2 = R.PCG32(42, 54), R.PCG32(42, 54)
    assert np.array_equal(np.concatenate([g1.random_raw(1000), g1.random_raw(37)]),
                          g2.random_raw(1037))
    u = R.PCG32(1, 54).random(10_000)
    assert u.min() > 0 and u.max() < 1


# ---------------------------------------------------------------- lattices and the spectral test

def test_spectral_test_against_brute_force():
    import itertools
    for m, a in [(16, 9), (64, 13), (64, 5), (32, 21)]:
        for d in (2, 3):
            h, nu = R.spectral_test(a, m, d)
            assert sum(int(h[j]) * a**j for j in range(d)) % m == 0
            best = min(sum(v * v for v in g) for g in itertools.product(range(-m, m + 1), repeat=d)
                       if any(g) and sum(g[j] * a**j for j in range(d)) % m == 0)
            assert nu**2 == pytest.approx(best)


def test_lattice_proposition_plane_counts():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = int(rng.integers(50, 600)); a = int(rng.integers(2, m)); c = int(rng.integers(0, m))
        u = R.lcg_sequence(m + 3, m, a, c, int(rng.integers(0, m))).astype(float) / m
        for d in (2, 3):
            h, _ = R.spectral_test(a, m, d)
            planes = np.unique(np.round(R.tuples(u, d) @ h.astype(float), 8))
            assert len(planes) <= np.abs(h).sum()                       # part (ii)
            gaps = np.diff(planes)                                       # planes one unit apart
            assert np.allclose(gaps, np.round(gaps), atol=1e-6)
            K = math.floor(m ** (1 / d) + 1e-12)                         # part (iii), via the
            assert np.linalg.norm(h) <= math.sqrt(d) * K + 1e-12        # pigeonhole vector


def test_course_pairs_on_four_lines():
    h, nu = R.spectral_test(9, 16, 2)
    assert nu**2 == pytest.approx(8) and np.abs(h).sum() == 4
    x = R.lcg_sequence(17, **R.COURSE_LCG, x0=0).astype(int)
    assert set((x[1:] - x[:-1]).tolist()) == {5, -3, 13, -11}


def test_randu_planes():
    h, nu = R.spectral_test(65539, 2**31, 3)
    assert list(h) == [9, -6, 1] and nu**2 == pytest.approx(118)
    assert 1 / nu == pytest.approx(0.0921, abs=5e-5)
    x = R.lcg_sequence(10_000, 2**31, 65539, 0, 1).astype(np.int64)
    assert np.all((x[2:] - 6 * x[1:-1] + 9 * x[:-2]) % 2**31 == 0)
    u = R.uniforms(300_001, "randu", seed=0)
    v = R.tuples(u, 3) @ np.array([9.0, -6.0, 1.0])
    vals = np.unique(np.round(v, 6))
    assert np.allclose(vals, np.arange(-5, 10)) and len(vals) == 15
    # RANDU visits only odd states: with the reduced modulus 2^30 the pairs lie on lines
    # h.x in 1/2 + Z, h = (16387, 16383), 4.3e-5 apart; nu_3 is unchanged.
    h2, nu2 = R.spectral_test(65539, 2**30, 2)
    assert list(h2) == [16387, 16383] and 1 / nu2 == pytest.approx(4.3e-5, rel=0.01)
    frac = np.mod(R.tuples(u, 2) @ h2.astype(float), 1.0)
    assert np.allclose(frac, 0.5, atol=1e-6)
    assert R.spectral_test(65539, 2**30, 3)[1] ** 2 == pytest.approx(118)
    assert 1 / R.spectral_test(65539, 2**31, 2)[1] == pytest.approx(2.2e-5, rel=0.03)


def test_small_lcg_good_pairs_bad_triples():
    m, a = 4096, 65
    _, nu2 = R.spectral_test(a, m, 2)
    assert round(nu2, 1) == 63.0
    best = max(R.spectral_test(b, m, 2)[1] for b in range(1, m) if R.hull_dobell(m, b, 1))
    assert round(best, 1) == 67.5                    # best full-period multiplier
    assert round(max(R.spectral_test(b, m, 2)[1] for b in range(1, m)), 1) == 68.1   # any a
    h, _ = R.spectral_test(a, m, 3)
    assert list(h) == [1, -2, 1]
    u = R.lcg_sequence(m + 2, m, a, 1, 0).astype(float) / m
    planes = np.unique(np.round(R.tuples(u, 3)[:m] @ h.astype(float), 9))
    assert len(planes) == 3


def test_spectral_values_quoted_and_marsaglia_bound():
    assert 1 / R.spectral_test(16807, 2**31 - 1, 3)[1] == pytest.approx(1.6e-3, rel=0.03)
    assert 1 / R.spectral_test(1664525, 2**32, 3)[1] == pytest.approx(6.6e-4, rel=0.01)
    assert (math.factorial(10) * 2**32) ** 0.1 == pytest.approx(41.6, abs=0.05)


# ---------------------------------------------------------------- tests of uniformity

def test_statistics_against_scipy():
    u = R.uniforms(20_000, "pcg64", seed=3)
    D, p = R.ks_uniform(u)
    ref = stats.kstest(u, "uniform")
    assert D == pytest.approx(ref.statistic, rel=1e-12)
    assert p == pytest.approx(ref.pvalue, abs=0.01)
    t = np.linspace(0.3, 2.5, 12)
    assert np.allclose(R.kolmogorov_sf(t), stats.kstwobign.sf(t), atol=1e-12)
    stat, p = R.chi_square_uniform(u, 50)
    counts = np.bincount((u * 50).astype(int), minlength=50)
    ref = stats.chisquare(counts)
    assert stat == pytest.approx(ref.statistic) and p == pytest.approx(ref.pvalue)
    r = R.autocorrelation(u, 5)
    assert np.all(np.abs(r) < 4 / math.sqrt(len(u)))


def test_table_of_tests():
    expected = {  # z_mean, p_ks, p_chi1, z_lag1, p_serial2, p_serial3 (as printed, 2 decimals)
        "randu": (-0.71, 0.54, 0.46, 0.91, 0.32, 0.0),
        "minstd": (-1.37, 0.28, 0.08, -0.64, 0.28, 0.09),
        "nr32": (0.97, 0.41, 0.49, 0.90, 0.74, 0.89),
        "pcg32": (-0.17, 0.86, 0.86, -0.67, 0.52, 0.32),
        "pcg64": (-0.08, 0.77, 0.73, -0.88, 0.35, 0.12),
    }
    for g, row in expected.items():
        u = R.uniforms(10**6, g, seed=1)
        b = R.run_battery(u)
        got = (b["z_mean"], b["p_ks"], b["p_chi1"], b["z_lag1"], b["p_serial2"], b["p_serial3"])
        assert np.allclose(np.round(got, 2), row, atol=1e-9), (g, got)
        if g == "randu":
            stat, p = R.serial_test(u, 3, 20)
            assert round(stat) == 247_846 and p < 1e-300
            P = R.tuples(u, 3, overlapping=False)
            idx = np.minimum((P * 20).astype(int), 19)
            counts = np.bincount(idx[:, 0] * 400 + idx[:, 1] * 20 + idx[:, 2], minlength=8000)
            assert (counts == 0).sum() == 2000 and counts.max() == 129
            assert len(P) / 8000 == pytest.approx(41.7, abs=0.05)


# ---------------------------------------------------------------- inversion

def test_inversion_exponential():
    u = R.uniforms(50_000, "pcg64", seed=4)
    x = R.exponential_inverse(u, 2.0)
    assert stats.kstest(x, "expon", args=(0, 0.5)).pvalue > 0.01


def test_semicircle_newton_proposition():
    u = np.linspace(0.5 + 1e-6, 1 - 1e-6, 4001)
    H = R.semicircle_inverse(u, 8, "newton", history=True)
    xs = H[-1]
    assert np.allclose(R.semicircle_cdf(xs), u, atol=1e-14)
    assert np.all(H[0] >= xs - 1e-13)                        # x_0 >= x*
    assert np.all(H[1] >= 0)                                 # x_1 >= 0
    assert np.all(np.diff(H[1:], axis=0) >= -1e-12)          # increasing from k = 1
    assert np.all(H[1:] <= xs + 1e-12)                       # below x*
    C = xs / (2 * (1 - xs**2))                               # quadratic rate
    for k in range(1, 4):
        assert np.all(np.abs(H[k + 1] - xs) <= C * (H[k] - xs) ** 2 + 1e-13)
    x = np.linspace(0, 1, 10_001)                            # h >= 0
    assert np.all(x * R.semicircle_pdf(x) - R.semicircle_cdf(x) + (1 + x) / 2 >= -1e-15)


def test_semicircle_numbers_quoted():
    u = np.linspace(1e-4, 1 - 1e-4, 6000)
    Hn = R.semicircle_inverse(u, 3, "newton", history=True)
    Hh = R.semicircle_inverse(u, 2, "halley", history=True)
    e2, e3 = (np.abs(R.semicircle_cdf(Hn[k]) - u) for k in (2, 3))
    assert e2.max() == pytest.approx(2.6e-4, rel=0.03) and e3.max() == pytest.approx(3.4e-6, rel=0.03)
    assert np.median(e2) == pytest.approx(1.1e-6, rel=0.05)
    assert np.median(e3) == pytest.approx(5e-13, rel=0.1)
    assert np.median(np.abs(R.semicircle_cdf(Hh[2]) - u)) == pytest.approx(6e-12, rel=0.1)


def test_normal_quantile_accuracy():
    u = np.concatenate([np.logspace(-300, -1, 3000), np.linspace(0.1, 0.9, 3000),
                        1 - np.logspace(-15, -1, 1000)])
    x = ndtri(u)
    rel = np.abs(R.norm_inv_acklam(u) - x) / np.abs(x)
    assert rel.max() < 1.15e-9 and rel.max() == pytest.approx(1.13e-9, rel=0.01)
    xs = np.linspace(-7, 7, 400_001)
    err = np.abs(R.norm_inv_bsm(ndtr(xs)) - ndtri(ndtr(xs)))
    assert err.max() < 3e-9 and round(err.max(), 10) == 3.0e-9
    assert abs(xs[err.argmax()]) == pytest.approx(1.405, abs=0.01)   # where the pieces meet
    xs = np.linspace(-8.2, 8.2, 6001)
    uu = ndtr(xs)
    assert np.abs(R.norm_inv(uu) - ndtri(uu)).max() < 4e-15


# ---------------------------------------------------------------- rejection

def test_envelope_acceptance_formula():
    assert R.envelope_acceptance(0.0) == pytest.approx(math.pi / 4)
    assert R.envelope_acceptance(math.pi / 6) == pytest.approx((3 + math.sqrt(3)) * math.pi / 16)
    assert R.envelope_acceptance(math.pi / 4) == pytest.approx((1 + math.sqrt(2)) * math.pi / 8)
    assert round(R.envelope_acceptance(math.pi / 6), 3) == 0.929
    assert round(R.envelope_acceptance(math.pi / 4), 3) == 0.948
    th = np.linspace(0.01, 1.55, 2000)
    assert th[np.argmax([R.envelope_acceptance(t) for t in th])] == pytest.approx(math.pi / 4, abs=1e-3)
    from scipy.integrate import quad
    for t in (0.2, math.pi / 6, math.pi / 4, 1.2):
        b = math.tan(t / 2)
        area = quad(lambda x: float(R.semicircle_envelope(x, t)), -1, 1, points=[-b, b])[0]
        assert 1 / area == pytest.approx(R.envelope_acceptance(t), rel=1e-9)
        x = np.linspace(-1, 1, 2001)
        assert np.all(R.semicircle_envelope(x, t) >= R.semicircle_pdf(x) - 1e-15)


def test_semicircle_rejection_sampler():
    for t, seed in ((0.0, 1), (math.pi / 6, 2), (math.pi / 4, 3)):
        n = 40_000
        s, tried = R.semicircle_rejection(n, t, seed=seed)
        p = R.envelope_acceptance(t)
        assert abs(n / tried - p) < 4 * math.sqrt(p * (1 - p) / tried) * 1.5
        assert stats.kstest(s, R.semicircle_cdf).pvalue > 0.001


# ---------------------------------------------------------------- normal variates

def test_box_muller_and_polar():
    u = R.uniforms(200_000, "pcg64", seed=8)
    z = R.normals_box_muller(u)
    assert stats.kstest(z, "norm").pvalue > 0.001
    assert abs(np.corrcoef(z[0::2], z[1::2])[0, 1]) < 4 / math.sqrt(len(z) / 2)
    zp, pairs = R.polar_marsaglia(u)
    assert len(zp) / 2 / pairs == pytest.approx(math.pi / 4, abs=4 * math.sqrt(0.17 / pairs))
    assert stats.kstest(zp, "norm").pvalue > 0.001
    assert math.sqrt(2 * 32 * math.log(2)) == pytest.approx(6.66, abs=0.005)
    assert math.sqrt(2 * 53 * math.log(2)) == pytest.approx(8.57, abs=0.005)
    assert 2 * ndtr(-6.66) == pytest.approx(2.7e-11, rel=0.02)


def test_diagnostics_numbers_quoted():
    u = R.uniforms(100_000, "pcg32", seed=2024)
    z = R.normals_box_muller(u)
    edges = np.round(np.arange(-4.0, 4.0001, 0.1), 10)
    b = R.binned_counts(z, edges, ndtr)
    assert len(b["z"]) == 80 and np.sum(np.abs(b["z"]) > 2) == 1
    assert 80 * 2 * ndtr(-2) == pytest.approx(3.6, abs=0.05)
    assert (b["below"], b["above"]) == (3, 4)
    assert 1e5 * ndtr(-4) == pytest.approx(3.2, abs=0.05)
    band = 1.96 / math.sqrt(len(z))
    assert np.sum(np.abs(R.autocorrelation(z, 20)) > band) == 1
    assert np.sum(np.abs(R.autocorrelation(z**2, 20)) > band) == 1
    assert np.allclose(b["counts"].sum() + b["below"] + b["above"], len(z))


# ---------------------------------------------------------------- Gaussian vectors

def test_cholesky_and_example():
    S = np.array([[1, .5, .3], [.5, 1, .6], [.3, .6, 1]])
    L = R.cholesky(S)
    assert np.allclose(L, np.linalg.cholesky(S))
    assert np.allclose(L[2], [0.3, 0.45 / math.sqrt(0.75), 0.8])
    assert round(L[1, 1], 3) == 0.866 and round(L[2, 1], 3) == 0.520
    A, lam = R.pca_factor(S)
    assert np.allclose(A @ A.T, S)
    assert np.allclose(np.round(lam, 3), [1.944, 0.707, 0.349])
    assert round(lam[0] / 3, 2) == 0.65 and round(lam[:2].sum() / 3, 2) == 0.88
    with pytest.raises(ValueError):
        R.cholesky(np.array([[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1]]))
    Z0 = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1.0]])   # zero pivot, yet not PSD
    assert np.linalg.eigvalsh(Z0).min() == pytest.approx(1 - math.sqrt(2))
    for f in (R.cholesky, R.pca_factor):
        with pytest.raises(ValueError):
            f(Z0)
    t = np.array([0.1, 0.4, 0.5, 1.0])                  # Brownian covariance: cumulative sums
    L = R.cholesky(np.minimum.outer(t, t))
    assert np.allclose(L, np.tril(np.tile(np.sqrt(np.diff(np.r_[0, t])), (4, 1))))


def test_pca_truncation_error_and_optimality():
    rng = np.random.default_rng(9)
    B = rng.standard_normal((5, 5))
    S = B @ B.T
    A, lam = R.pca_factor(S)
    Z = rng.standard_normal((200_000, 5))
    X = Z @ A.T
    for r in (1, 2, 4):
        Xr = Z[:, :r] @ A[:, :r].T
        err = np.mean(np.sum((X - Xr) ** 2, axis=1))
        assert err == pytest.approx(lam[r:].sum(), rel=0.02)
        for _ in range(20):                            # random r-dim projections do no better
            Q, _ = np.linalg.qr(rng.standard_normal((5, r)))
            P = Q @ Q.T
            assert np.trace((np.eye(5) - P) @ S) >= lam[r:].sum() - 1e-10
    X = R.correlated_normals(100_000, S, method="pca", seed=1)
    assert np.allclose(np.cov(X.T), S, atol=0.05 * np.abs(S).max())
