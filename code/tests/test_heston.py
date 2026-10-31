"""Tests for mfe230d.heston (Chapter 19): the characteristic function and its branches, moment explosion,
the three Fourier inversions, the QE scheme, the simulation schemes, and every number quoted in the notes."""
import numpy as np
import pytest
from scipy.special import ndtr

from mfe230d import bs
from mfe230d import heston as H
from mfe230d import stratified

R, Y, S = 0.01, 0.02, 100.0
LAB0 = dict(H.TEST, rho=0.0)
LAB7 = dict(H.TEST, rho=-0.7)
PUT0 = 8.153429952752      # ATM put, test case, rho = 0 (quoted in the notes and in Chapter 10)
PUT7 = 7.948021924422      # ATM put, test case, rho = -0.7


# ----------------------------------------------------------------- the characteristic function
@pytest.mark.parametrize("prm", [LAB0, LAB7, H.FO, H.CASE_I, dict(v0=0.02, kappa=1.0, theta=0.05, xi=0.8, rho=0.4)])
def test_stable_form_solves_riccati(prm):
    p = {k: prm[k] for k in ("kappa", "theta", "xi", "rho")}
    u = np.array([0.0, 0.3, 2.0, 7.5, 20.0, 0.3 - 1j, 4.0 - 1j, 1.0 - 0.5j])
    for T in (0.5, 3.0):
        C, D = H.riccati(u, T, **p)
        C2, D2 = H.riccati_rk4(u, T, **p, n=3000)
        assert np.max(np.abs(C - C2)) < 1e-9 and np.max(np.abs(D - D2)) < 1e-9


def test_charfn_basic_identities_and_decay_bound():
    u = np.linspace(-30, 30, 121)
    phi = H.charfn(u, S, 1.0, R, Y, **LAB7)
    assert H.charfn(0.0, S, 1.0, R, Y, **LAB7) == pytest.approx(1.0, abs=1e-15)
    assert H.charfn(-1j, S, 1.0, R, Y, **LAB7).real == pytest.approx(S * np.exp(R - Y), rel=1e-13)   # the forward
    assert np.all(np.abs(phi) <= 1 + 1e-13)
    assert np.allclose(H.charfn(-u, S, 1.0, R, Y, **LAB7), np.conj(phi), atol=1e-14)
    # |phi(u)| <= L(rho_bar^2 u^2 / 2), and the decay rate rho_bar (V0 + kappa theta T)/xi
    L = H.laplace_integrated_variance(0.5 * 0.51 * u * u, 1.0, 0.04, 2.0, 0.04, 0.3)
    assert np.all(np.abs(phi) <= L * (1 + 1e-12))
    rate = np.sqrt(0.51) * (0.04 + 2 * 0.04) / 0.3
    lam = 0.5 * 0.51 * np.array([400.0, 800.0]) ** 2
    lL = np.log(H.laplace_integrated_variance(lam, 1.0, 0.04, 2.0, 0.04, 0.3))
    assert (lL[1] - lL[0]) / 400 == pytest.approx(-rate, rel=1e-3)


def test_laplace_integrated_variance_against_exact_variance_paths():
    t = np.linspace(0, 1, 129)
    V = stratified.cir_exact(0.04, 2.0, 0.04, 0.3, t, 40_000, seed=11)
    Q = np.sum(0.5 * (V[:, 1:] + V[:, :-1]) * np.diff(t), axis=1)
    for lam in (5.0, 30.0):
        e = np.exp(-lam * Q)
        assert abs(e.mean() - H.laplace_integrated_variance(lam, 1.0, 0.04, 2.0, 0.04, 0.3)) < 4 * e.std() / 200


def test_rho_zero_is_the_laplace_transform_at_a_complex_argument():
    """Exercise sv-degenerate (c): for rho = 0, C + D v0 = log E[exp(-(u^2 + iu) Q_T / 2)]."""
    u = np.array([0.5, 3.0])
    lam = 0.5 * (u * u + 1j * u)
    C, D = H.riccati(u, 1.0, 2.0, 0.04, 0.3, 0.0)
    g = np.sqrt(4 + 2 * 0.09 * lam)
    em = np.exp(-g)
    den = (g + 2) * (1 - em) + 2 * g * em
    B = 2 * lam * (1 - em) / den
    logA = 2 * 2 * 0.04 / 0.09 * (np.log(2 * g) + 0.5 * (2 - g) - np.log(den))
    assert np.max(np.abs(C - logA)) < 1e-14 and np.max(np.abs(D + B)) < 1e-14


def test_g_inside_unit_disk_when_rho_xi_le_2kappa():
    """Theorem sv-cf (iii): |g(u)| < 1 on the real line if rho xi <= 2 kappa, at u - i if rho xi < kappa."""
    rng = np.random.default_rng(3)
    u = np.concatenate([np.linspace(0, 50, 501), [200.0, 1e3]])
    for _ in range(300):
        kappa, xi = rng.uniform(0.1, 5), rng.uniform(0.05, 3)
        rho = rng.uniform(-1, min(1, 2 * kappa / xi))
        for w, ok in ((u, rho * xi <= 2 * kappa), (u - 1j, rho * xi < kappa)):
            if not ok:
                continue
            beta = kappa - 1j * rho * xi * w
            d = np.sqrt(beta**2 + xi**2 * (w * w + 1j * w))
            assert np.all(d.real > 0) and np.all(np.abs((beta - d) / (beta + d)) < 1)
    # and |g| > 1 can happen for large u when rho xi > 2 kappa
    beta = 0.5 - 1j * 0.9 * 2.0 * 1e3
    d = np.sqrt(beta**2 + 4.0 * (1e6 + 1e3j))
    assert abs((beta - d) / (beta + d)) > 1


def test_heston_form_jumps_and_stable_form_is_continuous():
    u = np.linspace(0.001, 6, 6001)
    a = H.charfn(u, 1.0, 10.0, 0.0, 0.0, **H.FO)
    b = H.charfn(u, 1.0, 10.0, 0.0, 0.0, form="heston", **H.FO)
    assert np.max(np.abs(np.diff(a))) < 2e-3                       # smooth on a fine grid
    assert np.sum(np.abs(np.diff(b)) > 0.05) == 2                  # two jumps for u < 6 (Figure 19.1(a))
    assert np.allclose(a[u < 0.5], b[u < 0.5], atol=1e-12)          # identical before the first wrap
    # short maturity: the two forms agree
    s = H.charfn(u, 1.0, 1.0, 0.0, 0.0, **H.FO)
    h = H.charfn(u, 1.0, 1.0, 0.0, 0.0, form="heston", **H.FO)
    assert np.max(np.abs(s - h)) < 1e-12


def _cm(T, form, alpha=0.75, h=0.01, n=50_000):
    v = (np.arange(n) + 0.5) * h
    k = np.log(100.0)
    phi = np.exp(H.log_charfn(v - (alpha + 1) * 1j, 100.0, T, 0.025, 0.0, form=form, **H.FO))
    psi = np.exp(-0.025 * T) * phi / (alpha * alpha + alpha - v * v + 1j * (2 * alpha + 1) * v)
    return float(np.exp(-alpha * k) / np.pi * h * np.sum((np.exp(-1j * v * k) * psi).real))


def test_little_heston_trap_numbers():
    """Albrecher et al. (2007), Table 1: stable 18.77, 24.50, 43.17; Heston's form 23.92, 31.99 in our quadrature."""
    for T, stable in ((4.0, 18.77), (6.0, 24.50), (15.0, 43.17)):
        assert H.price(100.0, 100.0, T, 0.025, 0.0, "call", **H.FO) == pytest.approx(stable, abs=5e-3)
        assert _cm(T, "stable") == pytest.approx(stable, abs=5e-3)
    assert _cm(4.0, "heston") == pytest.approx(18.77, abs=5e-3)
    assert _cm(6.0, "heston") == pytest.approx(23.92, abs=5e-3)
    assert _cm(15.0, "heston") == pytest.approx(31.99, abs=5e-3)
    second = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.9)
    assert H.price(100.0, 100.0, 15.0, 0.025, 0.0, "call", **second) == pytest.approx(43.75, abs=5e-3)


# ----------------------------------------------------------------- moment explosion
def _blowup_rk4(a, kappa, xi, rho, Tmax=20.0, n=200_000):
    ka = kappa - rho * xi * a
    f = lambda D: 0.5 * xi * xi * D * D - ka * D + 0.5 * a * (a - 1)
    D, h = 0.0, Tmax / n
    for i in range(n):
        k1 = f(D); k2 = f(D + 0.5 * h * k1); k3 = f(D + 0.5 * h * k2); k4 = f(D + h * k3)
        D += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(D) or D > 1e8:
            return (i + 1) * h
    return np.inf


@pytest.mark.parametrize("a,prm", [(2.0, (1.0, 1.0, 0.8)), (20.0, (0.5, 1.0, -0.9)), (-3.0, (0.5, 1.0, -0.9)),
                                   (5.0, (1.0, 1.0, 0.8)), (3.0, (0.3, 0.5, 0.95)), (1.5, (0.1, 0.5, 0.95)),
                                   (2.0, (2.0, 0.3, -0.7))])
def test_explosion_time_formula(a, prm):
    T = H.explosion_time(a, *prm)
    ref = _blowup_rk4(a, *prm, n=40_000)
    if np.isinf(T):
        assert np.isinf(ref)
    else:
        assert T == pytest.approx(ref, abs=2e-3)


def test_quoted_moment_numbers():
    assert H.explosion_time(2.0, 1.0, 1.0, 0.8) == pytest.approx(1.7689, abs=5e-5)
    assert H.explosion_time(0.5, 1.0, 1.0, 0.8) == np.inf
    assert H.moment_bound(1.0, 2.0, 0.3, -0.7) == pytest.approx(38.5007, abs=5e-4)
    assert H.moment_bound(10.0, 2.0, 0.3, -0.7) == pytest.approx(24.1989, abs=5e-4)
    assert H.moment_bound(1.0, 1.5768, 0.5751, -0.5711) == pytest.approx(14.5012, abs=5e-4)
    assert H.moment_bound(10.0, 1.5768, 0.5751, -0.5711) == pytest.approx(7.7740, abs=5e-4)
    assert H.moment_bound(2.0, 1.0, 1.0, 0.8) == pytest.approx(1.8668, abs=5e-4)
    # T* nonincreasing on [1, inf)
    for prm in ((1.0, 1.0, 0.8), (0.5, 1.0, -0.9), (1.5768, 0.5751, -0.5711)):
        Ts = [H.explosion_time(a, *prm) for a in np.linspace(1.01, 40, 200)]
        assert all(t2 <= t1 + 1e-12 for t1, t2 in zip(Ts, Ts[1:]))


def test_carr_madan_needs_the_moment():
    """Beyond a_+(T) - 1 the damped characteristic function is not finite (or the price is wrong)."""
    prm = dict(v0=0.04, kappa=1.0, theta=0.04, xi=1.0, rho=0.8)
    T = 2.0
    ref = H.price(100.0, S, T, R, Y, "call", method="gil-pelaez", **prm)
    ok = H.carr_madan_price(100.0, S, T, R, Y, "call", alpha=0.5, h=0.01, n=60_000, **prm)
    assert ok == pytest.approx(ref, abs=1e-6)
    with np.errstate(all="ignore"):
        bad = H.carr_madan_price(100.0, S, T, R, Y, "call", alpha=1.2, h=0.01, n=60_000, **prm)
    assert not np.isfinite(bad) or abs(bad - ref) > 1e-3


# ----------------------------------------------------------------- Fourier pricing
def test_three_methods_agree_on_the_test_case():
    for prm, ref in ((LAB0, PUT0), (LAB7, PUT7)):
        vals = [H.price(100.0, S, 1.0, R, Y, "put", m, **prm) for m in ("cos", "gil-pelaez", "carr-madan")]
        assert max(vals) - min(vals) < 1e-12
        assert vals[0] == pytest.approx(ref, abs=1e-11)
    # Chapter 10's four-digit values
    assert round(PUT0, 4) == 8.1534 and round(PUT7, 4) == 7.9480
    # parity
    c = H.price(100.0, S, 1.0, R, Y, "call", **LAB7)
    assert c - PUT7 == pytest.approx(S * np.exp(-Y) - 100.0 * np.exp(-R), abs=1e-11)


def test_convergence_counts_quoted_in_the_notes():
    lo, hi = H.cos_interval(S, 100.0, 1.0, R, Y, **LAB7)
    assert (round(lo, 2), round(hi, 2)) == (-4.53, 1.35)
    err = lambda f: abs(f - PUT7)
    gp = lambda N: err(H.gil_pelaez_price(100.0, S, 1.0, R, Y, "put", h=125 / N, n=N, **LAB7))
    cm = lambda N, a: err(H.carr_madan_price(100.0, S, 1.0, R, Y, "put", alpha=a, h=125 / N, n=N, **LAB7))
    co = lambda N: err(H.cos_price(100.0, S, 1.0, R, Y, "put", N=N, interval=(lo, hi), **LAB7))
    assert gp(64) < 1e-8 < gp(48)
    assert co(128) < 1e-8 < co(96)
    assert cm(384, 1.5) < 1e-8 < cm(256, 1.5)
    assert cm(64, 9.4) < 1e-8 < cm(48, 9.4)
    assert (H.moment_bound(1.0, 2.0, 0.3, -0.7) - 1) / 4 == pytest.approx(9.4, abs=0.05)


def test_fang_oosterlee_values_and_their_interval():
    T1 = H.price(100.0, 100.0, 1.0, 0.0, 0.0, "call", **H.FO)
    T10 = H.price(100.0, 100.0, 10.0, 0.0, 0.0, "call", **H.FO)
    assert T10 == pytest.approx(22.318945791, abs=1e-9)
    assert T1 == pytest.approx(5.785155434, abs=1e-9)
    assert T1 - 5.785155450 == pytest.approx(-1.6e-8, abs=1e-9)
    stall = H.cos_price(100.0, 100.0, 1.0, 0.0, 0.0, "call", N=1024, L=12.0, **H.FO) - T1
    assert stall == pytest.approx(-3.9e-5, abs=5e-7)
    lo, hi = H.cos_interval(100.0, 100.0, 1.0, 0.0, 0.0, **H.FO)
    assert (round(lo, 2), round(hi, 2)) == (-7.40, 2.56)
    # the call priced directly on the twelve-standard-deviation interval converges (to about 8e-9)
    a, b = H.cos_range(100.0, 100.0, 1.0, 0.0, 0.0, 12.0, **H.FO)
    k = np.arange(512)
    w = k * np.pi / (b - a)
    phiY = H.charfn(w, 100.0, 1.0, 0.0, 0.0, **H.FO) * np.exp(-1j * w * np.log(100.0))
    chi, psi = H._chi_psi(k, a, b, 0.0, b)
    t = (phiY * np.exp(-1j * w * a)).real * 2 / (b - a) * 100.0 * (chi - psi)
    t[0] *= 0.5
    assert abs(t.sum() - T1) < 1e-8


def test_chernoff_interval_bounds_the_tails():
    """cos_interval: the Chernoff bounds of Q(Y < a) and Q(Y > b) are eps, and they do bound the true tails."""
    for prm in (LAB7, H.CASE_I):
        lo, hi = H.cos_interval(S, 100.0, 1.0, R, Y, eps=1e-3, **prm)
        pl, ph = H.tail_bounds(np.log(100.0) + lo, np.log(100.0) + hi, S, 1.0, R, Y, **prm)
        assert pl == pytest.approx(1e-3, rel=1e-6) and ph == pytest.approx(1e-3, rel=1e-6)
        u = (np.arange(40_000) + 0.5) * 0.005
        phi = H.charfn(u, S, 1.0, R, Y, **prm)
        for x, upper in ((np.log(100.0) + lo, False), (np.log(100.0) + hi, True)):
            gt = 0.5 + 0.005 / np.pi * np.sum((np.exp(-1j * u * x) * phi / (1j * u)).real)   # Q(X > x)
            tail = gt if upper else 1 - gt
            assert 0 < tail <= 1e-3
@pytest.mark.parametrize("rho", [-1.0, 1.0])
def test_perfect_correlation(rho):
    prm = dict(H.TEST, rho=rho)
    vals = [H.price(100.0, S, 1.0, R, Y, "put", m, **prm) for m in ("cos", "gil-pelaez", "carr-madan")]
    assert max(vals) - min(vals) < 1e-10


def test_degenerate_xi_zero_and_small_maturity():
    p = dict(v0=0.09, kappa=1.5, theta=0.04, rho=-0.5)
    w = 0.04 + 0.05 * (1 - np.exp(-1.5)) / 1.5
    ref = bs.put(S, 100.0, 1.0, R, Y, np.sqrt(w))
    assert ref == pytest.approx(10.561085, abs=5e-7)
    assert H.cos_price(S, 100.0, 1.0, R, Y, "put", xi=0.0, N=256, **p) == pytest.approx(ref, abs=1e-12)
    e = [abs(H.price(100.0, S, 1.0, R, Y, "put", xi=x, **p) - ref) for x in (1e-2, 1e-3)]
    assert e[1] < e[0] / 5                                         # O(xi)
    assert H.price(100.0, S, 1.0, R, Y, "put", xi=0.1, **p) == pytest.approx(10.477824, abs=5e-7)
    assert H.price(100.0, S, 1.0, R, Y, "put", xi=0.01, **p) == pytest.approx(10.555204, abs=5e-7)
    # V0 = 0 with Feller violated: the methods still agree
    q = dict(v0=0.0, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9)
    vals = [H.price(100.0, S, 1.0, R, Y, "put", m, **q) for m in ("cos", "gil-pelaez", "carr-madan")]
    assert max(vals) - min(vals) < 1e-10


def test_short_maturity_skew_heuristic():
    for T, val in ((0.01, -0.261), (1.0, -0.146)):
        Ks = 100 * np.exp((R - Y) * T) * np.exp(np.array([-0.01, 0.01]) * np.sqrt(T))
        iv = H.implied_vols(Ks, S, T, R, Y, **LAB7)
        assert (iv[1] - iv[0]) / (0.02 * np.sqrt(T)) == pytest.approx(val, abs=1e-3)
    assert -0.7 * 0.3 / (4 * 0.2) == pytest.approx(-0.2625)


def test_smile_shapes():
    Ks = np.array([70.0, 100.0, 140.0])
    neg = H.implied_vols(Ks, S, 1.0, R, Y, **LAB7)
    pos = H.implied_vols(Ks, S, 1.0, R, Y, **dict(H.TEST, rho=0.7))
    zero = H.implied_vols(Ks, S, 1.0, R, Y, **LAB0)
    assert neg[0] > neg[1] > neg[2] and pos[0] < pos[1] < pos[2]
    assert zero[0] > zero[1] < zero[2]


# ----------------------------------------------------------------- the variance step and QE
def test_cir_moments_match_noncentral_chi_square():
    v, h, kappa, theta, xi = 0.03, 0.25, 2.0, 0.04, 0.3
    m, s2 = H.cir_moments(v, h, kappa, theta, xi)
    e = np.exp(-kappa * h)
    c = xi**2 * (1 - e) / (4 * kappa)
    d, lam = 4 * kappa * theta / xi**2, v * e / c
    assert m == pytest.approx(c * (d + lam), rel=1e-14) and s2 == pytest.approx(c * c * (2 * d + 4 * lam), rel=1e-14)
    x = stratified.cir_exact(v, kappa, theta, xi, np.array([0.0, h]), 400_000, seed=1)[:, 1]
    assert x.mean() == pytest.approx(m, abs=4 * np.sqrt(s2 / 4e5))
    assert x.var() == pytest.approx(s2, rel=0.02)


def test_qe_matches_two_moments_exactly():
    v = np.array([0.0, 1e-4, 0.01, 0.04, 0.2])
    for h in (1.0, 0.1, 0.01):
        for psi_c in (1.0, 1.5, 2.0):
            q = H.qe_parameters(v, h, 0.5, 0.04, 1.0, psi_c)
            mean = np.where(q["quad"], q["a"] * (1 + q["b"] ** 2), (1 - q["p"]) / q["beta"])
            var = np.where(q["quad"], q["a"] ** 2 * (4 * q["b"] ** 2 + 2), (1 - q["p"]) * (1 + q["p"]) / q["beta"] ** 2)
            assert np.allclose(mean, q["m"], rtol=1e-12) and np.allclose(var, q["s2"], rtol=1e-11)
            assert np.all(q["psi"][q["quad"]] <= psi_c) and np.all(q["psi"][~q["quad"]] > psi_c)
    # sampled moments, both branches (psi < 1.5 for v = 0.04, h = 0.01; psi > 1.5 for v = 0)
    z = np.random.default_rng(8).standard_normal(1_000_000)
    for v0 in (0.04, 0.0):
        x, q = H.qe_step(np.full(z.size, v0), 0.01 if v0 else 0.5, 0.5, 0.04, 1.0, z)
        assert x.mean() == pytest.approx(float(q["m"][0]), rel=5e-3)
        assert x.var() == pytest.approx(float(q["s2"][0]), rel=2e-2)


def test_martingale_correction_is_exact():
    h, kappa, theta, xi, rho = 0.5, 0.5, 0.04, 1.0, -0.9
    K0, K1, K2, K3, K4 = H.andersen_k(h, kappa, theta, xi, rho)
    A = K2 + 0.5 * K4
    zq, wq = np.polynomial.hermite_e.hermegauss(80)
    wq = wq / wq.sum()
    for v in (0.0, 0.01, 0.04, 0.3):
        q = H.qe_parameters(np.array([v]), h, kappa, theta, xi)
        k0 = H.martingale_k0(q, np.array([v]), K1, K2, K3, K4)[0]
        if q["quad"][0]:
            M = np.sum(wq * np.exp(A * q["a"][0] * (q["b"][0] + zq) ** 2))
        else:
            p, b = q["p"][0], q["beta"][0]
            M = p + (1 - p) * b / (b - A)
        assert np.exp(k0 + (K1 + 0.5 * K3) * v) * M == pytest.approx(1.0, rel=1e-12)
    assert A < 0
    # rho > 0 and a long step: A >= beta on the exponential branch, so E[S_{t+h}] is infinite and no K0 exists
    K = H.andersen_k(4.0, 10.0, 0.04, 3.0, 0.9)
    q = H.qe_parameters(np.array([0.04]), 4.0, 10.0, 0.04, 3.0)
    assert not q["quad"][0] and K[2] + 0.5 * K[4] >= q["beta"][0]
    assert np.isnan(H.martingale_k0(q, np.array([0.04]), *K[1:])[0])


# ----------------------------------------------------------------- simulation
def test_schemes_preserve_the_forward_and_estimators_agree():
    for sch in ("euler", "qe"):
        s = H.simulate(S, 100.0, 1.0, R, Y, 8, 100_000, sch, estimator="plain", seed=[1, sch == "qe"], **LAB7)
        f = s["forward"]
        assert abs(f.mean() - S) < 4 * f.std() / np.sqrt(f.size)
        c = H.simulate(S, 100.0, 1.0, R, Y, 8, 100_000, sch, seed=[2, sch == "qe"], **LAB7)
        mp, sp = H.estimate(s, S, control=False)
        mc, sc = H.estimate(c, S)
        assert abs(mp - mc) < 4 * np.hypot(sp, sc)
        assert sp / sc > 4.5                                    # "a factor of about five"


def test_full_truncation_conditional_matches_plain_on_same_paths():
    """Proposition sv-full-truncation (ii): the conditional value given W is exact for the log-Euler scheme."""
    def var_path(dW, t):
        h = np.diff(t)
        V = np.empty(dW.shape)
        x = np.full(dW.shape[0], 0.04)
        for k in range(dW.shape[1]):
            V[:, k] = np.maximum(x, 0.0)
            x = x + 2.0 * (0.04 - V[:, k]) * h[k] + 0.3 * np.sqrt(V[:, k]) * dW[:, k]
        return V
    out = stratified.log_euler_mixing_check(S, 100.0, 1.0, R, Y, -0.7, var_path, m=8, n=200_000, seed=5)
    a, b = out["plain"], out["conditional"]
    assert abs(a.mean() - b.mean()) < 4 * np.sqrt(np.var(a - b) / a.size)
    sim = H.simulate(S, 100.0, 1.0, R, Y, 8, 200_000, "euler", seed=6, **LAB7)
    m, se = H.estimate(sim, S)
    assert abs(m - b.mean()) < 4 * np.hypot(se, b.std() / np.sqrt(b.size))


def test_chapter10_conditional_estimator_agrees_with_fourier():
    res = stratified.mixing_laboratory(S, 100.0, 1.0, R, Y, rhos=(0.0, -0.7), m=64, n=20_000, seed=19)
    for rho, ref in ((0.0, PUT0), (-0.7, PUT7)):
        mean, se = res[rho]["conditional"]
        assert abs(mean - ref) < 4 * se


def test_bias_numbers_quoted_in_the_notes():
    """Two points of Figure 19.2 with the seeds of the notes, recomputed: seed [19, i, j], i the step index."""
    def one(prm, n, i, m, j):
        sim = H.simulate(S, 100.0, 1.0, R, Y, m, n, ("euler", "qe")[j], seed=[19, i, j], **prm)
        return H.estimate(sim, S)[0]
    ref7 = H.price(100.0, S, 1.0, R, Y, "put", **LAB7)
    assert one(LAB7, 2**19, 2, 4, 0) - ref7 == pytest.approx(0.085, abs=5e-4)       # Euler, h = 1/4
    assert one(LAB7, 2**19, 0, 1, 1) - ref7 == pytest.approx(0.020, abs=5e-4)       # QE, h = 1
    refI = H.price(100.0, S, 1.0, R, Y, "put", **H.CASE_I)
    assert refI == pytest.approx(4.576954, abs=5e-7)
    assert one(H.CASE_I, 2**18, 0, 1, 1) - refI == pytest.approx(0.094, abs=5e-4)    # QE, h = 1
    assert one(H.CASE_I, 2**18, 2, 4, 1) - refI == pytest.approx(-0.021, abs=6e-4)   # QE, h = 1/4
