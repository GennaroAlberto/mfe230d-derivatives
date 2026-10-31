"""Checks of the Chapter 13 code against the closed forms, the results proved in the notes and
the numbers quoted there."""
import numpy as np
import pytest
from scipy.special import ndtr

from mfe230d import bs
from mfe230d import greeks as gr
from mfe230d import importance as im
from mfe230d.montecarlo import mc_estimate
from mfe230d.variance import strike_to_z

P = bs.TEST_CASE
S, K, T, r, y, SIG = (P[k] for k in ("S", "K", "T", "r", "y", "sigma"))
GREEKS = ("delta", "vega", "rho_r", "rho_y", "theta")


# ------------------------------------------------------------------- closed forms
def test_test_case_greeks():
    g = bs.greeks(**P)
    assert float(bs.call(**P)) == pytest.approx(7.36429, abs=5e-6)
    for k, v in dict(delta=0.509643, gamma=0.0195277, vega=39.0554, theta=-3.32226,
                     rho_r=43.6000, rho_y=-50.9643).items():
        assert float(g[k]) == pytest.approx(v, abs=6e-5 * max(1.0, abs(v)))
    assert float(bs.digital_call(**P)) == pytest.approx(0.436000, abs=1e-6)
    assert float(gr.exact_greek("digital", "delta", **P)) == pytest.approx(0.0195277, abs=1e-7)


@pytest.mark.parametrize("Sx", [80.0, 100.0, 125.0])
def test_call_gamma_and_digital_delta(Sx):
    # S Gamma_call = K Delta_dig, so they are equal at S = K.
    p = dict(P, S=Sx)
    assert Sx * float(bs.greeks(**p)["gamma"]) == pytest.approx(K * float(gr.exact_greek("digital", "delta", **p)), rel=1e-12)


@pytest.mark.parametrize("greek,param,h", [("delta", "S", 1e-3), ("gamma", "S", 1e-2), ("vega", "sigma", 1e-5),
                                           ("rho_r", "r", 1e-5), ("rho_y", "y", 1e-5), ("theta", "T", 1e-5)])
def test_digital_greeks_against_finite_differences(greek, param, h):
    f = lambda t: float(bs.digital_call(**dict(P, **{param: t})))  # noqa: E731
    x = P[param]
    fd = (f(x + h) - 2 * f(x) + f(x - h)) / h**2 if greek == "gamma" else (f(x + h) - f(x - h)) / (2 * h)
    if greek == "theta":
        fd = -fd
    assert float(gr.exact_greek("digital", greek, **P)) == pytest.approx(fd, rel=1e-5, abs=1e-9)


# ------------------------------------------------- pathwise and likelihood ratio: exact means
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("greek", GREEKS)
def test_pathwise_and_lr_are_unbiased(kind, greek):
    ex = float(bs.greeks(**P, kind=kind)[greek])
    m_pw, _ = gr.estimator_moments(gr.pathwise_samples, greek, kind, **P)
    m_lr, _ = gr.estimator_moments(gr.lr_samples, greek, kind, **P)
    assert m_pw == pytest.approx(ex, rel=1e-7, abs=1e-9)
    assert m_lr == pytest.approx(ex, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("greek", ("delta", "gamma", "vega", "rho_r", "rho_y", "theta"))
def test_lr_digital_unbiased_pathwise_digital_biased(greek):
    ex = float(gr.exact_greek("digital", greek, **P))
    assert gr.estimator_moments(gr.lr_samples, greek, "digital", **P)[0] == pytest.approx(ex, rel=1e-7)
    if greek != "gamma":
        m_pw = gr.estimator_moments(gr.pathwise_samples, greek, "digital", **P)[0]
        assert abs(m_pw - ex) > 0.01 * abs(ex)       # misses the density term
    m, v = gr.estimator_moments(gr.lr_samples, "delta", "digital", **P)
    assert np.sqrt(v) == pytest.approx(0.0290, abs=5e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("name", ["lr_lr", "pw_lr", "lr_pw"])
def test_gamma_estimators_unbiased(kind, name):
    ex = float(bs.greeks(**P, kind=kind)["gamma"])
    assert gr.estimator_moments(gr.gamma_samples, name, kind, **P)[0] == pytest.approx(ex, rel=1e-7)


def test_mixed_gammas_biased_for_the_digital():
    d = float(gr.exact_greek("digital", "delta", **P))
    assert gr.estimator_moments(gr.gamma_samples, "pw_lr", "digital", **P)[0] == 0.0
    assert gr.estimator_moments(gr.gamma_samples, "lr_pw", "digital", **P)[0] == pytest.approx(-d / S, rel=1e-7)


def test_per_sample_standard_deviations_quoted():
    sd = lambda fn, g, kind="call": np.sqrt(gr.estimator_moments(fn, g, kind, **P)[1])  # noqa: E731
    quoted = dict(delta=(0.582, 1.287), vega=(72.67, 246.70), rho_r=(49.15, 117.08),
                  rho_y=(58.22, 128.71), theta=(6.73, 23.45))
    for g, (a, b) in quoted.items():
        assert sd(gr.pathwise_samples, g) == pytest.approx(a, rel=1e-3)
        assert sd(gr.lr_samples, g) == pytest.approx(b, rel=1e-3)
        assert 2.2 <= sd(gr.lr_samples, g) / sd(gr.pathwise_samples, g) <= 3.5
    for name, v in dict(lr_lr=0.1234, pw_lr=0.0363, lr_pw=0.0290).items():
        assert sd(gr.gamma_samples, name) == pytest.approx(v, abs=6e-5)
    assert sd(gr.lr_samples, "rho_y") / sd(gr.pathwise_samples, "rho_y") == pytest.approx(2.21, abs=0.005)
    m = gr.fd_moments("call", "S", 1.0, "second", True, **P)
    assert np.sqrt(m["var"]) == pytest.approx(0.112, abs=5e-4)


def test_draw_by_draw_identities():
    z = np.random.default_rng(0).standard_normal(1000)
    pw, lr, gm = gr.pathwise_samples(z, "call", **P), gr.lr_samples(z, "call", **P), gr.gamma_samples(z, "call", **P)
    assert np.allclose(pw["rho_y"], -T * S * pw["delta"], atol=1e-12)
    assert np.allclose(pw["rho_r"], -T * K * pw["dual_delta"], atol=1e-12)
    assert np.allclose(lr["rho_y"], -T * S * lr["delta"], atol=1e-12)
    dig = gr.lr_samples(z, "digital", **P)["delta"]
    assert np.allclose(gm["lr_pw"], K / S * dig, atol=1e-14)        # LR-PW gamma of the call = (K/S) x LR digital delta
    fd = gr.fd_spot_greeks(z, "call", eps=0.01, **P)
    assert np.all(fd["delta_plus"] - fd["delta_minus"] >= -1e-12)     # convex payoff


# --------------------------------------------------------------------- finite differences
def test_fd_moments_against_simulation():
    f = gr.payoff_function("call", "S", **P)
    rng = np.random.default_rng(3)
    z, z1 = rng.standard_normal(400_000), rng.standard_normal(400_000)
    for crn in (True, False):
        m = gr.fd_moments("call", "S", 1.0, "central", crn, **P)
        x = gr.fd_samples(f, S, 1.0, z, "central", None if crn else [z1])
        assert x.var() == pytest.approx(m["var"], rel=0.02)
        assert abs(x.mean() - m["mean"]) < 4 * x.std() / np.sqrt(x.size)
    # bias is exact: the same difference of the closed form
    m = gr.fd_moments("call", "S", 2.0, "forward", True, **P)
    assert m["mean"] == pytest.approx((float(bs.call(**dict(P, S=102.0))) - float(bs.call(**P))) / 2.0, rel=1e-12)


def test_independent_variance_formula():
    # Proposition (ii): (sigma^2(theta+h) + sigma^2(theta-h)) / (4 h^2) for the central difference.
    h = 2.0
    v = lambda s: gr.estimator_moments(lambda z, kind, **p: dict(x=gr.discounted_payoffs(z, kind, **p)),  # noqa: E731
                                       "x", "call", **dict(P, S=s))[1]
    m = gr.fd_moments("call", "S", h, "central", False, **P)
    assert m["var"] == pytest.approx((v(S + h) + v(S - h)) / (4 * h * h), rel=1e-8)


def test_crn_lipschitz_bound():
    # Proposition (iii)(a): kappa = e^{-rT} G for the call in S, so Var <= E[kappa^2] for every h.
    Ek2 = np.exp(-2 * r * T) * np.exp(2 * (r - y) * T + SIG**2 * T)
    for h in (1e-3, 0.1, 1.0, 10.0):
        for scheme in ("forward", "central"):
            assert gr.fd_moments("call", "S", h, scheme, True, **P)["var"] <= Ek2


def test_digital_example_numbers():
    # Example gr-digital: second moment ~ 2 h e^{-rT} Delta_dig; optimal bumps and errors.
    d = float(gr.exact_greek("digital", "delta", **P))
    for h in (1e-3, 1e-2):
        zp = float(strike_to_z(K, S - h, T, r, y, SIG)); zm = float(strike_to_z(K, S + h, T, r, y, SIG))
        m2 = np.exp(-2 * r * T) * (ndtr(zp) - ndtr(zm))
        assert m2 / (2 * h) == pytest.approx(np.exp(-r * T) * d, rel=1e-4)
    o = gr.optimal_bump("digital", "S", 10**6, "central", True, bounds=(1e-3, 40), **P)
    assert o["h"] == pytest.approx(2.07, abs=0.01) and o["rmse"] == pytest.approx(7.39e-5, abs=1e-7)
    assert o["rmse"] / d == pytest.approx(0.0038, abs=5e-5)
    oi = gr.optimal_bump("digital", "S", 10**6, "central", False, bounds=(1e-3, 40), **P)
    assert oi["rmse"] == pytest.approx(1.35e-4, abs=1e-6)
    sd_lr = np.sqrt(gr.estimator_moments(gr.lr_samples, "delta", "digital", **P)[1])
    assert (o["rmse"] / (sd_lr / 1e3)) ** 2 == pytest.approx(6.5, abs=0.05)
    # asymptotic optimal bump from Proposition (iv)
    f = lambda s: float(bs.digital_call(**dict(P, S=s)))  # noqa: E731
    hh = 0.05
    a3 = (f(S + 2 * hh) - 2 * f(S + hh) + 2 * f(S - hh) - f(S - 2 * hh)) / (2 * hh**3)
    assert a3 == pytest.approx(-4.82e-5, abs=5e-8)
    b, v = a3 / 6, np.exp(-r * T) * d / 2
    c = (v / (4 * b * b)) ** 0.2
    assert c * 1e6 ** -0.2 == pytest.approx(2.06, abs=0.01)


def test_call_delta_bump_regimes():
    o = gr.optimal_bump("call", "S", 10**4, "forward", False, bounds=(1e-3, 60), **P)
    assert o["h"] == pytest.approx(4.4, abs=0.05) and o["rmse"] == pytest.approx(0.061, abs=5e-4)
    floor = 0.5822 / 100
    for e in (1e-4, 1e-3, 1e-2, 3e-2):
        assert gr.fd_rmse("call", "S", e * S, 10**4, "central", True, **P) == pytest.approx(floor, rel=0.03)
    for e in (1e-4, 1e-3):
        assert gr.fd_rmse("call", "S", e * S, 10**4, "forward", True, **P) == pytest.approx(floor, rel=0.02)
    assert gr.fd_rmse("call", "S", 1.0, 10**4, "forward", True, **P) > 1.9 * floor


def test_kernel_remark():
    z = np.random.default_rng(5).standard_normal(2000)
    G = gr.growth(z, T, r, y, SIG)
    for h in (0.1, 1.0, 10.0):
        g = gr.fd_spot_greeks(z, "call", eps=h / S, **P)["gamma_c"]
        k = np.maximum(h - np.abs(S - K / G), 0.0) / h**2
        assert np.allclose(g, np.exp(-r * T) * G * k, atol=1e-10)
    counts = []
    for e in (0.001, 0.01, 0.1):
        h = e * S
        p = ndtr(strike_to_z(K, S - h, T, r, y, SIG)) - ndtr(strike_to_z(K, S + h, T, r, y, SIG))
        counts.append(1024 * float(p))
    assert counts == pytest.approx([4.04, 40.4, 387.8], abs=0.05)


# --------------------------------------------------------------------- the laboratory table
def test_table_numbers():
    t = gr.greeks_table(n=100_000, seed=13)
    # (estimate, standard error, unit of the last quoted digit)
    quoted = dict(delta=dict(pw=(0.5108, 0.0018, 1e-4), lr=(0.5093, 0.0041, 1e-4), fd=(0.5108, 0.0018, 1e-4)),
                  gamma=dict(pw=(0.01954, 0.00011, 1e-5), lr=(0.01941, 0.00039, 1e-5), fd=(0.01977, 0.00036, 1e-5),
                             lr_pw=(0.019557, 0.000092, 1e-6)),
                  vega=dict(pw=(39.08, 0.23, 1e-2), lr=(38.82, 0.79, 1e-2), fd=(39.08, 0.23, 1e-2)),
                  rho_r=dict(pw=(43.71, 0.16, 1e-2), lr=(43.55, 0.37, 1e-2), fd=(43.71, 0.16, 1e-2)),
                  rho_y=dict(pw=(-51.08, 0.18, 1e-2), lr=(-50.93, 0.41, 1e-2), fd=(-51.08, 0.18, 1e-2)),
                  theta=dict(pw=(-3.324, 0.021, 1e-3), lr=(-3.299, 0.075, 1e-3), fd=(-3.324, 0.021, 1e-3)))
    for g, row in quoted.items():
        for k, (m, se, unit) in row.items():
            assert abs(t[g][k][0] - m) <= 0.5 * unit + 1e-12
            assert abs(t[g][k][1] - se) <= 0.5 * unit + 1e-12
            assert abs(t[g][k][0] - t[g]["exact"]) < 2 * t[g][k][1]


# ---------------------------------------------------------------------- degenerate cases
def test_share_variances_and_small_vol_limit():
    s2 = SIG**2 * T
    lr = np.exp(-2 * y * T) * (np.exp(s2) * (1 + 4 * s2) / s2 - 1)
    pw = np.exp(-2 * y * T) * (np.exp(s2) - 1)
    Q = dict(P, K=1e-300)
    assert gr.estimator_moments(gr.lr_samples, "delta", "call", **Q)[1] == pytest.approx(lr, rel=1e-6)
    assert gr.estimator_moments(gr.pathwise_samples, "delta", "call", **Q)[1] == pytest.approx(pw, rel=1e-6)
    assert (lr, pw, lr / pw) == pytest.approx((28.04, 0.0392, 715), rel=2e-3)
    # sigma^2 T Var(Delta_lr) -> e^{-2rT} f(F)^2 / S^2 for an in-the-money call (Remark gr-lr-degenerate)
    Kx = 80.0
    F = S * np.exp((r - y) * T)
    lim = np.exp(-2 * r * T) * (F - Kx) ** 2 / S**2
    vals = [s**2 * T * gr.estimator_moments(gr.lr_samples, "delta", "call", **dict(P, K=Kx, sigma=s))[1]
            for s in (0.02, 0.005)]
    assert abs(vals[1] - lim) < abs(vals[0] - lim) and vals[1] == pytest.approx(lim, rel=0.01)


def test_pathwise_delta_at_zero_spot():
    z = np.random.default_rng(1).standard_normal(100)
    assert np.all(gr.pathwise_samples(z, "call", **dict(P, S=0.0))["delta"] == 0.0)


# ------------------------------------------------------------------------------- the Asian
def test_geometric_asian_greeks_closed_form():
    g = gr.geometric_asian_greeks(S, K, T, r, y, SIG, 12)
    f = lambda s=S, v=SIG: float(im.geometric_asian_call(s, K, T, r, y, v, 12))  # noqa: E731
    assert g["price"] == pytest.approx(f(), rel=1e-12)
    assert g["delta"] == pytest.approx((f(s=S + 1e-3) - f(s=S - 1e-3)) / 2e-3, rel=1e-7)
    assert g["vega"] == pytest.approx((f(v=SIG + 1e-5) - f(v=SIG - 1e-5)) / 2e-5, rel=1e-7)
    assert (g["delta"], g["vega"]) == pytest.approx((0.486882, 22.4010), abs=1e-4)


def test_asian_pathwise_numbers():
    Z = np.random.default_rng(3).standard_normal((400_000, 12))
    g = gr.asian_samples(Z, S, K, T, r, y, SIG, "geometric")
    a = gr.asian_samples(Z, S, K, T, r, y, SIG, "arithmetic")
    ex = gr.geometric_asian_greeks(S, K, T, r, y, SIG, 12)
    for s, key, m, se, unit in ((g, "pw_delta", 0.4866, 0.0009, 1e-4), (g, "pw_vega", 22.42, 0.06, 1e-2),
                                (a, "pw_delta", 0.4951, 0.0009, 1e-4), (a, "pw_vega", 24.07, 0.07, 1e-2)):
        e = mc_estimate(s[key])
        assert abs(float(e["mean"]) - m) <= 0.5 * unit
        assert abs(float(e["se"]) - se) <= 0.5 * unit
    for key, exk in (("pw_delta", "delta"), ("pw_vega", "vega"), ("lr_delta", "delta"), ("lr_vega", "vega")):
        e = mc_estimate(g[key])
        assert abs(float(e["mean"]) - ex[exk]) < 3 * float(e["se"])


def test_asian_sd_ratios_quoted():
    # m = 1: the Asian is the call, exact ratios from quadrature (Table gr-lab)
    sd = lambda fn, g: np.sqrt(gr.estimator_moments(fn, g, "call", **P)[1])  # noqa: E731
    assert sd(gr.lr_samples, "delta") / sd(gr.pathwise_samples, "delta") == pytest.approx(2.2, abs=0.05)
    assert sd(gr.lr_samples, "vega") / sd(gr.pathwise_samples, "vega") == pytest.approx(3.4, abs=0.05)
    rng = np.random.default_rng(2026)
    for m in 2 ** np.arange(0, 8):
        s = gr.asian_samples(rng.standard_normal((100_000, m)), **P)
    ratio = lambda k: s["lr_" + k].std(ddof=1) / s["pw_" + k].std(ddof=1)  # noqa: E731
    assert ratio("delta") == pytest.approx(8.9, abs=0.05)
    assert ratio("vega") == pytest.approx(17, abs=0.5)


def test_rho_y_ratio_against_spot():
    r_ = []
    for s0 in (90.0, 95.0, 100.0, 105.0, 110.0):
        p = dict(P, S=s0)
        r_.append(np.sqrt(gr.estimator_moments(gr.lr_samples, "rho_y", "call", **p)[1]
                          / gr.estimator_moments(gr.pathwise_samples, "rho_y", "call", **p)[1]))
    assert r_ == pytest.approx([1.86, 2.01, 2.21, 2.47, 2.81], abs=0.005)


def test_theta_variance_as_maturity_shrinks():
    tv = lambda s0, t: t * gr.estimator_moments(gr.pathwise_samples, "theta", "call", **dict(P, S=s0, T=t))[1]  # noqa: E731
    assert tv(90.0, 1.0) == pytest.approx(37.6, abs=0.05)
    assert tv(90.0, 0.01) / 0.01 == pytest.approx(0.019, abs=5e-4)
    assert tv(90.0, 0.001) < 1e-10                                   # out of the money: -> 0
    lim_atm = 0.25 * S**2 * SIG**2 * (0.5 - 1 / (2 * np.pi))
    assert lim_atm == pytest.approx(34.1, abs=0.05)
    assert abs(tv(100.0, 1e-3) - lim_atm) < abs(tv(100.0, 1e-2) - lim_atm) < 1.5
    assert tv(110.0, 1e-3) == pytest.approx(0.25 * 110**2 * SIG**2, rel=1e-3)   # = 121


def test_second_difference_kernel_bound():
    # Proposition (iii)(c) for the call: |second difference| <= h e^{-rT} G 1{|S - K/G| < h},
    # with E[e^{-2rT} G^2 1_A] = O(h), so h * Var(one second difference) stays bounded.
    vals = []
    for h in (0.1, 0.01, 0.001):
        lo, hi = float(strike_to_z(K, S + h, T, r, y, SIG)), float(strike_to_z(K, S - h, T, r, y, SIG))
        g2 = lambda u: np.exp(-2 * r * T) * float(gr.growth(u, T, r, y, SIG)) ** 2  # noqa: E731
        from scipy.integrate import quad
        m2 = quad(lambda u: g2(u) * np.exp(-u * u / 2) / np.sqrt(2 * np.pi), lo, hi)[0]
        vals.append(m2 / h)
        assert h * gr.fd_moments("call", "S", h, "second", True, **P)["var"] <= m2 / h
    assert max(vals) / min(vals) < 1.01
    m = gr.fd_moments("call", "S", 1.0, "second", True, **P)
    assert abs(m["bias"]) / m["exact"] == pytest.approx(2e-4, abs=0.5e-4)


def test_lr_vega_weight_variance():
    # Var((Z^2 - 1)/sigma - sqrt(dt) Z) = 2/sigma^2 + dt
    z = np.random.default_rng(9).standard_normal(2_000_000)
    dt = 1 / 12
    assert np.var((z * z - 1) / SIG - np.sqrt(dt) * z) == pytest.approx(2 / SIG**2 + dt, rel=0.01)


# --------------------------------------------------------------------------------- adjoints
@pytest.mark.parametrize("average", ["arithmetic", "geometric"])
def test_adjoint_equals_tangent_and_pathwise(average):
    m = 12
    Z = np.random.default_rng(11).standard_normal((5000, m))
    sig = SIG * (1 + 0.1 * np.sin(np.arange(m)))       # a term structure of volatilities
    a = gr.asian_adjoint(Z, S, K, T, r, y, sig, average)
    assert np.allclose(gr.asian_tangent(Z, S, K, T, r, y, sig, {"S": 1.0}, average), a["delta"], atol=1e-12)
    assert np.allclose(gr.asian_tangent(Z, S, K, T, r, y, sig, {"r": 1.0}, average), a["rho_r"], atol=1e-11)
    assert np.allclose(gr.asian_tangent(Z, S, K, T, r, y, sig, {"y": 1.0}, average), a["rho_y"], atol=1e-11)
    for j in range(m):
        e = np.zeros(m); e[j] = 1.0
        assert np.allclose(gr.asian_tangent(Z, S, K, T, r, y, sig, {"sigma": e}, average), a["vega"][:, j], atol=1e-11)
    flat = gr.asian_adjoint(Z, S, K, T, r, y, SIG * np.ones(m), average)
    s = gr.asian_samples(Z, S, K, T, r, y, SIG, average)
    assert np.allclose(flat["vega"].sum(axis=1), s["pw_vega"], atol=1e-10)
    assert np.allclose(flat["delta"], s["pw_delta"], atol=1e-12)
    assert np.allclose(flat["price"], s["price"], atol=1e-11)


def test_adjoint_against_bumps_and_figure_total():
    m, n = 12, 100_000
    Z = np.random.default_rng(2027).standard_normal((n, m))
    sig = SIG * np.ones(m)
    a = gr.asian_adjoint(Z, S, K, T, r, y, sig)
    v = a["vega"].mean(axis=0)
    assert v.sum() == pytest.approx(23.97, abs=0.005)
    assert np.all(np.diff(v) < 0)                        # early periods matter most
    h = 1e-4
    for j in (0, 6, 11):
        e = np.zeros(m); e[j] = h
        b = (gr.asian_adjoint(Z, S, K, T, r, y, sig + e)["price"] - gr.asian_adjoint(Z, S, K, T, r, y, sig - e)["price"]) / (2 * h)
        assert abs(b.mean() - v[j]) < 1e-3
