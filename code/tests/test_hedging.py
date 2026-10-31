"""Tests for mfe230d.hedging (Chapter 18): closed forms, the exact variance of the discrete
hedge, the wrong-volatility identity, Leland's one-period balance, and every number quoted in
the notes (tables and text)."""
import numpy as np
import pytest
from scipy import integrate

from mfe230d import bs
from mfe230d import hedging as H
from mfe230d.paths import gbm_paths, time_grid
from mfe230d.variance import delta_hedge_control, gauss_expect, scalar_payoffs

P = dict(S=100.0, K=100.0, T=1.0, r=0.01, y=0.02, sigma=0.2)
G = np.exp(0.01)              # time-T money: e^{rT}


# ----------------------------------------------------------------------- claims and Greeks
def test_greeks_and_payoffs():
    x = np.array([70.0, 95.0, 100.0, 120.0])
    for kind in H.KINDS:
        V, D, Gm = H.price_delta_gamma(kind, x, 100.0, 0.4, 0.01, 0.02, 0.2)
        e = 1e-3
        up = H.price_delta_gamma(kind, x + e, 100.0, 0.4, 0.01, 0.02, 0.2)[0]
        dn = H.price_delta_gamma(kind, x - e, 100.0, 0.4, 0.01, 0.02, 0.2)[0]
        assert D == pytest.approx((up - dn) / (2 * e), rel=1e-6, abs=1e-9)
        assert Gm == pytest.approx((up - 2 * V + dn) / e**2, rel=1e-4, abs=1e-7)
        assert H.delta_only(kind, x, 100.0, 0.4, 0.01, 0.02, 0.2) == pytest.approx(D, abs=1e-14)
    assert H.payoff("put", [90.0, 110.0], 100.0).tolist() == [10.0, 0.0]
    with pytest.raises(ValueError):
        H.payoff("straddle", 1.0, 1.0)


# ------------------------------------------------------- the conventions of the experiment
def test_hedge_matches_chapter8_control_and_parity():
    z = np.random.default_rng(3).standard_normal((500, 52))
    Y, M = delta_hedge_control(z, **P)
    t = time_grid(1.0, 52)
    paths = gbm_paths(100.0, t, r=0.01, y=0.02, sigma=0.2, Z=z)
    out = H.hedge_paths(paths, t, 100.0, 0.01, 0.02, 0.2)
    assert out["disc_error"] == pytest.approx(bs.call(100, 100, 1, 0.01, 0.02, 0.2) + M - Y, abs=1e-11)
    # dividends reinvested: e^{-y(T - t)} shares hedge the forward exactly, so call and put errors agree
    put = H.hedge_paths(paths, t, 100.0, 0.01, 0.02, 0.2, kind="put")
    assert np.max(np.abs(out["error"] - put["error"])) < 1e-10
    # chunked simulation is the same as one block
    a = H.hedge_experiment(**P, m=12, n=3000, seed=5, chunk=1000)["error"]
    b = H.hedge_experiment(**P, m=12, n=3000, seed=5, chunk=10_000)["error"]
    assert np.max(np.abs(a - b)) < 1e-12


def test_unbiased_under_q_and_stream_agrees():
    e = H.hedge_experiment(**P, m=12, n=40_000, seed=11)["error"]
    s = H.summarize(e)
    assert abs(s["mean"]) < 4 * s["se"]
    f = H.summarize(H.hedge_stream(**P, m=12, n=40_000, seed=12)["error"])
    assert abs(f["mean"]) < 4 * f["se"]
    assert f["sd"] == pytest.approx(s["sd"], rel=0.03)


# ------------------------------------------------- Proposition hd-discrete-error: the constant
def test_gamma_moment_closed_form():
    g0 = H.price_delta_gamma("call", 100.0, 100.0, 1.0, 0.01, 0.02, 0.2)[2]
    assert H.gamma_moment(0.0, **P) == pytest.approx(100.0**4 * g0**2, rel=1e-12)
    z = np.random.default_rng(1).standard_normal(400_000)
    for t, mu in ((0.5, 0.01), (0.9, 0.10), (0.97, -0.10)):
        St = 100.0 * np.exp((mu - 0.02 - 0.02) * t + 0.2 * np.sqrt(t) * z)
        x = np.exp(-0.02 * t) * St**4 * H.price_delta_gamma("call", St, 100.0, 1 - t, 0.01, 0.02, 0.2)[2] ** 2
        assert H.gamma_moment(t, **P, mu=mu) == pytest.approx(x.mean(), abs=4 * x.std() / np.sqrt(x.size))
    # psi(t) ~ c (T - t)^{-1/2}: sqrt(tau) psi converges as tau -> 0
    a, b = (np.sqrt(tau) * H.gamma_moment(1 - tau, **P) for tau in (1e-6, 1e-8))
    assert a == pytest.approx(b, rel=1e-3)
    with pytest.raises(ValueError):
        H.gamma_moment(0.5, **P, kind="digital")


def test_leading_constants():
    I = H.gamma_integral(**P)
    assert I == pytest.approx(60391.4446, rel=1e-8)
    # plain quadrature in t (no substitution) agrees
    J = integrate.quad(lambda t: H.gamma_moment(t, **P), 0, 1, limit=500, points=[0.999, 0.99999])[0]
    assert J == pytest.approx(I, rel=1e-5)
    consts = {None: 7.0206, 0.10: 6.9991, 0.20: 6.5583, -0.10: 6.5458}
    for mu, c in consts.items():
        assert H.asymptotic_sd(**P, m=1, mu=mu) == pytest.approx(c, abs=5e-5)
    assert H.asymptotic_sd(**P, m=52, time_T=False) == pytest.approx(np.sqrt(0.5 * 0.2**4 * I / 52), rel=1e-12)


def test_exact_variance_reproduces_chapter8():
    e1, vy = H.exact_error_variance(**P, m=1)
    # independent route for m = 1 (Chapter 8): Var(Y - beta X) with beta = N(d_+) for the stock control
    p = scalar_payoffs()
    c, st, zs = p["call"], p["stock"], p["z_star"]
    b = float(bs.norm_cdf(bs.d_pm(100, 100, 1, 0.01, 0.02, 0.2)[0]))
    my, mx = gauss_expect(c, [zs]), gauss_expect(st)
    vY = gauss_expect(lambda u: c(u) ** 2, [zs]) - my * my
    vX = gauss_expect(lambda u: st(u) ** 2) - mx * mx
    cxy = gauss_expect(lambda u: c(u) * st(u), [zs]) - mx * my
    assert vy == pytest.approx(vY, rel=1e-9)
    assert e1 == pytest.approx(vY - 2 * b * cxy + b * b * vX, rel=1e-8)
    ratios = {1: 0.4757, 12: 0.1516, 52: 0.0748}         # Chapter 8 simulated 0.48, 0.15, 0.075
    for m, rho in ratios.items():
        assert np.sqrt(H.exact_error_variance(**P, m=m)[0] / vy) == pytest.approx(rho, abs=5e-5)
    lead = {1: 0.553, 12: 0.160, 52: 0.0767}
    for m, rho in lead.items():
        assert H.asymptotic_sd(**P, m=m, time_T=False) / np.sqrt(vy) == pytest.approx(rho, abs=5e-4)
    # quadrature converged
    assert H.exact_error_variance(**P, m=52)[0] == pytest.approx(H.exact_error_variance(**P, m=52, n_nodes=20001)[0],
                                                                  rel=1e-9)


def test_exact_variance_approaches_leading_term():
    c = H.asymptotic_sd(**P, m=1)
    vals = {m: np.sqrt(m) * G * np.sqrt(H.exact_error_variance(**P, m=m)[0]) for m in (16, 64, 256)}
    gaps = [c - vals[m] for m in (16, 64, 256)]
    assert all(g > 0 for g in gaps) and gaps[0] > gaps[1] > gaps[2]
    assert vals[256] == pytest.approx(c, rel=0.015)


# --------------------------------------------------------- Table hd-hw5 (n = 50,000, seed 2026)
TABLE1 = {  # m: (mean_Q, sd_Q, exact, leading, mae_Q, (mean, sd) for mu = 10%, 20%, -10%)
    0: (0.010, 12.651, 12.696, None, 9.342, ((-5.667, 16.830), (-14.453, 21.352), (4.286, 8.013))),
    1: (-0.000, 6.017, 6.040, 7.021, 4.749, ((-0.830, 7.238), (-3.694, 9.462), (-1.086, 6.041))),
    4: (0.010, 3.207, 3.223, 3.510, 2.459, ((-0.194, 3.367), (-0.855, 3.490), (-0.257, 3.046))),
    12: (-0.001, 1.942, 1.925, 2.027, 1.466, ((-0.062, 1.965), (-0.266, 1.911), (-0.097, 1.801))),
    52: (-0.001, 0.956, 0.949, 0.974, 0.715, ((-0.017, 0.954), (-0.064, 0.903), (-0.019, 0.888))),
    252: (-0.004, 0.437, 0.437, 0.442, 0.326, ((-0.006, 0.436), (-0.014, 0.410), (-0.008, 0.409))),
}


def test_table_discrete_hedging():
    for m, (mq, sq, ex, lead, mae, drifts) in TABLE1.items():
        s = H.summarize(H.hedge_experiment(**P, m=m, n=50_000, seed=2026)["error"])
        assert (round(s["mean"], 3), round(s["sd"], 3), round(s["mae"], 3)) == (mq, sq, mae)
        assert abs(s["mean"]) < 3.5 * s["se"]      # the m = 252 mean is 2.2 se from 0 (seed 2026)
        v = H.exact_error_variance(**P, m=max(m, 1))
        assert round(G * np.sqrt(v[1] if m == 0 else v[0]), 3) == ex
        if lead is not None:
            assert round(H.asymptotic_sd(**P, m=m), 3) == lead
        for mu, (mm, ss) in zip((0.10, 0.20, -0.10), drifts):
            d = H.summarize(H.hedge_experiment(**P, m=m, n=50_000, mu=mu, seed=2026)["error"])
            assert (round(d["mean"], 3), round(d["sd"], 3)) == (mm, ss)


# --------------------------------------------------------- Proposition hd-wrong-vol and the lab
def test_wrong_vol_identities():
    Vi, Vr = bs.call(100, 100, 1, 0.01, 0.02, 0.2), bs.call(100, 100, 1, 0.01, 0.02, 0.25)
    # under Q (mu = r) the expected P&L of the implied hedge is the price difference
    assert H.expected_vol_pnl(100, 100, 1, 0.01, 0.02, 0.2, 0.25, 0.01) == pytest.approx(Vr - Vi, rel=1e-10)
    z = np.random.default_rng(4).standard_normal(400_000)
    for t, mu in ((0.0, 0.1), (0.4, 0.1), (0.95, -0.1), (1.0, 0.2)):
        St = 100.0 * np.exp((mu - 0.02 - 0.5 * 0.25**2) * t + 0.25 * np.sqrt(t) * z)
        if t < 1.0:
            x = St**2 * H.price_delta_gamma("call", St, 100.0, 1 - t, 0.01, 0.02, 0.2)[2]
            assert H.gamma_s2_moment(t, 100, 100, 1, 0.01, 0.02, 0.2, 0.25, mu) == pytest.approx(
                x.mean(), abs=4 * x.std() / np.sqrt(x.size) + 1e-12)
    # at t = T the formula is K^2 times the density of S_T at K
    m_, v_ = np.log(100) + (0.2 - 0.02 - 0.5 * 0.0625), 0.0625
    dens = np.exp(-(np.log(100) - m_) ** 2 / (2 * v_)) / (100 * np.sqrt(2 * np.pi * v_))
    assert H.gamma_s2_moment(1.0, 100, 100, 1, 0.01, 0.02, 0.2, 0.25, 0.2) == pytest.approx(100**2 * dens, rel=1e-12)


def test_wrong_vol_pathwise_identity():
    """Discrete hedge at sigma_h of a stock with volatility sigma_r: the seller's P&L tends,
    path by path, to (1/2) int e^{-rt} Gamma_h S^2 (sigma_h^2 - sigma_r^2) dt."""
    rms = []
    for m in (250, 4000):
        t, paths = H.simulate_stock(100.0, 1.0, m, 200, 0.10, 0.02, 0.25, seed=9 if m == 250 else None,
                                    Z=None if m == 250 else np.random.default_rng(9).standard_normal((200, m)))
        disc = H.hedge_paths(paths, t, 100.0, 0.01, 0.02, 0.2)["disc_error"]
        cont = -H.vol_pnl_paths(paths, t, 100.0, 0.01, 0.02, 0.2, 0.25)
        rms.append(np.sqrt(np.mean((disc - cont) ** 2)))
    assert rms[1] < 0.2 and rms[0] / rms[1] > 2.5           # error of order m^{-1/2}


def test_implied_versus_realised_numbers():
    Vi, Vr = bs.call(100, 100, 1, 0.01, 0.02, 0.2), bs.call(100, 100, 1, 0.01, 0.02, 0.25)
    assert round(G * (Vr - Vi), 3) == 1.970 and round(Vr, 3) == 9.315
    epnl = {0.01: 1.970, 0.05: 1.987, 0.10: 1.984, 0.20: 1.899, -0.10: 1.839}
    for mu, v in epnl.items():
        assert round(G * H.expected_vol_pnl(100, 100, 1, 0.01, 0.02, 0.2, 0.25, mu), 3) == v
    t, paths = H.simulate_stock(100.0, 1.0, 252, 20_000, 0.10, 0.02, 0.25, seed=18)
    pr = -H.hedge_paths(paths, t, 100.0, 0.01, 0.02, 0.25, premium=Vi)["error"]
    pi = -H.hedge_paths(paths, t, 100.0, 0.01, 0.02, 0.2, premium=Vi)["error"]
    assert (round(pr.mean(), 3), round(pr.std(ddof=1), 3)) == (1.972, 0.545)
    assert (round(pi.mean(), 3), round(pi.std(ddof=1), 3)) == (1.983, 0.998)
    cont = G * H.vol_pnl_paths(paths, t, 100.0, 0.01, 0.02, 0.2, 0.25)
    assert (round(cont.std(ddof=1), 3), round(cont.min(), 3)) == (0.828, 0.216)
    mtm = H.mark_to_market(paths[:5], t, 100.0, 0.01, 0.02, 0.2, 0.2, Vi)
    assert mtm[:, 0] == pytest.approx(0.0, abs=1e-12)
    assert G * mtm[:, -1] == pytest.approx(pi[:5], abs=1e-10)


# ------------------------------------------------------------------ Theorem hd-leland: costs
def test_leland_one_period_balance():
    assert H.leland_vol(0.2, 0.0, 0.01) == 0.2
    for h in (1e-3, 1e-4):
        c = H.one_period_cost(100.0, 100.0, 1.0, 0.5, h, 0.01, 0.02, 0.2, mu=0.10)
        lead = H.one_period_cost_leading(100.0, 100.0, 1.0, 0.5, h, 0.01, 0.02, 0.2)
        assert c / lead == pytest.approx(1.0, abs=0.02 if h == 1e-3 else 0.006)
    # Leland's choice equates the leading cost k * lead with the extra time decay
    h, k = 1.0 / 52, 0.02
    sh = H.leland_vol(0.2, k, h)
    G_h = H.price_delta_gamma("call", 100.0, 100.0, 0.5, 0.01, 0.02, sh)[2]
    assert 0.5 * (sh**2 - 0.2**2) * 100.0**2 * G_h * h == pytest.approx(
        k * H.one_period_cost_leading(100.0, 100.0, 1.0, 0.5, h, 0.01, 0.02, 0.2, sigma_hedge=sh), rel=1e-12)
    assert H.expected_cost_rate(**P) == pytest.approx(15.5809, abs=5e-4)


TABLE2 = {  # m: (sigma_hat, V_hat, mean/sd hedged at sigma, mean/sd Leland, Leland at mu = 10%, 20%, -10%)
    1: (0.2078, 7.670, (0.031, 5.988), (0.339, 5.983), ((-0.447, 7.167), (-3.289, 9.343), (-0.805, 6.040))),
    4: (0.2154, 7.964, (-0.497, 3.336), (0.128, 3.336), ((-0.080, 3.487), (-0.784, 3.610), (-0.181, 3.150))),
    12: (0.2260, 8.377, (-1.027, 2.058), (0.058, 2.037), ((-0.012, 2.070), (-0.269, 2.019), (-0.062, 1.915))),
    52: (0.2510, 9.355, (-2.246, 1.337), (0.013, 1.063), ((-0.005, 1.066), (-0.076, 1.013), (-0.016, 0.998))),
    252: (0.3011, 11.303, (-4.992, 1.980), (-0.007, 0.555), ((-0.011, 0.553), (-0.028, 0.524), (-0.018, 0.517))),
}


def test_table_leland():
    k = 0.02
    costs = {52: (2.247, 2.270), 252: (4.992, 4.996)}
    for m, (sh, vh, plain, lel, drifts) in TABLE2.items():
        s_hat = H.leland_vol(0.2, k, 1.0 / m)
        assert round(s_hat, 4) == sh and round(float(bs.call(100, 100, 1, 0.01, 0.02, s_hat)), 3) == vh
        a = H.hedge_experiment(**P, m=m, n=50_000, k=k, seed=7)
        b = H.summarize(H.hedge_experiment(**P, m=m, n=50_000, k=k, sigma_hedge=s_hat, seed=7)["error"])
        sa = H.summarize(a["error"])
        assert (round(sa["mean"], 3), round(sa["sd"], 3)) == plain
        assert (round(b["mean"], 3), round(b["sd"], 3)) == lel
        if m in costs:
            assert round(a["costs"].mean(), 3) == costs[m][0]
            assert round(G * k * H.expected_cost_rate(**P) * np.sqrt(m), 3) == costs[m][1]
        for mu, (mm, ss) in zip((0.10, 0.20, -0.10), drifts):
            d = H.summarize(H.hedge_experiment(**P, m=m, n=50_000, k=k, sigma_hedge=s_hat, mu=mu, seed=7)["error"])
            assert (round(d["mean"], 3), round(d["sd"], 3)) == (mm, ss)


def test_leland_fine_grids():
    """Figure ch18-leland (b): with k = 10% the mean error keeps drifting down; with k = 2% it stays near 0."""
    ref = {(0.10, 1000): (-0.230, 0.696), (0.10, 10000): (-0.315, 0.454), (0.02, 10000): (-0.020, 0.175),
           (0.10, 100000): None}
    vals = {(0.10, 100000): (-0.365, 0.344)}
    for (k, m) in ref:
        e = H.hedge_stream(**P, m=m, n=2000, k=k, sigma_hedge=H.leland_vol(0.2, k, 1.0 / m), seed=1985 + m)["error"]
        if (k, m) in vals:   # quoted to two decimals in the notes
            assert e.mean() == pytest.approx(vals[(k, m)][0], abs=6e-4)
            assert e.std(ddof=1) == pytest.approx(vals[(k, m)][1], abs=6e-4)
            assert e.std(ddof=1) / np.sqrt(2000) < 0.008
        else:
            assert (round(e.mean(), 3), round(e.std(ddof=1), 3)) == ref[(k, m)]
