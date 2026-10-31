"""Tests for mfe230d.multiasset (Chapter 17): closed forms against each other and against
simulation, degenerate cases, and every number quoted in the notes."""
import numpy as np
import pytest

from mfe230d import bs
from mfe230d import multiasset as M

FX = M.FX
P = M.BASKET
LAW = (P["S"], P["w"], P["T"], P["r"], P["y"], P["sigma"], P["corr"])
REF = 7.4655807138


def basket_args(K=100.0, corr=None):
    return (P["S"], P["w"], K, P["T"], P["r"], P["y"], P["sigma"], P["corr"] if corr is None else corr)


# --------------------------------------------------------------------------- Black's function
def test_black_matches_bs_and_edge_cases():
    assert M.black(105.0, 100.0, 0.04, np.exp(-0.01)) == pytest.approx(
        float(bs.call(100.0, 100.0, 1.0, 0.01, 0.01 - np.log(1.05), 0.2)), rel=1e-12)
    assert M.black(105.0, 100.0, 0.0) == pytest.approx(5.0)
    assert M.black(95.0, 100.0, 0.0) == 0.0
    assert M.black(95.0, -3.0, 0.3, 0.9) == pytest.approx(0.9 * 98.0)
    assert M.black(95.0, 100.0, 0.04, kind="put") - M.black(95.0, 100.0, 0.04) == pytest.approx(5.0)
    # homogeneity of degree one in (F, K)
    assert M.black(3 * 95.0, 3 * 100.0, 0.04) == pytest.approx(3 * M.black(95.0, 100.0, 0.04), rel=1e-13)


# -------------------------------------------------------------------------- Garman--Kohlhagen
def test_course_cable_call():
    X, rd, rf, s = FX["X_gbp"], FX["r_usd"], FX["r_gbp"], FX["vol_gbp"]
    F = M.fx_forward(X, 1.0, rd, rf)
    C = float(M.garman_kohlhagen(X, F, 1.0, rd, rf, s))
    h = M.gk_hedge(X, F, 1.0, rd, rf, s)
    assert F == pytest.approx(1.206015, abs=5e-7)
    assert C == pytest.approx(0.0513343, abs=5e-8)
    assert C / X == pytest.approx(0.0427786, abs=5e-8)
    assert C / F == pytest.approx(0.042565, abs=5e-7)
    assert h["foreign"] == pytest.approx(0.509044, abs=5e-7)
    assert h["domestic"] == pytest.approx(-0.559519, abs=5e-7)
    assert X * h["foreign"] + h["domestic"] == pytest.approx(C, abs=1e-14)
    # ATMF: call = put = e^{-r_d T} F (2 N(sigma sqrt(T) / 2) - 1)
    atmf = np.exp(-rd) * F * (2 * bs.norm_cdf(0.5 * s) - 1)
    assert C == pytest.approx(atmf, rel=1e-12)
    assert float(M.garman_kohlhagen(X, F, 1.0, rd, rf, s, "put")) == pytest.approx(C, rel=1e-12)


def test_gk_put_hedge_and_finite_differences():
    X, K, T, rd, rf, s = 1.1, 1.15, 0.7, 0.04, 0.01, 0.13
    for kind in ("call", "put"):
        h = M.gk_hedge(X, K, T, rd, rf, s, kind)
        V = float(M.garman_kohlhagen(X, K, T, rd, rf, s, kind))
        e = 1e-6
        dV = (float(M.garman_kohlhagen(X + e, K, T, rd, rf, s, kind))
              - float(M.garman_kohlhagen(X - e, K, T, rd, rf, s, kind))) / (2 * e)
        assert h["foreign"] == pytest.approx(dV, rel=1e-6)
        assert X * h["foreign"] + h["domestic"] == pytest.approx(V, abs=1e-13)


# ------------------------------------------------------------------------ change of numeraire
def test_currency_symmetry_and_put_call_symmetry_of_chapter4():
    for X, K, T, rd, rf, s in [(1.2, 1.206015025, 1.0, 0.03, 0.025, 0.11), (0.8, 0.95, 2.0, 0.01, 0.04, 0.3),
                               (1.0, 1.3, 0.5, 0.02, 0.02, 0.2)]:
        c, dual = M.currency_symmetry(X, K, T, rd, rf, s)
        assert float(c) == pytest.approx(float(dual), rel=1e-12)
        F = M.fx_forward(X, T, rd, rf)
        assert float(c) == pytest.approx(K / F * float(bs.put(X, F**2 / K, T, rd, rf, s)), rel=1e-12)


def test_london_trader_numbers():
    X, rd, rf, s = FX["X_gbp"], FX["r_usd"], FX["r_gbp"], FX["vol_gbp"]
    K = M.fx_forward(X, 1.0, rd, rf)
    put = float(bs.put(1 / X, 1 / K, 1.0, rf, rd, s))
    delta = float(bs.greeks(1 / X, 1 / K, 1.0, rf, rd, s, "put")["delta"])
    assert 1 / X == pytest.approx(0.833333, abs=5e-7)
    assert put == pytest.approx(0.0354710, abs=5e-8)
    assert K * put == pytest.approx(0.0427786, abs=5e-8)
    assert delta == pytest.approx(-0.463940, abs=5e-7)
    assert K * delta == pytest.approx(-0.559519, abs=5e-7)
    assert K * (put - delta / X) == pytest.approx(0.509044, abs=5e-7)


def test_numeraire_drift_and_share_measure():
    assert M.numeraire_drift(0.01, [0.2, 0.1], [0.3, -0.5]) == pytest.approx(0.01 + 0.06 - 0.05)
    tc = bs.TEST_CASE
    S, K, T, r, y, s = (tc[k] for k in ("S", "K", "T", "r", "y", "sigma"))
    z = np.random.default_rng(3).standard_normal(2**20)
    q = np.exp(-r * T) * np.maximum(S * np.exp((r - y - s * s / 2) * T + s * np.sqrt(T) * z) - K, 0)
    qs = M.call_samples_share_measure(z, S, K, T, r, y, s)
    exact = float(bs.call(S, K, T, r, y, s))
    assert abs(qs.mean() - exact) < 3 * qs.std() / 1024
    assert q.std(ddof=1) == pytest.approx(12.575, abs=5e-4)
    assert qs.std(ddof=1) == pytest.approx(9.953, abs=5e-4)
    assert q.var() / qs.var() == pytest.approx(1.596, abs=5e-4)
    # density of the share measure: S_T / E[S_T] reweights Q-samples to Q^S
    ST = S * np.exp((r - y - s * s / 2) * T + s * np.sqrt(T) * z)
    LR = ST / (S * np.exp((r - y) * T))
    assert np.mean(LR * np.log(ST)) == pytest.approx(np.log(S) + (r - y + s * s / 2) * T, abs=3e-3)


# ---------------------------------------------------------------------------- exchange options
def test_margrabe_is_prewashed_black_scholes_and_parity():
    S1, S2, T, y1, y2, s1, s2, rho = 105.0, 95.0, 1.5, 0.03, 0.01, 0.25, 0.15, 0.4
    v = float(M.exchange_vol(s1, s2, rho))
    m = M.margrabe(S1, S2, T, y1, y2, s1, s2, rho)
    assert m == pytest.approx(float(bs.call(S1, S2, T, y2, y1, v)), rel=1e-13)
    assert m == pytest.approx(M.black(S1 * np.exp(-y1 * T), S2 * np.exp(-y2 * T), v * v * T), rel=1e-13)
    swap = M.margrabe(S2, S1, T, y2, y1, s2, s1, rho)
    assert m - swap == pytest.approx(S1 * np.exp(-y1 * T) - S2 * np.exp(-y2 * T), rel=1e-12)
    h = M.margrabe_hedge(S1, S2, T, y1, y2, s1, s2, rho)
    assert S1 * h["asset1"] + S2 * h["asset2"] == pytest.approx(m, rel=1e-12)
    e = 1e-5
    assert h["asset1"] == pytest.approx((M.margrabe(S1 + e, S2, T, y1, y2, s1, s2, rho)
                                         - M.margrabe(S1 - e, S2, T, y1, y2, s1, s2, rho)) / (2 * e), rel=1e-6)
    assert h["asset2"] == pytest.approx((M.margrabe(S1, S2 + e, T, y1, y2, s1, s2, rho)
                                         - M.margrabe(S1, S2 - e, T, y1, y2, s1, s2, rho)) / (2 * e), rel=1e-6)


def test_margrabe_degenerate_and_monotone():
    assert M.exchange_vol(0.2, 0.2, 1.0) == 0.0
    assert M.exchange_vol(0.3, 0.1, 1.0) == pytest.approx(0.2)
    assert M.exchange_vol(0.3, 0.1, -1.0) == pytest.approx(0.4)
    assert M.margrabe(100, 90, 1.0, 0.02, 0.05, 0.2, 0.2, 1.0) == pytest.approx(
        100 * np.exp(-0.02) - 90 * np.exp(-0.05))
    assert M.margrabe(90, 100, 1.0, 0.02, 0.02, 0.2, 0.2, 1.0) == 0.0
    vals = [M.margrabe(100, 100, 1.0, 0.02, 0.02, 0.2, 0.3, r) for r in np.linspace(-1, 1, 21)]
    assert np.all(np.diff(vals) < 0)
    # continuity as sigma_ex -> 0
    assert M.margrabe(100, 90, 1.0, 0.02, 0.05, 0.2, 0.2, 1 - 1e-12) == pytest.approx(
        M.margrabe(100, 90, 1.0, 0.02, 0.05, 0.2, 0.2, 1.0), abs=1e-6)


@pytest.mark.parametrize("r", [0.0, 0.05])
def test_margrabe_by_simulation_does_not_depend_on_r(r):
    S1, S2, T, y1, y2, s1, s2, rho = 100.0, 100.0, 1.0, 0.02, 0.01, 0.2, 0.3, 0.5
    ST = M.terminal_prices([S1, S2], T, r, [y1, y2], [s1, s2], [[1, rho], [rho, 1]], n=2**18, seed=21)
    pay = np.exp(-r * T) * np.maximum(ST[:, 0] - ST[:, 1], 0.0)
    m = M.margrabe(S1, S2, T, y1, y2, s1, s2, rho)
    assert abs(pay.mean() - m) < 4 * pay.std() / np.sqrt(pay.size)


def test_cross_currency_example():
    value, F12, s12 = M.cross_atmf_call(s12=0.08)
    assert F12 == pytest.approx(1.158955, abs=5e-7)
    assert FX["X_gbp"] / FX["X_eur"] == pytest.approx(1.176471, abs=5e-7)
    assert M.cross_vol_range(0.11, 0.10) == pytest.approx((0.01, 0.21))
    assert M.cross_atmf_call(s12=0.01)[0] == pytest.approx(0.004669, abs=5e-7)
    assert M.cross_atmf_call(s12=0.21)[0] == pytest.approx(0.097871, abs=5e-7)
    assert value == pytest.approx(0.037343, abs=5e-7)
    rho = M.implied_correlation(0.11, 0.10, 0.08)
    assert rho == pytest.approx(0.7136, abs=5e-5)
    assert float(M.exchange_vol(0.11, 0.10, rho)) == pytest.approx(0.08, rel=1e-12)
    # law of one price: USD value / X_EUR = Garman--Kohlhagen price of the EUR-based investor
    gk_eur = float(M.garman_kohlhagen(FX["X_gbp"] / FX["X_eur"], F12, 1.0, FX["r_eur"], FX["r_gbp"], 0.08))
    assert value / FX["X_eur"] == pytest.approx(gk_eur, rel=1e-12)


# ------------------------------------------------------------------------ correlated sampling
def test_terminal_prices_law_and_singular_correlation():
    S, y, s = [100.0, 50.0, 80.0], [0.02, 0.0, 0.01], [0.2, 0.3, 0.25]
    ST = M.terminal_prices(S, 1.0, 0.03, y, s, M.CORR3, n=2**17, seed=4)
    L = np.log(ST)
    Sig = M.log_covariance(s, M.CORR3, 1.0)
    assert np.allclose(np.cov(L.T), Sig, atol=2.5e-3)
    F = np.array(S) * np.exp((0.03 - np.array(y)))
    assert np.allclose(ST.mean(0) / F, 1.0, atol=6e-3)
    with pytest.raises(ValueError):
        M.terminal_prices([1, 1], 1.0, 0.0, [0, 0], [0.1, 0.1], [[1, 1], [1, 1]], n=4, seed=0)
    ST1 = M.terminal_prices([1, 1], 1.0, 0.0, [0, 0], [0.1, 0.2], [[1, 1], [1, 1]], n=8, seed=0, method="pca")
    assert np.allclose(np.log(ST1[:, 1]) + 0.02, 2 * (np.log(ST1[:, 0]) + 0.005))


# ------------------------------------------------------------------------------------ quantos
def test_quanto_numbers_and_rows():
    S, K, T, rd, rf, y, sS, sX, rho = 100.0, 100.0, 1.0, 0.03, 0.025, 0.02, 0.2, 0.11, -0.3
    assert M.quanto_yield(y, rd, rf, sS, sX, rho) == pytest.approx(0.0184)
    assert M.quanto_forward(S, T, rf, y, sS, sX, rho) == pytest.approx(101.1668, abs=5e-5)
    assert S * np.exp((rf - y) * T) == pytest.approx(100.5013, abs=5e-5)
    q = float(M.quanto_option(S, K, T, rd, rf, y, sS, sX, rho))
    assert q == pytest.approx(8.35441, abs=5e-6)
    assert q == pytest.approx(np.exp(-rd * T) * M.black(M.quanto_forward(S, T, rf, y, sS, sX, rho), K, sS**2 * T), rel=1e-12)
    assert float(M.quanto_option(S, K, T, rd, rf, y, sS, sX, 0.0)) == pytest.approx(7.99516, abs=5e-6)
    assert float(M.quanto_option(S, K, T, rd, rf, y, sS, sX, 0.3)) == pytest.approx(7.64664, abs=5e-6)
    assert float(bs.call(S, K, T, rf, y, sS)) == pytest.approx(8.03523, abs=5e-6)
    assert float(M.foreign_option(S, 1.2, K, T, rf, y, sS)) == pytest.approx(9.64228, abs=5e-6)
    assert float(M.composite_option(S, 1.2, 120.0, T, rd, y, sS, sX, rho)) == pytest.approx(9.79110, abs=5e-6)
    assert np.sqrt(sS**2 + 2 * rho * sS * sX + sX**2) == pytest.approx(0.1972, abs=5e-5)
    # quanto put-call parity
    qp = float(M.quanto_option(S, K, T, rd, rf, y, sS, sX, rho, kind="put"))
    assert q - qp == pytest.approx(np.exp(-rd * T) * (M.quanto_forward(S, T, rf, y, sS, sX, rho) - K), rel=1e-12)


def test_quanto_composite_foreign_by_simulation():
    S, K, T, rd, rf, y, sS, sX, rho = 100.0, 100.0, 1.0, 0.03, 0.025, 0.02, 0.2, 0.11, -0.3
    Sf, Xt = M.fx_equity_terminal(S, 1.2, T, rd, rf, y, sS, sX, rho, n=2**20, seed=2026)
    D = np.exp(-rd * T)
    cases = [(D * np.maximum(Sf - K, 0), float(M.quanto_option(S, K, T, rd, rf, y, sS, sX, rho)), 8.3457),
             (D * Xt * np.maximum(Sf - K, 0), float(M.foreign_option(S, 1.2, K, T, rf, y, sS)), 9.6313),
             (D * np.maximum(Xt * Sf - 120, 0), float(M.composite_option(S, 1.2, 120.0, T, rd, y, sS, sX, rho)), 9.7784)]
    for v, exact, quoted in cases:
        se = v.std(ddof=1) / np.sqrt(v.size)
        assert v.mean() == pytest.approx(quoted, abs=5e-5)
        assert abs(v.mean() - exact) < se < 0.016
    # the domestic measure: foreign money market and foreign share (dividends reinvested) earn r_d
    assert np.mean(D * Xt * np.exp(rf * T)) == pytest.approx(1.2, rel=2e-3)
    assert np.mean(D * Xt * Sf * np.exp(y * T)) == pytest.approx(120.0, rel=3e-3)


# ------------------------------------------------------------------------------------ baskets
def test_geometric_basket_law_and_pathwise_bound():
    g = M.geometric_basket_law(*LAW)
    assert g["Phi"] == pytest.approx(99.0050, abs=5e-5)
    assert np.sqrt(g["var"]) == pytest.approx(0.2023, abs=5e-5)
    assert g["EG"] < g["Phi"]
    s = M.basket_samples(*basket_args(), n=2**16, seed=1)
    assert np.all(s["G"] <= s["B"] * (1 + 1e-14))
    assert np.var(np.log(s["G"])) == pytest.approx(g["var"], rel=0.02)
    assert np.mean(np.log(s["G"])) == pytest.approx(g["mean"], abs=2e-3)
    se = s["geometric"].std() / np.sqrt(s["geometric"].size)
    assert abs(s["geometric"].mean() - M.geometric_basket_option(*basket_args())) < 4 * se


def test_basket_moments_and_lognormal_approximation():
    m1, m2 = M.basket_moments(*LAW)
    assert np.sqrt(np.log(m2 / m1**2)) == pytest.approx(0.2030, abs=5e-5)
    s = M.basket_samples(*basket_args(), n=2**18, seed=2)
    assert s["B"].mean() == pytest.approx(m1, rel=2e-3)
    assert np.mean(s["B"] ** 2) == pytest.approx(m2, rel=5e-3)
    # exact cases: one asset; perfectly correlated equal vols; K <= 0
    one = (np.array([100.0]), np.array([1.0]), 95.0, 1.0, 0.01, np.array([0.02]), np.array([0.2]), np.eye(1))
    assert M.basket_lognormal(*one) == pytest.approx(float(bs.call(100, 95, 1, 0.01, 0.02, 0.2)), rel=1e-12)
    assert M.basket_reference(*one) == pytest.approx(float(bs.call(100, 95, 1, 0.01, 0.02, 0.2)), rel=1e-12)
    ones = np.ones((3, 3))
    args1 = (np.array([100.0, 80.0, 120.0]), np.array([0.5, 0.2, 0.3]), 100.0, 1.0, 0.01,
             np.array([0.02, 0.0, 0.03]), np.full(3, 0.25), ones)
    F = args1[0] * np.exp((0.01 - args1[5]))
    exact = np.exp(-0.01) * M.black(args1[1] @ F, 100.0, 0.0625)
    assert M.basket_lognormal(*args1) == pytest.approx(exact, rel=1e-12)
    assert M.geometric_basket_option(*args1) == pytest.approx(exact, rel=1e-10)
    assert M.basket_reference(*args1) == pytest.approx(exact, rel=1e-10)
    for K in (0.0, -5.0):
        a = basket_args(K)
        assert M.basket_lognormal(*a) == pytest.approx(np.exp(-0.01) * (m1 - K), rel=1e-12)
        assert M.basket_reference(*a) == pytest.approx(np.exp(-0.01) * (m1 - K), rel=1e-10)


def test_mixed_signs_are_refused():
    spread = (np.array([100.0, 100.0]), np.array([1.0, -1.0]), 5.0, 1.0, 0.01, np.array([0.02, 0.02]),
              np.array([0.2, 0.25]), np.array([[1, 0.6], [0.6, 1]]))
    with pytest.raises(ValueError):
        M.basket_lognormal(*spread)
    with pytest.raises(ValueError):
        M.geometric_basket_law(*spread[:2], *spread[3:])
    s = M.basket_samples(*spread, n=2**12, seed=3)
    assert np.mean(s["B"] < 0) > 0.3 and "geometric" not in s


def test_basket_reference_and_quoted_numbers():
    assert M.basket_reference(*basket_args()) == pytest.approx(REF, abs=1e-9)
    assert abs(M.basket_reference(*basket_args(), nodes=8) - REF) < 1e-10
    assert abs(M.basket_reference(*basket_args(), nodes=16) - M.basket_reference(*basket_args(), nodes=64)) < 1e-13
    a = basket_args()
    assert M.basket_lognormal(*a) == pytest.approx(7.48004, abs=5e-6)
    assert M.gentle_call(*a) == pytest.approx(7.36489, abs=5e-6)
    assert M.geometric_basket_option(*a) == pytest.approx(6.87976, abs=5e-6)
    assert M.curran_lower_bound(*a) == pytest.approx(7.45406, abs=5e-6)
    assert M.basket_cv_bound(*LAW) == pytest.approx(2.8878, abs=5e-5)
    table = {80.0: (20.132481, 319.2, -325.6, -76.8, -9647.6), 90.0: (12.804691, 319.7, -700.3, -105.0, -7881.5),
             100.0: (7.465581, 144.6, -1006.9, -115.2, -5858.2), 110.0: (4.023289, -68.7, -1085.2, -110.5, -3991.3),
             120.0: (2.027888, -199.0, -945.9, -95.5, -2517.6)}
    for K, (ref, ln, ge, cu, geo) in table.items():
        b = basket_args(K)
        c = M.basket_reference(*b)
        assert c == pytest.approx(ref, abs=5e-7)
        for f, val in ((M.basket_lognormal, ln), (M.gentle_call, ge), (M.curran_lower_bound, cu),
                       (M.geometric_basket_option, geo)):
            assert 1e4 * (f(*b) - c) == pytest.approx(val, abs=0.05)
        # bounds are bounds
        assert M.geometric_basket_option(*b) <= M.curran_lower_bound(*b) <= c


def test_basket_monte_carlo_numbers():
    a = basket_args()
    plain = M.basket_mc(*a, n=2**20, seed=2026)
    geo = M.basket_mc(*a, n=2**20, seed=2026, controls=("geometric",))
    three = M.basket_mc(*a, n=2**20, seed=2026, controls=("geometric", "forward", "strip"))
    assert (plain["mean"], plain["se"]) == pytest.approx((7.46603, 0.01256), abs=5e-6)
    assert (geo["mean"], geo["se"]) == pytest.approx((7.46541, 0.00101), abs=5e-6)
    assert geo["beta"] == pytest.approx(1.0498, abs=5e-5) and geo["rho"] == pytest.approx(0.99679, abs=5e-6)
    assert (three["mean"], three["se"]) == pytest.approx((7.46585, 0.00071), abs=5e-6)
    assert three["R2"] == pytest.approx(0.99678, abs=5e-6)
    for o in (plain, geo, three):
        assert abs(o["mean"] - REF) < 2 * o["se"]
    s = M.basket_samples(*a, n=2**20, seed=2026)
    vY = s["Y"].var(ddof=1)
    assert vY == pytest.approx(165.5, abs=0.05)
    assert M.basket_cv_bound(*LAW) / vY == pytest.approx(0.0175, abs=5e-4)
    assert (s["Y"] - s["geometric"]).var() <= M.basket_cv_bound(*LAW)
    # the bound formula is E[(D(A - G))^2], checked by simulation
    assert np.mean((np.exp(-0.01) * (s["B"] - s["G"])) ** 2) == pytest.approx(M.basket_cv_bound(*LAW), rel=0.01)


def test_basket_figure_correlations():
    Z = np.random.default_rng(17).standard_normal((2**16, 3))
    one_minus = [1 - M.basket_mc(*basket_args(K), Z=Z, controls=("geometric",))["rho"] ** 2
                 for K in np.arange(80.0, 120.01, 2.5)]
    assert min(one_minus) == pytest.approx(0.0042, abs=1e-4)
    assert max(one_minus) == pytest.approx(0.0117, abs=1e-4)


def test_two_asset_reference_against_simulation():
    a = (np.array([100.0, 60.0]), np.array([0.4, 0.6]), 70.0, 2.0, 0.02, np.array([0.01, 0.03]),
         np.array([0.3, 0.15]), np.array([[1, 0.2], [0.2, 1]]))
    ref = M.basket_reference(*a)
    o = M.basket_mc(*a, n=2**18, seed=9, controls=("geometric",))
    assert abs(o["mean"] - ref) < 4 * o["se"]
