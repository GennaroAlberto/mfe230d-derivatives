"""Checks of the Chapter 2 implied-volatility solver against bs.py (closed forms and Brent's method),
and of the numbers quoted in the notes."""
import numpy as np
import pytest

from mfe230d import bs
from mfe230d import impliedvol as iv

TC = bs.TEST_CASE
S, K, T, r, y, SIG = (TC[k] for k in ("S", "K", "T", "r", "y", "sigma"))


def test_normalised_price_and_derivatives():
    k = np.linspace(70, 140, 8)
    m = np.log(S / k) + (r - y) * T
    assert np.allclose(iv.normalised_call(m, SIG * np.sqrt(T)) * k * np.exp(-r * T), bs.call(S, k, T, r, y, SIG),
                       atol=1e-12)
    h = 1e-4
    for mm in (-0.7, 0.0, 0.4):
        for w in (0.1, 0.8, 2.0):
            f = lambda x: float(iv.normalised_call(mm, x))
            e = 1e-6
            assert iv.normalised_vega(mm, w) == pytest.approx((f(w + e) - f(w - e)) / (2 * e), rel=1e-7, abs=1e-10)
            assert iv.normalised_volga(mm, w) == pytest.approx((f(w + h) - 2 * f(w) + f(w - h)) / h**2,
                                                               rel=1e-4, abs=1e-8)
    # limits and the value at the inflection point
    assert iv.normalised_call(0.3, 0.0) == pytest.approx(np.expm1(0.3))
    assert iv.normalised_call(-0.3, np.inf) == pytest.approx(np.exp(-0.3))
    for mm in (-0.01, -0.5, -2.0):
        ws = iv.inflection_point(mm)
        assert iv.normalised_call(mm, ws) == pytest.approx(0.5 * np.exp(mm) - bs.norm_cdf(-ws), abs=1e-15)


def test_inflection_point_and_concavity_of_the_log():
    for mm in (-1.2, -0.05, 0.05, 1.2):
        ws = float(iv.inflection_point(mm))
        assert iv.normalised_volga(mm, 0.9 * ws) > 0 > iv.normalised_volga(mm, 1.1 * ws)
    assert np.all(iv.normalised_volga(0.0, np.linspace(0.01, 5, 50)) < 0)
    # log c is concave for m <= 0 (second differences on a fine grid), and the stable evaluation agrees
    w = np.linspace(0.05, 4.0, 400)
    for mm in (0.0, -0.05, -0.5, -2.0):
        lc, dlc = iv.log_normalised_call(mm, w)
        c = iv.normalised_call(mm, w)
        ok = c > 1e-300
        assert np.allclose(lc[ok], np.log(c[ok]), rtol=1e-9, atol=1e-12)
        assert np.all(np.diff(lc, 2) < 0)
        assert np.allclose(dlc[ok], iv.normalised_vega(mm, w[ok]) / c[ok], rtol=1e-8)
    # ... but not for m > 0: the notes quote (log c)'' = 1.73 at m = 0.5, w = 0.3
    h = 1e-4
    f = lambda x: np.log(float(iv.normalised_call(0.5, x)))
    assert (f(0.3 + h) - 2 * f(0.3) + f(0.3 - h)) / h**2 == pytest.approx(1.73, abs=0.005)
    # no underflow far below the money
    lc, _ = iv.log_normalised_call(-3.0, 0.02)
    assert np.isfinite(lc) and lc < -10_000


def test_put_call_symmetry_in_normalised_form():
    for mm in (0.2, 1.0):
        for w in (0.1, 0.7, 2.5):
            put_hat = float(iv.normalised_call(mm, w)) - np.expm1(mm)          # parity
            assert put_hat == pytest.approx(np.exp(mm) * float(iv.normalised_call(-mm, w)), rel=1e-11)


def test_test_case():
    assert iv.implied_vol(bs.call(**TC), S, K, T, r, y, full_output=True) == (pytest.approx(0.2, abs=1e-15), 5)
    assert iv.implied_vol(bs.put(**TC), S, K, T, r, y, kind="put") == pytest.approx(0.2, abs=1e-15)
    # m = -0.01, w* = sqrt(0.02); the target lies above c(m, w*), so the start is w*
    m = np.log(bs.forward(S, T, r, y) / K)
    beta = bs.call(**TC) / (K * np.exp(-r * T))
    assert (float(m), float(beta)) == pytest.approx((-0.01, 0.0743830), abs=5e-8)
    assert float(iv.initial_guess(m, beta)) == pytest.approx(np.sqrt(0.02))
    assert 0.5 * np.exp(m) - bs.norm_cdf(-np.sqrt(0.02)) == pytest.approx(0.0512564, abs=5e-8)


def test_agrees_with_brent_on_a_grid():
    for kind, f in (("call", bs.call), ("put", bs.put)):
        for k in (60.0, 85.0, 100.0, 115.0, 160.0):
            for tt in (0.05, 0.5, 2.0):
                for s in (0.05, 0.2, 0.6, 1.5):
                    p = float(f(S, k, tt, r, y, s))
                    otm = (kind == "call") == (k >= bs.forward(S, tt, r, y))
                    if (otm and p < 1e-8) or (not otm and p - max(f(S, k, tt, r, y, 1e-9), 0) < 1e-6):
                        continue           # no volatility information left in the price
                    ours = iv.implied_vol(p, S, k, tt, r, y, kind=kind)
                    brent = bs.implied_vol(p, S, k, tt, r, y, kind=kind)
                    assert ours == pytest.approx(brent, abs=1e-9)
                    assert ours == pytest.approx(s, abs=1e-7 if otm else 1e-5)


def test_random_out_of_the_money_round_trip():
    rng = np.random.default_rng(7)
    n = 5000
    k = S * np.exp(rng.uniform(-1.5, 1.5, n))
    tt = rng.uniform(0.02, 5.0, n)
    s = rng.uniform(0.03, 1.5, n)
    call_otm = k >= bs.forward(S, tt, r, y)
    p = np.where(call_otm, bs.call(S, k, tt, r, y, s), bs.put(S, k, tt, r, y, s))
    keep = p > 1e-10 * k
    sig_c = iv.implied_vol(p[keep & call_otm], S, k[keep & call_otm], tt[keep & call_otm], r, y)
    sig_p = iv.implied_vol(p[keep & ~call_otm], S, k[keep & ~call_otm], tt[keep & ~call_otm], r, y, kind="put")
    assert np.max(np.abs(sig_c / s[keep & call_otm] - 1)) < 1e-9
    assert np.max(np.abs(sig_p / s[keep & ~call_otm] - 1)) < 1e-9


def test_edges():
    lo = max(S * np.exp(-y * T) - 80 * np.exp(-r * T), 0.0)
    assert iv.implied_vol(lo, S, 80.0, T, r, y) == 0.0
    assert iv.implied_vol(S * np.exp(-y * T), S, 120.0, T, r, y) == np.inf
    assert iv.implied_vol(K * np.exp(-r * T), S, K, T, r, y, kind="put") == np.inf
    assert np.isnan(iv.implied_vol(lo - 0.01, S, 80.0, T, r, y))                 # below the bound: arbitrage
    assert np.isnan(iv.implied_vol(S, S, 120.0, T, r, y))                       # above the bound
    assert np.isnan(iv.implied_vol(5.0, S, 0.0, T, r, y))                       # K = 0
    assert np.isnan(iv.implied_vol(5.0, S, K, 0.0, r, y))                       # T = 0
    with pytest.raises(ValueError):
        iv.implied_total_vol(0.1, 0.5)


def test_newton_on_the_price_from_the_inflection_point_is_monotone():
    for mm, w_true in ((-0.5, 0.4), (-0.5, 2.0), (-2.0, 0.3), (-0.02, 0.9)):
        beta = float(iv.normalised_call(mm, w_true))
        path = iv.newton_iterates(mm, beta, iv.inflection_point(mm), 60)
        steps = np.diff(path)
        assert np.all(np.sign(steps[np.abs(steps) > 1e-14]) == np.sign(w_true - path[0]))
        assert path[-1] == pytest.approx(w_true, rel=1e-12)


def test_newton_on_the_price_can_fail():
    beta = float(iv.normalised_call(0.0, 0.2))
    assert iv.newton_iterates(0.0, beta, 2.0, 1)[1] == pytest.approx(-0.492177, abs=5e-7)
    beta = float(iv.normalised_call(-0.5, 0.4))
    assert beta == pytest.approx(0.0155453, abs=5e-8)
    assert iv.newton_iterates(-0.5, beta, 3.0, 1)[1] == pytest.approx(-1.904369, abs=5e-7)
    w, n = iv.newton_price(-0.5, beta, 3.0)
    assert np.isnan(w)
    # the log iteration from the same bad starts converges
    assert iv.implied_total_vol(0.0, float(iv.normalised_call(0.0, 0.2)), w0=2.0)[0] == pytest.approx(0.2, rel=1e-13)
    assert iv.implied_total_vol(-0.5, beta, w0=3.0)[0] == pytest.approx(0.4, rel=1e-13)


def test_log_newton_converges_from_any_start():
    for mm, w_true in ((-0.5, 0.4), (-0.5, 2.5), (0.0, 0.05), (-1.5, 0.1), (-0.001, 1.0)):
        beta = float(iv.normalised_call(mm, w_true))
        for w0 in (1e-3, 0.1, 1.0, 10.0, 100.0):
            w, n = iv.implied_total_vol(mm, beta, w0=w0)
            assert w == pytest.approx(w_true, rel=1e-12) and n < 60


def test_iteration_study_quoted_in_the_notes():
    res = iv.iteration_study()                      # seed 2026, 20,000 draws
    assert len(res["m"]) == 12454
    assert np.max(np.abs(res["w_log"] / res["w"] - 1)) < 1e-12
    assert np.max(np.abs(res["w_mk"] / res["w"] - 1)) < 1e-12
    assert res["it_log"].mean() == pytest.approx(6.31, abs=0.005)
    assert res["it_log"].max() == 17 and np.percentile(res["it_log"], 99) == 9
    assert res["it_mk"].mean() == pytest.approx(9.30, abs=0.005)
    assert res["it_mk"].max() == 34 and np.percentile(res["it_mk"], 99) == 31
