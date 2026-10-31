"""Checks of the Chapter 3 code (arbitrage in the strike and the smile) against the notes."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from mfe230d import bs
from mfe230d import smile as sm

TC = bs.TEST_CASE
S, T, r, y = TC["S"], TC["T"], TC["r"], TC["y"]
# the course's "problem set parameters"
CS = dict(S=3900.0, T=1.0, r=0.03, y=0.013)


def ssvi_call(K, T=T):
    return sm.smile_call(K, S, T, r, y, sm.ssvi_smile(K, S, T, r, y)[0])


def ssvi_density(K, T=T):
    return sm.smile_density(K, S, T, r, y, *sm.ssvi_smile(K, S, T, r, y))


def course_call(K):
    return sm.smile_call(K, CS["S"], CS["T"], CS["r"], CS["y"], sm.course_smile(K, CS["S"])[0])


# ---------------------------------------------------------------------------------------
# Strike conditions
# ---------------------------------------------------------------------------------------

def test_lognormal_quotes_pass_and_bounds_hold():
    K = np.linspace(5.0, 300.0, 60)
    C = bs.call(S, K, T, r, y, 0.2)
    chk = sm.strike_checks(K, C, S, T, r, y)
    assert chk["ok"]
    assert np.all(C <= S * np.exp(-y * T)) and np.all(C >= np.maximum(S * np.exp(-y * T) - K * np.exp(-r * T), 0))
    assert np.all(chk["slopes"] >= -np.exp(-r * T) - 1e-12) and np.all(chk["slopes"] <= 0)


def test_checks_catch_each_violation():
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    C = bs.call(S, K, T, r, y, 0.2)
    bad = C.copy(); bad[2] += 1.2           # middle call too dear: a butterfly arbitrage at K = 100
    chk = sm.strike_checks(K, bad, S, T, r, y)
    assert not chk["ok"] and chk["slope_low"] and chk["monotone"]
    assert list(chk["convex"]) == [True, True, False, True]   # flies at 80, 90, 100, 110
    bad = C.copy(); bad[0] = S * np.exp(-y * T) - 80.0 * np.exp(-r * T) - 0.1   # below the lower bound
    assert not sm.strike_checks(K, bad, S, T, r, y)["slope_low"]
    bad = C.copy(); bad[-1] = bad[-2] + 0.01   # increasing in the strike
    assert not sm.strike_checks(K, bad, S, T, r, y)["monotone"]


def test_butterfly_weights_are_butterfly_prices():
    # equal spacing h: s_{i+1} - s_i is the butterfly price divided by h
    K = np.array([90.0, 100.0, 110.0])
    C = bs.call(S, K, T, r, y, 0.2)
    fly = sm.strike_checks(K, C, S, T, r, y)["butterflies"][1]
    assert fly == pytest.approx((C[0] - 2 * C[1] + C[2]) / 10.0, rel=1e-12)


# ---------------------------------------------------------------------------------------
# Breeden--Litzenberger
# ---------------------------------------------------------------------------------------

def test_second_differences_recover_lognormal_density():
    K = np.array([60.0, 100.0, 150.0])
    q = bs.norm_pdf(bs.d_pm(S, K, T, r, y, 0.2)[1]) / (K * 0.2 * np.sqrt(T))
    est = sm.fd_density(lambda k: bs.call(S, k, T, r, y, 0.2), K, 1e-2, r, T)
    assert est == pytest.approx(q, rel=1e-5)


def test_atom_kink_and_second_difference_spike():
    p, a, sig = 0.2, 120.0, 0.2
    f = lambda k: sm.atom_call(k, S, T, r, y, sig, p, a)
    # the call with strike 0 is the prepaid forward, and the slopes jump by e^{-rT} p at the atom
    assert f(1e-12) == pytest.approx(S * np.exp(-y * T), rel=1e-12)
    h = 1e-6
    right, left = (f(a + h) - f(a)) / h, (f(a) - f(a - h)) / h
    assert right - left == pytest.approx(np.exp(-r * T) * p, rel=1e-4)
    assert -np.exp(r * T) * right == pytest.approx(sm.atom_tail(a, S, T, r, y, sig, p, a), abs=1e-5)
    # h times the second difference tends to e^{-rT} Q(S_T = a)
    for hh in (1.0, 0.1, 0.01):
        assert hh * sm.second_difference(f, a, hh) == pytest.approx(np.exp(-r * T) * p, abs=0.02 * hh)


def test_default_atom_slope_at_zero():
    # S_T = 0 with probability p: the right slope at K = 0 is -e^{-rT} (1 - p), not -e^{-rT}
    p = 0.1
    f = lambda k: sm.atom_call(k, S, T, r, y, 0.2, p, 0.0)
    h = 1e-6
    assert (f(h) - f(1e-300)) / h == pytest.approx(-np.exp(-r * T) * (1 - p), rel=1e-6)


def test_fd_error_bound_and_optimal_step():
    M4, eps = 2.763e-4, 0.005
    h = sm.fd_optimal_step(M4, eps)
    assert h == pytest.approx(5.4289, abs=1e-3)
    assert sm.fd_density_error_bound(h, M4, eps) == pytest.approx(2 * np.sqrt(M4 * eps / 3), rel=1e-12)
    # the bound holds for cent-rounded SSVI quotes (numbers quoted in the notes)
    rounded = lambda k: np.round(ssvi_call(k), 2)
    errs = {}
    for hh in (1.0, 5.0):
        Kc = np.arange(60.0, 140.0 + 1e-9, hh)
        errs[hh] = np.abs(sm.second_difference(rounded, Kc, hh) - ssvi_density(Kc)).max()
        assert errs[hh] <= sm.fd_density_error_bound(hh, M4, eps)
    assert errs[1.0] == pytest.approx(0.0169, abs=5e-5)
    assert errs[5.0] == pytest.approx(0.00101, abs=1e-5)


def test_fourth_derivative_constant_quoted():
    Kg = np.linspace(50.0, 150.0, 2001)
    d4 = sm.second_difference(ssvi_density, Kg, 0.05)
    assert np.abs(d4).max() == pytest.approx(2.763e-4, rel=2e-3)


# ---------------------------------------------------------------------------------------
# The smile: digitals and densities
# ---------------------------------------------------------------------------------------

def test_flat_smile_reduces_to_black_scholes():
    K = np.array([80.0, 100.0, 130.0])
    z = np.zeros_like(K)
    d = sm.smile_digital(K, S, T, r, y, 0.2, z)
    assert d["price"] == pytest.approx(bs.digital_call(S, K, T, r, y, 0.2), rel=1e-12)
    q = sm.smile_density(K, S, T, r, y, 0.2, z, z)
    assert q == pytest.approx(bs.greeks(S, K, T, r, y, 0.2)["gamma"] * S**2 / K**2, rel=1e-12)
    assert sm.smile_asset_digital(K, S, T, r, y, 0.2, z) == pytest.approx(
        S * np.exp(-y * T) * bs.norm_cdf(bs.d_pm(S, K, T, r, y, 0.2)[0]), rel=1e-12)


@pytest.mark.parametrize("which", ["ssvi", "course"])
def test_skew_and_density_formulas_match_finite_differences(which):
    if which == "ssvi":
        Ss, Tt, rr, yy, K = S, T, r, y, np.array([70.0, 100.0, 125.0])
        vol = lambda k: sm.ssvi_smile(k, Ss, Tt, rr, yy)
    else:
        Ss, Tt, rr, yy, K = CS["S"], CS["T"], CS["r"], CS["y"], np.array([3000.0, 3900.0, 4800.0])
        vol = lambda k: sm.course_smile(k, Ss)
    f = lambda k: sm.smile_call(k, Ss, Tt, rr, yy, vol(k)[0])
    sig, ds, d2 = vol(K)
    h = 2e-5 * K
    assert sm.smile_digital(K, Ss, Tt, rr, yy, sig, ds)["price"] == pytest.approx(-(f(K + h) - f(K - h)) / (2 * h), rel=1e-7)
    h = 2e-3 * K
    assert sm.smile_density(K, Ss, Tt, rr, yy, sig, ds, d2) == pytest.approx(sm.second_difference(f, K, h), rel=1e-3)
    # sigma' and sigma'' themselves
    hs = 1e-4 * K
    assert ds == pytest.approx((vol(K + hs)[0] - vol(K - hs)[0]) / (2 * hs), rel=1e-6)
    assert d2 == pytest.approx((vol(K + 20 * hs)[0] - 2 * sig + vol(K - 20 * hs)[0]) / (20 * hs) ** 2, rel=1e-4)


def test_course_atm_digital_midterm():
    # the course's midterm number: e^{-rT} N(d_-) + 0.28 e^{-yT} N'(d_+) sqrt(T) = 0.571394
    sig, ds, _ = sm.course_smile(CS["S"], CS["S"])
    d = sm.smile_digital(CS["S"], CS["S"], CS["T"], CS["r"], CS["y"], sig, ds)
    assert d["bs"] == pytest.approx(0.463167, abs=1e-6)
    assert d["skew"] == pytest.approx(0.108227, abs=1e-6)
    assert d["price"] == pytest.approx(0.571394, abs=1e-6)
    dp = bs.d_pm(CS["S"], CS["S"], CS["T"], CS["r"], CS["y"], 0.25)[0]
    assert d["skew"] == pytest.approx(0.28 * np.exp(-CS["y"]) * bs.norm_pdf(dp), rel=1e-12)
    a = sm.smile_asset_digital(CS["S"], CS["S"], CS["T"], CS["r"], CS["y"], sig, ds)
    assert a == pytest.approx(2641.47, abs=0.01)
    assert course_call(CS["S"]) == pytest.approx(413.04, abs=0.01)
    assert course_call(CS["S"]) + CS["S"] * d["bs"] == pytest.approx(2219.39, abs=0.01)


def test_course_smile_density_turns_negative_far_out():
    br = lambda k: sm.smile_bracket(k, CS["S"], CS["T"], CS["r"], CS["y"], *sm.course_smile(k, CS["S"]))
    k0 = brentq(br, 50.0, 200.0)
    assert k0 == pytest.approx(84.33, abs=0.01)
    assert sm.second_difference(course_call, 60.0, 2.0) < 0 < sm.second_difference(course_call, 100.0, 2.0)
    # positive on the rest of the range where sigma > 0 (sigma vanishes at S e^{0.25/0.28})
    Kg = np.linspace(85.0, 9500.0, 20000)
    assert np.all(br(Kg) > 0)
    assert CS["S"] * np.exp(0.25 / 0.28) == pytest.approx(9524.18, abs=0.01)


def test_ssvi_numbers_quoted_in_the_notes():
    sig, ds, d2 = sm.ssvi_smile(100.0, S, T, r, y)
    assert sig == pytest.approx(0.196570, abs=1e-6)
    d = sm.smile_digital(100.0, S, T, r, y, sig, ds)
    assert d["bs"] == pytest.approx(0.436329, abs=1e-6)
    assert d["skew"] == pytest.approx(0.133830, abs=1e-6)
    assert d["price"] == pytest.approx(0.570159, abs=1e-6)
    assert bs.digital_call(S, 100.0, T, r, y, 0.2) == pytest.approx(0.436000, abs=1e-6)
    assert ssvi_call(100.0) == pytest.approx(7.23033, abs=1e-5)
    assert np.exp(r * T) * ssvi_density(100.0) == pytest.approx(0.021125, abs=1e-6)
    # ATM-forward skew rho phi(theta) sqrt(theta) / (2 sqrt T) at T = 1/4, 1, 2
    for Tt, val in ((0.25, -0.6965), (1.0, -0.3432), (2.0, -0.2381)):
        F = bs.forward(S, Tt, r, y)
        _, dsF, _ = sm.ssvi_smile(F, S, Tt, r, y)
        assert dsF * F == pytest.approx(val, abs=1e-4)


def test_ssvi_is_free_of_arbitrage():
    rho = -0.7
    for Tt in (0.1, 0.25, 1.0, 2.0, 5.0):
        theta = 0.04 * Tt
        c = sm.ssvi_conditions(theta, rho)
        assert 0 <= c["calendar"] <= c["calendar_max"] and c["fly1"] < 4 and c["fly2"] <= 4
        F = bs.forward(S, Tt, r, y)
        k = np.linspace(-6.0, 4.0, 4001)
        w, w1, w2 = sm.ssvi_total_variance(k, theta, rho, sm.phi_power_law(theta))
        g = sm.durrleman_g(k, w, w1, w2)
        assert g.min() > 0
        # the bracket of the strike formula is Durrleman's g
        K = F * np.exp(k)
        assert sm.smile_bracket(K, S, Tt, r, y, *sm.ssvi_smile(K, S, Tt, r, y)) == pytest.approx(g, abs=1e-10)
    # density integrates to one, with mean F
    dens = lambda k: np.exp(r * T) * ssvi_density(k)
    assert quad(dens, 1e-8, 2000.0, limit=400)[0] == pytest.approx(1.0, abs=1e-8)
    assert quad(lambda k: k * dens(k), 1e-8, 5000.0, limit=400)[0] == pytest.approx(bs.forward(S, T, r, y), rel=1e-8)
    # total variance nondecreasing in T at fixed log-moneyness (calendar condition)
    k = np.linspace(-3, 2, 501)
    ws = [sm.ssvi_total_variance(k, 0.04 * Tt, rho, sm.phi_power_law(0.04 * Tt))[0] for Tt in (0.25, 0.5, 1, 2)]
    assert np.all(np.diff(np.array(ws), axis=0) >= 0)


def test_call_spread_bounds_bracket_digital_and_converge():
    sig_fn = lambda k: sm.ssvi_smile(k, S, T, r, y)
    sig, ds, _ = sig_fn(100.0)
    D = sm.smile_digital(100.0, S, T, r, y, sig, ds)["price"]
    gaps = []
    for h in (5.0, 1.0, 0.2):
        lo, hi = sm.call_spread_bounds(ssvi_call, 100.0, h)
        assert lo <= D <= hi
        gaps.append(hi - lo)
    # the gap is about e^{-rT} q(K) h: first order in h
    assert gaps[1] / gaps[2] == pytest.approx(5.0, rel=0.02)
    assert gaps[2] == pytest.approx(ssvi_density(100.0) * 0.2, rel=0.02)
