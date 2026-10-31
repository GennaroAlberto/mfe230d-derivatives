import numpy as np
import pytest

from mfe230d import bs

TC = bs.TEST_CASE


def test_test_case_price():
    assert bs.call(**TC) == pytest.approx(7.36429, abs=5e-6)


def test_put_call_parity():
    S, K, T, r, y, s = 95.0, 103.0, 0.7, 0.03, 0.01, 0.3
    lhs = bs.call(S, K, T, r, y, s) - bs.put(S, K, T, r, y, s)
    assert lhs == pytest.approx(S * np.exp(-y * T) - K * np.exp(-r * T), abs=1e-12)


def test_digitals_sum_to_bond():
    p = bs.digital_call(**TC) + bs.digital_put(**TC)
    assert p == pytest.approx(np.exp(-TC["r"] * TC["T"]), abs=1e-14)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    S, K, T, r, y, s = 100.0, 110.0, 0.8, 0.02, 0.01, 0.25
    f = bs.call if kind == "call" else bs.put
    g = bs.greeks(S, K, T, r, y, s, kind)
    h = 1e-4
    assert g["delta"] == pytest.approx((f(S + h, K, T, r, y, s) - f(S - h, K, T, r, y, s)) / (2 * h), rel=1e-6)
    k = 1e-2  # larger step for the second difference, to keep round-off small
    assert g["gamma"] == pytest.approx((f(S + k, K, T, r, y, s) - 2 * f(S, K, T, r, y, s) + f(S - k, K, T, r, y, s)) / k**2, rel=1e-6)
    assert g["vega"] == pytest.approx((f(S, K, T, r, y, s + h) - f(S, K, T, r, y, s - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(S, K, T + h, r, y, s) - f(S, K, T - h, r, y, s)) / (2 * h), rel=1e-6)
    assert g["rho_r"] == pytest.approx((f(S, K, T, r + h, y, s) - f(S, K, T, r - h, y, s)) / (2 * h), rel=1e-6)
    assert g["rho_y"] == pytest.approx((f(S, K, T, r, y + h, s) - f(S, K, T, r, y - h, s)) / (2 * h), rel=1e-6)


def test_implied_vol_roundtrip():
    for s in (0.05, 0.2, 0.8):
        p = float(bs.call(100.0, 120.0, 0.5, 0.01, 0.0, s))
        assert bs.implied_vol(p, 100.0, 120.0, 0.5, 0.01, 0.0) == pytest.approx(s, abs=1e-9)


def test_vectorised():
    K = np.linspace(80, 120, 5)
    assert bs.call(100.0, K, 1.0, 0.01, 0.02, 0.2).shape == (5,)
