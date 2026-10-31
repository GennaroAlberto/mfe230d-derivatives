"""Checks of the Chapter 2 transform code, and of the Black--Scholes numbers quoted in Chapter 2,
against the closed forms of bs.py."""
import numpy as np
import pytest
from scipy.integrate import quad

from mfe230d import bs
from mfe230d import transforms as tr

TC = bs.TEST_CASE
S, K, T, r, y, SIG = (TC[k] for k in ("S", "K", "T", "r", "y", "sigma"))


# --- Theorem bsf-formula and the numbers of the notes ------------------------------------------

def test_test_case_prices_and_digitals():
    dp, dm = bs.d_pm(**TC)
    assert (float(dp), float(dm)) == pytest.approx((0.05, -0.15), abs=1e-15)
    assert bs.call(**TC) == pytest.approx(7.364290, abs=5e-7)
    assert bs.put(**TC) == pytest.approx(8.349406, abs=5e-7)
    assert bs.digital_call(**TC) == pytest.approx(0.436000, abs=5e-7)
    aon = S * np.exp(-y * T) * bs.norm_cdf(dp)
    assert aon == pytest.approx(50.964333, abs=5e-7)
    assert bs.call(**TC) == pytest.approx(aon - K * bs.digital_call(**TC), abs=1e-12)


def test_limits_of_the_formula():
    # sigma -> 0: discounted intrinsic value at the forward; sigma -> inf: S e^{-yT}
    for k in (90.0, 99.0049833749168, 110.0):
        lim = max(S * np.exp(-y * T) - k * np.exp(-r * T), 0.0)
        assert bs.call(S, k, T, r, y, 1e-10) == pytest.approx(lim, abs=1e-8)
    assert bs.call(S, K, T, r, y, 60.0) == pytest.approx(S * np.exp(-y * T), abs=1e-9)
    # tau -> 0 at and around the strike
    for x in (95.0, 100.0, 105.0):
        assert bs.call(x, K, 1e-12, r, y, SIG) == pytest.approx(max(x - K, 0.0), abs=1e-5)
    # K -> 0 and S -> 0
    assert bs.call(S, 1e-12, T, r, y, SIG) == pytest.approx(S * np.exp(-y * T), abs=1e-9)
    assert bs.put(1e-12, K, T, r, y, SIG) == pytest.approx(K * np.exp(-r * T), abs=1e-9)


def test_greeks_of_the_notes_and_their_relations():
    c, p = bs.greeks(**TC, kind="call"), bs.greeks(**TC, kind="put")
    quoted = dict(delta=0.509643, gamma=0.0195277, vega=39.0554, theta=-3.32226, rho_r=43.6000, rho_y=-50.9643)
    for k, v in quoted.items():
        assert float(c[k]) == pytest.approx(v, abs=6e-5 if abs(v) > 1 else 6e-7)
    quoted_put = dict(delta=-0.470555, theta=-4.29260, rho_r=-55.4049, rho_y=47.0555)
    for k, v in quoted_put.items():
        assert float(p[k]) == pytest.approx(v, abs=6e-5 if abs(v) > 1 else 6e-7)
    dK = -np.exp(-r * T) * bs.norm_cdf(bs.d_pm(**TC)[1])
    # relations (a)-(e) of the proposition, for the call and the put
    for g, V, dKV in ((c, bs.call(**TC), dK), (p, bs.put(**TC), dK + np.exp(-r * T))):
        assert V == pytest.approx(S * g["delta"] + K * dKV, abs=1e-12)
        assert g["rho_y"] == pytest.approx(-T * S * g["delta"], abs=1e-12)
        assert g["rho_r"] == pytest.approx(-T * K * dKV, abs=1e-12)
        assert g["rho_r"] + g["rho_y"] == pytest.approx(-T * V, abs=1e-12)
        assert g["vega"] == pytest.approx(SIG * T * S**2 * g["gamma"], abs=1e-12)
        assert g["theta"] == pytest.approx(-(r * g["rho_r"] + y * g["rho_y"] + 0.5 * SIG * g["vega"]) / T, abs=1e-12)
        assert g["theta"] + (r - y) * S * g["delta"] + 0.5 * SIG**2 * S**2 * g["gamma"] == pytest.approx(r * V, abs=1e-12)


# --- the heat-equation reduction --------------------------------------------------------------

def test_heat_reduction_reproduces_the_formula():
    s, xi = tr.heat_variables(**TC)
    assert (float(s), float(xi)) == pytest.approx((0.04, -0.03), abs=1e-15)
    k = np.linspace(60, 160, 11)
    for tt in (0.05, 1.0, 7.0):
        s, xi = tr.heat_variables(S, k, tt, r, y, SIG)
        assert np.allclose(tr.from_heat(tr.heat_call_exact(s, xi), k, tt, r), bs.call(S, k, tt, r, y, SIG),
                           atol=1e-12)


def test_heat_solution_solves_the_heat_equation():
    s, xi, h = 0.3, 0.2, 1e-4
    u = lambda a, b: float(tr.heat_call_exact(a, b))
    us = (u(s + h, xi) - u(s - h, xi)) / (2 * h)
    uxx = (u(s, xi + h) - 2 * u(s, xi) + u(s, xi - h)) / h**2
    assert us == pytest.approx(0.5 * uxx, rel=1e-6)


# --- Fourier ------------------------------------------------------------------------------------

def test_payoff_and_kernel_transforms():
    for a, om in ((1.5, 0.0), (1.5, 2.3), (3.0, -1.1)):
        z = om - 1j * a
        re = quad(lambda x: np.exp(-a * x) * np.expm1(x) * np.cos(om * x), 0, 60, limit=400)[0]
        im = quad(lambda x: -np.exp(-a * x) * np.expm1(x) * np.sin(om * x), 0, 60, limit=400)[0]
        assert complex(tr.call_payoff_transform(z)) == pytest.approx(re + 1j * im, abs=1e-9)
    s, z = 0.7, 0.4 - 1.3j
    kern = lambda x: np.exp(-x**2 / (2 * s)) / np.sqrt(2 * np.pi * s)
    re = quad(lambda x: (np.exp(-1j * z * x) * kern(x)).real, -30, 30)[0]
    im = quad(lambda x: (np.exp(-1j * z * x) * kern(x)).imag, -30, 30)[0]
    assert complex(tr.heat_kernel_transform(z, s)) == pytest.approx(re + 1j * im, abs=1e-10)


def test_fourier_call_matches_closed_form():
    assert tr.fourier_call(**TC) == pytest.approx(bs.call(**TC), abs=1e-13)
    k = np.array([50.0, 80.0, 100.0, 125.0, 200.0])
    for tt in (0.01, 0.25, 1.0, 5.0, 30.0):
        assert np.allclose(tr.fourier_call(S, k, tt, r, y, SIG), bs.call(S, k, tt, r, y, SIG), atol=1e-11)


def test_fourier_error_follows_the_nearest_pole():
    # error of the trapezoidal rule ~ S e^{-yT} e^{-2 pi (a - 1)/h} (pole at z = -i), as in the figure
    exact = bs.call(**TC)
    for a, h in ((1.5, 0.2), (2.0, 0.4), (1.25, 0.1), (3.0, 0.8)):
        err = abs(tr.fourier_call(**TC, a=a, h=h) - exact)
        pred = S * np.exp(-y * T) * np.exp(-2 * np.pi * (a - 1) / h)
        assert 0.5 < err / pred < 2.0
    with pytest.raises(ValueError):
        tr.fourier_call(**TC, a=0.9)


# --- Laplace ------------------------------------------------------------------------------------

def test_laplace_transform_of_the_heat_kernel():
    lam, xi = 0.7, 0.3
    num = quad(lambda s: np.exp(-lam * s - xi**2 / (2 * s)) / np.sqrt(2 * np.pi * s), 0, np.inf)[0]
    assert complex(tr.laplace_heat_kernel(lam, xi)).real == pytest.approx(num, rel=1e-9)


def test_laplace_transform_of_the_call():
    lam = 1.3
    for xi in (-0.4, 0.0, 0.25):
        num = quad(lambda s: np.exp(-lam * s) * float(tr.heat_call_exact(s, xi)), 0, 80, limit=400)[0]
        assert complex(tr.laplace_heat_call(lam, xi)).real == pytest.approx(num, rel=1e-8)
    # C^1 across 0 and the ODE  ubar''/2 - lam ubar = -(e^xi - 1)^+  away from 0
    f = lambda x: complex(tr.laplace_heat_call(lam, x)).real
    e = 1e-7
    assert f(-e) == pytest.approx(f(e), abs=1e-6)
    assert (f(0) - f(-e)) / e == pytest.approx((f(e) - f(0)) / e, abs=1e-5)
    h = 1e-4
    for x in (-0.5, 0.4):
        ode = 0.5 * (f(x + h) - 2 * f(x) + f(x - h)) / h**2 - lam * f(x)
        assert ode == pytest.approx(-max(np.expm1(x), 0.0), abs=1e-6)


def test_talbot_on_a_known_pair():
    t = np.array([0.1, 1.0, 3.0])
    assert np.allclose(tr.talbot(lambda lam: 1.0 / (lam + 1.0), t), np.exp(-t), atol=1e-12)
    with pytest.raises(ValueError):
        tr.talbot(lambda lam: 1.0 / lam, np.array([0.0]))


def test_laplace_call_matches_closed_form():
    assert abs(tr.laplace_call(**TC, M=20) - bs.call(**TC)) < 1e-12
    k = np.array([50.0, 80.0, 100.0, 125.0, 200.0])
    for tt in (0.01, 0.25, 1.0, 5.0, 30.0):
        assert np.allclose(tr.laplace_call(S, k, tt, r, y, SIG), bs.call(S, k, tt, r, y, SIG), atol=1e-9)
    errs = [abs(tr.laplace_call(**TC, M=M) - bs.call(**TC)) for M in (8, 12, 16, 20)]
    assert errs[0] > errs[1] > errs[2] > errs[3]
