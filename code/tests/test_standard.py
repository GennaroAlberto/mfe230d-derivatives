"""Checks of the Chapter 1 code against the results proved in the notes and the numbers quoted there."""
import sys
import pathlib

import numpy as np
import pytest
from scipy import integrate

from mfe230d import bs
from mfe230d.standard import (
    backward_residual,
    discounted_payoffs,
    fk_gbm_check,
    fk_residual,
    fk_vasicek_check,
    forward_residual,
    girsanov_price,
    green,
    log_drift,
    log_grid,
    lognormal_moment,
    market_price_of_risk,
    max_relative_residuals,
    risk_neutral_density_ratio,
    sup_moment_bound,
    superposition_price,
    transition_density,
    vasicek_bond,
)
from mfe230d.paths import gbm_paths, time_grid

TC = bs.TEST_CASE
S, K, T, R, Y, SIG = TC["S"], TC["K"], TC["T"], TC["r"], TC["y"], TC["sigma"]
VAS = dict(kappa=0.5, theta=0.04, xi=0.02)  # Example exa:sm-vasicek and Figure fig:sm-feynman-kac


# ---------------------------------------------------------------------------------------
# The model, moments and the pricing formula (lem:sm-girsanov, prop:sm-risk-neutral, rem:sm-forward)
# ---------------------------------------------------------------------------------------

def test_test_case_forward_and_prices():
    # E^Q[S_T] = 100 e^{-0.01} = 99.005 (the forward), and the call 7.36429 by superposition.
    assert lognormal_moment(1.0, S, T, R - Y, SIG) == pytest.approx(99.00498337, abs=1e-8)
    xi = log_grid(S, T, SIG, points=[K])
    call = superposition_price(lambda z: np.maximum(z - K, 0.0), S, T, R, Y, SIG, xi)
    assert call == pytest.approx(7.36429, abs=5e-6)
    assert call == pytest.approx(float(bs.call(S, K, T, R, Y, SIG)), abs=2e-6)
    put = superposition_price(lambda z: np.maximum(K - z, 0.0), S, T, R, Y, SIG, xi)
    assert put == pytest.approx(float(bs.put(S, K, T, R, Y, SIG)), abs=2e-6)
    # digital: midpoint value at the jump keeps the trapezoid rule second order
    dig = superposition_price(lambda z: (z > K) + 0.5 * (z == K), S, T, R, Y, SIG, xi)
    assert dig == pytest.approx(float(bs.digital_call(S, K, T, R, Y, SIG)), abs=2e-6)
    # prepaid forward: f(x) = x is worth S e^{-yT}; the call with K = 0 is the same claim
    pre = superposition_price(lambda z: z, S, T, R, Y, SIG, xi)
    assert pre == pytest.approx(S * np.exp(-Y * T), rel=1e-7)


def test_green_value_quoted():
    # G(0, 100; 1, 100) = e^{-r} N'(d_-) / (K sigma sqrt T), quoted in the notes as 0.019528
    dp, dm = bs.d_pm(S, K, T, R, Y, SIG)
    expected = np.exp(-R * T) * bs.norm_pdf(dm) / (K * SIG * np.sqrt(T))
    assert float(green(K, S, T, R, Y, SIG)) == pytest.approx(float(expected), rel=1e-12)
    assert float(green(K, S, T, R, Y, SIG)) == pytest.approx(0.019528, abs=5e-7)


@pytest.mark.parametrize("q", [-2.0, -0.5, 0.0, 1.0, 2.5])
def test_density_mass_and_moments(q):
    tau, g = 0.7, R - Y
    f = lambda z: z**q * transition_density(z, S, tau, g, SIG)  # noqa: E731
    quad = lambda h: sum(integrate.quad(h, a, b, limit=200, epsabs=0, epsrel=1e-11)[0]  # noqa: E731
                         for a, b in ((0, S), (S, np.inf)))
    assert quad(f) == pytest.approx(float(lognormal_moment(q, S, tau, g, SIG)), rel=1e-8)
    assert quad(lambda z: green(z, S, tau, R, Y, SIG)) == pytest.approx(np.exp(-R * tau), rel=1e-9)


def test_density_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        transition_density(100.0, 100.0, 0.0, 0.0, 0.2)
    with pytest.raises(ValueError):
        transition_density(100.0, 100.0, 1.0, 0.0, 0.0)
    assert float(transition_density(-1.0, 100.0, 1.0, 0.0, 0.2)) == 0.0
    with pytest.raises(ValueError):
        market_price_of_risk(0.05, 0.01, 0.0)


def test_log_drift_is_nu():
    assert log_drift(R - Y, SIG) == pytest.approx(R - Y - 0.5 * SIG**2)


def test_density_ratio_has_mean_one_and_reprices_the_call():
    # dQ/dP on F_T: E^P[Z_T] = 1 and e^{-rT} E^P[Z_T (S_T - K)^+] = call (lem:sm-girsanov, prop:sm-risk-neutral)
    mu = 0.07
    z = np.random.default_rng(11).standard_normal(400_000)
    ST = S * np.exp(log_drift(mu - Y, SIG) * T + SIG * np.sqrt(T) * z)
    Z = risk_neutral_density_ratio(ST, S, T, mu, R, Y, SIG)
    assert Z.mean() == pytest.approx(1.0, abs=4 * Z.std() / np.sqrt(z.size))
    est = girsanov_price(lambda s: np.maximum(s - K, 0.0), S, T, mu, R, Y, SIG, n=400_000, seed=12)
    assert abs(est["mean"] - float(bs.call(S, K, T, R, Y, SIG))) < 4 * est["se"]


def test_sup_moment_bound_holds():
    # E^Q[sup_{t<=T} S_t^q] is finite and below the Doob bound of lem:sm-girsanov(iv)
    t = time_grid(T, 250)
    P = gbm_paths(S, t, r=R, y=Y, sigma=SIG, n_paths=20_000, seed=5)
    for q in (-2.0, 1.0, 3.0):
        emp = np.mean(np.max(P**q, axis=1))
        assert emp < float(sup_moment_bound(q, S, T, R - Y, SIG))
        assert emp >= np.mean(P[:, -1] ** q) * 0.99


# ---------------------------------------------------------------------------------------
# The Green's function (prop:sm-green and cor:sm-superposition)
# ---------------------------------------------------------------------------------------

def test_backward_and_forward_residuals_are_second_order():
    back, fwd = max_relative_residuals([1e-2, 1e-3])
    assert back[1] < 3e-5 and fwd[1] < 3e-5
    # truncation error O(h^2): ten times smaller steps, a hundred times smaller residual
    assert back[0] / back[1] == pytest.approx(100.0, rel=0.02)
    assert fwd[0] / fwd[1] == pytest.approx(100.0, rel=0.02)


def test_residuals_at_other_points():
    x = np.array([60.0, 100.0, 150.0])
    rb, sb = backward_residual(0.3, x, 1.3, 110.0, 0.05, 0.0, 0.35, h=1e-3)
    rf, sf = forward_residual(1.3, x, 0.3, 110.0, -0.01, 0.03, 0.35, h=1e-3)
    assert np.all(np.abs(rb) < 1e-4 * np.max(sb)) and np.all(np.abs(rf) < 1e-4 * np.max(sf))


@pytest.mark.parametrize("tau", [1e-2, 1e-4])
def test_dirac_limits(tau):
    # forward variable: int G(t,x;T,xi) h(xi) d xi -> h(x); backward variable: int G phi dx -> phi(xi)
    h = lambda z: np.cos(z / 30.0) + 0.001 * z  # noqa: E731
    x = 100.0
    xi = log_grid(x, tau, SIG, n=40_001)
    fwd = np.trapezoid(green(xi, x, tau, R, Y, SIG) * h(xi), xi)
    bwd = np.trapezoid(green(x, xi, tau, R, Y, SIG) * h(xi), xi)
    tol = 50 * tau
    assert fwd == pytest.approx(h(x), abs=tol) and bwd == pytest.approx(h(x), abs=tol)


def test_strike_derivatives():
    # d/dK digital = -G(0, S; T, K) and d/dK call = -digital (rem:sm-arrow-debreu)
    Ks = np.array([80.0, 100.0, 130.0])
    e = 1e-3
    dD = (bs.digital_call(S, Ks + e, T, R, Y, SIG) - bs.digital_call(S, Ks - e, T, R, Y, SIG)) / (2 * e)
    dC = (bs.call(S, Ks + e, T, R, Y, SIG) - bs.call(S, Ks - e, T, R, Y, SIG)) / (2 * e)
    assert np.allclose(dD, -green(Ks, S, T, R, Y, SIG), atol=1e-8)
    assert np.allclose(dC, -bs.digital_call(S, Ks, T, R, Y, SIG), atol=1e-7)


def test_call_solves_pricing_equation():
    # the Black-Scholes call with yield solves eq:sm-bs-pde, the PDE of cor:sm-bs-pde
    v = lambda t, x: bs.call(x, K, T - t, R, Y, SIG)  # noqa: E731
    x = np.array([70.0, 100.0, 140.0])
    res = fk_residual(v, 0.2, x, lambda t, z: (R - Y) * z, lambda t, z: SIG * z, R, ht=1e-4, hx=1e-2)
    assert np.all(np.abs(res) < 1e-5)


# ---------------------------------------------------------------------------------------
# Feynman-Kac by simulation (thm:sm-feynman-kac, exa:sm-vasicek, fig:sm-feynman-kac)
# ---------------------------------------------------------------------------------------

def test_vasicek_bond_solves_its_equation_and_matches_gaussian_formula():
    k, th, xi = VAS["kappa"], VAS["theta"], VAS["xi"]
    u = lambda t, x: vasicek_bond(x, 5.0 - t, k, th, xi)  # noqa: E731
    x = np.linspace(-0.05, 0.15, 9)
    res = fk_residual(u, 2.0, x, lambda t, z: k * (th - z), lambda t, z: xi + 0 * z, lambda t, z: z,
                      ht=1e-3, hx=1e-3)
    assert np.max(np.abs(res)) < 1e-6
    # E exp(-int X) = exp(-m + v/2) with m, v the mean and variance of the Gaussian int_0^T X
    T5, x0 = 5.0, 0.04
    B = -np.expm1(-k * T5) / k
    m = th * T5 + (x0 - th) * B
    v = xi**2 / k**2 * (T5 - 2 * B + (1 - np.exp(-2 * k * T5)) / (2 * k))
    assert float(vasicek_bond(x0, T5, k, th, xi)) == pytest.approx(np.exp(-m + v / 2), rel=1e-12)
    assert float(vasicek_bond(x0, T5, k, th, xi)) == pytest.approx(0.820253, abs=5e-7)  # quoted
    assert float(vasicek_bond(x0, 0.0, k, th, xi)) == 1.0


def test_vasicek_monte_carlo_number_quoted():
    # our implementation (numpy default_rng, seed 2029): 10^5 exact OU paths, 50 steps, trapezoid rule
    res = fk_vasicek_check(0.04, 5.0, n_steps=50, n_paths=100_000, seed=2029, **VAS)
    assert res["mean"] == pytest.approx(0.820203, abs=5e-7)
    assert res["se"] == pytest.approx(1.59e-4, abs=5e-7)
    assert abs(res["error"]) < 1.96 * res["se"]


def test_discounted_payoffs_with_running_cost():
    # constant rate and cost on a deterministic path: e^{-rT} g + c (1 - e^{-rT}) / r
    t = time_grid(2.0, 2000)
    X = np.full((1, t.size), 3.0)
    val = discounted_payoffs(t, X, lambda s, z: 0.05 + 0 * z, lambda z: z, cost=lambda s, z: 0.5 + 0 * z)
    exact = np.exp(-0.1) * 3.0 + 0.5 * (1 - np.exp(-0.1)) / 0.05
    assert float(val[0]) == pytest.approx(exact, rel=1e-7)


def test_gbm_check_digital():
    pde = float(bs.digital_call(S, K, T, R, Y, SIG))
    res = fk_gbm_check(lambda s: (s > K) * 1.0, pde, S, T, R, Y, SIG, n_paths=200_000, seed=3)
    assert abs(res["error"]) < 4 * res["se"]


def test_figure_errors_within_intervals():
    # every point of the left panel of fig:sm-feynman-kac lies within its 95% interval of zero
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "figures"))
    import ch01_standard as fig

    rows = fig.vasicek_errors()  # the figure's paths and seeds
    assert np.all(np.abs(rows[:, 0]) < 1.96 * rows[:, 1])
