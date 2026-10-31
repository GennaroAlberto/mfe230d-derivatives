"""Checks of the Chapter 7 code against the results proved in the notes and the numbers quoted there."""
import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr, zeta

from mfe230d import bs
from mfe230d.paths import (
    PiecewiseConstant,
    bridge_order,
    brownian_paths,
    construction_matrix,
    effective_parameters,
    euler,
    explained_variance,
    fitted_order,
    gbm_coefficients,
    gbm_from_brownian,
    gbm_paths,
    gbm_scheme_errors,
    gbm_scheme_terminal,
    gbm_scheme_weak_error,
    integral,
    milstein,
    mse_model,
    optimal_allocation,
    ou_paths,
    pca_factors,
    richardson,
    softplus_call,
    standard_normals,
    term_rate,
    term_vol,
    time_grid,
)

# The course's term-structure example (Example 7.3): y = 0, r(t) = 0.02 t,
# sigma = 20% before t = 1/2 and 10% after, monthly dates.
SIG = PiecewiseConstant([0.5], [0.2, 0.1])
RATE = lambda u: 0.02 * u  # noqa: E731
MONTHLY = time_grid(1.0, 12)


# ------------------------------------------------------------------ term structures

def test_piecewise_constant_and_quadrature():
    f = PiecewiseConstant([0.3, 0.7], [1.0, 2.0, 5.0])
    assert f(np.array([0.0, 0.3, 0.69, 0.7, 2.0])).tolist() == [1.0, 2.0, 2.0, 5.0, 5.0]
    assert f.integral(0.1, 0.9) == pytest.approx(0.2 * 1 + 0.4 * 2 + 0.2 * 5)
    assert f.integral(0.0, 1.0) == pytest.approx(integral(f, 0.0, 1.0))
    # Gauss-Legendre is exact for polynomials
    assert integral(lambda u: 3 * u**5 - u, 0.2, 1.3) == pytest.approx((1.3**6 - 0.2**6) / 2 - (1.3**2 - 0.2**2) / 2)
    assert integral(0.05, 0.0, 2.0) == pytest.approx(0.1)


def test_course_term_structure_example():
    p = effective_parameters(MONTHLY, r=RATE, y=0.0, sigma=SIG)
    assert np.allclose(p["var"], [0.04] * 6 + [0.01] * 6)
    assert np.allclose(p["r"], 0.01 * (2 * np.arange(1, 13) - 1) / 12)
    assert p["clock"][-1] == pytest.approx(0.025)
    assert term_vol(SIG, 1.0) == pytest.approx(np.sqrt(0.025))
    assert round(float(term_vol(SIG, 1.0)), 4) == 0.1581
    assert term_rate(RATE, 1.0) == pytest.approx(0.01)
    # forward variances recovered from term vols (eq. 7.2)
    tv = term_vol(SIG, MONTHLY[1:])
    tot = MONTHLY[1:] * tv**2
    assert np.allclose(np.diff(np.concatenate([[0], tot])) / np.diff(MONTHLY), p["var"])
    assert round(float(bs.call(100, 100, 1, 0.01, 0.0, np.sqrt(0.025))), 4) == 6.7799


def test_frozen_parameters_numbers():
    left = MONTHLY[:-1]
    R = np.sum(0.02 * left) / 12
    assert round(R, 5) == 0.00917
    F = 100 * np.exp(R)
    assert round(F, 2) == 100.92 and round(100 * np.exp(0.01), 2) == 101.01
    V_left = np.sum(np.where(left <= 0.5, 0.04, 0.01)) / 12  # sigma(1/2) = 20%
    V_right = np.sum(SIG(left) ** 2) / 12                      # sigma(1/2) = 10%
    assert V_left == pytest.approx(0.0275) and V_right == pytest.approx(0.025)
    price = lambda V: np.exp(-0.01) * bs.call(F, 100, 1.0, 0.0, 0.0, np.sqrt(V))  # noqa: E731
    assert round(float(price(V_left)), 4) == 7.0384
    assert round(float(price(V_right)), 4) == 6.7337


def test_time_change_numbers():
    lam = 0.01 / 0.025
    assert lam == pytest.approx(0.4)
    assert (lam * 0.04, lam * 0.01) == pytest.approx((0.016, 0.004))


# ------------------------------------------------------------------ exact GBM and OU

def test_gbm_constant_parameters_equals_closed_form():
    t = time_grid(2.0, 10)
    Z = standard_normals(5, 10, seed=1)
    S = gbm_paths(100.0, t, r=0.03, y=0.01, sigma=0.25, Z=Z)
    W = np.concatenate([np.zeros((5, 1)), np.cumsum(np.sqrt(np.diff(t)) * Z, axis=1)], axis=1)
    assert np.allclose(S, gbm_from_brownian(100.0, t, W, 0.02, 0.25))


def test_gbm_term_structure_forward_and_call():
    S = gbm_paths(100.0, MONTHLY, r=RATE, y=0.0, sigma=SIG, n_paths=400_000, seed=3)
    ST = S[:, -1]
    assert abs(ST.mean() - 100 * np.exp(0.01)) < 3 * ST.std() / np.sqrt(ST.size)
    pay = np.exp(-0.01) * np.maximum(ST - 100, 0)
    exact = bs.call(100, 100, 1, 0.01, 0.0, np.sqrt(0.025))
    assert abs(pay.mean() - exact) < 3 * pay.std() / np.sqrt(pay.size)
    # log-variance at the six-month date is 0.02 (= 0.04 x 1/2)
    assert np.log(S[:, 6]).var() == pytest.approx(0.02, rel=0.02)


@pytest.mark.parametrize("method", ["increments", "bridge", "pca"])
def test_gbm_methods_have_the_same_law(method):
    # Exact: the Gaussian part is built on the variance clock by a square root of min(Theta_i, Theta_j).
    clock = effective_parameters(MONTHLY, sigma=SIG)["clock"]
    A = construction_matrix(clock, method)
    assert np.allclose(A @ A.T, np.minimum.outer(clock[1:], clock[1:]), atol=1e-14)
    # and the paths agree with the increments construction in distribution (spot check of the mean)
    S = gbm_paths(100.0, MONTHLY, r=RATE, sigma=SIG, n_paths=200_000, seed=4, method=method)
    assert abs(S[:, -1].mean() - 100 * np.exp(0.01)) < 4 * S[:, -1].std() / np.sqrt(2e5)


def test_degenerate_volatility():
    S = gbm_paths(100.0, time_grid(1.0, 4), r=0.05, sigma=0.0, n_paths=3, seed=0, method="bridge")
    assert np.allclose(S, 100 * np.exp(0.05 * time_grid(1.0, 4)))


def test_ou_exact_moments():
    k, th, xi, x0, T = 2.0, 0.03, 0.02, 0.10, 1.5
    X = ou_paths(x0, time_grid(T, 3), k, th, xi, n_paths=200_000, seed=5)
    mean = th + (x0 - th) * np.exp(-k * T)
    var = xi**2 * (1 - np.exp(-2 * k * T)) / (2 * k)
    assert X[:, -1].mean() == pytest.approx(mean, abs=4 * np.sqrt(var / 2e5))
    assert X[:, -1].var() == pytest.approx(var, rel=0.02)
    # kappa = 0: Brownian motion with volatility xi
    Y = ou_paths(0.0, time_grid(1.0, 4), 0.0, 0.0, 0.3, n_paths=100_000, seed=6)
    assert Y[:, -1].var() == pytest.approx(0.09, rel=0.02)


def test_ou_euler_stationary_variance():
    # Exercise: Euler's stationary variance is xi^2 / (kappa (2 - kappa h)).
    k, th, xi, h = 2.0, 0.03, 0.02, 1.0 / 12
    X = euler(lambda t, x: k * (th - x), lambda t, x: xi + 0 * x, th, time_grid(25.0, 300),
              n_paths=20_000, seed=3)
    assert X[:, -100:].var() / (xi**2 / (2 * k)) == pytest.approx(2 / (2 - k * h), rel=0.03)


# ------------------------------------------------------------------ Brownian constructions

@pytest.mark.parametrize("t", [time_grid(1.0, 8), time_grid(2.0, 7), np.array([0, 0.1, 0.15, 0.5, 0.9, 1.3])])
@pytest.mark.parametrize("method", ["increments", "bridge", "pca"])
def test_constructions_are_square_roots_of_min(t, method):
    A = construction_matrix(t, method)
    assert np.allclose(A @ A.T, np.minimum.outer(t[1:], t[1:]), atol=1e-13)


def test_bridge_order_is_dyadic():
    idx, left, right = bridge_order(8)
    assert idx.tolist() == [8, 4, 2, 6, 1, 3, 5, 7]
    assert left.tolist() == [0, 0, 0, 4, 0, 2, 4, 6] and right.tolist() == [-1, 8, 4, 8, 2, 4, 6, 8]
    assert sorted(bridge_order(13)[0].tolist()) == list(range(1, 14))


def test_bridge_law_by_simulation():
    # Proposition 7.7(i): B = W_u - interpolation is N(0, (u-s)(t-u)/(t-s)), independent of the outside path.
    t = np.array([0.0, 0.2, 0.5, 0.9, 1.4])
    W = brownian_paths(t, n_paths=400_000, seed=8, method="increments")
    s, u, tt = 0.2, 0.5, 0.9
    B = W[:, 2] - ((tt - u) * W[:, 1] + (u - s) * W[:, 3]) / (tt - s)
    assert B.var() == pytest.approx((u - s) * (tt - u) / (tt - s), rel=0.01)
    for j in (1, 3, 4):
        assert abs(np.corrcoef(B, W[:, j])[0, 1]) < 0.006


def test_explained_variance_table():
    m = 64
    t = time_grid(1.0, m)
    k = [0, 1, 3, 15]
    table = {"increments": [0.031, 0.061, 0.120, 0.435],
             "bridge": [0.672, 0.836, 0.918, 0.981],
             "pca": [0.811, 0.901, 0.950, 0.988]}
    for method, vals in table.items():
        assert np.round(explained_variance(t, method)[k], 3).tolist() == vals
    assert explained_variance(t, "increments")[0] == pytest.approx(2 / (m + 1))
    assert explained_variance(t, "bridge")[0] == pytest.approx((2 * m + 1) / (3 * m))


def test_pca_eigenvalues_closed_form_and_optimality():
    m = 64
    t = time_grid(1.0, m)
    h = 1.0 / m
    _, lam = pca_factors(t)
    k = np.arange(1, m + 1)
    assert np.allclose(lam, h / (4 * np.sin((2 * k - 1) * np.pi / (2 * (2 * m + 1))) ** 2))
    assert lam[0] / lam.sum() == pytest.approx(0.8107, abs=1e-4)
    assert abs(lam[0] / lam.sum() - 8 / np.pi**2) < 0.002
    # Proposition 7.8: no square root beats principal components
    C = np.minimum.outer(t[1:], t[1:])
    w, V = np.linalg.eigh(C)
    root = V @ np.diag(np.sqrt(w)) @ V.T
    rng = np.random.default_rng(0)
    best = np.cumsum(lam) / lam.sum()
    for _ in range(5):
        U, _ = np.linalg.qr(rng.standard_normal((m, m)))
        A = root @ U
        share = np.cumsum(np.sum(A**2, axis=0)) / np.trace(C)
        assert np.all(share <= best + 1e-12)


# ------------------------------------------------------------------ schemes

def test_schemes_agree_with_product_form():
    t = time_grid(1.0, 16)
    a, b, bx = gbm_coefficients(-0.01, 0.2)
    dW = np.sqrt(1 / 16) * standard_normals(50, 16, seed=9)
    assert np.allclose(euler(a, b, 100.0, t, dW=dW)[:, -1], gbm_scheme_terminal(100.0, -0.01, 0.2, 1.0, dW, "euler"))
    assert np.allclose(milstein(a, b, bx, 100.0, t, dW=dW)[:, -1],
                       gbm_scheme_terminal(100.0, -0.01, 0.2, 1.0, dW, "milstein"))
    # additive noise: Milstein = Euler
    A, B, Bx = (lambda t, x: -x), (lambda t, x: 0.3 + 0 * x), (lambda t, x: 0 * x)
    assert np.allclose(euler(A, B, 1.0, t, dW=dW), milstein(A, B, Bx, 1.0, t, dW=dW))


def test_euler_exact_moments_exercise():
    S0, mu, sig, T, n = 100.0, -0.01, 0.2, 1.0, 12
    h = T / n
    ex2 = S0**2 * np.exp((2 * mu + sig**2) * T)
    e2 = lambda n, extra=0.0: S0**2 * ((1 + mu * T / n) ** 2 + sig**2 * T / n + extra * (T / n) ** 2) ** n  # noqa: E731
    assert round(e2(12) - ex2, 4) == -0.0850 and round(e2(24) - ex2, 4) == -0.0425
    assert abs(2 * e2(24) - e2(12) - ex2) < 3e-5
    assert (e2(12, 0.5 * sig**4) - ex2) / ex2 == pytest.approx(-(mu**2 + 2 * mu * sig**2) * T * h, rel=0.01)
    dW = np.sqrt(h) * standard_normals(400_000, n, seed=10)
    X = gbm_scheme_terminal(S0, mu, sig, T, dW, "euler")
    assert X.mean() == pytest.approx(S0 * (1 + mu * h) ** n, abs=4 * X.std() / np.sqrt(X.size))
    assert ndtr(-(1 + mu * 0.25) / (1.0 * 0.5)) == pytest.approx(0.0230, abs=1e-4)


def test_discrete_maximum_expansion_exercise():
    EM = lambda n: np.sum(np.arange(1, n + 1) ** -0.5) / np.sqrt(2 * np.pi * n)  # noqa: E731
    ex = np.sqrt(2 / np.pi)
    c, d = zeta(0.5) / np.sqrt(2 * np.pi), 1 / (2 * np.sqrt(2 * np.pi))
    assert c == pytest.approx(-0.5826, abs=1e-4)
    assert EM(64) - ex == pytest.approx(c / 8 + d / 64, abs=1e-5)
    r_half = (np.sqrt(2) * EM(64) - EM(32)) / (np.sqrt(2) - 1) - ex
    assert r_half == pytest.approx(-np.sqrt(2) * d / 64, rel=0.01)
    # Spitzer's formula against simulation
    W = np.cumsum(standard_normals(200_000, 16, seed=11) / 4.0, axis=1)
    M = np.maximum(W.max(axis=1), 0.0)
    assert M.mean() == pytest.approx(EM(16), abs=4 * M.std() / np.sqrt(M.size))


# ------------------------------------------------------------------ the laboratory of Section 7.3

@pytest.fixture(scope="module")
def lab():
    return gbm_scheme_errors(S0=100.0, mu=-0.01, sigma=0.2, T=1.0, levels=range(7), n_paths=10**6, seed=2026)


def test_lab_strong_errors(lab):
    assert lab["m"].tolist() == [1, 2, 4, 8, 16, 32, 64]
    assert round(lab["strong"]["euler"][-1], 3) == 0.279
    assert round(lab["strong"]["milstein"][-1], 4) == 0.0047
    assert round(fitted_order(lab["h"][2:], lab["strong"]["euler"][2:]), 2) == 0.49
    assert round(fitted_order(lab["h"][2:], lab["strong"]["milstein"][2:]), 2) == 0.99
    # Exercise: the Euler strong error constant sigma^2 E[X_T] sqrt(T/pi)
    const = 0.04 * 100 * np.exp(-0.01) / np.sqrt(np.pi)
    assert lab["strong"]["euler"][-1] * 8 == pytest.approx(const, rel=0.005)


def test_exact_weak_errors_table():
    # Exact weak errors by lattice convolution (Section 7.3 and 7.4 of the notes).
    e = {m: gbm_scheme_weak_error(m, "euler") for m in (2, 4, 8, 16, 32, 64, 128)}
    mi = {m: gbm_scheme_weak_error(m, "milstein") for m in (8, 16, 32, 64)}
    assert [round(e[m], 5) for m in (16, 32, 64)] == [0.01059, 0.00540, 0.00273]
    assert [round(mi[m], 5) for m in (16, 32, 64)] == [0.00451, 0.00221, 0.00110]
    rm = [2 * mi[m] - mi[m // 2] for m in (16, 32, 64)]
    assert [round(x, 6) for x in rm] == [-0.000329, -0.000083, -0.000021]
    re = [2 * e[m] - e[m // 2] for m in (4, 8, 16, 32, 64)]
    assert [round(re[0], 4), round(re[1], 4), round(re[2], 5), round(re[3], 5), round(re[4], 6)] == \
        [0.0162, 0.0038, 0.00091, 0.00022, 0.000055]
    # Euler bias constant: 0.169, 0.173 at m = 16, 32; the limit (extrapolated) is 0.176
    assert [round(16 * e[16], 3), round(32 * e[32], 3)] == [0.169, 0.173]
    assert round(2 * 128 * e[128] - 64 * e[64], 3) == 0.176
    # insensitive to the lattice
    assert gbm_scheme_weak_error(32, "euler", dx=5e-5, half_width=6.0) == pytest.approx(e[32], abs=1e-9)


def test_lab_weak_errors_agree_with_exact(lab):
    for s in ("euler", "milstein"):
        exact = np.array([gbm_scheme_weak_error(int(m), s) for m in lab["m"]])
        z = (lab["weak"][s] - exact) / lab["weak_se"][s]
        assert np.all(np.abs(z[4:]) < 2.1)  # m = 16, 32, 64, as quoted in the notes
        assert np.all(np.abs(z) < 3)
    ex_m = np.array([gbm_scheme_weak_error(int(m), "milstein") for m in lab["m"]])
    z = (lab["rich"]["milstein"][1:] - (2 * ex_m[1:] - ex_m[:-1])) / lab["rich_se"]["milstein"][1:]
    assert np.all(np.abs(z) < 3)
    assert np.round(lab["weak_se"]["euler"][4:], 4).tolist() == [0.0005, 0.0004, 0.0003]
    assert lab["var_f"] == pytest.approx(154, abs=1.5)


def test_smooth_payoff_exact_moments():
    S0, mu, sig, T = 100.0, -0.01, 0.2, 1.0
    dens = lambda z: np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)  # noqa: E731
    X = lambda z: S0 * np.exp((mu - 0.5 * sig**2) * T + sig * np.sqrt(T) * z)  # noqa: E731
    m1 = integrate.quad(lambda z: softplus_call(X(z)) * dens(z), -12, 12, limit=400)[0]
    m2 = integrate.quad(lambda z: softplus_call(X(z)) ** 2 * dens(z), -12, 12, limit=400)[0]
    assert round(m1, 3) == 8.206
    assert round(m2 - m1**2) == 154


# ------------------------------------------------------------------ Richardson and allocation

def test_richardson_cancels_leading_term():
    theta, c, d = 1.0, 0.7, -2.0
    y = lambda h: theta + c * h + d * h**2  # noqa: E731
    assert richardson(y(0.01), y(0.02)) - theta == pytest.approx(-2 * d * 0.01**2)
    ys = lambda h: theta + c * np.sqrt(h) + d * h  # noqa: E731
    # with an expansion in powers of h^(1/2), p = 1 leaves an h^(1/2) error and p = 1/2 an O(h) one
    assert richardson(ys(0.01), ys(0.02)) - theta == pytest.approx(c * (2 - np.sqrt(2)) * 0.1)
    assert richardson(ys(0.01), ys(0.02), p=0.5) - theta == pytest.approx(-np.sqrt(2) * d * 0.01)


def test_optimal_allocation_formula_and_numbers():
    c, v = 0.17, 154.0
    for beta in (1.0, 2.0):
        for C in (1e5, 1e7):
            opt = optimal_allocation(C, c, v, beta=beta)
            grid = np.geomspace(0.5, 500, 20001)
            assert opt["m"] == pytest.approx(grid[np.argmin(mse_model(grid, C, c, v, beta=beta))], rel=1e-3)
            assert opt["sd"] ** 2 == pytest.approx(2 * beta * opt["bias"] ** 2)
    a = optimal_allocation(1e6, c, v)
    assert round(a["m"]) == 7 and round(a["n"], -4) == 1.4e5 and round(a["rmse"], 3) == 0.041
    assert round(np.sqrt(v / 1e6), 3) == 0.012
    b = optimal_allocation(1e8, c, v)
    assert round(b["m"]) == 33 and round(b["n"], -6) == 3e6 and round(b["rmse"], 4) == 0.0088
    assert round(np.sqrt(v / 1e8), 4) == 0.0012


# ------------------------------------------------------------------ API conventions

def test_dates_get_zero_prepended_and_one_dimensional_normals():
    W = brownian_paths([0.5, 1.0], n_paths=4, seed=1)
    assert W.shape == (4, 3) and np.all(W[:, 0] == 0)
    z = np.array([0.3, -1.2])
    assert ou_paths(0.0, [0.5, 1.0], 1.0, 0.0, 0.2, Z=z).shape == (1, 3)
    assert gbm_paths(100.0, [0.5, 1.0], Z=z).shape == (1, 3)
    assert term_vol(SIG, 0.7) == pytest.approx(np.sqrt((0.5 * 0.04 + 0.2 * 0.01) / 0.7))
