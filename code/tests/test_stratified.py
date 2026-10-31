"""Checks of the Chapter 10 code against the results proved in the notes and the numbers they quote."""
import numpy as np
import pytest
from scipy import integrate
from scipy.optimize import minimize
from scipy.special import ndtr

from mfe230d.bs import TEST_CASE, call, put
from mfe230d.paths import time_grid
from mfe230d.rng import norm_inv
from mfe230d.stratified import (
    COURSE_OTM,
    HESTON_LAB,
    allocation_factors,
    asian_call_values,
    bs_conditional,
    call_stratum_moments,
    cir_exact,
    course_otm_experiment,
    discounted_call,
    equiprobable_edges,
    heston_put_samples,
    heston_variance_integrals,
    inverse_count_moment,
    laboratory,
    latin_hypercube,
    linear_direction,
    log_euler_mixing_check,
    mixing_parameters,
    neyman_allocation,
    normal_edges,
    open_uniforms,
    poststratified_estimate,
    poststratified_mse,
    proportional_allocation,
    stratified_estimate,
    stratified_normals,
    stratified_projection,
    stratified_uniforms,
    strata_of,
    stratum_moments,
    terminal_stratified_normals,
    variance_decomposition,
)

P = TEST_CASE
S, K, T, r, y, SIG = (P[k] for k in ("S", "K", "T", "r", "y", "sigma"))
ZS = (np.log(K / S) - (r - y - 0.5 * SIG**2) * T) / (SIG * np.sqrt(T))
F = lambda z: discounted_call(z, S, K, T, r, y, SIG)


def _otm():
    Po = COURSE_OTM
    f = lambda z: discounted_call(z, Po["S"], Po["K"], Po["T"], Po["r"], Po["y"], Po["sigma"])
    zs = (np.log(Po["K"] / Po["S"]) - (Po["r"] - Po["y"] - 0.5 * Po["sigma"] ** 2) * Po["T"]) / (Po["sigma"] * np.sqrt(Po["T"]))
    return f, zs


@pytest.fixture(scope="module")
def lab():
    """A reduced run of the laboratory: the quoted (full-size) ratios are checked within sampling
    tolerance, the exact ratios exactly."""
    return laboratory(n_call=100_000, n_otm=100_000, n_asian=2**15, lhs_n=2**10, lhs_reps=300, n_sd=2**18,
                      n_heston=50_000)


# ------------------------------------------------------------------ strata and their moments
def test_uniforms_in_strata_and_labels():
    rng = np.random.default_rng(0)
    v = open_uniforms(rng, 10**5)
    assert v.min() > 0 and v.max() < 1
    u, lab_ = stratified_uniforms([3, 0, 5, 2], edges=[0, 0.1, 0.5, 0.9, 1.0], rng=rng)
    e = np.array([0, 0.1, 0.5, 0.9, 1.0])
    assert np.all((u > e[lab_]) & (u <= e[lab_ + 1]))
    assert np.array_equal(np.bincount(lab_, minlength=4), [3, 0, 5, 2])
    assert np.array_equal(strata_of(u, e), lab_)
    assert proportional_allocation(1000, np.full(10, 0.1)).tolist() == [100] * 10
    assert proportional_allocation(7, [0.5, 0.5]).sum() == 7


def test_stratum_moments_against_closed_forms_and_decomposition():
    e = normal_edges(10)
    p, m, v = stratum_moments(F, e, kinks=(ZS,))
    p2, m2, v2 = call_stratum_moments(e, S, K, T, r, y, SIG)
    assert np.allclose(p, 0.1) and np.allclose(m, m2, atol=1e-11) and np.allclose(v, v2, rtol=1e-9, atol=1e-12)
    d = variance_decomposition(p, m, v)                                # Theorem st-variance (i)
    assert d["mean"] == pytest.approx(7.36429, abs=5e-6)
    assert d["total"] == pytest.approx(158.006, abs=1e-3)
    _, _, v1 = stratum_moments(F, np.array([-np.inf, np.inf]), kinks=(ZS,))
    assert d["total"] == pytest.approx(v1[0], rel=1e-9)
    assert ZS == pytest.approx(0.15)


def test_example_call_variance_factors():
    """Example st-call: proportional factors, Neyman factors, and the out-of-the-money strata."""
    expect = {2: (0.657, None), 10: (0.1007, 0.0234), 100: (0.0081, 0.00044), 1000: (0.00073, None)}
    for H, (prop, ney) in expect.items():
        p, m, v = stratum_moments(F, normal_edges(H), kinks=(ZS,))
        a = allocation_factors(p, m, v)
        assert a["proportional"] == pytest.approx(prop, rel=0.006)
        if ney is not None:
            assert a["neyman"] == pytest.approx(ney, rel=0.01)
        assert np.all(v[normal_edges(H)[1:] <= ZS] == 0)               # strata below z* are worthless
    se = {10: 0.317, 100: 0.090, 1000: 0.027}
    for H, s in se.items():
        p, m, v = stratum_moments(F, normal_edges(H), kinks=(ZS,))
        assert np.sqrt(allocation_factors(p, m, v)["proportional"]) == pytest.approx(s, abs=6e-4)
    p, m, v = stratum_moments(F, normal_edges(10), kinks=(ZS,))
    assert np.sqrt(allocation_factors(p, m, v)["neyman"]) == pytest.approx(0.153, abs=6e-4)


def test_stratified_estimator_unbiased_with_the_variance_of_theorem():
    """Theorem st-variance (ii)-(iii): replicate the estimator and compare with sum p^2 s^2 / n_h."""
    H, n, R = 8, 80, 4000
    p, m, v = stratum_moments(F, normal_edges(H), kinks=(ZS,))
    rng = np.random.default_rng(7)
    counts = proportional_allocation(n, p)
    runs = [stratified_estimate(F(z), lab_, p) for z, lab_ in (stratified_normals(counts, rng=rng) for _ in range(R))]
    est = np.array([o["mean"] for o in runs])
    var = np.sum(p**2 * v / counts)
    assert var == pytest.approx(np.sum(p * v) / n, rel=1e-12)
    assert abs(est.mean() - 7.36429) < 4 * np.sqrt(var / R)
    assert est.var(ddof=1) == pytest.approx(var, rel=0.1)
    # the variance estimate of Proposition st-error (i) is unbiased
    assert np.mean([o["se"] ** 2 for o in runs]) == pytest.approx(var, rel=0.05)
    # a stratum without draws is refused; one draw per stratum gives no standard error
    with pytest.raises(ValueError):
        stratified_estimate([1.0, 2.0], [0, 0], [0.5, 0.5])
    assert np.isnan(stratified_estimate([1.0, 2.0], [0, 1], [0.5, 0.5])["se"])


def test_a_bad_allocation_can_increase_the_variance():
    p, s = np.array([0.5, 0.5]), np.array([0.0, 1.0])
    n = 10
    assert np.sum(p**2 * s**2 / np.array([n - 1, 1])) > 0.5 / n                # sigma^2 = s_2^2 / 2


def test_neyman_allocation():
    p = np.full(4, 0.25)
    sd = np.array([0.0, 1.0, 2.0, 5.0])
    counts = neyman_allocation(100, p, sd, minimum=1)
    assert counts.sum() == 100 and counts[0] == 1
    assert counts[1:].tolist() == [13, 25, 61]                     # one draw each, then 96 split as 1 : 2 : 5
    sbar = np.sum(p * sd)
    var = np.sum(p**2 * sd**2 / counts)
    assert var == pytest.approx(sbar**2 / 99, rel=0.01)                         # bar sigma^2 / (n - k)
    # Proposition st-optimal (ii): the gap decomposition, on the test-case call
    pp, m, v = stratum_moments(F, normal_edges(10), kinks=(ZS,))
    a = allocation_factors(pp, m, v)
    sd = np.sqrt(v)
    gap = np.sum(pp * (m - np.sum(pp * m)) ** 2) + np.sum(pp * (sd - np.sum(pp * sd)) ** 2)
    assert (1 - a["neyman"]) * a["total"] == pytest.approx(gap, rel=1e-9)


def test_many_strata_limit_and_refinement():
    """Theorem st-variance (iv)-(v): refinement never hurts; sum p_h sigma_h^2 -> E Var(Y | U)."""
    prev = np.inf
    for H in (1, 2, 4, 8, 16, 32, 64, 128):
        p, m, v = stratum_moments(F, normal_edges(H), kinks=(ZS,))
        w = np.sum(p * v)
        assert w <= prev + 1e-12
        prev = w
    assert prev / 158.006 < 0.0065


# ------------------------------------------------------------------ the course's experiment
def test_course_otm_experiment():
    f, zs = _otm()
    assert float(call(**COURSE_OTM)) == pytest.approx(0.35963, abs=5e-6)
    assert zs == pytest.approx(1.877, abs=5e-4)
    assert 1 - ndtr(zs) == pytest.approx(0.030, abs=5e-4)
    for H, fac in ((10, 0.856), (50, 0.318), (100, 0.158)):
        assert allocation_factors(*stratum_moments(f, normal_edges(H), kinks=(zs,)))["proportional"] == \
            pytest.approx(fac, abs=6e-4)
    out = course_otm_experiment(seed=5)
    got = [out[k]["var"] * 1e5 for k in ("plain", 10, 50, 100)]
    exact = [out[k]["exact_var"] * 1e5 for k in ("plain", 10, 50, 100)]
    assert np.allclose(got, [9.61, 8.05, 2.64, 1.46], atol=0.006)
    assert np.allclose(exact, [8.08, 6.92, 2.57, 1.28], atol=0.006)


# ------------------------------------------------------------------ post-stratification
def test_binomial_inverse_moment_expansion():
    for p in (0.1, 0.25, 0.5):
        for n in (200, 400):
            exact = inverse_count_moment(n, p)
            approx = 1 / (n * p) + (1 - p) / (n * n * p * p)
            assert abs(exact - approx) * n**3 < 40 / p**3                       # O(n^-3)


def test_poststratification_mse():
    H, n, R = 4, 40, 8000
    p, m, v = stratum_moments(F, normal_edges(H), kinks=(ZS,))
    rng = np.random.default_rng(11)
    Z = rng.standard_normal((R, n))
    est = np.array([poststratified_estimate(F(z), strata_of(ndtr(z), equiprobable_edges(H)), p)["mean"] for z in Z])
    mse = poststratified_mse(n, p, v)
    assert np.mean((est - 7.36429) ** 2) == pytest.approx(mse, rel=0.08)
    for nn in (400, 800):                                                        # Proposition st-post (iii)
        expansion = np.sum(p * v) / nn + np.sum((1 - p) * v) / nn**2
        assert abs(poststratified_mse(nn, p, v) - expansion) * nn**3 < 30 * np.sum(v)
    assert abs(est.mean() - 7.36429) < 4 * np.sqrt(mse / R)


# ------------------------------------------------------------------ paths
def test_projection_has_the_conditional_law():
    rng = np.random.default_rng(3)
    v = rng.standard_normal(5)
    v /= np.linalg.norm(v)
    xi = np.full(200_000, 1.3)
    Z = stratified_projection(v, xi, rng=rng)
    assert np.allclose(Z @ v, 1.3)
    C = np.cov((Z - np.outer(xi, v)).T)
    assert np.allclose(C, np.eye(5) - np.outer(v, v), atol=0.01)


def test_terminal_stratification_bridge_and_increments():
    t = time_grid(T, 12)
    v = linear_direction(t, w=np.r_[np.zeros(11), 1.0])
    assert np.allclose(v, np.sqrt(np.diff(t) / T))                              # W_T / sqrt(T) = v^T Z
    from mfe230d.paths import brownian_paths
    Z, lab_ = terminal_stratified_normals(np.ones(16, dtype=int), 12, seed=1)
    W = brownian_paths(t, Z=Z, method="bridge")
    assert np.allclose(W[:, -1], np.sqrt(T) * Z[:, 0])
    e = normal_edges(16)
    assert np.all((Z[:, 0] > e[lab_]) & (Z[:, 0] <= e[lab_ + 1]))


def test_asian_example_numbers():
    from mfe230d.stratified import asian_reference, geometric_asian_call
    assert geometric_asian_call(S, K, T, r, y, SIG, 12) == pytest.approx(4.40286, abs=5e-6)   # the course's value
    ref = asian_reference(S, K, T, r, y, SIG, n=2**20, seed=2026)
    assert abs(ref["mean"] - 4.5612893) < 4 * ref["se"] and ref["se"] < 2e-4    # exact value of exa:as-reference
    assert abs(4.56058 - 4.5612893) < 0.0008                                     # course estimate within its se
    # the direction of sum_j W_{t_j} is the GHS drift direction (cosine 0.9999)
    t = time_grid(T, 12)
    def neg(z):
        A = asian_call_values(z[None, :], S, K, T, r, y, SIG)[0] * np.exp(r * T)
        return np.inf if A <= 0 else -(np.log(A) - 0.5 * z @ z)
    mu = minimize(neg, 0.3 * np.ones(12), method="BFGS").x
    v = linear_direction(t)
    assert mu @ v / np.linalg.norm(mu) == pytest.approx(0.9999, abs=5e-5)
    # plain sd 7.49; about 0.055 with 256 strata along that direction
    rng = np.random.default_rng(2)
    sd = np.std(asian_call_values(rng.standard_normal((2**18, 12)), S, K, T, r, y, SIG), ddof=1)
    assert sd == pytest.approx(7.49, abs=0.03)
    p = np.full(256, 1.0 / 256)
    xi, lab_ = stratified_normals(proportional_allocation(2**16, p), rng=rng)
    o = stratified_estimate(asian_call_values(stratified_projection(v, xi, rng=rng), S, K, T, r, y, SIG), lab_, p)
    assert 0.045 < o["se"] * 2**8 / sd < 0.065


# ------------------------------------------------------------------ Latin hypercube
def test_latin_hypercube_structure_and_additive_variance():
    rng = np.random.default_rng(4)
    V = latin_hypercube(16, 3, rng=rng)
    for j in range(3):
        assert np.array_equal(np.sort(np.floor(16 * V[:, j]).astype(int)), np.arange(16))
    # additive f: variance equals the one-point-per-stratum formula of Proposition st-lhs (ii)
    n, R = 8, 20_000
    g = lambda U: np.exp(U).sum(axis=1)
    est = np.array([g(latin_hypercube(n, 2, rng=rng)).mean() for _ in range(R)])
    e = np.linspace(0, 1, n + 1)
    mu_k = (np.exp(e[1:]) - np.exp(e[:-1])) * n
    var_k = (np.exp(2 * e[1:]) - np.exp(2 * e[:-1])) / 2 * n - mu_k**2
    exact = 2 * var_k.sum() / n**2
    assert est.mean() == pytest.approx(2 * (np.e - 1), abs=4 * np.sqrt(exact / R))
    assert est.var(ddof=1) == pytest.approx(exact, rel=0.07)


def test_lhs_figure_limits():
    ve = 0.5 * (np.e**2 - 1) - (np.e - 1) ** 2
    assert (36 / 144) / (5 * ve + 36 / 144) == pytest.approx(0.17, abs=0.005)
    assert 1 - (5 / 3) / ((4 / 3) ** 5 - 1) == pytest.approx(0.48, abs=0.005)


# ------------------------------------------------------------------ conditional Monte Carlo
def _heston_put_fourier(S_, K_, T_, r_, y_, v0, kappa, theta, xi, rho):
    """Heston (1993) put by Gil-Pelaez inversion, in the 'little trap' form (test reference only)."""
    def cf(u):
        b, a = kappa, kappa * theta
        d = np.sqrt((rho * xi * 1j * u - b) ** 2 + xi**2 * (1j * u + u**2))
        g = (b - rho * xi * 1j * u - d) / (b - rho * xi * 1j * u + d)
        C = (r_ - y_) * 1j * u * T_ + a / xi**2 * ((b - rho * xi * 1j * u - d) * T_
                                                     - 2 * np.log((1 - g * np.exp(-d * T_)) / (1 - g)))
        D = (b - rho * xi * 1j * u - d) / xi**2 * (1 - np.exp(-d * T_)) / (1 - g * np.exp(-d * T_))
        return np.exp(C + D * v0 + 1j * u * np.log(S_))
    k, Fw = np.log(K_), S_ * np.exp((r_ - y_) * T_)
    P2 = 0.5 + integrate.quad(lambda u: (np.exp(-1j * u * k) * cf(u) / (1j * u)).real, 1e-10, 200, limit=500)[0] / np.pi
    P1 = 0.5 + integrate.quad(lambda u: (np.exp(-1j * u * k) * cf(u - 1j) / (1j * u * Fw)).real, 1e-10, 200,
                              limit=500)[0] / np.pi
    c = S_ * np.exp(-y_ * T_) * P1 - K_ * np.exp(-r_ * T_) * P2
    return c - S_ * np.exp(-y_ * T_) + K_ * np.exp(-r_ * T_)


def test_fourier_reference_prices():
    H_ = HESTON_LAB
    assert _heston_put_fourier(S, K, T, r, y, H_["v0"], H_["kappa"], H_["theta"], H_["xi"], 0.0) == \
        pytest.approx(8.1534, abs=5e-5)
    assert _heston_put_fourier(S, K, T, r, y, H_["v0"], H_["kappa"], H_["theta"], H_["xi"], -0.7) == \
        pytest.approx(7.9480, abs=5e-5)
    # xi -> 0: the Fourier price tends to Black-Scholes at volatility sqrt(theta)
    assert _heston_put_fourier(S, K, T, r, y, 0.04, 2.0, 0.04, 1e-3, 0.0) == pytest.approx(float(put(S, K, T, r, y, 0.2)),
                                                                                         abs=1e-4)


def test_cir_exact_moments_and_heston_identity():
    t = time_grid(1.0, 16)
    V = cir_exact(0.04, 2.0, 0.04, 0.3, t, 200_000, seed=5)
    assert V.min() >= 0
    assert V[:, -1].mean() == pytest.approx(0.04, abs=4 * V[:, -1].std() / np.sqrt(V.shape[0]))
    var_exact = 0.04 * 0.09 / 4 * (1 - np.exp(-4.0))                            # theta xi^2 (1 - e^{-2 kappa T}) / (2 kappa)
    assert V[:, -1].var() == pytest.approx(var_exact, rel=0.02)
    Q, I = heston_variance_integrals(V, t, 2.0, 0.04, 0.3)
    assert abs(I.mean()) < 4 * I.std() / np.sqrt(I.size)                        # E int sqrt(V) dW = 0
    assert I.var() == pytest.approx(Q.mean(), rel=0.02)                           # Ito isometry


def test_mixing_identity_on_a_grid():
    """Proposition st-mixing (iv): the conditional value is E[plain | W] exactly, path by path in law."""
    def lognormal_variance(dW, t):                       # V_{t_k} = 0.04 exp(0.8 W_{t_k} - 0.32 t_k), left points
        W = np.concatenate([np.zeros((dW.shape[0], 1)), np.cumsum(dW, axis=1)[:, :-1]], axis=1)
        return 0.04 * np.exp(0.8 * W - 0.32 * t[:-1])
    out = log_euler_mixing_check(S, K, T, r, y, -0.6, lognormal_variance, m=32, n=200_000, seed=8)
    d = out["plain"] - out["conditional"]
    assert abs(d.mean()) < 4 * d.std() / np.sqrt(d.size)
    assert out["conditional"].var() < out["plain"].var()
    # rho = +-1: sigma~ = 0 and the conditional value is the discounted intrinsic value of the forward
    St, sg = mixing_parameters(S, T, 1.0, np.array([0.04]), np.array([0.1]))
    assert sg[0] == 0
    assert bs_conditional("put", St, K, T, r, y, sg)[0] == pytest.approx(np.exp(-r * T) * max(K - St[0] * np.exp((r - y) * T), 0))
    # sigma~ > 0 matches the Black-Scholes call
    assert bs_conditional("call", [100.0], K, T, r, y, [0.2])[0] == pytest.approx(7.36429, abs=5e-6)


def test_mixing_calibration_against_fourier():
    """The quoted calibration: 40 runs of 5000 paths, seeds 777..816, both correlations."""
    H_ = HESTON_LAB
    refs = {0.0: 8.15342995262597, -0.7: _heston_put_fourier(S, K, T, r, y, H_["v0"], H_["kappa"], H_["theta"],
                                                              H_["xi"], -0.7)}
    zs = {0.0: [], -0.7: []}
    for seed in range(777, 817):
        s = heston_put_samples(S, K, T, r, y, (0.0, -0.7), H_["v0"], H_["kappa"], H_["theta"], H_["xi"], m=64,
                               n=5000, seed=seed)
        for rho in zs:
            c = s[rho]["conditional"]
            zs[rho].append((c.mean() - refs[rho]) / (c.std(ddof=1) / np.sqrt(c.size)))
    z0, z7 = np.array(zs[0.0]), np.array(zs[-0.7])
    assert (round(z0.mean(), 2), round(z0.std(ddof=1), 2)) == (0.16, 1.04)
    assert (round(z7.mean(), 2), round(z7.std(ddof=1), 2)) == (0.15, 1.01)


# ------------------------------------------------------------------ the laboratory table
def test_laboratory_table(lab):
    exact = {("call", "proportional 10"): 0.317, ("call", "neyman 10"): 0.153, ("call", "post 10"): 0.317,
             ("call", "proportional 100"): 0.090, ("otm", "proportional 10"): 0.925, ("otm", "proportional 100"): 0.398}
    for k, e in exact.items():
        assert round(lab[k]["exact"], 3) == e
    measured = {("call", "proportional 10"): 0.318, ("call", "neyman 10"): 0.152, ("call", "post 10"): 0.311,
                ("call", "proportional 100"): 0.085, ("otm", "proportional 10"): 0.939,
                ("otm", "proportional 100"): 0.386, ("asian", "terminal (bridge)"): 0.530,
                ("asian", "first increment"): 0.917, ("asian", "post-stratified W_T"): 0.530,
                ("asian", "average direction"): 0.108, ("asian", "LHS (increments)"): 0.54,
                ("asian", "LHS (bridge)"): 0.354, ("heston", "conditional 0.0"): 0.163,
                ("heston", "conditional -0.7"): 0.728}
    for k, m in measured.items():                       # full-size values, reproduced within sampling error
        assert lab[k]["ratio"] == pytest.approx(m, rel=0.1), k
    # efficiencies 1 / (ratio^2 cost) with the quoted costs (exact ratio where there is one)
    cost = {("call", "proportional 10"): 4.5, ("call", "neyman 10"): 5, ("call", "post 10"): 3,
            ("call", "proportional 100"): 4.5, ("otm", "proportional 10"): 4.5, ("otm", "proportional 100"): 4.5,
            ("asian", "terminal (bridge)"): 1.6, ("asian", "first increment"): 1.2, ("asian", "post-stratified W_T"): 1.5,
            ("asian", "average direction"): 1.5, ("asian", "LHS (increments)"): 4.2, ("asian", "LHS (bridge)"): 4.8,
            ("heston", "conditional 0.0"): 1.0, ("heston", "conditional -0.7"): 1.0}
    quoted = {("call", "proportional 10"): 2.2, ("call", "neyman 10"): 8.6, ("call", "post 10"): 3.3,
              ("call", "proportional 100"): 27, ("otm", "proportional 10"): 0.26, ("otm", "proportional 100"): 1.4,
              ("asian", "terminal (bridge)"): 2.2, ("asian", "first increment"): 1.0,
              ("asian", "post-stratified W_T"): 2.4, ("asian", "average direction"): 57,
              ("asian", "LHS (increments)"): 0.8, ("asian", "LHS (bridge)"): 1.7,
              ("heston", "conditional 0.0"): 37, ("heston", "conditional -0.7"): 1.9}
    for k, e in quoted.items():
        rho = lab[k]["exact"] if k in exact else measured[k]
        assert 1 / (rho**2 * cost[k]) == pytest.approx(e, rel=0.05), k
    # Example st-asian: terminal stratification removes about 72% of the variance
    assert 1 - 0.530**2 == pytest.approx(0.72, abs=0.005)
    # every estimate is consistent with its exact value
    assert abs(lab[("call", "proportional 100")]["mean"] - 7.36429) < 4 * lab[("call", "proportional 100")]["se"]
    assert abs(lab[("otm", "proportional 100")]["mean"] - 0.35963) < 4 * lab[("otm", "proportional 100")]["se"]
    assert abs(lab[("asian", "average direction")]["mean"] - 4.5612893) < 4 * lab[("asian", "average direction")]["se"]


def test_post_stratified_exact_ratio():
    """Proposition st-post (iii) at n = 2 x 10^5 gives 0.3174, as proportional allocation does."""
    p, m, v = stratum_moments(F, normal_edges(10), kinks=(ZS,))
    n = 200_000
    post = np.sqrt((np.sum(p * v) / n + np.sum((1 - p) * v) / n**2) / (158.00619 / n))
    prop = np.sqrt(np.sum(p * v) / 158.00619)
    assert round(post, 4) == round(prop, 4) == 0.3174 and post > prop
