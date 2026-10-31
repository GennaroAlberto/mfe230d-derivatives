"""Checks of the Chapter 11 code against the results proved in the notes and the numbers quoted there."""
import math

import numpy as np
import pytest
from scipy.stats import qmc as scipy_qmc

from mfe230d import bs, qmc
from mfe230d.montecarlo import sine_product
from mfe230d.rng import ks_uniform

P = tuple(bs.TEST_CASE[k] for k in ("S", "K", "r", "y", "sigma"))


# ------------------------------------------------------------------ van der Corput and Halton

def test_radical_inverse_values():
    assert qmc.van_der_corput(8, 2).tolist() == [0, .5, .25, .75, .125, .625, .375, .875]
    assert np.allclose(qmc.van_der_corput(9, 3), [0, 1/3, 2/3, 1/9, 4/9, 7/9, 2/9, 5/9, 8/9])
    i = np.arange(500)
    for b in (2, 3, 5, 7):
        K = 6
        assert np.allclose(qmc.radical_inverse_int(i, b, K) / b**K, qmc.radical_inverse(i, b), atol=b**-K)


@pytest.mark.parametrize("b", [2, 3, 5])
def test_any_b_to_the_k_consecutive_points_fill_the_cells(b):
    """Proposition (van der Corput and Halton), part (i), checked with exact integer cells."""
    K = 7 if b == 2 else 4
    for k in range(K + 1):
        for j in (0, 1, 5, 17, 123):
            N = qmc.radical_inverse_int(np.arange(j, j + b**k), b, K)
            cells = N // b ** (K - k)
            assert np.array_equal(np.sort(cells), np.arange(b**k))


def test_van_der_corput_discrepancy_bound_and_powers():
    for b, nmax in ((2, 4096), (3, 729)):
        x = qmc.van_der_corput(nmax, b)
        nD = np.array([n * qmc.star_discrepancy(x[:n]) for n in range(1, nmax + 1)])
        bound = np.array([n * qmc.van_der_corput_bound(n, b) for n in range(1, nmax + 1)])
        assert np.all(nD <= bound + 1e-12)
        for k in range(0, int(round(math.log(nmax, b))) + 1):
            assert nD[b**k - 1] == pytest.approx(1.0)
        if b == 2:   # the numbers quoted after the proposition
            assert nD.max() == pytest.approx(4.7778, abs=1e-4) and nD.argmax() + 1 == 3413
            assert bound[3412] == pytest.approx(12.0)


def test_halton_chinese_remainder_blocks():
    """Part (iii): any prod b_j^{k_j} consecutive Halton points fill the boxes once."""
    for bases, ks in (((2, 3), (2, 1)), ((2, 3, 5), (1, 1, 1)), ((3, 5), (2, 1))):
        B = int(np.prod([b**k for b, k in zip(bases, ks)]))
        for j in (0, 7, 100):
            idx = np.arange(j, j + B)
            cell = np.zeros(B, dtype=np.int64)
            for b, k in zip(bases, ks):
                cell = cell * b**k + qmc.radical_inverse_int(idx, b, k)
            assert np.array_equal(np.sort(cell), np.arange(B))
    assert qmc.primes(6) == [2, 3, 5, 7, 11, 13]
    assert qmc.primes(16)[14:16] == [47, 53]


# ------------------------------------------------------------------ discrepancy

def test_one_dimensional_discrepancy_is_kolmogorov_smirnov():
    u = np.random.default_rng(3).random(1000)
    assert qmc.star_discrepancy(u) == pytest.approx(ks_uniform(u)[0], abs=1e-15)
    i = np.arange(1, 51)
    assert qmc.star_discrepancy((2 * i - 1) / 100) == pytest.approx(1 / 100)


def test_two_dimensional_discrepancy_against_enumeration():
    rng = np.random.default_rng(4)
    for n in (5, 12, 30):
        Pts = rng.random((n, 2))
        fast = qmc.star_discrepancy(Pts)
        # brute force over all candidate corners (point coordinates and 1), open and closed boxes
        best = 0.0
        g1, g2 = np.append(Pts[:, 0], 1.0), np.append(Pts[:, 1], 1.0)
        for a1 in g1:
            for a2 in g2:
                vol = a1 * a2
                best = max(best, vol - np.mean((Pts[:, 0] < a1) & (Pts[:, 1] < a2)),
                           np.mean((Pts[:, 0] <= a1) & (Pts[:, 1] <= a2)) - vol)
        assert fast == pytest.approx(best)
        # and no anchored box on a fine grid does better
        fine = np.linspace(0, 1, 201)
        grid = max(abs(qmc.local_discrepancy(Pts, [a, b])) for a in fine for b in fine)
        assert grid <= fast + 1e-12
    assert qmc.star_discrepancy(np.zeros((4, 0))) == 0.0
    P3 = rng.random((6, 3))
    assert qmc.star_discrepancy(P3) >= max(abs(qmc.local_discrepancy(P3, a)) for a in rng.random((200, 3)))


def test_discrepancies_quoted_in_the_notes():
    n = 2048
    assert qmc.star_discrepancy(np.random.default_rng(1).random((n, 2))) == pytest.approx(0.0200, abs=5e-5)
    assert qmc.star_discrepancy(qmc.halton(n, 2)) == pytest.approx(0.0037, abs=5e-5)
    assert qmc.star_discrepancy(qmc.sobol(n, 2)) == pytest.approx(0.0024, abs=5e-5)


# ------------------------------------------------------------------ Sobol'

def test_sobol_matches_scipy_in_gray_code_order():
    n, d = 2**11, qmc.MAX_SOBOL_DIM
    X = qmc.sobol(n, d)
    S = scipy_qmc.Sobol(d, scramble=False, bits=32).random(n)
    i = np.arange(n)
    assert np.array_equal(S, X[i ^ (i >> 1)])


def test_direction_numbers():
    assert qmc.direction_integers(2, 8) == [1, 3, 5, 15, 17, 51, 85, 255]
    for j in (1, 2, 5, 40, 128):
        m = qmc.direction_integers(j, 32)
        assert all(mk % 2 == 1 and mk < 2 ** (k + 1) for k, mk in enumerate(m))
    assert qmc.sobol_degrees(8) == [1, 1, 2, 3, 3, 4, 4, 5]
    assert np.array_equal(qmc.sobol(16, 1)[:, 0], qmc.van_der_corput(16, 2))


def test_generator_matrices_and_t_values():
    for C in qmc.generator_matrices(6, 12):
        assert np.array_equal(C, np.triu(C)) and np.all(np.diag(C) == 1)
    # numbers quoted in the notes: first 2^10 points, first 2, 3, 5, 8 coordinates
    t = {d: qmc.t_value(qmc.generator_matrices(d, 10)) for d in (2, 3, 5, 8)}
    assert t == {2: 0, 3: 1, 5: 3, 8: 5}
    for d in (2, 3, 5, 8):
        assert t[d] <= sum(e - 1 for e in qmc.sobol_degrees(d))
    for m in range(1, 15):
        assert qmc.t_value(qmc.generator_matrices(2, m)) == 0


def test_rank_criterion_agrees_with_counting():
    for d, m in ((2, 8), (3, 8), (4, 7)):
        t = qmc.t_value(qmc.generator_matrices(d, m))
        X = qmc.sobol_integers(2**m, d)
        assert qmc.is_net(X, t)
        if t > 0:
            assert not qmc.is_net(X, t - 1)
        # a later aligned block is a net too (proposition, part (ii))
        assert qmc.is_net(qmc.sobol_integers(2**m, d, start=3 * 2**m), t)


def test_shift_by_half_a_cell_keeps_the_net():
    m, d = 8, 3
    t = qmc.t_value(qmc.generator_matrices(d, m))
    X = qmc.sobol_integers(2**m, d)
    Y = X + np.uint64(2 ** (32 - m - 1))
    assert qmc.is_net(Y, t)
    assert (Y.astype(float) / 2**32).min() > 0 and (Y.astype(float) / 2**32).max() < 1


def test_origin_is_refused_by_the_normal_map():
    with pytest.raises(ValueError):
        qmc.to_normals(qmc.sobol(4, 2))
    Z = qmc.to_normals(qmc.sobol(4, 2, start=1))
    assert np.all(np.isfinite(Z))


# ------------------------------------------------------------------ randomisation

def test_randomisations_preserve_nets_and_make_points_uniform():
    m, d = 8, 3
    X = qmc.sobol_integers(2**m, d)
    t = qmc.t_value(qmc.generator_matrices(d, m))
    for seed in range(3):
        for Y in (qmc.owen_scramble(X, seed=seed), qmc.digital_shift(X, seed=seed)):
            assert qmc.is_net((Y * 2**32).astype(np.uint64), t)
    # one fixed point under many independent scrambles (as 400 columns) is uniform
    x = qmc.sobol_integers(16, 1)
    Y = qmc.owen_scramble(np.repeat(x, 400, axis=1), seed=9)
    assert ks_uniform(Y[5])[1] > 1e-3 and ks_uniform(Y[0])[1] > 1e-3


def test_scrambled_one_dimensional_points_are_stratified():
    N = 2**6
    X = np.repeat(qmc.sobol_integers(N, 1), 20, axis=1)
    Y = qmc.owen_scramble(X, seed=2)
    assert np.all(np.sort(np.floor(Y * N), axis=0) == np.arange(N)[:, None])
    for bits in (10, 32):                      # works whatever the resolution of the integers
        Z = qmc.owen_scramble(qmc.sobol_integers(N, 2, bits=bits), bits=bits, seed=1)
        assert np.all(np.sort(np.floor(Z * N), axis=0) == np.arange(N)[:, None])


def test_variance_examples_of_part_ii():
    n, R = 16, 4000
    # (a) random shift of i/n with cos(2 pi n y): every replicate equals cos(2 pi n U)
    x = (np.arange(n) / n)[:, None]
    est = np.array([np.mean(np.cos(2 * np.pi * n * qmc.random_shift(x, seed=s))) for s in range(R)])
    assert est.var() == pytest.approx(0.5, rel=0.06)
    # (b) digital shift and the sign of digit m + 1: every replicate is +1 or -1
    X = (np.arange(n, dtype=np.uint64) << np.uint64(32 - 4))[:, None]   # i / n, n = 2^4
    for s in range(50):
        y = qmc.digital_shift(X, seed=s)
        sign = np.where(np.floor(y * 2 * n) % 2 == 0, 1.0, -1.0)
        assert abs(sign.mean()) == 1.0
        # the nested scramble does not have this defect: the signs are independent across points
    signs = np.array([np.where(np.floor(qmc.owen_scramble(X, seed=s) * 2 * n) % 2 == 0, 1.0, -1.0).mean()
                      for s in range(400)])
    assert signs.var() == pytest.approx(1 / n, rel=0.2)


def test_rqmc_estimate_is_unbiased_with_valid_error():
    f = lambda U: np.prod(1 + (U - 0.5), axis=1) * U[:, 0]   # noqa: E731  integral 1/2 + 1/12
    exact = 0.5 + 1 / 12
    for method, seq in (("owen", "sobol"), ("digital", "sobol"), ("shift", "sobol"), ("shift", "halton")):
        e = qmc.rqmc_estimate(f, 256, 4, R=30, method=method, sequence=seq, seed=1)
        assert abs(e["mean"] - exact) < 4 * e["se"] + 1e-12
    e = qmc.rqmc_estimate(lambda U: np.ones(len(U)), 100, 3, R=5, seed=0)   # constant integrand
    assert e["mean"] == 1.0 and e["se"] == 0.0 and e["lo"] == e["hi"] == 1.0
    mc = qmc.rqmc_estimate(f, 256, 4, R=30, sequence="random", seed=1)
    ow = qmc.rqmc_estimate(f, 256, 4, R=30, method="owen", seed=1)
    assert ow["se"] < mc["se"] / 5


def test_n_not_a_power_of_two():
    X = qmc.sobol_integers(1000, 5)
    assert [qmc._distinct_prefix_length(X[:, j], 32) for j in range(5)] == [10] * 5
    Y = qmc.owen_scramble(X, seed=3)
    assert Y.shape == (1000, 5) and Y.min() > 0 and Y.max() < 1


def test_convergence_numbers_quoted_in_the_notes():
    """Same seeds and replicate counts as code/figures/ch11_qmc.py."""
    ns = 2 ** np.arange(4, 17)
    d = 5
    eH = abs(np.cumsum(sine_product(qmc.halton(ns[-1], d)))[-1] / ns[-1] - 1)
    eS = abs(np.cumsum(sine_product(qmc.sobol(ns[-1], d)))[-1] / ns[-1] - 1)
    assert eH == pytest.approx(2.8e-5, rel=0.03) and eS == pytest.approx(1.9e-4, rel=0.03)
    assert math.sqrt((np.pi**2 / 8) ** d - 1) / 2**8 == pytest.approx(5.3e-3, rel=0.01)
    E = qmc.rqmc_nested(sine_product, ns, d, R=128, seed=7)
    rmse = np.sqrt(np.mean((E - 1) ** 2, axis=0))
    assert rmse[-1] == pytest.approx(8.8e-5, rel=0.01)
    sel = ns >= 2**8
    slope = lambda r: np.polyfit(np.log(ns[sel]), np.log(r[sel]), 1)[0]   # noqa: E731
    assert slope(rmse) == pytest.approx(-0.99, abs=0.005)
    chunks = [slope(np.sqrt(np.mean((E[c:c + 32] - 1) ** 2, axis=0))) for c in range(0, 128, 32)]
    assert min(chunks) == pytest.approx(-1.03, abs=0.005) and max(chunks) == pytest.approx(-0.95, abs=0.005)
    nall = np.arange(1, 2**12 + 1)
    E2 = qmc.rqmc_nested(sine_product, nall, 2, R=64, seed=8)
    nr = nall * np.sqrt(np.mean((E2 - 1) ** 2, axis=0))
    assert nr[2**8 - 1] == pytest.approx(0.36, abs=0.005) and nr[-1] == pytest.approx(0.10, abs=0.005)
    assert np.median(nr[2**11:2**12 - 1]) == pytest.approx(1.2, abs=0.05)


# ------------------------------------------------------------------ effective dimension

def test_geometric_asian_course_value():
    assert qmc.geometric_asian_call(qmc.asian_grid(12), *P) == pytest.approx(4.40286, abs=5e-6)
    # m = 1 is the Black--Scholes call
    assert qmc.geometric_asian_call(qmc.asian_grid(1), *P) == pytest.approx(7.36429, abs=5e-6)


def test_pick_freeze_identity_for_a_linear_payoff():
    m = 8
    w = np.arange(1, m + 1, dtype=float)
    w /= np.linalg.norm(w)
    est = qmc.pick_freeze_shares(lambda Z: Z @ w, m, [1, 3, 8], n=2**15, seed=0)
    assert np.allclose(est, np.cumsum(w**2)[[0, 2, 7]], atol=0.02)


def test_average_direction_and_shares_table():
    t = qmc.asian_grid(64)
    w2 = {me: qmc.average_direction(t, me)[0] ** 2 for me in ("increments", "bridge", "pca")}
    m = 64
    assert w2["increments"] == pytest.approx(6 * m / ((m + 1) * (2 * m + 1)))
    assert (round(w2["increments"], 3), round(w2["bridge"], 3), round(w2["pca"], 3)) == (0.046, 0.756, 0.986)
    share = lambda c: qmc.lognormal_call_share(c, t, *P)   # noqa: E731
    assert share([0.0])[0] == pytest.approx(0.0, abs=1e-12) and share([1.0])[0] == 1.0
    c = np.linspace(0.05, 0.95, 7)
    assert np.all(share(c) <= c)                              # kappa_g(c) <= c
    table = {"increments": ([0.032, 0.063, 0.127, 0.253], 53, 0.684),
             "bridge": ([0.685, 0.914, 0.978, 0.994], 7, 0.860),
             "pca": ([0.979, 0.997, 1.000, 1.000], 2, 0.989)}
    for me, (vals, dT, add) in table.items():
        R = qmc.ridge_shares(qmc.average_direction(t, me), share)
        assert np.round(R["truncation"][[0, 1, 3, 7]], 3).tolist() == vals
        assert int(np.argmax(R["truncation"] >= 0.99)) + 1 == dT
        assert round(R["additive"], 3) == add


def test_exact_shares_agree_with_pick_freeze_for_the_geometric_call():
    t = qmc.asian_grid(16)
    share = lambda c: qmc.lognormal_call_share(c, t, *P)   # noqa: E731
    for me in ("increments", "bridge"):
        R = qmc.ridge_shares(qmc.average_direction(t, me), share, ks=np.array([1, 4]))
        fG = lambda Z, me=me: qmc.asian_call_payoffs(Z, t, *P, method=me)[1]   # noqa: E731
        est = qmc.pick_freeze_shares(fG, 16, [1, 4], n=2**15, seed=1)
        assert np.allclose(est, R["truncation"], atol=0.03)


def test_course_example_and_asian_laboratory():
    """Example "Asian options by randomised QMC": the numbers quoted in the notes."""
    E = qmc.asian_rqmc(12, [2**14], 16, 2026, *P, geometric=True)[:, 0, :]
    mean, se = E.mean(axis=0), E.std(axis=0, ddof=1) / 4
    assert mean[2] == pytest.approx(4.56133, abs=5e-6) and se[2] == pytest.approx(0.00011, abs=5e-6)
    assert mean[1] == pytest.approx(4.56111, abs=5e-6) and se[1] == pytest.approx(0.00038, abs=5e-6)
    assert mean[5] == pytest.approx(4.40291, abs=5e-6) and se[5] == pytest.approx(0.00011, abs=5e-6)
    assert abs(mean[5] - 4.40286) < 3 * se[5] and abs(mean[2] - 4.56058) < 1.5 * 7.5e-4
    # m = 64 (the factors quoted in the notes use 64 scrambles and are printed by the figure script;
    # here a smaller run checks the ranking and the estimate of sigma from the second moment)
    E = qmc.asian_rqmc(64, [2**12], 8, 11, *P, square=True)[:, 0, :]
    sigma = math.sqrt(E[:, 3].mean() - E[:, 2].mean() ** 2)
    assert sigma == pytest.approx(7.12, abs=0.02)
    v = E[:, :3].var(axis=0, ddof=1)
    assert v[0] > v[1] > v[2] and sigma**2 / 2**12 / v[0] > 5
