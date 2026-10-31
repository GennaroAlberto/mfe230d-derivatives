"""Checks of the Chapter 14 code against the results proved in the notes and the numbers they quote.

The full reproduction of Table 1 of Longstaff and Schwartz (2001) takes about 30 seconds
and the nested Andersen--Broadie run with 500 inner paths about 10 seconds; they run only
when the environment variable MFE230D_SLOW is set. The default run checks three rows.
"""
import os

import numpy as np
import pytest

from mfe230d import bs
from mfe230d.american import (
    LS_TABLE1,
    american_binomial,
    andersen_broadie,
    bermudan_grid,
    bermudan_reference,
    bias_study,
    binomial,
    design_condition,
    discounted_payoffs,
    dual_samples,
    duality_study,
    european_martingale,
    laguerre_basis,
    ls_table1_lsm,
    ls_table1_references,
    lsm_fit,
    lsm_price,
    mean_se,
    policy_cashflows,
    policy_grid,
    policy_martingale,
    poly_basis,
    put_payoff,
    simulate,
)

P = bs.TEST_CASE
S, K, T, r, y, SIG = (P[k] for k in ("S", "K", "T", "r", "y", "sigma"))
SLOW = pytest.mark.skipif(not os.environ.get("MFE230D_SLOW"), reason="slow: set MFE230D_SLOW=1")


# ------------------------------------------------------------------ bases
def test_laguerre_basis_is_the_longstaff_schwartz_basis():
    x = np.array([0.3, 0.9, 1.4])
    w = np.exp(-x / 2)
    expected = np.column_stack([np.ones(3), w, w * (1 - x), w * (1 - 2 * x + x**2 / 2)])
    assert np.allclose(laguerre_basis(3, scale=40.0)(40.0 * x), expected, rtol=1e-14)
    assert np.allclose(poly_basis(2, scale=2.0)(np.array([2.0, 4.0])), [[1, 1, 1], [1, 2, 4]])


def test_scaling_the_basis_improves_conditioning():
    # The numbers quoted in the notes (in-the-money prices at t = 0.5, LS case S0 = 36).
    s = simulate(36.0, 1.0, 50, 0.06, 0.0, 0.2, 100_000, seed=0)[:, 25]
    s = s[s < 40.0]
    assert design_condition(poly_basis(3, 1.0), s) == pytest.approx(2.552e7, rel=1e-3)
    assert design_condition(poly_basis(3, 40.0), s) == pytest.approx(5914.0, rel=1e-3)
    assert design_condition(laguerre_basis(3, 1.0), s) == pytest.approx(1.115e8, rel=1e-3)
    assert design_condition(laguerre_basis(3, 40.0), s) == pytest.approx(1.449e5, rel=1e-3)
    assert design_condition(poly_basis(8, 40.0), s) == pytest.approx(2.65e9, rel=1e-2)


# ------------------------------------------------------------------ reference values
def test_grid_with_one_date_is_the_european_option():
    for kind, price in (("put", bs.put), ("call", bs.call)):
        exact = float(price(S, K, T, r, y, SIG))
        assert bermudan_grid(S, K, T, r, y, SIG, 1, kind, h=5e-4) == pytest.approx(exact, abs=5e-6)
        assert bermudan_reference(S, K, T, r, y, SIG, 1, kind) == pytest.approx(exact, abs=1e-8)


def test_grid_agrees_with_a_bermudan_tree():
    # Two independent methods: 400 tree steps between exercise dates against the grid.
    tree = binomial(40.0, 40.0, 1.0, 0.06, 0.0, 0.2, 20_000, "put", 50)
    grid = bermudan_reference(40.0, 40.0, 1.0, 0.06, 0.0, 0.2, 50, h=2e-3)
    assert grid == pytest.approx(2.3140678, abs=1e-7)
    finer = bermudan_reference(40.0, 40.0, 1.0, 0.06, 0.0, 0.2, 50, h=5e-4)
    assert finer == pytest.approx(grid, abs=1e-6)
    assert tree == pytest.approx(grid, abs=2e-5)


def test_bermudan_prices_increase_with_dates_towards_the_american_price():
    K_, r_, s_ = 40.0, 0.06, 0.2
    euro = float(bs.put(40.0, K_, 1.0, r_, 0.0, s_))
    b = [bermudan_reference(40.0, K_, 1.0, r_, 0.0, s_, m, h=2e-3) for m in (1, 2, 10, 50, 250)]
    am = american_binomial(40.0, K_, 1.0, r_, 0.0, s_, n=2000)
    assert b[0] == pytest.approx(euro, abs=1e-8)
    assert np.all(np.diff(b) > 0) and b[-1] < am
    assert am == pytest.approx(2.319575, abs=3e-5)       # converged BBSR value (n = 16000)
    assert np.round(b, 4).tolist() == [2.0664, 2.1991, 2.2930, 2.3141, 2.3185] and round(am, 4) == 2.3196
    assert np.round(am - np.array(b[2:]), 4).tolist() == [0.0266, 0.0055, 0.0011]


def test_american_call_without_dividends_is_european():
    # y = 0, r > 0: the discounted call payoff is a submartingale (remark on degenerate cases).
    euro_tree = binomial(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, 500, "call", "european")
    am_tree = binomial(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, 500, "call", "american")
    assert am_tree == pytest.approx(euro_tree, abs=1e-12)
    g = bermudan_grid(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, 50, "call", h=1e-3)
    assert g == pytest.approx(float(bs.call(100.0, 100.0, 1.0, 0.05, 0.0, 0.3)), abs=2e-5)


def test_test_case_values():
    assert float(bs.put(S, K, T, r, y, SIG)) == pytest.approx(8.349406, abs=1e-6)
    assert bermudan_reference(S, K, T, r, y, SIG, 50, "put", h=2e-3) == pytest.approx(8.349408, abs=1e-6)
    assert american_binomial(S, K, T, r, y, SIG, 2000, "put") == pytest.approx(8.349408, abs=1e-6)
    assert bermudan_reference(S, K, T, r, y, SIG, 50, "call", h=2e-3) == pytest.approx(7.456678, abs=1e-6)
    assert american_binomial(S, K, T, r, y, SIG, 2000, "call") == pytest.approx(7.459750, abs=1e-6)
    # quoted to four or six decimals in the text: 4000 steps (converged value 7.459733)
    assert american_binomial(S, K, T, r, y, SIG, 4000, "call") == pytest.approx(7.459739, abs=1e-6)
    assert round(american_binomial(S, K, T, r, y, SIG, 4000, "put"), 6) == 8.349408
    assert round(american_binomial(40.0, 40.0, 1.0, 0.06, 0.0, 0.2, 4000), 4) == 2.3196
    # Remark (b): the exercise region lies below K (1 - e^{-r dt}) / (1 - e^{-y dt})
    dt = T / 50
    sbar = K * (1 - np.exp(-r * dt)) / (1 - np.exp(-y * dt))
    assert sbar == pytest.approx(50.005, abs=1e-3)
    b = bermudan_grid(S, K, T, r, y, SIG, 50, "put", h=1e-3, details=True)["boundary"]
    assert b[1] == pytest.approx(44.94, abs=0.01) and b[49] == pytest.approx(50.005, abs=1e-3)
    assert np.all(b[1:50] <= sbar + 1e-3)
    assert np.nanmin(b[1:50]) == b[1] and np.nanmax(b[1:50]) == b[49]


def test_ls_table_references():
    rows = ls_table1_references(h=2e-3, n_tree=2000)
    berm = [round(row["bermudan"], 3) for row in rows]
    amer = [round(row["american"], 3) for row in rows]
    euro = [round(row["european"], 3) for row in rows]
    assert berm == [4.478, 4.840, 7.101, 8.507, 3.250, 3.745, 6.148, 7.668, 2.314, 2.885,
                    5.312, 6.917, 1.617, 2.212, 4.582, 6.244, 1.110, 1.690, 3.948, 5.641]
    assert amer == [4.487, 4.848, 7.109, 8.514, 3.257, 3.751, 6.155, 7.675, 2.320, 2.890,
                    5.318, 6.923, 1.621, 2.217, 4.588, 6.250, 1.113, 1.693, 3.953, 5.647]
    assert euro == [row[3] for row in LS_TABLE1]                 # LS's closed-form column
    fd = np.array([row[4] for row in LS_TABLE1])
    b = np.array([row["bermudan"] for row in rows])
    a = np.array([row["american"] for row in rows])
    sig2 = np.array([row[1] == 0.2 for row in LS_TABLE1])
    assert np.abs(fd - b)[sig2].max() < 5e-4                     # LS's "FD" values are Bermudan
    long4 = np.array([row[1] == 0.4 and row[2] == 2 for row in LS_TABLE1])
    assert np.abs(fd - b)[~long4].max() < 5e-4                   # 15 rows with sigma = 20% or T = 1
    assert np.all(fd[long4] >= b[long4] - 5e-4) and np.all(fd[long4] <= a[long4] + 5e-4)
    assert np.abs(fd - b).max() == pytest.approx(0.0058, abs=1e-4)
    assert (a - fd).max() == pytest.approx(0.0087, abs=1e-4)
    assert round((a - b).min(), 3) == 0.003 and round((a - b).max(), 3) == 0.009


# ------------------------------------------------------------------ Longstaff--Schwartz
@pytest.fixture(scope="module")
def ls_rows():
    """Rows 0, 8 and 19 of the LS table with the setup of ``ls_table1_lsm`` (same seeds)."""
    out = {}
    for i in (0, 8, 19):
        S0, sig, T_ = LS_TABLE1[i][:3]
        out[i] = lsm_price(S0, 40.0, T_, 0.06, 0.0, sig, 50 * T_, 100_000, laguerre_basis(3, 40.0), "put",
                           seed=[2001, i], antithetic=True, control=False)
    return out


def test_ls_table_rows(ls_rows):
    want = {0: (4.4816, 0.0061, 4.4704, 0.0061), 8: (2.3091, 0.0053, 2.3103, 0.0053),
            19: (5.6341, 0.0116, 5.6438, 0.0115)}
    for i, res in ls_rows.items():
        got = (res["in_sample"], res["in_se"], res["out_sample"], res["out_se"])
        assert got == pytest.approx(want[i], abs=6e-5)


@SLOW
def test_ls_table_all_rows():
    got = ls_table1_lsm()
    ins = [round(row["in_sample"], 3) for row in got]
    outs = [round(row["out_sample"], 3) for row in got]
    assert ins == [4.482, 4.839, 7.104, 8.517, 3.247, 3.745, 6.138, 7.675, 2.309, 2.879,
                   5.309, 6.906, 1.624, 2.207, 4.578, 6.231, 1.111, 1.690, 3.950, 5.634]
    assert outs == [4.470, 4.841, 7.088, 8.507, 3.253, 3.751, 6.145, 7.674, 2.310, 2.877,
                    5.305, 6.919, 1.607, 2.219, 4.559, 6.254, 1.112, 1.690, 3.931, 5.644]
    se = [round(row["out_se"], 3) for row in got]
    assert min(se) == 0.005 and max(se) == 0.012
    berm = np.array([row["bermudan"] for row in ls_table1_references(h=2e-3, n_tree=2000)])
    z_in = (np.array([row["in_sample"] for row in got]) - berm) / np.array([row["in_se"] for row in got])
    z_out = (np.array([row["out_sample"] for row in got]) - berm) / np.array([row["out_se"] for row in got])
    assert np.abs(z_in).max() < 1.3 and 2.3 < np.abs(z_out).max() < 2.4
    assert (round(z_in.mean(), 1), round(z_out.mean(), 1)) == (-0.2, -0.3)


@SLOW
def test_in_sample_bias_at_ten_thousand_paths():
    res = bias_study(degree=2, n_train=10_000, reps=20)
    assert (res["in_sample"] - res["true"]).mean() == pytest.approx(-0.011, abs=1e-3)


def test_naive_and_paired_standard_errors(ls_rows):
    # LS's standard errors match the naive formula; antithetic pairs are negatively correlated.
    for i, naive_want in ((0, 0.0092), (8, 0.0086), (19, 0.0206)):
        Y = ls_rows[i]["fit"]["samples"]
        assert Y.std(ddof=1) / np.sqrt(Y.size) == pytest.approx(naive_want, abs=5e-5)
        h = Y.size // 2
        assert -0.70 < np.corrcoef(Y[:h], Y[h:])[0, 1] < -0.55
        assert mean_se(Y, antithetic=True)[1] == pytest.approx(ls_rows[i]["in_se"])


def test_interpolating_regression_is_perfect_foresight():
    # Prop. (in-sample bias): when the basis interpolates the in-the-money points, the
    # in-sample cash flow of every path is max_{k>=1} Z_k.
    paths = simulate(40.0, 1.0, 4, 0.06, 0.0, 0.3, 6, seed=3)
    g = put_payoff(40.0)
    fit = lsm_fit(paths, g, poly_basis(5, scale=40.0), 0.06, 1.0, min_points=1)
    Z = discounted_payoffs(paths, g, 0.06, 1.0)
    assert np.allclose(fit["samples"], Z[:, 1:].max(axis=1))
    assert fit["value"] == pytest.approx(max(Z[0, 0], Z[:, 1:].max(axis=1).mean()))


def test_out_of_sample_estimator_is_unbiased_for_the_policy_value_and_below_the_price():
    K_, r_, s_ = 40.0, 0.06, 0.2
    g = put_payoff(K_)
    basis = poly_basis(2, scale=K_)
    true = bermudan_reference(36.0, K_, 1.0, r_, 0.0, s_, 10, h=2e-3)
    for seed in range(3):
        fit = lsm_fit(simulate(36.0, 1.0, 10, r_, 0.0, s_, 300, seed=[5, seed]), g, basis, r_, 1.0)
        value = policy_grid(fit, basis, 36.0, K_, 1.0, r_, 0.0, s_, h=1e-3)["price"]
        assert value <= true + 1e-6
        Y, _ = policy_cashflows(simulate(36.0, 1.0, 10, r_, 0.0, s_, 200_000, seed=[6, seed]), fit, g, basis)
        mean, se = mean_se(Y)
        assert abs(mean - value) < 4 * se


def test_test_case_lsm_and_european_control():
    out = {}
    for kind in ("put", "call"):
        res = lsm_price(S, K, T, r, y, SIG, 50, 100_000, laguerre_basis(3, K), kind, seed=1, control=True)
        pol = policy_grid(res["fit"], laguerre_basis(3, K), S, K, T, r, y, SIG, kind, h=1e-3)["price"]
        out[kind] = (res, pol)
    res, pol = out["put"]
    assert (res["in_sample"], res["out_sample"], res["out_se"]) == pytest.approx((8.2729, 8.2766, 0.0308), abs=1e-4)
    assert (res["cv"], res["cv_se"], res["rho"]) == pytest.approx((8.3024, 0.0110, 0.9345), abs=1e-4)
    assert (res["cv_se"] / res["out_se"]) ** 2 == pytest.approx(0.127, abs=1e-3)
    assert pol == pytest.approx(8.3079, abs=1e-4)
    assert np.mean(res["tau"] < 50) == pytest.approx(0.350, abs=1e-3)
    test = simulate(S, T, 50, r, y, SIG, 100_000, seed=[1, 1])          # the evaluation paths
    early = res["tau"] < 50
    Sx = test[np.flatnonzero(early), res["tau"][early]]
    assert np.quantile(Sx, [0.05, 0.5, 0.95]) == pytest.approx([75.9, 82.6, 87.7], abs=0.1)
    assert np.quantile(res["tau"][early], 0.05) >= 37                   # last quarter of the year
    t_ex = res["tau"][early] / 50.0
    tv = bs.put(Sx, K, T - t_ex, r, y, SIG) - (K - Sx)
    assert np.quantile(tv, [0.05, 0.95]) == pytest.approx([0.0156, 0.189], abs=1e-3)
    res, pol = out["call"]
    assert (res["in_sample"], res["out_sample"], res["out_se"]) == pytest.approx((7.4949, 7.4237, 0.0330), abs=1e-4)
    assert pol == pytest.approx(7.4462, abs=1e-4)
    ls = lsm_price(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 50, 100_000, laguerre_basis(3, 40.0), "put", seed=3)
    assert (ls["out_sample"], ls["out_se"], ls["cv"], ls["cv_se"], ls["rho"]) == pytest.approx(
        (4.4855, 0.0091, 4.4812, 0.0076, 0.562), abs=1e-3)
    assert (ls["cv_se"] / ls["out_se"]) ** 2 == pytest.approx(0.68, abs=0.01)


def test_bias_study_at_one_hundred_paths():
    res = bias_study(degree=2, n_train=100, reps=150)
    assert res["true"] == pytest.approx(2.314068, abs=1e-6)
    a, b = res["in_sample"] - res["true"], res["policy"] - res["true"]
    assert (a.mean(), b.mean()) == pytest.approx((0.280, -0.157), abs=1e-3)
    assert b.max() < 0                                   # every policy is worth less than the price


# ------------------------------------------------------------------ duality
def test_european_martingale():
    paths = simulate(36.0, 1.0, 10, 0.06, 0.0, 0.2, 200_000, seed=11)
    M = european_martingale(paths, 40.0, 1.0, 0.06, 0.0, 0.2, "put")
    assert np.all(M[:, 0] == 0)
    m, se = M.mean(axis=0), M.std(axis=0, ddof=1) / np.sqrt(M.shape[0])
    assert np.all(np.abs(m[1:]) < 4 * se[1:])


def test_doob_martingale_has_almost_zero_variance_and_bounds():
    res = duality_study(nested=((400, 100),))
    sm = res["summary"]
    assert res["true"] == pytest.approx(4.477811, abs=1e-6)
    assert res["policy"] == pytest.approx(4.474417, abs=1e-6)
    assert res["out_sample"] == pytest.approx((4.4793, 0.0094), abs=1e-4)
    assert sm["zero"] == pytest.approx((7.779, 0.010), abs=1e-3)              # 10^5 paths
    assert sm["european"] == pytest.approx((4.7322, 0.0014), abs=1e-4)
    assert round(sm["european"][0] / res["true"] - 1, 3) == 0.057
    assert sm["ab_exact"] == pytest.approx((4.47861, 0.00024), abs=1e-5)
    assert sm["ab_100"] == pytest.approx((4.705, 0.017), abs=1e-3)
    assert sm["doob"][0] == pytest.approx(res["true"], abs=1e-5) and sm["doob"][1] < 1e-6
    # Every path of the policy martingale satisfies max_k (Z_k - M_k) >= L_0 (Prop. A--B).
    assert res["samples"]["ab_exact"].min() >= res["policy"] - 1e-9
    assert np.mean(res["samples"]["ab_exact"] - res["policy"] < 1e-9) == pytest.approx(0.75, abs=0.01)
    assert res["samples"]["doob"].std(ddof=1) == pytest.approx(1.6e-5, abs=0.1e-5)
    lo = res["out_sample"][0] - 1.96 * res["out_sample"][1]
    hi = sm["ab_exact"][0] + 1.96 * sm["ab_exact"][1]
    assert (round(lo, 3), round(hi, 3)) == (4.461, 4.479)
    # Weak duality: every upper estimate is above the price by more than two standard errors.
    for key in ("zero", "european", "ab_exact", "ab_100"):
        assert sm[key][0] - 2 * sm[key][1] > res["true"]


@SLOW
def test_nested_estimator_with_more_inner_paths():
    res = duality_study(nested=((200, 500),))
    assert res["summary"]["ab_500"] == pytest.approx((4.537, 0.011), abs=1e-3)
    hi = res["summary"]["ab_500"][0] + 1.96 * res["summary"]["ab_500"][1]
    assert round(hi, 3) == 4.558


def test_andersen_broadie_decomposition():
    # Z_k - M_k = L_0 + (Z_k - L_k) + sum_{j<k} 1_{ex_j} (c_j - Z_j), with any continuation estimates.
    g = put_payoff(40.0)
    basis = laguerre_basis(3, scale=40.0)
    fit = lsm_fit(simulate(38.0, 1.0, 10, 0.06, 0.0, 0.2, 20_000, seed=2), g, basis, 0.06, 1.0)
    outer = simulate(38.0, 1.0, 10, 0.06, 0.0, 0.2, 60, seed=4)
    ab = andersen_broadie(outer, fit, g, basis, 0.0, 0.2, 50, seed=5)
    Z = discounted_payoffs(outer, g, 0.06, 1.0)
    c, L, M, ex = ab["c"], ab["L"], ab["M"], ab["exercise"]
    mistakes = np.zeros_like(Z)
    mistakes[:, 1:] = np.cumsum(np.where(ex[:, :-1], c - Z[:, :-1], 0.0), axis=1)
    assert np.allclose(Z - M, L[:, :1] + (Z - L) + mistakes)
    # with exact continuation values the same martingale has mean zero and beats the price
    pol = policy_grid(fit, basis, 38.0, 40.0, 1.0, 0.06, 0.0, 0.2, h=1e-3)
    Me = policy_martingale(outer, fit, g, basis, pol)
    assert dual_samples(Z, Me).min() >= pol["price"] - 1e-6
