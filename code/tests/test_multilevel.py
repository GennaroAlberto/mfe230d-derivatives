"""Checks of the Chapter 12 code against the results proved in the notes and the numbers quoted there."""
import numpy as np
import pytest

from mfe230d import bs
from mfe230d.multilevel import (
    GBMLevels,
    SDELevels,
    asian_call_payoff,
    call_payoff,
    complexity_bound,
    complexity_levels,
    coupled_sum,
    digital_payoff,
    euler_gbm_level_variance,
    gbm_level_exact,
    geometric_level_probabilities,
    level_cost,
    level_statistics,
    mlmc,
    mlmc_fixed,
    optimal_samples,
    rmse_at_budget,
    single_term,
    single_term_work,
    standard_mc_cost,
)
from mfe230d.paths import coarsen, gbm_coefficients, gbm_scheme_terminal, gbm_scheme_weak_error, optimal_allocation, softplus_call

TC = bs.TEST_CASE
D = np.exp(-0.01)
CALL = bs.call(**TC)
DIGITAL = bs.digital_call(**TC)
STOCK = lambda x: D * x  # noqa: E731


# ------------------------------------------------------------------ samplers and the coupling

def test_costs():
    assert [level_cost(l) for l in range(5)] == [1, 3, 6, 12, 24]
    assert GBMLevels(n0_steps=12).cost(2) == 48 + 24
    assert level_cost(2, M=4) == 20


@pytest.mark.parametrize("scheme", ["euler", "milstein"])
def test_gbm_levels_use_summed_increments(scheme):
    # fine and coarse members are the scheme on the fine increments and on their pairwise sums
    l, n = 3, 7
    s = GBMLevels(payoff=STOCK, scheme=scheme)
    Pf, Pc = s(l, n, np.random.default_rng(5))
    dW = np.sqrt(1.0 / 8) * np.random.default_rng(5).standard_normal((n, 8))
    assert np.allclose(Pf, D * gbm_scheme_terminal(100.0, -0.01, 0.2, 1.0, dW, scheme))
    assert np.allclose(Pc, D * gbm_scheme_terminal(100.0, -0.01, 0.2, 1.0, coarsen(dW, 2), scheme))
    Sf, Sc = s.terminal_pair(l, n, np.random.default_rng(5))
    assert np.allclose(D * Sf, Pf) and np.allclose(D * Sc, Pc)
    # level 0: one step, no coarse member
    P0, C0 = s(0, 4, np.random.default_rng(1))
    assert np.all(C0 == 0)


@pytest.mark.parametrize("scheme", ["euler", "milstein"])
def test_generic_sde_sampler_agrees(scheme):
    a, b, bx = gbm_coefficients(-0.01, 0.2)
    g = SDELevels(a, b, 100.0, 1.0, STOCK, scheme=scheme, b_x=bx)
    s = GBMLevels(payoff=STOCK, scheme=scheme)
    x = g(4, 50, np.random.default_rng(9))
    y = s(4, 50, np.random.default_rng(9))
    assert np.allclose(x[0], y[0]) and np.allclose(x[1], y[1])


def test_coarse_member_has_the_law_of_the_level_below():
    # Definition 12.1: E[coarse member at level l] = E[P_{l-1}] (here exactly known)
    s = GBMLevels(payoff=STOCK, scheme="euler")
    Pf, Pc = s(3, 200_000, np.random.default_rng(3))
    ex = gbm_level_exact(2, "euler", discount=D)
    assert abs(Pc.mean() - ex["mean_f"]) < 4 * Pc.std() / np.sqrt(Pc.size)
    assert Pc.var() == pytest.approx(ex["var_P"], rel=0.02)


def test_asian_levels_apply_the_same_rule_on_every_grid():
    f = asian_call_payoff(100.0, D)
    s = GBMLevels(payoff=f, scheme="milstein")
    Pf, Pc = s(2, 5, np.random.default_rng(2))
    dW = 0.5 * np.random.default_rng(2).standard_normal((5, 4))
    dC = coarsen(dW, 2)
    fac = 1 - 0.01 * 0.5 + 0.2 * dC + 0.5 * 0.04 * (dC**2 - 0.5)
    S = np.concatenate([np.full((5, 1), 100.0), 100 * np.cumprod(fac, axis=1)], axis=1)
    A = (0.5 * S[:, 0] + S[:, 1] + 0.5 * S[:, 2]) / 2
    assert np.allclose(Pc, D * np.maximum(A - 100, 0))


# ------------------------------------------------------------------ Proposition 12.2

def test_optimal_samples_meet_the_variance_and_the_cost_bound():
    V = np.array([158.0, 1.6, 0.9, 0.5, 0.0])        # includes a degenerate level V = 0
    C = np.array([level_cost(l) for l in range(5)])
    eps = 0.01
    n = optimal_samples(V, C, eps)
    S = np.sum(np.sqrt(V * C))
    assert n[-1] == 1                                 # one sample on the degenerate level
    assert np.sum(V / n) <= eps**2 / 2 + 1e-15
    assert np.sum(n * C) <= 2 * S**2 / eps**2 + C.sum()
    # Cauchy-Schwarz: any allocation has variance x cost >= S^2, equality iff n ~ sqrt(V/C)
    Vp, rng = V[:4], np.random.default_rng(0)
    Sp = np.sum(np.sqrt(Vp * C[:4]))
    for _ in range(100):
        m = rng.uniform(1, 1e6, 4)
        assert np.sum(Vp / m) * np.sum(m * C[:4]) >= Sp**2 * (1 - 1e-12)
    m = 7.3 * np.sqrt(Vp / C[:4])
    assert np.sum(Vp / m) * np.sum(m * C[:4]) == pytest.approx(Sp**2)


def test_multilevel_estimator_is_unbiased_for_the_finest_level():
    s = GBMLevels(payoff=STOCK, scheme="euler")
    R = mlmc_fixed(s, [200_000, 20_000, 10_000, 5_000], seed=4)
    assert abs(R["estimate"] - gbm_level_exact(3, "euler", discount=D)["mean_f"]) < 4 * R["se"]
    assert R["cost"] == 200_000 + 3 * 20_000 + 6 * 10_000 + 12 * 5_000


# ------------------------------------------------------------------ Example 12.5 and Proposition 12.4

def test_exact_euler_level_variance():
    for l in range(0, 12):
        assert euler_gbm_level_variance(l, discount=D) == pytest.approx(gbm_level_exact(l, "euler", discount=D)["var"], rel=1e-10)
    assert round(euler_gbm_level_variance(1, discount=D), 4) == 3.9306
    assert round(euler_gbm_level_variance(4, discount=D), 5) == 0.49890
    assert round(euler_gbm_level_variance(7, discount=D), 6) == 0.062483
    # E[P^2] = 10^4 exactly for the test case, and V_l = 8 h_l + O(h_l^2)
    assert (D * 100) ** 2 * np.exp(-0.02 + 0.04) == pytest.approx(1e4)
    for l in (10, 14):
        h = 2.0**-l
        assert abs(euler_gbm_level_variance(l, discount=D) / h - 8.0) < 0.3 * h * 2


def test_exact_milstein_level_variance_is_of_order_h2():
    r = [gbm_level_exact(l, "milstein", discount=D)["var"] * 4.0**l for l in (6, 8, 10, 12)]
    assert max(r) - min(r) < 2e-3 and 0.35 < r[-1] < 0.37


@pytest.mark.parametrize("scheme", ["euler", "milstein"])
def test_exact_level_moments_against_simulation(scheme):
    s = GBMLevels(payoff=STOCK, scheme=scheme)
    st = level_statistics(s, range(1, 5), 200_000, seed=8)
    for i, l in enumerate(range(1, 5)):
        ex = gbm_level_exact(l, scheme, discount=D)
        assert st["var"][i] == pytest.approx(ex["var"], rel=0.06)
        assert abs(st["mean"][i] - ex["mean"]) < 4 * np.sqrt(ex["var"] / 2e5)


def test_digital_bound_of_the_proposition_is_consistent():
    # (ii): the mismatch probability of the coupled digital is at most P(|X-K|<=d) + P(|Yf-X|>d) + P(|Yc-X|>d)
    rng = np.random.default_rng(12)
    n, m, h = 50_000, 64, 1 / 64
    dW = np.sqrt(h) * rng.standard_normal((n, m))
    X = 100 * np.exp(-0.03 + 0.2 * dW.sum(axis=1))
    Yf = gbm_scheme_terminal(100.0, -0.01, 0.2, 1.0, dW, "euler")
    Yc = gbm_scheme_terminal(100.0, -0.01, 0.2, 1.0, coarsen(dW, 2), "euler")
    for d in (0.5, 1.0, 2.0):
        lhs = np.mean((Yf > 100) != (Yc > 100))
        rhs = np.mean(np.abs(X - 100) <= d) + np.mean(np.abs(Yf - X) > d) + np.mean(np.abs(Yc - X) > d)
        assert lhs <= rhs


# ------------------------------------------------------------------ Figure 12.1 (level diagnostics)

def _figure_level(scheme, l):
    """One level of Figure 12.1, generated exactly as by ch12_multilevel.level_data."""
    s_i = 0 if scheme == "euler" else 1
    n = 500_000 if l <= 7 else 40_000
    Sf, Sc = GBMLevels(scheme=scheme).terminal_pair(l, n, np.random.default_rng([2027, s_i, l]))
    out = {}
    for name, f in (("call", call_payoff(100.0, D)), ("digital", digital_payoff(100.0, D))):
        Y = f(Sf) - f(Sc)
        c = Y - Y.mean()
        out[name] = dict(var=Y.var(ddof=1), mean=Y.mean(), var_P=f(Sf).var(ddof=1),
                         kurt=np.mean(c**4) / np.mean(c**2) ** 2, nonzero=np.count_nonzero(Y))
    return out


def test_figure_levels_quoted_numbers():
    e10, m10 = _figure_level("euler", 10)["call"], _figure_level("milstein", 10)["call"]
    assert round(e10["var"], 4) == 0.0046 and round(m10["var"] * 1e7, 1) == 2.0
    assert 3e4 < 155 / e10["var"] < 4e4 and 7e8 < 160 / m10["var"] < 1e9
    m7 = _figure_level("milstein", 7)        # the Euler kurtoses are checked in test_full_figure_numbers
    assert round(m7["digital"]["kurt"], -3) == 14000
    assert 30 <= m7["digital"]["nonzero"] <= 40         # "about 36 nonzero corrections"
    assert 157 < m7["call"]["var_P"] < 160


# ------------------------------------------------------------------ Theorem 12.6

def test_complexity_bound_regimes():
    eps = np.array([1e-2, 1e-3, 1e-4, 1e-5])
    b1 = np.array([complexity_bound(e, 1.0, 2.0, 1.0)[0] for e in eps]) * eps**2
    b2 = np.array([complexity_bound(e, 1.0, 1.0, 1.0)[0] for e in eps]) * eps**2 / np.log(eps) ** 2
    b3 = np.array([complexity_bound(e, 1.0, 0.5, 1.0)[0] for e in eps]) * eps**2.5
    for b in (b1, b2, b3):
        assert b.max() / b.min() < 3.0                   # bounded: the rate is right
    # without alpha >= min(beta, gamma)/2 the ceilings dominate: alpha = 0.25, beta = 2, gamma = 1
    b4 = np.array([complexity_bound(e, 0.25, 2.0, 1.0)[0] for e in eps]) * eps**2
    assert b4[-1] / b4[0] > 1e4
    assert complexity_levels(0.01, 1.0, 1.0) == 8 and complexity_levels(1.0, 0.5, 1.0) == 0
    assert 2.0**-7 > np.sqrt(0.5) * 0.01 >= 2.0**-8          # smallest L with 2^-L <= eps/sqrt2


def test_bias_lemma_on_the_exact_example():
    # Lemma 12.8: E[P_L] - E[P] = -E[Y_L]/(2^alpha - 1) (1 + o(1)), alpha = 1
    for L in (6, 10):
        ex = gbm_level_exact(L, "euler", discount=D)
        assert ex["bias"] / (-ex["mean"]) == pytest.approx(1.0, abs=2.0**-L * 2)


# ------------------------------------------------------------------ Section 12.4 (numbers quoted)

def test_adaptive_runs_quoted_in_the_notes():
    eps = (0.05, 0.02, 0.01, 0.005)
    e = [mlmc(GBMLevels(scheme="euler"), x, alpha=1.0, beta=1.0, N0=1000, seed=11) for x in eps]
    m = [mlmc(GBMLevels(scheme="milstein"), x, alpha=1.0, beta=2.0, N0=1000, seed=11) for x in eps]
    assert [R["L"] for R in e] == [3, 4, 5, 6] and [R["L"] for R in m] == [3, 4, 5, 5]
    ce = [x * x * R["cost"] for x, R in zip(eps, e)]
    cm = [x * x * R["cost"] for x, R in zip(eps, m)]
    assert round(ce[0]) == 675 and round(ce[-1]) == 1370
    assert 371 <= min(cm) and max(cm) <= 375 and round(np.mean(cm)) == 373
    R = e[-1]
    assert R["N"].tolist() == [23497780, 1547621, 827227, 434917, 226194, 114576, 57780]
    assert round(R["estimate"], 4) == 7.3703 and round(R["se"], 4) == 0.0035
    assert round(R["cost"] / 1e7, 1) == 5.5
    # plain Monte Carlo on the same finest level (Var P_L from Figure 12.1)
    vE = _figure_level("euler", 6)["call"]["var_P"]
    vM = GBMLevels(scheme="milstein").terminal_pair(5, 500_000, np.random.default_rng([2027, 1, 5]))[0]
    vM = call_payoff(100.0, D)(vM).var(ddof=1)
    assert round(standard_mc_cost(vE, 6, 0.005) / R["cost"]) == 15
    assert round(standard_mc_cost(vM, 5, 0.005) / m[-1]["cost"]) == 27


def test_single_level_columns_of_the_comparison():
    # best single level with exact biases, integer m, v = 154 (Var f(X_T) = 153.95)
    for scheme, want in (("euler", ((7, 0.0400), (34, 0.00885))), ("milstein", ((5, 0.0318), (19, 0.00659)))):
        b = {m: gbm_scheme_weak_error(m, scheme) for m in range(1, 46)}
        for C, (m_star, rmse) in zip((1e6, 1e8), want):
            r, m = min((np.sqrt(b[m] ** 2 + 153.95 * m / C), m) for m in b)
            assert m == m_star and abs(r - rmse) < 5e-5 * (1 if C == 1e6 else 0.2)


def test_chapter7_laboratory_comparison():
    # the Duffie-Glynn numbers of Chapter 7 and the multilevel numbers of the table
    assert round(optimal_allocation(1e6, 0.17, 154.0)["rmse"], 4) == 0.0408
    assert round(optimal_allocation(1e8, 0.17, 154.0)["rmse"], 5) == 0.00879
    assert round(np.sqrt(154 / 1e6), 4) == 0.0124
    rows = {}
    for s_i, scheme in enumerate(("euler", "milstein")):
        s = GBMLevels(payoff=softplus_call, r=0.0, y=0.01, scheme=scheme)
        st = level_statistics(s, range(11), [max(int(4e5 / 2**l), 20_000) for l in range(11)], seed=3 + s_i)
        bias = np.array([gbm_scheme_weak_error(2**l, scheme) for l in range(11)])
        rows[scheme] = [rmse_at_budget(st["var"], st["cost"], bias, B) for B in (1e6, 1e8)]
    assert [rows["euler"][0]["L"], rows["euler"][1]["L"]] == [4, 8]
    assert [rows["milstein"][0]["L"], rows["milstein"][1]["L"]] == [6, 10]
    assert round(rows["euler"][0]["rmse"], 4) == 0.0227 and round(rows["euler"][1]["rmse"], 5) == 0.00307
    assert round(rows["milstein"][0]["rmse"], 4) == 0.0136 and round(rows["milstein"][1]["rmse"], 5) == 0.00137


# ------------------------------------------------------------------ Section 12.5

def test_geometric_randomisation_costs():
    p, tail = geometric_level_probabilities(1.5, 200)
    assert p.sum() == pytest.approx(1.0) and tail[3] == pytest.approx(2.0**-4.5)
    cost = sum(pk * level_cost(k) for k, pk in enumerate(p))
    assert round(cost, 3) == 2.987
    assert round(sum(2.0**k * tk for k, tk in enumerate(tail)), 3) == 3.414


def test_single_term_window():
    # Proposition 12.10(iii): finite variance iff r < beta, finite cost iff r > gamma
    m2 = 4.0 ** -np.arange(8)
    costs = np.array([level_cost(l) for l in range(8)])
    assert np.isfinite(single_term_work(m2, costs, 1.5, tail_rate=(2.0, 1.0))[2])
    assert single_term_work(m2, costs, 2.0, tail_rate=(2.0, 1.0))[0] == np.inf
    assert single_term_work(m2, costs, 1.0, tail_rate=(2.0, 1.0))[1] == np.inf


@pytest.mark.parametrize("estimator", [single_term, coupled_sum])
def test_capped_randomised_estimators_target_the_capped_level(estimator):
    s = GBMLevels(payoff=STOCK, scheme="milstein")
    R = estimator(s, 200_000, 0.8, seed=3, max_level=6)
    assert R["levels"].max() == 6
    assert abs(R["mean"] - gbm_level_exact(6, "milstein", discount=D)["mean_f"]) < 4 * R["se"]


@pytest.mark.parametrize("estimator", [single_term, coupled_sum])
def test_randomised_estimators_are_unbiased(estimator):
    s = GBMLevels(payoff=STOCK, scheme="milstein")
    R = estimator(s, 300_000, 1.5, seed=21)
    assert abs(R["mean"] - 100 * np.exp(-0.02)) < 4 * R["se"]


def test_rhee_glynn_numbers_quoted():
    s = GBMLevels(scheme="milstein")
    a = single_term(s, 400_000, 1.5, seed=[5, 150])
    b = coupled_sum(s, 400_000, 1.5, seed=[6, 150])
    assert round(a["mean"], 4) == 7.4075 and round(a["se"], 4) == 0.0267 and round(a["mean_cost"], 2) == 3.01
    assert round(b["mean"], 4) == 7.3238 and round(b["se"], 4) == 0.0199 and round(b["mean_cost"], 2) == 3.71
    z = []
    for r in (1.1, 1.25, 1.4, 1.5, 1.6, 1.75, 1.9):
        for est, sd in ((single_term, 5), (coupled_sum, 6)):
            R = est(s, 400_000, r, seed=[sd, round(100 * r)])
            z.append((R["mean"] - CALL) / R["se"])
            if est is coupled_sum and r in (1.75, 1.9):
                assert 335 < R["Z"].var(ddof=1) * R["mean_cost"] < 395
    assert round(min(z), 1) == -2.4 and round(max(z), 1) == 1.7


# ------------------------------------------------------------------ path-dependent payoffs (Section 12.2)

def test_asian_variance_ratios_quoted():
    f = asian_call_payoff(100.0, D)
    n = [max(int(4e5 / 2**l), 20_000) for l in range(9)]
    rE = level_statistics(GBMLevels(payoff=f, scheme="euler"), range(9), n, seed=4)["var"]
    rM = level_statistics(GBMLevels(payoff=f, scheme="milstein"), range(9), n, seed=4)["var"]
    rE, rM = rE[:-1] / rE[1:], rM[:-1] / rM[1:]
    assert np.all((rM[1:] > 3.8 - 0.05) & (rM[1:] < 4.2))          # 1 <= l <= 7
    assert np.all((rE[:4] > 3.15) & (rE[:4] < 3.45))                # l <= 3
    assert [round(x, 1) for x in rE[6:]] == [2.3, 2.2]              # l = 6, 7


def test_exact_simulation_variance_of_the_call():
    from mfe230d.variance import call_payoff_moments
    mean, var = call_payoff_moments(100.0, 100.0, 1.0, 0.01, 0.02, 0.2)   # undiscounted payoff
    assert round(D**2 * var, 1) == 158.0                                 # Figure 12.3, exact simulation


# ------------------------------------------------------------------ the full figure data (slow, opt-in)

@pytest.mark.skipif(not __import__("os").environ.get("MFE230D_SLOW"),
                    reason="reproduces Figures 12.1-12.3 in full (about 30 s); set MFE230D_SLOW=1")
def test_full_figure_numbers():
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "figures"))
    import ch12_multilevel as fig
    from mfe230d.multilevel import _fit_rate
    lv = fig.level_data()
    ec, mc = lv["euler"]["call"], lv["milstein"]["call"]
    ed, md = lv["euler"]["digital"], lv["milstein"]["digital"]
    assert round(_fit_rate(range(4, 11), ec["var"][4:]), 2) == 0.99
    assert round(_fit_rate(range(4, 11), mc["var"][4:]), 2) == 1.99
    assert round(_fit_rate(range(2, 11), mc["mean"][2:]), 2) == 1.06
    for l in (8, 9, 10):
        assert abs(ec["mean"][l]) < 2 * np.sqrt(ec["var"][l] / 40_000)
    assert round(_fit_rate(range(4, 8), ed["var"][4:]), 2) == 0.50
    assert round(ec["kurt"][7], 1) == 7.6 and round(ed["kurt"][7], -1) == 260
    ratios = md["var"][4:7] / md["var"][5:8]
    assert round(ratios.min(), 1) == 1.5 and round(ratios.max(), 1) == 2.3
    assert round(_fit_rate(range(4, 8), md["var"][4:]), 2) == 0.85
    m1 = mc["var"][1] + mc["mean"][1] ** 2
    assert round(m1, 3) == 0.047
    m0 = mc["var"][0] + mc["mean"][0] ** 2
    assert 290 < m0 / (1 - 2.0**-1.9) and m0 / (1 - 2.0**-1.5) < 335   # "about 300" for r in [1.5, 1.9]
    # the adaptive algorithm on the Euler digital, eight seeds (Section 12.4)
    sd = GBMLevels(payoff=digital_payoff(100.0, D), scheme="euler")
    for eps, c2, rm in ((0.004, 10.8, 0.0033), (0.002, 17.4, 0.0022), (0.001, 32.5, 0.0010)):
        runs = [mlmc(sd, eps, alpha=1.0, beta=0.5, N0=2000, seed=k) for k in range(11, 19)]
        assert round(eps**2 * np.mean([R["cost"] for R in runs]), 1) == c2
        assert round(np.sqrt(np.mean([(R["estimate"] - DIGITAL) ** 2 for R in runs])), 4) == rm
    lab = fig.lab_data()
    assert [round(x, 3) for x in (lab["euler"]["reps"][-1], lab["milstein"]["reps"][-1])] == [0.026, 0.012]
