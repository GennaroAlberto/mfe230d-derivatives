"""Figures for Chapter 12 (Multilevel Monte Carlo)."""
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import bs
from mfe230d.multilevel import (
    GBMLevels,
    call_payoff,
    coupled_sum,
    digital_payoff,
    level_cost,
    level_statistics,
    mlmc,
    mlmc_fixed,
    rmse_at_budget,
    single_term,
    single_term_work,
    standard_mc_cost,
)
from mfe230d.paths import gbm_scheme_weak_error, optimal_allocation, softplus_call
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
TC = bs.TEST_CASE
D = np.exp(-TC["r"] * TC["T"])
CALL = bs.call(**TC)
SCHEMES = (("euler", GOLD, "Euler"), ("milstein", COBALT, "Milstein"))

# Level diagnostics (Figure 12.1): levels 0..10 for the call, 0..7 for the digital.
LEVELS_CALL = range(11)
LEVELS_DIG = range(8)
N_SHARED = 500_000          # samples per level for l <= 7 (call and digital on the same paths)
N_FINE = 40_000             # samples per level for l = 8, 9, 10 (call only)
SEED_LEVELS = 2027
# Adaptive runs (Figure 12.2, left) and the budget comparison (right, Chapter 7's laboratory).
EPS = (0.05, 0.02, 0.01, 0.005)
SEED_ADAPT = 11
LAB_MEAN = 8.206013337678918        # E f(X_T) for the softplus payoff of Chapter 7
BUDGETS = np.geomspace(1e4, 1e8, 17)
BUDGET_REPS = (1e4, 1e5, 1e6)
REPS = 40
SEED_LAB = 3
# Rhee-Glynn (Figure 12.3).
R_GRID = (1.1, 1.25, 1.4, 1.5, 1.6, 1.75, 1.9)
N_RG = 400_000
SEED_RG = 5


def level_data():
    """Level statistics of the test-case call and digital, Euler and Milstein.

    For l <= 7 the call and the digital are evaluated on the same N_SHARED coupled paths
    (generator default_rng([SEED_LEVELS, l]), one per scheme and level); for l = 8..10 the
    call alone on N_FINE paths. Returns {scheme: {"call": stats, "digital": stats}}.
    """
    out = {}
    for s_i, (scheme, _, _) in enumerate(SCHEMES):
        sampler = GBMLevels(scheme=scheme)
        stats = {k: {q: [] for q in ("mean", "var", "mean_P", "var_P", "kurt")} for k in ("call", "digital")}
        for l in LEVELS_CALL:
            rng = np.random.default_rng([SEED_LEVELS, s_i, l])
            n = N_SHARED if l in LEVELS_DIG else N_FINE
            Sf, Sc = sampler.terminal_pair(l, n, rng)
            pays = (("call", call_payoff(100.0, D)), ("digital", digital_payoff(100.0, D)))
            for name, f in pays:
                if name == "digital" and l not in LEVELS_DIG:
                    continue
                Pf = f(Sf)
                Pc = f(Sc) if l > 0 else np.zeros(n)
                Y = Pf - Pc
                c = Y - Y.mean()
                v2 = np.mean(c**2)
                st = stats[name]
                st["mean"].append(Y.mean()); st["var"].append(Y.var(ddof=1))
                st["mean_P"].append(Pf.mean()); st["var_P"].append(Pf.var(ddof=1))
                st["kurt"].append(np.mean(c**4) / v2**2 if v2 > 0 else np.nan)
        out[scheme] = {k: {q: np.array(v) for q, v in d.items()} for k, d in stats.items()}
    return out


def adaptive_data(levels=None):
    """Adaptive MLMC on the test-case call for each eps, and the plain Monte Carlo cost on
    the same finest level (model: Var P_L from ``levels``)."""
    out = {}
    for scheme, _, _ in SCHEMES:
        sampler = GBMLevels(scheme=scheme)
        beta = 1.0 if scheme == "euler" else 2.0
        runs = [mlmc(sampler, eps, alpha=1.0, beta=beta, N0=1000, seed=SEED_ADAPT) for eps in EPS]
        mc = None
        if levels is not None:
            vP = levels[scheme]["call"]["var_P"]
            mc = np.array([standard_mc_cost(vP[R["L"]], R["L"], eps) for R, eps in zip(runs, EPS)])
        out[scheme] = dict(runs=runs, mc=mc)
    return out


def lab_data():
    """Chapter 7's laboratory (GBM, mu = -1%, sigma = 20%, softplus payoff, undiscounted):
    level variances (estimated), exact Euler/Milstein biases, best MLMC RMSE per budget,
    the Duffie-Glynn RMSE, and REPS replications of MLMC at BUDGET_REPS."""
    out = {}
    for s_i, (scheme, _, _) in enumerate(SCHEMES):
        sampler = GBMLevels(payoff=softplus_call, r=0.0, y=0.01, scheme=scheme)
        L = range(11)
        st = level_statistics(sampler, L, [max(int(4e5 / 2**l), 20_000) for l in L], seed=SEED_LAB + s_i)
        bias = np.array([gbm_scheme_weak_error(2**l, scheme) for l in L])
        model = [rmse_at_budget(st["var"], st["cost"], bias, B) for B in BUDGETS]
        reps = []
        for B in BUDGET_REPS:
            R = rmse_at_budget(st["var"], st["cost"], bias, B)
            N = np.maximum(1, np.round(R["N"])).astype(int)
            err = [mlmc_fixed(sampler, N, seed=[SEED_LAB, s_i, int(B), k])["estimate"] - LAB_MEAN
                   for k in range(REPS)]
            reps.append(np.sqrt(np.mean(np.square(err))))
        out[scheme] = dict(stats=st, bias=bias, model=model, reps=np.array(reps))
    out["dg"] = [optimal_allocation(B, 0.17, 154.0)["rmse"] for B in BUDGETS]
    out["exact"] = np.sqrt(154.0 / BUDGETS)
    return out


def rg_data(levels):
    """Rhee-Glynn estimators of the Milstein call for each r in R_GRID: simulated variance x
    mean cost (single-term and coupled-sum), and the single-term model from the level data
    (E[Y_n^2] = V_n + (E Y_n)^2, levels 0..10, geometric tails beta = 2, gamma = 1)."""
    st = levels["milstein"]["call"]
    m2 = st["var"] + st["mean"] ** 2
    costs = np.array([level_cost(l) for l in LEVELS_CALL])
    rr = np.linspace(1.02, 1.98, 97)
    model = np.array([single_term_work(m2, costs, r, mean=CALL, tail_rate=(2.0, 1.0))[2] for r in rr])
    sampler = GBMLevels(scheme="milstein")
    sims = {"single": [], "coupled": []}
    for r in R_GRID:
        a = single_term(sampler, N_RG, r, seed=[SEED_RG, round(100 * r)])
        b = coupled_sum(sampler, N_RG, r, seed=[SEED_RG + 1, round(100 * r)])
        sims["single"].append((a["mean"], a["se"], a["Z"].var(ddof=1) * a["mean_cost"]))
        sims["coupled"].append((b["mean"], b["se"], b["Z"].var(ddof=1) * b["mean_cost"]))
    return dict(r=rr, model=model, sims={k: np.array(v) for k, v in sims.items()})


# ---------------------------------------------------------------------------------------


def fig_levels(out, lv):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    lc, ld = np.array(LEVELS_CALL), np.array(LEVELS_DIG)
    ax.plot(lc, np.log2(lv["milstein"]["call"]["var_P"]), color=GREY, lw=1.0, ls="--", label=r"Var $P_\ell$, call")
    ax.plot(ld, np.log2(lv["milstein"]["digital"]["var_P"]), color=GREY, lw=1.0, ls=":", label=r"Var $P_\ell$, digital")
    for scheme, color, name in SCHEMES:
        ax.plot(lc[1:], np.log2(lv[scheme]["call"]["var"][1:]), color=color, marker="o", ms=3, label=f"{name}, call")
        v = lv[scheme]["digital"]["var"][1:]
        ax.plot(ld[1:], np.log2(v), color=color, marker="s", ms=3, ls="--", mfc="white", label=f"{name}, digital")
    ax.set(xlabel=r"level $\ell$", ylabel=r"$\log_2 V_\ell$", title=r"variance of the corrections $Y_\ell$",
           xticks=range(0, 11, 2))
    ax.legend(fontsize=6.3, loc="lower left", ncol=2, handlelength=1.8, columnspacing=0.8)
    for scheme, color, name in SCHEMES:
        m = np.abs(lv[scheme]["call"]["mean"][1:])
        ax2.plot(lc[1:], np.log2(m), color=color, marker="o", ms=3, label=f"{name}, call")
    m = np.abs(lv["euler"]["digital"]["mean"][1:])
    ax2.plot(ld[1:], np.log2(m), color=GOLD, marker="s", ms=3, ls="--", mfc="white", label="Euler, digital")
    ref = np.array([1, 10])
    ax2.plot(ref, -4.5 - (ref - 1), color=INK, lw=0.8, ls=":", label="slope $-1$")
    ax2.set(xlabel=r"level $\ell$", ylabel=r"$\log_2|\mathrm{mean}\ Y_\ell|$", title=r"mean of the corrections $Y_\ell$",
            xticks=range(0, 11, 2))
    ax2.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch12-levels.pdf")
    plt.close(fig)


def fig_complexity(out, ad, lab):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    eps = np.array(EPS)
    for scheme, color, name in SCHEMES:
        cost = np.array([R["cost"] for R in ad[scheme]["runs"]])
        ax.plot(eps, eps**2 * cost, color=color, marker="o", ms=3.5, label=f"MLMC, {name}")
        if ad[scheme]["mc"] is not None:
            ax.plot(eps, eps**2 * ad[scheme]["mc"], color=color, marker="^", ms=3.5, ls="--", mfc="white",
                    label=f"plain MC, {name}")
    ax.set(xscale="log", yscale="log", xlabel=r"accuracy $\varepsilon$", ylabel=r"$\varepsilon^2\times$ cost (steps)",
           title="adaptive MLMC, test-case call", ylim=(200, 1.5e5))
    ax.set_xticks(eps, [f"{e:g}" for e in eps])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=6.5, loc="upper right", ncol=2, columnspacing=0.8)

    ax2.plot(BUDGETS, lab["exact"], color=GREEN, lw=1.4, label="exact simulation")
    ax2.plot(BUDGETS, lab["dg"], color=GREY, lw=1.4, ls="--", label="Euler, optimal allocation")
    for scheme, color, name in SCHEMES:
        ax2.plot(BUDGETS, [R["rmse"] for R in lab[scheme]["model"]], color=color, lw=1.4, label=f"MLMC, {name}")
        ax2.plot(BUDGET_REPS, lab[scheme]["reps"], color=color, marker="o", ms=3.5, ls="none")
    ax2.set(xscale="log", yscale="log", xlabel=r"budget $\mathcal{C}$ (steps)", ylabel="root-mean-square error",
            title="Chapter 7's laboratory")
    ax2.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch12-complexity.pdf")
    plt.close(fig)


def fig_unbiased(out, rg, ad):
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.axvspan(1.0, 2.0, color=COBALT, alpha=0.05, lw=0)
    ax.plot(rg["r"], rg["model"], color=COBALT, lw=1.4, label="single term (model)")
    r = np.array(R_GRID)
    ax.plot(r, rg["sims"]["single"][:, 2], color=COBALT, marker="o", ms=3.5, ls="none", label="single term (simulated)")
    ax.plot(r, rg["sims"]["coupled"][:, 2], color=GOLD, marker="s", ms=3.5, ls="none", label="coupled sum (simulated)")
    ml = np.mean([R["cost"] * e**2 for R, e in zip(ad["milstein"]["runs"], EPS)])
    ax.axhline(ml, color=INK, lw=0.8, ls=":", label=r"MLMC: $\varepsilon^2\times$cost")
    ax.axhline(CALL_VAR, color=GREEN, lw=0.8, ls="--", label="exact simulation")
    ax.axvline(1.0, color=GREY, lw=0.8); ax.axvline(2.0, color=GREY, lw=0.8)
    ax.text(0.95, 300, "mean cost $=\\infty$", rotation=90, fontsize=6.5, color=INK, ha="center", va="center")
    ax.text(2.05, 300, "variance $=\\infty$", rotation=90, fontsize=6.5, color=INK, ha="center", va="center")
    ax.set(xlim=(0.9, 2.1), ylim=(100, 1e4), yscale="log", xlabel=r"rate $r$ of $P(J=j)\propto2^{-rj}$",
           ylabel="variance $\\times$ mean cost", title=r"Rhee--Glynn, Milstein call ($\gamma=1$, $\beta=2$)")
    ax.legend(fontsize=6.3, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "ch12-unbiased.pdf")
    plt.close(fig)


CALL_VAR = 158.006      # Var of the discounted test-case call payoff (Chapter 8, exact)


def compute():
    lv = level_data()
    ad = adaptive_data(lv)
    lab = lab_data()
    rg = rg_data(lv)
    return lv, ad, lab, rg


def main(out):
    use_style()
    out.mkdir(parents=True, exist_ok=True)
    lv, ad, lab, rg = compute()
    fig_levels(out, lv)
    fig_complexity(out, ad, lab)
    fig_unbiased(out, rg, ad)
