"""Figures for Chapter 7 (Simulating paths)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d.paths import (
    brownian_paths,
    explained_variance,
    fitted_order,
    gbm_scheme_errors,
    gbm_scheme_terminal,
    gbm_scheme_weak_error,
    mse_model,
    optimal_allocation,
    softplus_call,
    time_grid,
)
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"

# The laboratory of Section 7.3: GBM with the test-case drift r - y and volatility.
LAB = dict(S0=100.0, mu=-0.01, sigma=0.2, T=1.0, levels=range(7), n_paths=10**6, seed=2026)
C_EULER = 0.17      # Euler bias constant read off the lab (weak error x m at m = 16, 32)
V_PAYOFF = 154.0    # Var f(X_T) for the smooth payoff (exact value 153.95)
EXACT_MEAN = 8.206013337678918  # E f(X_T) by quadrature (see the tests)


def bridge(out):
    """Bridge construction stage by stage, and the variance carried by the first normals."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    K = 8
    t = time_grid(1.0, 2**K)
    W = brownian_paths(t, n_paths=1, seed=12, method="bridge")[0]
    stages = [(0, GREY, 0.9), (1, GREEN, 0.9), (2, GOLD, 1.0), (4, COBALT, 1.0), (8, INK, 0.7)]
    for level, color, lw in stages:
        step = 2 ** (K - level)
        ax.plot(t[::step], W[::step], color=color, lw=lw, marker="o" if level <= 4 else None,
                ms=2.5, label=f"{2**level} normal{'s' if level else ''}")
    lo, hi = W.min(), W.max()
    ax.set(xlabel="$t$", ylabel="$W_t$", xlim=(0, 1), ylim=(lo - 0.55 * (hi - lo), hi + 0.08 * (hi - lo)),
           title="bridge construction, coarse to fine")
    ax.legend(fontsize=6.5, loc="lower center", ncol=3, handlelength=1.5, columnspacing=0.8)

    m = 64
    tg = time_grid(1.0, m)
    k = np.arange(1, m + 1)
    for method, color, label in (("increments", GOLD, "increments"),
                                 ("bridge", COBALT, "Brownian bridge"),
                                 ("pca", GREEN, "principal components")):
        ax2.plot(k, explained_variance(tg, method), color=color, marker="o", ms=2.2, lw=1.4, label=label)
    ax2.set(xscale="log", xlabel="first $k$ normals", ylabel=r"share of variance $\pi_k$",
            ylim=(0, 1.02), title=f"$m={m}$ steps")
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ch07-bridge.pdf")
    plt.close(fig)


def orders(out, R=None):
    """Strong and weak errors of Euler and Milstein for GBM on common Brownian paths."""
    if R is None:
        R = gbm_scheme_errors(**LAB)
    h = R["h"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for s, color, label in (("euler", GOLD, "Euler"), ("milstein", COBALT, "Milstein")):
        ax.plot(h, R["strong"][s], color=color, marker="o", ms=3, label=label)
    hh = np.array([h[-1], h[2]])
    ax.plot(hh, 0.8 * R["strong"]["euler"][-1] * np.sqrt(hh / h[-1]), color=INK, lw=0.8, ls="--")
    ax.plot(hh, 0.6 * R["strong"]["milstein"][-1] * hh / h[-1], color=INK, lw=0.8, ls=":")
    ax.text(h[3], 0.55 * R["strong"]["euler"][3], "slope 1/2", fontsize=7, color=INK)
    ax.text(h[4], 0.35 * R["strong"]["milstein"][4], "slope 1", fontsize=7, color=INK)
    ax.set(xscale="log", yscale="log", xlabel="step $h$", ylabel=r"$E|X_T-Y^h_T|$", title="strong error")
    ax.legend(fontsize=7, loc="upper left")

    for s, color, label in (("euler", GOLD, "Euler"), ("milstein", COBALT, "Milstein")):
        ax2.errorbar(h, R["weak"][s], yerr=2 * R["weak_se"][s], color=color, marker="o", ms=3,
                     capsize=1.5, lw=1.4, label=label)
    sel = ~np.isnan(R["rich"]["milstein"])
    ax2.errorbar(h[sel], np.abs(R["rich"]["milstein"][sel]), yerr=2 * R["rich_se"]["milstein"][sel],
                 color=GREEN, marker="s", ms=3, capsize=1.5, lw=1.4, label="Milstein + Richardson")
    exact = {s: np.array([gbm_scheme_weak_error(int(m), s) for m in R["m"]]) for s in ("euler", "milstein")}
    rich_exact = 2 * exact["milstein"][1:] - exact["milstein"][:-1]
    for s, color in (("euler", GOLD), ("milstein", COBALT)):
        ax2.plot(h, exact[s], "x", color=INK, ms=4, mew=0.8)
    ax2.plot(h[1:], np.abs(rich_exact), "x", color=INK, ms=4, mew=0.8)
    ax2.plot(hh, 0.6 * R["weak"]["milstein"][-1] * hh / h[-1], color=INK, lw=0.8, ls=":")
    ax2.plot(hh, 0.5 * abs(R["rich"]["milstein"][-2]) * (hh / h[-2]) ** 2, color=INK, lw=0.8, ls="-.")
    ax2.text(h[4], 0.3 * R["weak"]["milstein"][4], "slope 1", fontsize=7, color=INK)
    ax2.text(h[2], 0.18 * abs(R["rich"]["milstein"][2]), "slope 2", fontsize=7, color=INK)
    ax2.set(xscale="log", yscale="log", xlabel="step $h$", ylabel=r"$E f(Y^h_T)-E f(X_T)$",
            title="weak error, smooth payoff")
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ch07-orders.pdf")
    plt.close(fig)
    return R


def _replicated_rmse(budget, m, reps, seed, exact):
    """RMSE over independent replications of the Euler estimator (m=0: exact sampling)."""
    rng = np.random.default_rng(seed)
    S0, mu, sigma, T = LAB["S0"], LAB["mu"], LAB["sigma"], LAB["T"]
    est = np.empty(reps)
    for i in range(reps):
        if m == 0:
            Z = rng.standard_normal(int(budget))
            X = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        else:
            n = int(round(budget / m))
            dW = np.sqrt(T / m) * rng.standard_normal((n, m))
            X = gbm_scheme_terminal(S0, mu, sigma, T, dW, "euler")
        est[i] = softplus_call(X).mean()
    return float(np.sqrt(np.mean((est - exact) ** 2)))


def allocation(out, R=None):
    """Duffie-Glynn: the mean-square error against the steps, and the error against the budget."""
    if R is None:
        R = gbm_scheme_errors(**LAB)
    c, v = C_EULER, V_PAYOFF
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    budget = 1e6
    m = np.geomspace(1, 64, 200)
    ax.plot(m, (c / m) ** 2, color=GOLD, label="squared bias $c^2h^2$")
    ax.plot(m, v * m / budget, color=GREEN, label="variance $v\\,m/C$")
    ax.plot(m, mse_model(m, budget, c, v), color=COBALT, lw=2, label="mean-square error")
    ax.plot(R["m"], np.array([gbm_scheme_weak_error(int(k), "euler") for k in R["m"]]) ** 2, "o",
            color=GOLD, ms=3)
    opt = optimal_allocation(budget, c, v)
    ax.axvline(opt["m"], color=INK, lw=0.8, ls="--")
    ax.text(opt["m"] * 1.08, 2e-2, f"$m^*\\approx{opt['m']:.1f}$", fontsize=7, color=INK)
    ax.set(xscale="log", yscale="log", xlabel="steps $m$", ylabel="mean-square error terms",
           ylim=(1e-6, 1e-1),
           title=f"budget $C=10^6$")
    ax.legend(fontsize=7, loc="lower left")

    budgets = np.geomspace(1e4, 1e9, 60)
    ax2.plot(budgets, np.sqrt(v / budgets), color=GREEN, label="exact simulation")
    rm = [optimal_allocation(C, c, v)["rmse"] for C in budgets]
    ax2.plot(budgets, rm, color=COBALT, label="Euler, optimal $m$")
    bias4 = gbm_scheme_weak_error(4, "euler")
    ax2.plot(budgets, np.sqrt(bias4**2 + v * 4 / budgets), color=GOLD, label="Euler, $m=4$")
    sim_b = np.array([1e4, 1e5, 1e6])
    reps = 100
    ax2.plot(sim_b, [_replicated_rmse(C, 0, reps, 70 + i, EXACT_MEAN) for i, C in enumerate(sim_b)],
             "o", color=GREEN, ms=3.5)
    ax2.plot(sim_b, [_replicated_rmse(C, max(1, round(optimal_allocation(C, c, v)["m"])), reps, 80 + i,
                                      EXACT_MEAN) for i, C in enumerate(sim_b)], "o", color=COBALT, ms=3.5)
    ax2.plot(sim_b, [_replicated_rmse(C, 4, reps, 90 + i, EXACT_MEAN) for i, C in enumerate(sim_b)],
             "o", color=GOLD, ms=3.5)
    ax2.set(xscale="log", yscale="log", xlabel="budget $C$ (steps)", ylabel="root-mean-square error",
            title="error against cost")
    ax2.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch07-allocation.pdf")
    plt.close(fig)


def main(out):
    use_style()
    bridge(out)
    R = orders(out)
    allocation(out, R)
    print("strong slopes (m >= 4):",
          {s: round(fitted_order(R["h"][2:], R["strong"][s][2:]), 2) for s in ("euler", "milstein")})
