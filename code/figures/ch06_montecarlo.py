"""Figures for Chapter 6 (Monte Carlo pricing and its error)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mfe230d import bs
from mfe230d import montecarlo as mc
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
TC = bs.TEST_CASE
NS = 2 ** np.arange(1, 21)          # n = 2, 4, ..., 2^20


def convergence(out, seed=2024):
    """Bare-bones Monte Carlo on the test case: errors against CLT and batch standard errors."""
    res = mc.bs_call_bbmc(**TC, n=NS[-1], seed=seed)
    disc = np.exp(-TC["r"] * TC["T"])
    series = (
        (res["samples"]["ST"], float(bs.forward(TC["S"], TC["T"], TC["r"], TC["y"])),
         r"$\mathbb{E}^{\mathbb{Q}}[S_T]$"),
        (disc * res["samples"]["CT"], float(bs.call(**TC)), r"call price $C$"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, (x, exact, title) in zip(axes, series):
        est = mc.nested_estimates(x, NS, m=1024)
        ax.plot(NS, np.abs(est["mean"] - exact), "o", color=COBALT, ms=3.2,
                label=r"$|\hat X_n-\mu|$")
        ax.plot(NS, est["se"], color=COBALT, lw=1.4, label=r"CLT: $\hat s_n/\sqrt{n}$")
        ax.plot(NS, est["se_batch"], color=GOLD, lw=1.4, ls="--",
                label=r"batch, $m=\min(2^{10},n)$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([2**k for k in (1, 5, 10, 15, 20)])
        ax.set_xlabel("number of samples $n$")
        ax.set_title(title)
    axes[0].set_ylabel("error and standard error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "ch06-convergence.pdf")
    plt.close(fig)


def coverage_and_tails(out):
    """Left: coverage of nominal 95% intervals. Right: the sample standard deviation under heavy tails."""
    S, K, T, r, y, sigma = (TC[k] for k in ("S", "K", "T", "r", "y", "sigma"))
    F = float(bs.forward(S, T, r, y))
    k_dig = F * np.exp(-0.5 * sigma**2 * T + sigma * np.sqrt(T) * mc.normal_quantile(0.98))
    p_dig = float(np.exp(r * T) * bs.digital_call(S, k_dig, T, r, y, sigma))   # = 1%
    disc = np.exp(-r * T)

    def call(rng, shape):
        return disc * np.maximum(mc.terminal_prices(S, T, r, y, sigma, rng.standard_normal(shape)) - K, 0.0)

    def digital(rng, shape):
        return (mc.terminal_prices(S, T, r, y, sigma, rng.standard_normal(shape)) > k_dig).astype(float)

    def pareto15(rng, shape):
        return mc.pareto(1.5, shape, rng=rng)

    ns = 2 ** np.arange(3, 16)
    reps = 2000
    cases = ((call, float(bs.call(**TC)), COBALT, "o", "at-the-money call"),
             (digital, p_dig, GOLD, "s", r"digital, $\mathbb{Q}(S_T>K)=1\%$"),
             (pareto15, mc.pareto_mean(1.5), GREEN, "^", r"Pareto, $\alpha=1.5$"))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.75))
    for j, (sampler, exact, color, marker, label) in enumerate(cases):
        cov = [mc.coverage(sampler, int(n), reps, exact, seed=100 * j + i) for i, n in enumerate(ns)]
        ax.plot(ns, cov, color=color, marker=marker, ms=3.2, lw=1.3, label=label)
    ax.axhline(0.95, color=INK, lw=0.8, ls=":", alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**k for k in (3, 6, 9, 12, 15)])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("number of samples $n$")
    ax.set_ylabel("coverage of the 95% interval")
    ax.legend(loc="lower right", fontsize=7.5)

    n_max = 2**20
    grid = 2 ** np.arange(4, 21)
    for i, seed in enumerate((11, 12, 13)):
        x = mc.pareto(1.5, n_max, seed=seed)
        s = [x[:n].std(ddof=1) for n in grid]
        ax2.plot(grid, s, color=GREEN, lw=1.2, alpha=1.0 if i == 0 else 0.55,
                 label=r"$\alpha=1.5$ (three runs)" if i == 0 else None)
    x = mc.pareto(3.0, n_max, seed=14)
    ax2.plot(grid, [x[:n].std(ddof=1) for n in grid], color=COBALT, lw=1.3, label=r"$\alpha=3$")
    ax2.axhline(np.sqrt(0.75), color=COBALT, lw=0.8, ls=":", label=r"$\sigma=\sqrt{3}/2$ for $\alpha=3$")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks([2**k for k in (4, 8, 12, 16, 20)])
    ax2.set_xlabel("number of samples $n$")
    ax2.set_ylabel(r"sample standard deviation $\hat s_n$")
    ax2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch06-coverage.pdf")
    plt.close(fig)


def dimension(out):
    """Left: 1-D rules on the course's integrand. Right: product midpoint against Monte Carlo in d dimensions."""
    f1 = lambda x: 0.5 * np.pi * np.sin(np.pi * x)
    ks = 2 ** np.arange(1, 13)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.85))
    ax.plot(ks, [abs(1 - mc.riemann_1d(f1, k, "left")) for k in ks], color=GOLD, marker="s", ms=3,
            lw=1.3, label="Riemann sums (= trapezoid)")
    ax.plot(ks, [abs(1 - mc.midpoint_1d(f1, k)) for k in ks], color=COBALT, marker="o", ms=3,
            lw=1.3, label="midpoint")
    var1 = np.pi**2 / 8 - 1
    ax.plot(ks, np.sqrt(var1 / ks), color=GREEN, lw=1.3, ls="--", label="Monte Carlo (RMSE)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2**k for k in (1, 4, 8, 12)])
    ax.set_xlabel("number of points $N$")
    ax.set_ylabel("absolute error")
    ax.set_title(r"$d=1$: $\int_0^1\frac{\pi}{2}\sin(\pi x)\,\mathrm{d}x=1$")
    ax.legend(loc="lower left", fontsize=7.5)

    plan = ((2, (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), COBALT),
            (4, (2, 3, 4, 6, 8, 12, 16, 24, 32), GOLD),
            (8, (2, 3, 4, 5, 6), GREEN))
    Nline = np.logspace(np.log10(4), 6.3, 50)
    for d, kk, color in plan:
        N = np.array([k**d for k in kk])
        err = [abs(mc.product_midpoint(mc.sine_product, d, k) - 1) for k in kk]
        ax2.plot(N, err, color=color, marker="o", ms=3, lw=1.3)
        ax2.plot(Nline, np.sqrt(((np.pi**2 / 8) ** d - 1) / Nline), color=color, lw=1.1, ls="--")
    handles = [Line2D([], [], color=c, lw=1.3, label=f"$d={d}$") for d, _, c in plan]
    handles += [Line2D([], [], color=GREY, lw=1.2, marker="o", ms=3, label="product midpoint"),
                Line2D([], [], color=GREY, lw=1.1, ls="--", label="Monte Carlo (RMSE)")]
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("number of points $N$")
    ax2.set_title(r"$f(x)=\prod_{j=1}^d\frac{\pi}{2}\sin(\pi x_j)$ on $[0,1]^d$")
    ax2.set_ylim(1e-9, 3)
    ax2.legend(handles=handles, loc="lower left", fontsize=7.0, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "ch06-dimension.pdf")
    plt.close(fig)


def main(out):
    use_style()
    convergence(out)
    coverage_and_tails(out)
    dimension(out)
