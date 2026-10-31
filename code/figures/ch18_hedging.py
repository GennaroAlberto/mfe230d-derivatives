"""Figures for Chapter 18 (Hedging in practice): the error of discrete delta hedging, hedging
at the implied or at the realised volatility, and Leland's volatility under transaction costs.
All on the course's test case S = K = 100, T = 1, r = 1%, y = 2%, sigma = 20%."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import bs
from mfe230d import hedging as H
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
P = dict(S=100.0, K=100.0, T=1.0, r=0.01, y=0.02, sigma=0.2)


def discrete_figure(out, n_hist=20_000, n_rms=10_000):
    """(a) Histograms of the hedging error under Q for m = 12, 52, 252;
    (b) sqrt(m) times the RMS error against m, for three drifts, with the exact values
    under Q and the leading-order constants of Proposition hd-discrete-error."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    bins = np.linspace(-6, 6, 121)
    for m, col in ((12, COBALT), (52, GOLD), (252, GREEN)):
        e = H.hedge_experiment(**P, m=m, n=n_hist, seed=1812)["error"]
        ax.hist(e, bins=bins, density=True, histtype="step", color=col, lw=1.2, label=f"$m={m}$")
    ax.set(xlabel="hedging error $X_T-(S_T-K)^+$", ylabel="density", title="(a) error under $\\mathbb{Q}$",
           xlim=(-6, 6))
    ax.legend(fontsize=7.5)

    ms = 2 ** np.arange(0, 11)
    exact = np.array([np.exp(P["r"]) * np.sqrt(H.exact_error_variance(**P, m=int(m))[0]) for m in ms])
    bx.semilogx(ms, np.sqrt(ms) * exact, color=COBALT, lw=1.2, label="exact, $\\mu=r$")
    for mu, col, mk, lab in ((None, COBALT, "o", "$\\mu=r$"), (0.20, GOLD, "s", "$\\mu=20\\%$"),
                             (-0.10, GREEN, "^", "$\\mu=-10\\%$")):
        rms = [H.summarize(H.hedge_stream(**P, m=int(m), n=n_rms, mu=mu, seed=18 + int(m))["error"])["rms"]
               for m in ms]
        bx.semilogx(ms, np.sqrt(ms) * np.array(rms), ls="none", marker=mk, ms=3.5, color=col, label=lab)
        bx.axhline(H.asymptotic_sd(**P, m=1, mu=mu), color=col, ls="--", lw=0.9)
    bx.set(xlabel="rebalancing dates $m$", ylabel="$\\sqrt{m}\\times$ RMS error", title="(b) the rate $m^{-1/2}$",
           ylim=(5.5, 7.4))
    bx.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "ch18-discrete.pdf")
    plt.close(fig)


def implied_figure(out, n=20_000, m=252, sigma_i=0.20, sigma_r=0.25, mu=0.10):
    """A long call bought at the implied volatility, hedged daily at the implied or at the
    realised volatility: (a) mark-to-market P&L along three paths; (b) final P&L."""
    S, K, T, r, y = (P[k] for k in ("S", "K", "T", "r", "y"))
    Vi, Vr = bs.call(S, K, T, r, y, sigma_i), bs.call(S, K, T, r, y, sigma_r)
    t, paths = H.simulate_stock(S, T, m, n, mu, y, sigma_r, seed=18)
    g = np.exp(r * T)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    sel = paths[:3]
    mi = g * H.mark_to_market(sel, t, K, r, y, sigma_i, sigma_i, Vi)
    mr = g * H.mark_to_market(sel, t, K, r, y, sigma_r, sigma_i, Vi)
    for j, col in enumerate((COBALT, GOLD, GREEN)):
        ax.plot(t, mi[j], color=col, lw=1.1, label="hedged at implied" if j == 0 else None)
        ax.plot(t, mr[j], color=col, lw=0.8, ls="--", label="hedged at realised" if j == 0 else None)
    ax.axhline(g * (Vr - Vi), color=GREY, lw=0.8, ls=":")
    ax.set(xlabel="time $t$", ylabel="P&L marked at implied vol", title="(a) three paths")
    leg = ax.legend(fontsize=7.5, loc="upper left")
    for hnd in leg.legend_handles:
        hnd.set_color(INK)

    pr = -H.hedge_paths(paths, t, K, r, y, sigma_r, premium=Vi)["error"]
    pi = -H.hedge_paths(paths, t, K, r, y, sigma_i, premium=Vi)["error"]
    bins = np.linspace(-1, 7, 121)
    bx.hist(pr, bins=bins, density=True, histtype="step", color=GOLD, lw=1.2, label="hedged at realised")
    bx.hist(pi, bins=bins, density=True, histtype="step", color=COBALT, lw=1.2, label="hedged at implied")
    bx.axvline(g * (Vr - Vi), color=GREY, lw=0.8, ls=":")
    bx.axvline(g * H.expected_vol_pnl(S, K, T, r, y, sigma_i, sigma_r, mu), color=COBALT, lw=0.8, ls="--")
    bx.set(xlabel="final P&L (time-$T$ money)", ylabel="density", title="(b) final P&L, daily hedge")
    bx.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch18-implied.pdf")
    plt.close(fig)


def leland_figure(out, n=10_000, n_large=2_000, k=0.02):
    """(a) Mean and standard deviation of the hedging error with round-trip cost k = 2%,
    hedging at sigma or at Leland's volatility, and without costs; (b) the mean error of
    Leland's strategy for k = 2% and 10% on finer grids, with +- 2 standard errors."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ms = 2 ** np.arange(0, 10)
    runs = (("no costs", GREY, dict(k=0.0)),
            ("costs, hedged at $\\sigma$", GOLD, dict(k=k)),
            ("costs, Leland $\\hat\\sigma$", COBALT, "leland"))
    for lab, col, kw in runs:
        mean, sd = [], []
        for m in ms:
            opts = dict(k=k, sigma_hedge=H.leland_vol(P["sigma"], k, P["T"] / m)) if kw == "leland" else kw
            e = H.hedge_stream(**P, m=int(m), n=n, seed=180 + int(m), **opts)["error"]
            mean.append(e.mean())
            sd.append(e.std(ddof=1))
        ax.semilogx(ms, mean, color=col, lw=1.2, marker="o", ms=2.5, label=lab)
        ax.semilogx(ms, sd, color=col, lw=1.0, ls="--")
    ax.axhline(0.0, color=GREY, lw=0.6)
    ax.set(xlabel="rebalancing dates $m$", ylabel="hedging error", title="(a) mean (solid), sd (dashed), $k=2\\%$",
           ylim=(-8, 7))
    ax.legend(fontsize=7, loc="lower left")

    ml = np.array([10, 30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000])
    for kk, col, mk in ((0.02, COBALT, "o"), (0.10, GOLD, "s")):
        mean, se = [], []
        for m in ml:
            e = H.hedge_stream(**P, m=int(m), n=n_large, k=kk, sigma_hedge=H.leland_vol(P["sigma"], kk, P["T"] / m),
                               seed=1985 + int(m))["error"]
            mean.append(e.mean())
            se.append(e.std(ddof=1) / np.sqrt(n_large))
        bx.errorbar(ml, mean, yerr=2 * np.array(se), color=col, marker=mk, ms=3.5, lw=1.0, capsize=2,
                    label=f"$k={kk:.0%}$".replace("%", "\\%"))
    bx.axhline(0.0, color=GREY, lw=0.8, ls=":")
    bx.set_xscale("log")
    bx.set(xlabel="rebalancing dates $m$", ylabel="mean hedging error", title="(b) Leland's strategy, fine grids")
    bx.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch18-leland.pdf")
    plt.close(fig)


def main(out):
    use_style()
    discrete_figure(out)
    implied_figure(out)
    leland_figure(out)


if __name__ == "__main__":
    import pathlib

    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
