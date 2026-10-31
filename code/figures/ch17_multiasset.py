"""Figures for Chapter 17 (FX and multi-asset options): the cross-rate volatility triangle and
exchange options against the correlation, and the basket call's control variates and
approximations against the strike."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import multiasset as M
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def cross_figure(out, n=2**14, seed=17):
    """(a) Cross volatility against rho; (b) ATMF GBP/EUR call in USD: Margrabe and Monte Carlo."""
    fx = M.FX
    s1, s2 = fx["vol_gbp"], fx["vol_eur"]
    rhos = np.linspace(-1.0, 1.0, 401)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.7))
    ax.plot(rhos, 100 * M.exchange_vol(s1, s2, rhos), color=COBALT)
    r8 = M.implied_correlation(s1, s2, 0.08)
    ax.plot([r8], [8.0], "o", color=GOLD, ms=5, label=f"quote 8%: $\\rho$ = {r8:.3f}")
    for lev in (100 * abs(s1 - s2), 100 * (s1 + s2)):
        ax.axhline(lev, color=GREY, lw=0.7, ls=":")
    ax.set(xlabel="correlation $\\rho$ of GBP/USD and EUR/USD", ylabel="GBP/EUR volatility (%)",
           title="(a) the volatility triangle", xlim=(-1, 1))
    ax.legend(fontsize=7.5, loc="lower left")

    values = [M.cross_atmf_call(rho=r)[0] for r in rhos]
    bx.plot(rhos, values, color=COBALT, label="Margrabe")
    X1, X2 = fx["X_gbp"], fx["X_eur"]
    F12 = M.cross_atmf_call(rho=0.0)[1]
    Z = np.random.default_rng(seed).standard_normal((n, 2))
    D = np.exp(-fx["r_usd"])
    rgrid = np.linspace(-1.0, 1.0, 9)
    est, err = [], []
    for r in rgrid:
        ST = M.terminal_prices([X1, X2], 1.0, fx["r_usd"], [fx["r_gbp"], fx["r_eur"]], [s1, s2],
                               [[1.0, r], [r, 1.0]], Z=Z, method="pca")
        pay = D * np.maximum(ST[:, 0] - F12 * ST[:, 1], 0.0)
        est.append(pay.mean())
        err.append(2 * pay.std(ddof=1) / np.sqrt(n))
    bx.errorbar(rgrid, est, yerr=err, fmt="o", color=GOLD, ms=3.5, capsize=2, label="Monte Carlo $\\pm2$ s.e.")
    bx.set(xlabel="correlation $\\rho$", ylabel="value (USD per GBP notional)",
           title="(b) ATMF GBP/EUR call", xlim=(-1.05, 1.05))
    bx.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch17-cross.pdf")
    plt.close(fig)


def basket_figure(out, n=2**16, seed=17):
    """(a) Standard deviation per draw against K; (b) errors of the approximations against K."""
    P = M.BASKET
    S, w, T, r, y, sig, corr = (P[k] for k in ("S", "w", "T", "r", "y", "sigma", "corr"))
    Ks = np.arange(80.0, 120.01, 2.5)
    Z = np.random.default_rng(seed).standard_normal((n, 3))
    sd = {"plain": [], "geo": [], "three": []}
    rho = []
    for K in Ks:
        a = (S, w, K, T, r, y, sig, corr)
        sd["plain"].append(M.basket_mc(*a, Z=Z)["se"] * np.sqrt(n))
        g = M.basket_mc(*a, Z=Z, controls=("geometric",))
        sd["geo"].append(g["se"] * np.sqrt(n))
        rho.append(g["rho"])
        sd["three"].append(M.basket_mc(*a, Z=Z, controls=("geometric", "forward", "strip"))["se"] * np.sqrt(n))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax.semilogy(Ks, sd["plain"], color=COBALT, label="plain")
    ax.semilogy(Ks, sd["geo"], color=GOLD, label="geometric control")
    ax.semilogy(Ks, sd["three"], color=GREEN, label="three controls")
    ax.set(xlabel="strike $K$", ylabel="standard deviation per draw", title="(a) control variates")
    ax.legend(fontsize=7.5, loc="center left")

    Kf = np.arange(80.0, 120.01, 1.0)
    ref, ln, ge, cu = [], [], [], []
    for K in Kf:
        a = (S, w, K, T, r, y, sig, corr)
        ref.append(M.basket_reference(*a, nodes=24))
        ln.append(M.basket_lognormal(*a))
        ge.append(M.gentle_call(*a))
        cu.append(M.curran_lower_bound(*a))
    ref = np.array(ref)
    bx.plot(Kf, np.array(ln) - ref, color=COBALT, label="two moments")
    bx.plot(Kf, np.array(ge) - ref, color=GOLD, label="Gentle")
    bx.plot(Kf, np.array(cu) - ref, color=GREEN, label="conditioning bound")
    bx.axhline(0.0, color=INK, lw=0.6)
    bx.set(xlabel="strike $K$", ylabel="price minus reference", title="(b) approximations")
    bx.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ch17-basket.pdf")
    plt.close(fig)
    return dict(K=Ks, rho=np.array(rho), sd=sd)


def main(out):
    use_style()
    cross_figure(out)
    basket_figure(out)
