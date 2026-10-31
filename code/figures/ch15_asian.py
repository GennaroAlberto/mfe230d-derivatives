"""Figures for Chapter 15 (Asian options): control variates, and the accuracy of the
moment-matching and conditioning approximations, on the course's monthly test case."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import asian as A
from mfe230d.style import COBALT, GOLD, GREY, use_style
from mfe230d.variance import control_variate, regression_control_variates

GREEN = "#1BAF7A"
S, R, Y = 100.0, 0.01, 0.02
T12 = A.TEST_DATES


def control_figure(out, n=2**16, seed=15):
    """(a) Standard deviation per path against the strike; (b) variances against sigma."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 12))
    Ks = np.arange(70.0, 141.0, 5.0)
    sd = {"plain": [], "geo": [], "three": []}
    for K in Ks:
        s = A.payoff_samples(S, K, T12, R, Y, 0.2, Z=Z)
        mu = A.control_means(S, K, T12, R, Y, 0.2)
        y = s["arithmetic"]
        sd["plain"].append(y.std(ddof=1))
        sd["geo"].append(control_variate(y, s["geometric"], mu["geometric"])["se"] * np.sqrt(n))
        X = np.column_stack([s[c] for c in ("geometric", "vanilla", "strip")])
        sd["three"].append(regression_control_variates(y, X, [mu[c] for c in ("geometric", "vanilla", "strip")])["se"]
                           * np.sqrt(n))
    ax.semilogy(Ks, sd["plain"], color=COBALT, label="plain")
    ax.semilogy(Ks, sd["geo"], color=GOLD, label="geometric control")
    ax.semilogy(Ks, sd["three"], color=GREEN, label="three controls")
    ax.set(xlabel="strike $K$", ylabel="standard deviation per path", title="(a) $\\sigma=20\\%$")
    ax.legend(fontsize=7.5, loc="center left")

    sigs = np.arange(0.05, 0.61, 0.05)
    vY, v1, vopt, bound = [], [], [], []
    for sg in sigs:
        s = A.payoff_samples(S, 100.0, T12, R, Y, sg, Z=Z)
        y, x = s["arithmetic"], s["geometric"]
        rho = np.corrcoef(x, y)[0, 1]
        vY.append(y.var(ddof=1))
        v1.append((y - x).var(ddof=1))
        vopt.append((1 - rho**2) * y.var(ddof=1))
        bound.append(A.dispersion_bound(S, T12, R, Y, sg))
    bx.semilogy(sigs, vY, color=GREY, ls=":", label="Var $Y$")
    bx.semilogy(sigs, bound, color=COBALT, ls="--", label="bound $D^2\\mathbb{E}(A-G)^2$")
    bx.semilogy(sigs, v1, color=GOLD, label="Var$(Y-X)$")
    bx.semilogy(sigs, vopt, color=GREEN, label="$(1-\\rho^2)$Var $Y$")
    bx.set(xlabel="volatility $\\sigma$", ylabel="variance per path", title="(b) $K=100$")
    bx.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ch15-control.pdf")
    plt.close(fig)


def approx_figure(out):
    """Errors of the approximations against the strike: (a) moment matching; (b) conditioning."""
    Ks = np.arange(60.0, 160.1, 2.5)
    ref = np.array([A.recursion_price(S, K, T12, R, Y, 0.2, N=2001, nq=48) for K in Ks])
    err = lambda f: np.array([f(S, K, T12, R, Y, 0.2) for K in Ks]) - ref
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax.axhline(0.0, color=GREY, lw=0.6)
    ax.plot(Ks, err(A.lognormal_price), color=COBALT, label="two moments")
    ax.plot(Ks, err(A.shifted_lognormal_price), color=GOLD, label="three moments")
    ax.plot(Ks, err(A.edgeworth_price), color=GREEN, label="Edgeworth")
    ax.set(xlabel="strike $K$", ylabel="price minus reference", title="(a) moment matching")
    ax.legend(fontsize=7.5)
    bx.semilogy(Ks, np.abs(err(A.curran_first_term)), color=COBALT, label="first term")
    bx.semilogy(Ks, np.abs(err(A.conditioning_lower_bound)), color=GOLD, label="bound $L(x^*)$")
    bx.semilogy(Ks, np.abs(err(A.curran_refined)), color=GREEN, label="conditional lognormal")
    bx.set(xlabel="strike $K$", ylabel="absolute error", title="(b) conditioning on $G$", ylim=(1e-9, 1e-2))
    bx.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ch15-approx.pdf")
    plt.close(fig)


def main(out):
    use_style()
    control_figure(out)
    approx_figure(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
