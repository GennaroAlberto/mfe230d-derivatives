"""Figures for Chapter 8 (antithetic variates, moment matching and control variates)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d.bs import TEST_CASE, call
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style
from mfe230d.variance import (
    DEVIOUS,
    antithetic,
    antithetic_correlation,
    cv_bias_constant,
    mm_bias_constant,
    moment_match,
    scalar_payoffs,
    sqrt_control_mean,
    sqrt_option_samples,
    strike_to_z,
    terminal_price,
)

GREEN = "#1BAF7A"
P = TEST_CASE
S, K, T, r, y, SIG = (P[k] for k in ("S", "K", "T", "r", "y", "sigma"))
DISC = np.exp(-r * T)


def antithetic_figure(out):
    """(a) The call payoff, its antithetic image and their even part; (b) rho against the strike."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    zs = float(strike_to_z(K, S, T, r, y, SIG))
    u = np.linspace(-3, 3, 601)
    f = lambda v: DISC * np.maximum(terminal_price(v, S, T, r, y, SIG) - K, 0.0)
    ax.plot(u, f(u), color=COBALT, label="$f(z)$")
    ax.plot(u, f(-u), color=GOLD, label="$f(-z)$")
    ax.plot(u, 0.5 * (f(u) + f(-u)), color=INK, lw=1.0, ls="--", label="even part $f_e(z)$")
    for v in (zs, -zs):
        ax.axvline(v, color=GREY, lw=0.6, ls=":")
    ax.set(xlabel="$z$", ylabel="discounted payoff", xlim=(-3, 3), ylim=(0, 60),
           title="(a) call payoff and its antithetic image")
    ax.legend(loc="upper center", fontsize=7.5)

    med = S * np.exp((r - y - 0.5 * SIG**2) * T)
    strikes = np.linspace(60, 150, 61)
    payoffs = {
        "call": (lambda k: (lambda v: max(float(terminal_price(v, S, T, r, y, SIG)) - k, 0.0)), COBALT),
        "put": (lambda k: (lambda v: max(k - float(terminal_price(v, S, T, r, y, SIG)), 0.0)), GOLD),
        "straddle": (lambda k: (lambda v: abs(float(terminal_price(v, S, T, r, y, SIG)) - k)), GREEN),
    }
    rng = np.random.default_rng(81)
    z = rng.standard_normal(20_000)
    for name, (mk, color) in payoffs.items():
        rho = [antithetic_correlation(mk(k), [float(strike_to_z(k, S, T, r, y, SIG))]) for k in strikes]
        bx.plot(strikes, rho, color=color, label=name)
        for k in (70, 85, 100, 115, 130):
            g = mk(k)
            sim = antithetic(np.vectorize(g), z)["rho"]
            bx.plot(k, sim, "o", ms=3, color=color)
    bx.axvline(med, color=GREY, lw=0.8, ls=":")          # the median S e^{nu T}
    bx.axhline(0, color=INK, lw=0.6, alpha=0.4)
    bx.set(xlabel="strike $K$", ylabel=r"$\rho=\mathrm{Corr}(f(Z),f(-Z))$", ylim=(-1.05, 1.05),
           title="(b) antithetic correlation")
    bx.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch08-antithetic.pdf")
    plt.close(fig)


def _beta(yy, xx):
    xc = xx - xx.mean(1, keepdims=True)
    return (xc * (yy - yy.mean(1, keepdims=True))).sum(1) / (xc * xc).sum(1)


def bias_figure(out):
    """n x bias of moment matching and of the in-sample control-variate estimator, and splitting."""
    p = scalar_payoffs()
    c_mm = mm_bias_constant(p["call"], [p["z_star"]])
    c_cv = cv_bias_constant(p["call"], p["stock"], [p["z_star"]])
    price = float(call(**P))
    mux = S * np.exp(-y * T)
    rng = np.random.default_rng(82)
    ns = np.array([4, 8, 16, 32, 64, 128, 256])
    res = {k: [] for k in ("mm", "cv", "split")}
    budget = 6_400_000
    for n in ns:
        R = budget // n
        z = rng.standard_normal((R, n))
        est = (DISC * np.maximum(terminal_price(moment_match(z, "log"), S, T, r, y, SIG) - K, 0)).mean(1)
        Y = DISC * np.maximum(terminal_price(z, S, T, r, y, SIG) - K, 0)
        X = DISC * terminal_price(z, S, T, r, y, SIG)
        cv = Y.mean(1) - _beta(Y, X) * (X.mean(1) - mux)
        h = n // 2
        b1, b2 = _beta(Y[:, :h], X[:, :h]), _beta(Y[:, h:], X[:, h:])
        sp = 0.5 * (Y[:, :h].mean(1) - b2 * (X[:, :h].mean(1) - mux)
                    + Y[:, h:].mean(1) - b1 * (X[:, h:].mean(1) - mux))
        for key, e in (("mm", est), ("cv", cv), ("split", sp)):
            res[key].append((n * (e.mean() - price), 2 * n * e.std() / np.sqrt(R)))
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    for key, color, label, const in (
            ("mm", COBALT, "moment matching (log)", c_mm),
            ("cv", GOLD, r"control variate, in-sample $\hat\beta_n$", c_cv),
            ("split", GREEN, r"control variate, split sample", 0.0)):
        m, e = np.array(res[key]).T
        ax.errorbar(ns, m, yerr=e, color=color, marker="o", ms=3, lw=1.4, capsize=2, label=label)
        ax.axhline(const, color=color, lw=0.8, ls="--", alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set(xlabel="sample size $n$", ylabel=r"$n\times$ bias", ylim=(-11, 4))
    ax.legend(loc="center right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch08-bias.pdf")
    plt.close(fig)


def control_figure(out):
    """The square-root call and its control: (a) payoffs against S_T; (b) samples and fitted lines."""
    d = DEVIOUS
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    s = np.linspace(0.4e4, 2.6e4, 400)
    yv = np.maximum(np.sqrt(s) - np.sqrt(d["K"]), 0)
    xv = np.maximum(s - d["K"], 0) / (2 * np.sqrt(d["K"]))
    ax.plot(s / 1e4, yv, color=COBALT, label=r"$(\sqrt{S_T}-\sqrt{K})^+$")
    ax.plot(s / 1e4, xv, color=GOLD, label=r"$(S_T-K)^+/(2\sqrt{K})$")
    ax.fill_between(s / 1e4, yv, xv, color=GOLD, alpha=0.12, lw=0)
    ax.set(xlabel=r"$S_T$ (units of $10^4$)", ylabel="payoff", title="(a) payoff and control")
    ax.legend(loc="upper left", fontsize=7.5)

    z = np.random.default_rng(83).standard_normal(2_000)
    Y, X = sqrt_option_samples(z, **d)
    bx.plot(X, Y, "o", ms=2.4, color=GREY, alpha=0.6, mec="none", label="samples $(X_i,Y_i)$")
    mx, my = sqrt_control_mean(**d), 7.36429
    xs = np.linspace(0, X.max(), 50)
    xc = X - X.mean()
    bhat = float(xc @ (Y - Y.mean()) / (xc @ xc))
    bx.plot(xs, my + 1.0 * (xs - mx), color=GOLD, lw=1.4, label=r"slope $\beta=1$")
    bx.plot(xs, my + bhat * (xs - mx), color=COBALT, lw=1.4, label=rf"slope $\hat\beta={bhat:.2f}$")
    bx.set(xlabel="control $X$", ylabel="payoff $Y$", title="(b) 2,000 samples")
    bx.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch08-control.pdf")
    plt.close(fig)


def main(out):
    use_style()
    antithetic_figure(out)
    bias_figure(out)
    control_figure(out)
