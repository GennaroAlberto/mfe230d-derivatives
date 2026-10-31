"""Figures for Chapter 9 (importance sampling)."""
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import importance as im
from mfe230d.bs import TEST_CASE
from mfe230d.style import COBALT, GOLD, GREY, INK, PALETTE, use_style
from mfe230d.variance import strike_to_z

GREEN, PURPLE = PALETTE[2], PALETTE[3]
P = TEST_CASE
S, K0, T, r, y, SIG = (P[k] for k in ("S", "K", "T", "r", "y", "sigma"))
NU, SD = r - y - 0.5 * SIG**2, SIG * np.sqrt(T)
DISC = np.exp(-r * T)


def _lognormal_pdf(s, alpha=0.0, beta=1.0):
    """Density of S_T when log(S_T/S) ~ N((nu + alpha) T, beta^2 sigma^2 T)."""
    x = np.log(s / S)
    v = (beta * SD) ** 2
    return np.exp(-0.5 * (x - (NU + alpha) * T) ** 2 / v) / (s * np.sqrt(2 * np.pi * v))


def densities_figure(out):
    """(a) Sampling densities for drift shifts alpha and the zero-variance density; (b) the weighted payoff."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.35))
    s = np.linspace(50, 220, 700)
    alphas = (0.0, 0.1, 0.2, 0.3)
    for a, col in zip(alphas, PALETTE):
        lab = r"$\mathbb{Q}$, $\alpha=0$" if a == 0 else rf"$\alpha={a:g}$"
        ax.plot(s, _lognormal_pdf(s, a), color=col, label=lab)
        L = _lognormal_pdf(s) / _lognormal_pdf(s, a)
        bx.plot(s, DISC * np.maximum(s - K0, 0) * L, color=col, label=lab)
    zero_var = np.maximum(s - K0, 0) * _lognormal_pdf(s)
    zero_var /= np.trapezoid(zero_var, s)
    ax.plot(s, zero_var, color=INK, lw=1.0, ls="--", label="zero variance")
    ax.set(xlabel="$S_T$", ylabel="density", title="(a) sampling densities, $K=100$", xlim=(50, 220))
    ax.legend(fontsize=7)
    bx.axhline(im.is_moments("call", S, K0, T, r, y, SIG)[0], color=GREY, lw=0.8, ls=":")
    bx.set(xlabel="$S_T$", ylabel=r"$e^{-rT}(S_T-K)^+L$", title="(b) weighted payoff", xlim=(50, 220), ylim=(0, 60))
    bx.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch09-densities.pdf")
    plt.close(fig)


def contours_figure(out):
    """Variance ratio plain/IS: (a) against (K, alpha) at beta = 1; (b) against (alpha, beta) at K = 100."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    Ks, As = np.linspace(90, 110, 81), np.linspace(-0.1, 0.5, 121)
    R = np.array([[im.variance_ratio("call", S, k, T, r, y, SIG, a) for k in Ks] for a in As])
    levels = [1, 2, 4, 6, 8, 10, 12, 14]
    cs = ax.contour(Ks, As, R, levels=levels, cmap="Blues", vmin=-6, vmax=14, linewidths=1.0)
    ax.clabel(cs, fontsize=6.5, fmt="%g")
    astar = [im.optimal_alpha("call", S, k, T, r, y, SIG) for k in Ks]
    ax.plot(Ks, astar, color=GOLD, lw=1.4, label=r"$\alpha^*(K)$")
    ax.set(xlabel="$K$", ylabel=r"$\alpha$", title=r"(a) $\beta=1$")
    ax.legend(fontsize=7, loc="lower right")

    As2, Bs = np.linspace(0.0, 0.5, 101), np.linspace(0.68, 1.1, 85)
    R2 = np.array([[im.variance_ratio("call", S, K0, T, r, y, SIG, a, b) if 2 * b * b > 1 else np.nan
                    for b in Bs] for a in As2])
    cs2 = bx.contour(Bs, As2, R2, levels=[1, 2, 5, 10, 15, 20, 25, 30], cmap="Blues", vmin=-10, vmax=30,
                     linewidths=1.0)
    bx.clabel(cs2, fontsize=6.5, fmt="%g")
    bx.axvspan(0.68, 1 / np.sqrt(2), color=GOLD, alpha=0.18, lw=0)
    bx.axvline(1 / np.sqrt(2), color=GOLD, lw=1.2)
    bx.text(0.69, 0.04, "infinite\nvariance", fontsize=6.5, color=INK)
    bx.set(xlabel=r"$\beta$", ylabel=r"$\alpha$", title=r"(b) $K=100$", xlim=(0.68, 1.1))
    fig.tight_layout()
    fig.savefig(out / "ch09-contours.pdf")
    plt.close(fig)


def rare_figure(out):
    """Relative error per sample against the strike: (a) digital call, (b) call."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    Ks = np.linspace(100, 260, 81)
    d = strike_to_z(Ks, S, T, r, y, SIG)
    re = lambda kind, k, c: np.sqrt(im.is_moments(kind, S, k, T, r, y, SIG, c * SD / T)[1]  # noqa: E731
                                    / im.is_moments(kind, S, k, T, r, y, SIG)[0] ** 2 - 1)
    for ax, kind in zip(axes, ("digital", "call")):
        ghs = d if kind == "digital" else np.array([im.ghs_shift_call(dd, SD) for dd in d])
        ce = np.array([im.ce_shift_digital(dd) if kind == "digital" else im.ce_shift_call(dd, SD) for dd in d])
        opt = np.array([im.optimal_alpha(kind, S, k, T, r, y, SIG) * T / SD for k in Ks])
        ax.plot(Ks, [re(kind, k, 0.0) for k in Ks], color=GOLD, label="plain")
        ax.plot(Ks, [re(kind, k, c) for k, c in zip(Ks, ghs)], color=COBALT, label="GHS drift")
        ax.plot(Ks, [re(kind, k, c) for k, c in zip(Ks, opt)], color=GREEN, ls="--", label="optimal shift")
        ax.plot(Ks, [re(kind, k, c) for k, c in zip(Ks, ce)], color=INK, lw=0.9, ls=":", label="cross-entropy")
        if kind == "digital":
            ax.plot(Ks, np.sqrt(im.tail_exponential_relative_moment(d) - 1), color=PURPLE, label="exponential")
        for k in (150.0, 200.0, 250.0):        # Monte Carlo check of the GHS curve
            dk = float(strike_to_z(k, S, T, r, y, SIG))
            c = dk if kind == "digital" else im.ghs_shift_call(dk, SD)
            Yk, _, _ = im.sample_terminal_is(kind, S, k, T, r, y, SIG, c * SD / T, 1.0, 100_000, seed=int(k))
            ax.plot(k, Yk.std(ddof=1) / Yk.mean(), "o", ms=3.5, color=COBALT)
        ax.set(yscale="log", xlabel="$K$", title="(a) digital call" if kind == "digital" else "(b) call")
    axes[0].set_ylabel("relative error per sample")
    axes[0].legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch09-rare.pdf")
    plt.close(fig)


def asian_figure(out, lab=None):
    """(a) GHS and cross-entropy drifts for the Asian call; (b) variance reduction against the strike."""
    m = 16
    lab = im.asian_laboratory(m=m) if lab is None else lab
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.45))
    j = np.arange(1, m + 1)
    score = lambda Z: im.asian_dates_prices(Z, S, T, r, y, SIG).mean(axis=1)  # noqa: E731
    for row, col in zip(lab[1:], (COBALT, GOLD, GREEN)):
        ax.plot(j, row["mu"], color=col, label=f"$K={row['K']:.0f}$")
        hist, _ = im.cross_entropy_drift(score, row["K"], m, n=10_000, n_iter=8, seed=int(row["K"]))
        ax.plot(j, hist[-1], "o", ms=2.8, color=col)
    ax.set(xlabel="date $j$", ylabel=r"drift $\mu_j$", title="(a) GHS (lines), cross-entropy (dots)")
    ax.legend(fontsize=7)
    Ks = [row["K"] for row in lab]
    var_plain = [row["plain_var"] for row in lab]
    for key, col, name in (("plain_cv", GOLD, "control variate"), ("is", COBALT, "importance sampling"),
                           ("is_cv", GREEN, "both")):
        var = [row["cv_var"] if key == "plain_cv" else row[key]["se"] ** 2 * row["n"] for row in lab]
        bx.plot(Ks, [vp / v for vp, v in zip(var_plain, var)], "o-", color=col, label=name)
    bx.set(yscale="log", xlabel="$K$", ylabel="variance reduction", title="(b) arithmetic Asian, 16 dates")
    bx.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch09-asian.pdf")
    plt.close(fig)


def degeneracy_figure(out):
    """(a) Effective sample size against the dimension; (b) running sample standard deviations."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.45))
    dims = np.unique(np.round(np.logspace(0, np.log10(256), 25)).astype(int))
    c, energy, beta = 0.25, 1.0, 0.9
    cases = ((COBALT, rf"shift ${c}$ per date", lambda m: np.full(m, c), 1.0),
             (GOLD, r"total shift $|\mu|=1$", lambda m: np.full(m, energy / np.sqrt(m)), 1.0),
             (GREEN, rf"scale $\beta={beta}$ per date", lambda m: np.zeros(m), beta))
    rng = np.random.default_rng(9)
    for col, name, drift, b in cases:
        theory = [np.exp(-np.sum(np.square(drift(m)))) if b == 1 else (np.sqrt(2 * b * b - 1) / b**2) ** m
                  for m in dims]
        ax.plot(dims, theory, color=col, label=name)
        for m in dims[::4]:
            mu = drift(m)
            x = mu + b * rng.standard_normal((10_000, m))
            logw = np.sum(im.gaussian_log_lr(x, 0.0, 1.0, mu, b), axis=1)
            w = np.exp(logw - logw.max())
            ax.plot(m, im.effective_sample_size(w) / w.size, "o", ms=3, color=col)
    ax.set(xscale="log", yscale="log", xlabel="dimension $m$", ylabel="ESS$/n$", title="(a) weight degeneracy",
           ylim=(1e-5, 2))
    ax.legend(fontsize=6.8, loc="lower left")

    n_max = 2**22
    ns = np.unique(np.round(np.logspace(2, np.log10(n_max), 60)).astype(int))
    for b, col, seeds in ((1.0, COBALT, (1,)), (0.75, GREEN, (1,)), (0.55, GOLD, (1, 2, 3))):
        p, m2 = im.is_moments("call", S, K0, T, r, y, SIG, 0.27, b)
        if np.isfinite(m2):
            bx.axhline(np.sqrt(m2 - p * p), color=col, lw=0.8, ls="--")
        for k, sd in enumerate(seeds):
            Y, _, _ = im.sample_terminal_is("call", S, K0, T, r, y, SIG, 0.27, b, n_max, seed=sd)
            c1, c2 = np.cumsum(Y), np.cumsum(Y * Y)
            sdev = np.sqrt((c2[ns - 1] - c1[ns - 1] ** 2 / ns) / (ns - 1))
            bx.plot(ns, sdev, color=col, lw=1.1, label=rf"$\beta={b:g}$" if k == 0 else None)
    bx.set(xscale="log", yscale="log", xlabel="$n$", ylabel=r"$\hat s_n$", title=r"(b) $\alpha=0.27$, $K=100$")
    bx.set_yticks([2, 3, 5, 10, 20, 30])
    bx.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    bx.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch09-degeneracy.pdf")
    plt.close(fig)


def main(out):
    use_style()
    densities_figure(out)
    contours_figure(out)
    rare_figure(out)
    asian_figure(out)
    degeneracy_figure(out)
