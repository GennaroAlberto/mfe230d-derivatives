"""Figures for Chapter 13 (Monte Carlo Greeks)."""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import bs
from mfe230d import greeks as gr
from mfe230d.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN, PURPLE = PALETTE[2], PALETTE[3]
P = bs.TEST_CASE
S0 = P["S"]


def _mc_rmse(kind, scheme, crn, eps, n, reps, seed, exact):
    """Monte Carlo RMSE of the n-sample finite-difference estimator from ``reps`` replications."""
    rng = np.random.default_rng(seed)
    f = gr.payoff_function(kind, "S", **P)
    err = np.empty(reps)
    for k in range(reps):
        z = rng.standard_normal(n)
        zi = None if crn else [rng.standard_normal(n) for _ in range(2)]
        err[k] = gr.fd_samples(f, S0, eps * S0, z, scheme, zi).mean() - exact
    return np.sqrt(np.mean(err**2))


def bump_figure(out):
    """(a) Call delta: RMSE against the relative bump; (b) digital delta and call gamma;
    (c) optimal RMSE against n for the digital delta."""
    warnings.simplefilter("ignore")
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(6.9, 2.45))
    n = 10_000
    eps = np.logspace(-5, np.log10(0.3), 45)
    styles = [("forward", False, GOLD, "--", "forward, indep."), ("central", False, COBALT, "--", "central, indep."),
              ("forward", True, GOLD, "-", "forward, CRN"), ("central", True, COBALT, "-", "central, CRN")]
    for scheme, crn, col, ls, lab in styles:
        rm = [gr.fd_rmse("call", "S", e * S0, n, scheme, crn, **P) for e in eps]
        ax.plot(eps, rm, color=col, ls=ls, lw=1.3, label=lab)
        exact = float(gr.exact_greek("call", "delta", **P))
        for i, e in enumerate((1e-4, 1e-3, 1e-2, 1e-1)):
            ax.plot(e, _mc_rmse("call", scheme, crn, e, n, 200, 100 + i, exact), "o", ms=3, color=col)
    ax.set(xscale="log", yscale="log", xlabel=r"relative bump $\varepsilon=h/S$", ylabel="RMSE",
           title=r"(a) call delta, $n=10^4$")
    ax.legend(fontsize=6.3, loc="upper center")

    for kind, scheme, col, lab in (("digital", "central", COBALT, "digital delta"),
                                   ("call", "second", GOLD, "call gamma")):
        ex = abs(float(gr.exact_greek(kind, "delta" if scheme == "central" else "gamma", **P)))
        for crn, ls in ((True, "-"), (False, "--")):
            rm = np.array([gr.fd_rmse(kind, "S", e * S0, n, scheme, crn, **P) for e in eps]) / ex
            bx.plot(eps, rm, color=col, ls=ls, lw=1.3, label=f"{lab}, {'CRN' if crn else 'indep.'}")
    e_ref = np.array([1e-4, 1e-2])
    bx.plot(e_ref, 0.06 * (e_ref / 1e-4) ** -0.5, color=GREY, lw=0.8)
    bx.text(1.2e-4, 0.011, r"slope $-1/2$", fontsize=6.5, color=INK)
    bx.set(xscale="log", yscale="log", xlabel=r"relative bump $\varepsilon=h/S$", ylabel="RMSE / exact value",
           title=r"(b) $n=10^4$", ylim=(5e-3, 3e3))
    bx.legend(fontsize=6.3, loc="upper right")

    ns = np.logspace(2, 12, 21)
    for scheme, crn, col, ls, lab in styles:
        rm = [gr.optimal_bump("digital", "S", m, scheme, crn, bounds=(1e-5, 60), **P)["rmse"] for m in ns]
        cx.plot(ns, rm, color=col, ls=ls, lw=1.3, label=lab)
    for slope, y0, txt in ((-0.25, 2e-3, r"$-1/4$"), (-1 / 3, 6e-4, r"$-1/3$"), (-0.4, 1.2e-4, r"$-2/5$")):
        xs = np.array([1e9, 1e11])
        cx.plot(xs, y0 * 2e-3 * (xs / 1e9) ** slope / 2e-3, color=GREY, lw=0.8)
        cx.text(1.3e11, y0 * (100) ** slope, txt, fontsize=7, color=INK, va="center")
    cx.set(xscale="log", yscale="log", xlabel="$n$", ylabel="RMSE at the optimal bump",
           title="(c) digital delta: rates")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "ch13-bump.pdf")
    plt.close(fig)


def course_figure(out, n=1024, seed=1024):
    """The course's shoot-out against the spot, one set of n draws for every S_0."""
    z = np.random.default_rng(seed).standard_normal(n)
    Ss = np.linspace(90, 110, 801)
    q = {k: [] for k in ("dp", "dm", "dc", "g1", "g01", "g001", "pw", "lr", "lrlr", "pwlr")}
    for s in Ss:
        p = dict(P, S=s)
        fd = gr.fd_spot_greeks(z, "call", eps=0.01, **p)
        q["dp"].append(fd["delta_plus"].mean()); q["dm"].append(fd["delta_minus"].mean())
        q["dc"].append(fd["delta_c"].mean())
        for key, e in (("g1", 0.1), ("g01", 0.01), ("g001", 0.001)):
            q[key].append(gr.fd_spot_greeks(z, "call", eps=e, **p)["gamma_c"].mean())
        q["pw"].append(gr.pathwise_samples(z, "call", **p)["delta"].mean())
        q["lr"].append(gr.lr_samples(z, "call", **p)["delta"].mean())
        g = gr.gamma_samples(z, "call", **p)
        q["lrlr"].append(g["lr_lr"].mean()); q["pwlr"].append(g["pw_lr"].mean())
    q = {k: np.array(v) for k, v in q.items()}
    G = bs.greeks(Ss, P["K"], P["T"], P["r"], P["y"], P["sigma"])
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.6), sharex=True)
    ax, bx, cx, dx = axs.ravel()
    ax.plot(Ss, G["delta"], color=INK, lw=1.0, label="Black--Scholes")
    ax.plot(Ss, q["dp"], color=GOLD, lw=1.0, label=r"$\Delta_+$")
    ax.plot(Ss, q["dm"], color=GREEN, lw=1.0, label=r"$\Delta_-$")
    ax.plot(Ss, q["dc"], color=COBALT, lw=1.0, label=r"$\Delta_c$")
    ax.set(ylabel="delta", title=r"(a) bumped deltas, $\varepsilon=0.01$")
    ax.legend(fontsize=6.5)
    bx.plot(Ss, q["g001"], ".", ms=1.2, color=GREY, label=r"$\varepsilon=0.001$")
    bx.plot(Ss, q["g01"], color=GOLD, lw=0.8, label=r"$\varepsilon=0.01$")
    bx.plot(Ss, q["g1"], color=COBALT, lw=1.2, label=r"$\varepsilon=0.1$")
    bx.plot(Ss, G["gamma"], color=INK, lw=1.0, label="Black--Scholes")
    bx.set(ylabel="gamma", title=r"(b) bumped gammas $\Gamma_c$", ylim=(0, 0.075))
    bx.legend(fontsize=6, ncol=4, loc="upper center", handlelength=1.2, columnspacing=0.8)
    cx.plot(Ss, G["delta"], color=INK, lw=1.0, label="Black--Scholes")
    cx.plot(Ss, q["pw"], color=COBALT, lw=1.0, label="pathwise")
    cx.plot(Ss, q["lr"], color=GOLD, lw=1.0, label="likelihood ratio")
    cx.set(xlabel="$S_0$", ylabel="delta", title="(c) pathwise and LR deltas")
    cx.legend(fontsize=6.5)
    dx.plot(Ss, G["gamma"], color=INK, lw=1.0, label="Black--Scholes")
    dx.plot(Ss, q["pwlr"], color=COBALT, lw=1.0, label="PW-LR")
    dx.plot(Ss, q["lrlr"], color=GOLD, lw=1.0, label="LR-LR")
    dx.set(xlabel="$S_0$", ylabel="gamma", title="(d) PW-LR and LR-LR gammas", ylim=(0, 0.03))
    dx.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "ch13-course.pdf")
    plt.close(fig)


def asian_figure(out, n=100_000, seed=2026):
    """(a) Per-sample standard deviations of Asian deltas and vegas against the number of fixings;
    (b) bucketed vegas by the adjoint and by bumping, 12 fixings."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ms = 2 ** np.arange(0, 8)
    sd = {k: [] for k in ("pw_delta", "lr_delta", "pw_vega", "lr_vega")}
    rng = np.random.default_rng(seed)
    for m in ms:
        Z = rng.standard_normal((n, m))
        s = gr.asian_samples(Z, **P)
        for k in sd:
            sd[k].append(s[k].std(ddof=1))
    sd = {k: np.array(v) for k, v in sd.items()}
    ax.plot(ms, sd["pw_vega"] / sd["pw_vega"][0], "o-", color=COBALT, ms=3, label="vega, pathwise")
    ax.plot(ms, sd["lr_vega"] / sd["pw_vega"][0], "s--", color=COBALT, ms=3, label="vega, LR")
    ax.plot(ms, sd["pw_delta"] / sd["pw_delta"][0], "o-", color=GOLD, ms=3, label="delta, pathwise")
    ax.plot(ms, sd["lr_delta"] / sd["pw_delta"][0], "s--", color=GOLD, ms=3, label="delta, LR")
    mref = ms[ms >= 16]
    c = 1.6 * sd["lr_vega"][-1] / sd["pw_vega"][0] / np.sqrt(ms[-1])
    ax.plot(mref, c * np.sqrt(mref), color=GREY, lw=0.8)
    ax.text(20, c * np.sqrt(20) * 1.3, r"$\propto\sqrt{m}$", fontsize=7, color=INK)
    ax.set(xscale="log", yscale="log", xlabel="fixings $m$", ylabel="sd per sample / pathwise at $m=1$",
           title="(a) arithmetic Asian call")
    ax.set_xticks(ms, [str(m) for m in ms])
    ax.legend(fontsize=6.3, loc="upper left")

    m = 12
    Z = np.random.default_rng(seed + 1).standard_normal((n, m))
    sig = P["sigma"] * np.ones(m)
    a = gr.asian_adjoint(Z, P["S"], P["K"], P["T"], P["r"], P["y"], sig)
    v, se = a["vega"].mean(axis=0), a["vega"].std(axis=0, ddof=1) / np.sqrt(n)
    h = 1e-4
    bumped = []
    for j in range(m):
        e = np.zeros(m); e[j] = h
        up = gr.asian_adjoint(Z, P["S"], P["K"], P["T"], P["r"], P["y"], sig + e)["price"]
        dn = gr.asian_adjoint(Z, P["S"], P["K"], P["T"], P["r"], P["y"], sig - e)["price"]
        bumped.append(((up - dn) / (2 * h)).mean())
    j = np.arange(1, m + 1)
    bx.bar(j, v, color=COBALT, alpha=0.75, width=0.7, label="adjoint (one sweep)")
    bx.errorbar(j, v, yerr=1.96 * se, fmt="none", ecolor=INK, lw=0.8, capsize=2)
    bx.plot(j, bumped, "o", color=GOLD, ms=3.5, label="central bumps, CRN")
    bx.set(xlabel="period $j$ (month)", ylabel=r"$\partial P/\partial\sigma_j$",
           title=f"(b) bucketed vegas, $m={m}$; total {v.sum():.2f}")
    bx.set_xticks(j)
    bx.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch13-asian.pdf")
    plt.close(fig)


def main(out):
    use_style()
    bump_figure(out)
    course_figure(out)
    asian_figure(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
