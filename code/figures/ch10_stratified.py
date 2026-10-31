"""Figures for Chapter 10 (stratified sampling, Latin hypercube sampling and conditional Monte Carlo)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

from mfe230d.bs import TEST_CASE, put
from mfe230d.paths import brownian_paths, time_grid
from mfe230d.rng import norm_inv
from mfe230d.stratified import (
    COURSE_OTM,
    HESTON_LAB,
    allocation_factors,
    asian_call_values,
    discounted_call,
    heston_put_samples,
    latin_hypercube,
    linear_direction,
    neyman_allocation,
    normal_edges,
    proportional_allocation,
    stratified_estimate,
    stratified_normals,
    stratified_projection,
    stratum_moments,
    terminal_stratified_normals,
)
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
P = TEST_CASE
S, K, T, r, y, SIG = (P[k] for k in ("S", "K", "T", "r", "y", "sigma"))


def _payoff(Pd):
    S_, K_, T_, r_, y_, s_ = (Pd[k] for k in ("S", "K", "T", "r", "y", "sigma"))
    zs = (np.log(K_ / S_) - (r_ - y_ - 0.5 * s_**2) * T_) / (s_ * np.sqrt(T_))
    return (lambda z: discounted_call(z, S_, K_, T_, r_, y_, s_)), zs


def strata_figure(out):
    """(a) The call against u with 8 equiprobable strata; (b) variance factors against H."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    f, zs = _payoff(P)
    u = np.linspace(1e-4, 1 - 1e-4, 2001)
    ax.plot(u, f(norm_inv(u)), color=COBALT, label="payoff $g(u)$")
    H = 8
    p, m, v = stratum_moments(f, normal_edges(H), kinks=(zs,))
    e = np.linspace(0, 1, H + 1)
    for h in range(H):
        ax.plot([e[h], e[h + 1]], [m[h], m[h]], color=GOLD, lw=2.0, label="stratum mean $\\mu_h$" if h == 0 else None)
        ax.axvline(e[h + 1], color=GREY, lw=0.5, ls=":")
    ax.set(xlabel="$u$", ylabel="discounted payoff", xlim=(0, 1), ylim=(0, 45),
           title="(a) the call against $u$, 8 strata")
    ax.legend(loc="upper left", fontsize=7.5)

    Hs = np.unique(np.round(np.logspace(0, 3, 40)).astype(int))
    cases = [(P, "test-case call", COBALT), (COURSE_OTM, "deep OTM call ($K=150$)", GOLD)]
    rng = np.random.default_rng(1010)
    n, R = 2000, 400
    for Pd, name, color in cases:
        f, zs = _payoff(Pd)
        prop, ney = [], []
        for H in Hs:
            fac = allocation_factors(*stratum_moments(f, normal_edges(H), kinks=(zs,)))
            prop.append(fac["proportional"]); ney.append(fac["neyman"])
        bx.plot(Hs, prop, color=color, label=f"{name}, proportional")
        if Pd is P:
            bx.plot(Hs, ney, color=color, ls="--", label=f"{name}, Neyman")
        tot = allocation_factors(*stratum_moments(f, normal_edges(1), kinks=(zs,)))["total"]
        for H in (2, 10, 50, 200):
            pH = np.full(H, 1.0 / H)
            est = []
            for _ in range(R):
                z, lab = stratified_normals(proportional_allocation(n, pH), rng=rng)
                est.append(stratified_estimate(f(z), lab, pH)["mean"])
            bx.plot(H, n * np.var(est, ddof=1) / tot, "o", ms=3.5, color=color)
            if Pd is P and H <= 50:
                pp, mm, vv = stratum_moments(f, normal_edges(H), kinks=(zs,))
                counts = neyman_allocation(n, pp, np.sqrt(vv), minimum=1)
                est = []
                for _ in range(R):
                    z, lab = stratified_normals(counts, rng=rng)
                    est.append(stratified_estimate(f(z), lab, pp)["mean"])
                bx.plot(H, n * np.var(est, ddof=1) / tot, "o", ms=3.5, mfc="white", color=color)
    bx.set(xscale="log", yscale="log", xlabel="number of strata $H$", ylabel="$n\\,\\mathrm{Var}/\\sigma^2$",
           title="(b) variance factor against plain sampling")
    bx.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch10-strata.pdf")
    plt.close(fig)


def paths_figure(out):
    """(a) Brownian paths stratified on W_T and filled in by the bridge; (b) the Asian call."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    t = time_grid(1.0, 64)
    H = 16
    rng = np.random.default_rng(1020)
    Z, lab = terminal_stratified_normals(np.ones(H, dtype=int), 64, rng=rng)
    W = brownian_paths(t, Z=Z, method="bridge")
    for w in W:
        ax.plot(t, w, color=COBALT, lw=0.7, alpha=0.8)
    ax.plot(np.ones(H), W[:, -1], "o", ms=2.5, color=GOLD)
    for e in normal_edges(H)[1:-1]:
        ax.plot([1.0, 1.04], [e, e], color=INK, lw=0.6)
    ax.set(xlabel="$t$", ylabel="$W_t$", xlim=(0, 1.05), title="(a) one path in each of 16 strata of $W_1$")

    rng = np.random.default_rng(1021)
    m = 12
    sd = np.std(asian_call_values(rng.standard_normal((2**20, m)), S, K, T, r, y, SIG), ddof=1)
    n = 2**17
    Hs = 2 ** np.arange(0, 9)
    v = linear_direction(time_grid(T, m))
    for meth, name, color in (("bridge", "$W_T$, bridge", COBALT), ("increments", "$Z_1$, increments", GOLD),
                              ("average", "$\\Sigma_j W_{t_j}$, projection", GREEN)):
        ratios = []
        for H in Hs:
            pH = np.full(H, 1.0 / H)
            counts = proportional_allocation(n, pH)
            if meth == "average":
                xi, lab = stratified_normals(counts, rng=rng)
                vals = asian_call_values(stratified_projection(v, xi, rng=rng), S, K, T, r, y, SIG, "increments")
            else:
                Z, lab = terminal_stratified_normals(counts, m, rng=rng)
                vals = asian_call_values(Z, S, K, T, r, y, SIG, meth)
            ratios.append(stratified_estimate(vals, lab, pH)["se"] * np.sqrt(n) / sd)
        bx.plot(Hs, ratios, "o-", ms=3, color=color, label=name)
    nl, reps = 2**10, 2000
    for meth, ls, name in (("bridge", "--", "LHS, bridge"), ("increments", ":", "LHS, increments")):
        est = [asian_call_values(norm_inv(latin_hypercube(nl, m, rng=rng)), S, K, T, r, y, SIG, meth).mean()
               for _ in range(reps)]
        bx.axhline(np.std(est, ddof=1) * np.sqrt(nl) / sd, color=GREY, ls=ls, lw=1.2, label=name)
    bx.set(xscale="log", xlabel="number of strata $H$", ylabel="standard-error ratio", ylim=(0, 1.42),
           title="(b) Asian call, 12 fixings")
    bx.set_xticks(Hs, [str(h) for h in Hs])
    bx.legend(fontsize=6.5, loc="upper right", ncol=2, handlelength=1.6, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out / "ch10-paths.pdf")
    plt.close(fig)


def mixing_figure(out):
    """(a) Plain and conditional put values against the realised volatility; (b) ratio against rho."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    H_ = HESTON_LAB
    s = heston_put_samples(S, K, T, r, y, 0.0, H_["v0"], H_["kappa"], H_["theta"], H_["xi"], m=64, n=2000, seed=1030)
    vol = np.sqrt(s["Q"] / T)
    ax.plot(vol, s["plain"], ".", ms=2, color=COBALT, alpha=0.5, label="plain payoffs")
    ax.plot(vol, s["conditional"], ".", ms=2, color=GOLD, label="conditional values")
    ax.set(xlabel="realised volatility $\\sqrt{Q_T/T}$", ylabel="discounted put",
           title="(a) $\\rho=0$: 2000 variance paths")
    ax.legend(fontsize=7.5, loc="upper left", markerscale=4)

    rhos = np.array([0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -0.95])
    for xi, color in ((0.3, COBALT), (0.6, GOLD)):
        many = heston_put_samples(S, K, T, r, y, tuple(rhos), H_["v0"], H_["kappa"], H_["theta"], xi, m=64,
                                  n=50_000, seed=1031)
        ratios = [many[float(rho)]["conditional"].std(ddof=1) / many[float(rho)]["plain"].std(ddof=1)
                  for rho in rhos]
        bx.plot(-rhos, ratios, "o-", ms=3, color=color, label=f"$\\xi={xi}$")
    bx.set(xlabel="$-\\rho$", ylabel="standard-error ratio", ylim=(0, 1.02),
           title="(b) conditional against plain")
    bx.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch10-mixing.pdf")
    plt.close(fig)


def lhs_figure(out):
    """(a) A Latin hypercube design of 10 points; (b) n Var / sigma^2 against n for three integrands."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw=dict(width_ratios=[1, 1.5]))
    rng = np.random.default_rng(1040)
    n = 10
    V = latin_hypercube(n, 2, rng=rng)
    for k in range(n + 1):
        ax.axhline(k / n, color=GREY, lw=0.4)
        ax.axvline(k / n, color=GREY, lw=0.4)
    ax.plot(V[:, 0], V[:, 1], "o", ms=4, color=COBALT)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="$u_1$", ylabel="$u_2$", title="(a) 10 points")
    ax.grid(False)

    d = 5
    ve = 0.5 * (np.e**2 - 1) - (np.e - 1) ** 2          # Var e^U for U uniform
    c = 6.0                                              # weight of the interaction (u1 - 1/2)(u2 - 1/2)
    tests = {
        "additive": (lambda U: np.exp(U).sum(axis=1), 0.0, COBALT),
        "additive + interaction": (lambda U: np.exp(U).sum(axis=1) + c * (U[:, 0] - 0.5) * (U[:, 1] - 0.5),
                                   (c * c / 144) / (d * ve + c * c / 144), GOLD),
        "product $\\prod_j 2u_j$": (lambda U: np.prod(2 * U, axis=1),
                                    1 - (d / 3) / ((4 / 3) ** d - 1), GREEN),
    }
    ns = np.array([4, 8, 16, 32, 64, 128, 256])
    R = 400
    for name, (g, limit, color) in tests.items():
        var = float(np.var(g(rng.random((2**20, d)))))
        vals = []
        for nn in ns:
            est = [g(latin_hypercube(nn, d, rng=rng)).mean() for _ in range(R)]
            vals.append(nn * np.var(est, ddof=1) / var)
        bx.plot(ns, vals, "o-", ms=3, color=color, label=name)
        if limit > 0:
            bx.axhline(limit, color=color, ls="--", lw=0.8)
    bx.axhline(1.0, color=INK, lw=0.6)
    bx.set(xscale="log", yscale="log", xlabel="$n$", ylabel="$n\\,\\mathrm{Var}/\\sigma^2$",
           title="(b) Latin hypercube in $d=5$")
    bx.set_xticks(ns, [str(k) for k in ns])
    bx.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch10-lhs.pdf")
    plt.close(fig)


def main(out):
    use_style()
    strata_figure(out)
    paths_figure(out)
    mixing_figure(out)
    lhs_figure(out)
