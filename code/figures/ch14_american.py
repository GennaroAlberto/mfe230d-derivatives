"""Figures for Chapter 14 (American options by simulation)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d.american import (
    LS_CASE,
    bermudan_grid,
    bias_study,
    cash_flows_at,
    duality_study,
    laguerre_basis,
    lsm_boundary,
    lsm_fit,
    poly_basis,
    put_payoff,
    simulate,
)
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
PURPLE = "#6250D6"
K, R = LS_CASE["K"], LS_CASE["r"]


def boundary_figure(out):
    """(a) The regression at t = 0.5; (b) exercise boundaries of LSM and of the grid."""
    S0, sig, T, m = 36.0, 0.20, 1.0, 50
    g = put_payoff(K)
    det = bermudan_grid(S0, K, T, R, 0.0, sig, m, h=1e-3, details=True)
    basis = laguerre_basis(3, scale=K)
    train = simulate(S0, T, m, R, 0.0, sig, 100_000, seed=[14, 0], antithetic=True)
    fit = lsm_fit(train, g, basis, R, T, antithetic=True)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    # (a) the regression at k = 25 (t = 0.5), in time-t_k units, minus the exercise value
    k = 25
    tk = det["t"][k]
    Y = cash_flows_at(train, fit, g, basis, k)
    Sk = train[:, k]
    itm = np.flatnonzero(Sk < K)
    pick = np.random.default_rng(1).choice(itm, 1500, replace=False)
    ax.scatter(Sk[pick], np.exp(R * tk) * Y[pick] - (K - Sk[pick]), s=2, color=GREY, alpha=0.35, lw=0,
               label="realised cash flow $-$ $(K-s)$")
    sg = np.linspace(26.0, K, 300)
    ax.plot(sg, np.exp(R * tk) * (basis(sg) @ fit["coefs"][k]) - (K - sg), color=COBALT,
            label="fitted $\\hat c_k(s)-(K-s)$")
    ax.plot(det["s"], det["C"][k] - (K - det["s"]), color=GOLD, ls="--", label="exact $c_k(s)-(K-s)$")
    ax.axhline(0.0, color=INK, lw=0.8)
    ax.set(xlim=(26, K), ylim=(-3, 5), xlabel="$s$", ylabel="value minus exercise value",
           title="(a) regression at $t_k=0.5$")
    ax.legend(loc="upper left", fontsize=6.8, markerscale=4)

    # (b) exercise boundaries
    s_grid = np.linspace(20.0, K, 4001)
    t = det["t"]
    bx.plot(t[1:], det["boundary"][1:], color=GOLD, label="grid (exact)")
    b_lsm = lsm_boundary(fit, g, basis, s_grid)
    bx.plot(t[1:m], b_lsm[1:m], color=COBALT, marker="o", ms=2.2, lw=0.9, label="LSM, Laguerre")
    quad = poly_basis(2, scale=K)
    b_poly = lsm_boundary(lsm_fit(train, g, quad, R, T, antithetic=True), g, quad, s_grid)
    bx.plot(t[1:m], b_poly[1:m], color=GREEN, marker="s", ms=2.0, lw=0.8, label="LSM, $1,x,x^2$")
    bx.set(xlabel="$t$", ylabel="critical price $b(t)$", xlim=(0, 1), ylim=(31, 40.5),
           title="(b) exercise boundary")
    bx.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch14-boundary.pdf")
    plt.close(fig)


def bias_figure(out):
    """In-sample LSM and the exact value of the fitted policy against the number of paths."""
    design = ((100, 150), (300, 80), (1000, 40), (3000, 30), (10000, 20))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for deg, color in ((2, COBALT), (8, GOLD)):
        ns, ins, ine, outs, oute = [], [], [], [], []
        for N, reps in design:
            res = bias_study(degree=deg, n_train=N, reps=reps)
            a, b = res["in_sample"] - res["true"], res["policy"] - res["true"]
            ns.append(N)
            ins.append(a.mean())
            ine.append(2 * a.std(ddof=1) / np.sqrt(reps))
            outs.append(b.mean())
            oute.append(2 * b.std(ddof=1) / np.sqrt(reps))
        J = deg + 1
        ax.errorbar(ns, ins, yerr=ine, color=color, marker="o", ms=3, capsize=2,
                    label=f"in-sample, $J={J}$")
        ax.errorbar(ns, outs, yerr=oute, color=color, marker="s", ms=3, capsize=2, ls="--",
                    label=f"out-of-sample, $J={J}$")
    ax.axhline(0.0, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set(xlabel="training paths $N$", ylabel="mean estimate $-$ Bermudan price")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch14-bias.pdf")
    plt.close(fig)


def duality_figure(out):
    """Empirical distributions of max_k (Z_k - M_k) for four martingales M (LS case S0 = 36)."""
    res = duality_study()
    U0, L0 = res["true"], res["policy"]
    smp = res["samples"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    curves = (("european", "European put martingale", COBALT),
              ("ab_100", "Andersen--Broadie, 100 inner paths", GOLD),
              ("ab_exact", "Andersen--Broadie, exact $c_k$", GREEN),
              ("doob", "Doob martingale $M^*$ (grid)", PURPLE))
    for axis in (ax, bx):
        for key, label, color in curves:
            v = np.sort(smp[key])
            axis.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=color, label=label)
        axis.axvline(U0, color=INK, lw=0.8, ls=":")
    bx.axvline(L0, color=GREY, lw=0.8, ls="--")
    ax.set(xlim=(3.9, 5.8), ylim=(0, 1.02), xlabel="$\\max_k\\,(Z_k-M_k)$", ylabel="empirical distribution",
           title="(a) four martingales")
    bx.set(xlim=(4.470, 4.535), ylim=(0, 1.02), xlabel="$\\max_k\\,(Z_k-M_k)$", title="(b) detail")
    bx.set_xticks([4.47, 4.49, 4.51, 4.53])
    ax.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch14-duality.pdf")
    plt.close(fig)


def main(out):
    use_style()
    boundary_figure(out)
    bias_figure(out)
    duality_figure(out)
