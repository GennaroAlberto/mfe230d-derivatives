"""Figures for Chapter 19 (Stochastic volatility): the branch cut of Heston's original formula and the
convergence of three Fourier inversions; smiles produced by rho, xi and kappa; the bias of the full-truncation
Euler and QE schemes against the Fourier price."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from mfe230d import heston as H
from mfe230d.style import COBALT, GOLD, GREY, use_style

GREEN = "#1BAF7A"
S, R, Y = 100.0, 0.01, 0.02
LAB = dict(H.TEST, rho=-0.7)


def fourier_figure(out):
    """(a) Re phi along the real axis, stable and original forms (Fang--Oosterlee parameters, T = 10, r = y = 0);
    (b) error of the ATM put against the number of nodes N for the three inversions (test case, rho = -0.7)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    u = np.linspace(0.0, 6.0, 3001)
    a = H.charfn(u, 1.0, 10.0, 0.0, 0.0, **H.FO)
    b = H.charfn(u, 1.0, 10.0, 0.0, 0.0, form="heston", **H.FO)
    jump = np.abs(np.diff(b)) > 0.05
    bb = b.real.copy()
    bb[1:][jump] = np.nan                       # do not draw the vertical jumps
    ax.plot(u, a.real, color=COBALT, label="stable form")
    ax.plot(u, bb, color=GOLD, ls="--", label="Heston (1993) form")
    ax.set(xlabel="$u$", ylabel=r"Re $\varphi(u)$", title="(a) $T=10$, principal logarithms")
    ax.legend(fontsize=7.5)

    ref = H.price(100.0, S, 1.0, R, Y, "put", **LAB)
    lo, hi = H.cos_interval(S, 100.0, 1.0, R, Y, **LAB)
    U = 125.0
    Ns = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    floor = 1e-15
    err = {
        "Gil-Pelaez": [H.gil_pelaez_price(100.0, S, 1.0, R, Y, "put", h=U / N, n=N, **LAB) for N in Ns],
        r"Carr--Madan, $\alpha=1.5$": [H.carr_madan_price(100.0, S, 1.0, R, Y, "put", alpha=1.5, h=U / N, n=N, **LAB)
                                       for N in Ns],
        r"Carr--Madan, $\alpha=9.4$": [H.carr_madan_price(100.0, S, 1.0, R, Y, "put", alpha=9.4, h=U / N, n=N, **LAB)
                                       for N in Ns],
        "COS": [H.cos_price(100.0, S, 1.0, R, Y, "put", N=N, interval=(lo, hi), **LAB) for N in Ns],
    }
    styles = [(COBALT, "-", "o"), (GOLD, "-", "s"), (GOLD, "--", "^"), (GREEN, "-", "D")]
    for (name, vals), (c, ls, mk) in zip(err.items(), styles):
        bx.semilogy(Ns, np.maximum(np.abs(np.array(vals) - ref), floor), color=c, ls=ls, marker=mk, ms=3, label=name)
    bx.axhline(1e-8, color=GREY, lw=0.8, ls=":")
    bx.set(xscale="log", xlabel="number of nodes $N$", ylabel="absolute error", ylim=(1e-15, 1e2),
           title="(b) ATM put, $\\rho=-0.7$")
    bx.set_xticks([16, 32, 64, 128, 256, 512])
    bx.set_xticklabels(["16", "32", "64", "128", "256", "512"])
    bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch19-fourier.pdf")
    plt.close(fig)


def smile_figure(out):
    """Implied volatility against the strike at T = 1 (test case, V0 = theta = 0.04): (a) rho, (b) xi, (c) kappa."""
    Ks = np.linspace(60.0, 150.0, 46)
    base = dict(LAB)
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=True)
    panels = [("rho", [-0.7, 0.0, 0.7], r"(a) $\rho$", r"$\rho={}$"),
              ("xi", [0.1, 0.3, 0.6], r"(b) $\xi$ ($\rho=-0.7$)", r"$\xi={}$"),
              ("kappa", [0.5, 2.0, 8.0], r"(c) $\kappa$ ($\rho=-0.7$)", r"$\kappa={}$")]
    for ax, (name, vals, title, lab) in zip(axs, panels):
        for v, c in zip(vals, (COBALT, GOLD, GREEN)):
            prm = dict(base, **{name: v})
            ax.plot(Ks, 100 * H.implied_vols(Ks, S, 1.0, R, Y, **prm), color=c, label=lab.format(v))
        ax.axhline(20.0, color=GREY, lw=0.6, ls=":")
        ax.set(xlabel="strike $K$", title=title)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("implied volatility (%)")
    fig.tight_layout()
    fig.savefig(out / "ch19-smiles.pdf")
    plt.close(fig)


def bias_figure(out, n_test=2**19, n_case=2**18):
    """Absolute bias of the ATM put against the step h for full-truncation Euler and QE; the dotted line is twice
    the standard error of the QE estimates."""
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))
    cases = [(LAB, n_test, r"(a) test case, $\rho=-0.7$ (Feller holds)"),
             (H.CASE_I, n_case, r"(b) $\kappa=0.5$, $\xi=1$, $\rho=-0.9$ (Feller fails)")]
    results = []
    for ax, (prm, n, title) in zip(axs, cases):
        b = H.bias_study(n_paths=n, **prm)
        results.append(b)
        for sch, c, mk, name in (("euler", GOLD, "s", "full truncation"), ("qe", COBALT, "o", "QE")):
            h, bias, se = b[sch]["h"], b[sch]["bias"], b[sch]["se"]
            sig = np.abs(bias) > 2 * se
            ax.loglog(h, np.abs(bias), color=c, lw=1.0)
            ax.loglog(h[sig], np.abs(bias[sig]), ls="none", marker=mk, ms=4, color=c, label=name)
            ax.loglog(h[~sig], np.abs(bias[~sig]), ls="none", marker=mk, ms=4, mfc="white", color=c)
        ax.loglog(b["qe"]["h"], 2 * b["qe"]["se"], color=GREY, ls=":", lw=1.0, label="2 standard errors")
        ax.loglog(b["euler"]["h"], 0.4 * b["euler"]["h"], color=GREY, lw=0.6, ls="--", label="slope 1")
        ax.set(xlabel="time step $h$", title=title, ylim=(5e-4, 10))
    axs[0].legend(fontsize=7, loc="upper left")
    axs[0].set_ylabel("|bias| of the ATM put")
    fig.tight_layout()
    fig.savefig(out / "ch19-bias.pdf")
    plt.close(fig)
    return results


def main(out):
    use_style()
    fourier_figure(out)
    smile_figure(out)
    return bias_figure(out)


if __name__ == "__main__":
    import pathlib
    import sys
    res = main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "latex/figures"))
    for b in res:
        print("reference", b["reference"])
        for s in ("euler", "qe"):
            print(s, np.round(b[s]["bias"], 4), np.round(b[s]["se"], 4))
