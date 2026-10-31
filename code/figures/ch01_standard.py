"""Figures for Chapter 1 (The standard model, revisited)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import bs
from mfe230d.standard import fk_vasicek_check, green, max_relative_residuals
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
TC = bs.TEST_CASE

# The bond of Example exa:sm-vasicek: OU short rate, kappa = 0.5, theta = 4%, xi = 2%, T = 5 years.
VASICEK = dict(T=5.0, kappa=0.5, theta=0.04, xi=0.02)
X0S = np.array([-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
N_PATHS, N_STEPS, SEED0 = 100_000, 50, 2026


def dirac(out):
    """Arrow-Debreu density, digital and call as functions of the strike, for shrinking maturities."""
    S, r, y, sig = TC["S"], TC["r"], TC["y"], TC["sigma"]
    K = np.linspace(50.0, 150.0, 801)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.15))
    for T, color in ((1.0, COBALT), (0.25, GOLD), (0.02, GREEN)):
        lab = f"$T={T:g}$"
        axes[0].plot(K, green(K, S, T, r, y, sig), color=color, lw=1.3, label=lab)
        axes[1].plot(K, bs.digital_call(S, K, T, r, y, sig), color=color, lw=1.3, label=lab)
        axes[2].plot(K, bs.call(S, K, T, r, y, sig), color=color, lw=1.3, label=lab)
    lim = dict(color=INK, lw=0.9, ls="--")
    axes[0].annotate("", xy=(S, 0.16), xytext=(S, 0.0),
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, ls="--"))
    axes[0].text(S + 4, 0.15, r"$\delta_{100}$", color=INK, fontsize=9)
    axes[1].plot(K, np.where(K < S, 1.0, 0.0), **lim, label="$T\\downarrow0$")
    axes[2].plot(K, np.maximum(S - K, 0.0), **lim, label="$T\\downarrow0$")
    axes[0].set(title="Arrow–Debreu density", ylim=(0, 0.17))
    axes[1].set(title="digital call", ylim=(-0.03, 1.08))
    axes[2].set(title="call", ylim=(-2, 52))
    for ax in axes:
        ax.set_xlabel("strike $K$")
        ax.set_xlim(50, 150)
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "ch01-dirac.pdf")
    plt.close(fig)


def vasicek_errors(n_paths=N_PATHS, n_steps=N_STEPS, seed0=SEED0):
    """Monte Carlo minus PDE for the bond at each initial rate, one seed per rate."""
    rows = []
    for i, x0 in enumerate(X0S):
        res = fk_vasicek_check(x0, n_steps=n_steps, n_paths=n_paths, seed=seed0 + i, **VASICEK)
        rows.append((res["error"], res["se"], res["pde"]))
    return np.array(rows)


def feynman_kac(out):
    """Two checks: the bond by Monte Carlo against the PDE, and the residuals of the equations for G."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    rows = vasicek_errors()
    ax.errorbar(100 * X0S, 1e4 * rows[:, 0], yerr=1e4 * 1.96 * rows[:, 1], color=COBALT, marker="o",
                ms=3.5, lw=0, elinewidth=1.2, capsize=3)
    ax.axhline(0.0, color=INK, lw=0.8)
    ax.set(xlabel="initial short rate $x_0$ (%)", ylabel=r"MC $-$ PDE ($\times10^{-4}$)",
           title="bond price: Monte Carlo minus PDE")

    hs = np.logspace(-6, -0.5, 34)
    back, fwd = max_relative_residuals(hs)
    ax2.loglog(hs, back, color=COBALT, marker="o", ms=2.5, lw=1.2, label="backward equation")
    ax2.loglog(hs, fwd, color=GOLD, marker="s", ms=2.5, lw=1.0, ls="--", label="forward equation")
    ref = hs[hs > 1e-4]
    ax2.loglog(ref, 2.5 * ref**2, color=GREY, lw=0.9, ls=":", label="$h^2$")
    ax2.set(xlabel="relative step $h$", ylabel="largest relative residual",
            title="finite-difference residuals of $G$")
    ax2.legend(fontsize=7, loc="upper left")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "ch01-feynman-kac.pdf")
    plt.close(fig)


def main(out):
    use_style()
    dirac(out)
    feynman_kac(out)
