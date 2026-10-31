"""Figures for Chapter 4 (Static replication and the Kolmogorov equations)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import bs
from mfe230d.replication import (bs_market, fd_grid, forward_solve, grid_payoff, grid_price, grid_replication,
                                 local_vol_terminal, power_call, skew_local_vol, sqrt_call, sqrt_call_digital)
from mfe230d.standard import green
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
TC = bs.TEST_CASE
S, T, R, Y, SIG = TC["S"], TC["T"], TC["r"], TC["y"], TC["sigma"]
H_JUMP = 120.0


def _grid(h):
    """Strikes 100 - h, 100, ..., 400."""
    return 100.0 + h * np.arange(-1, int(round(300.0 / h)) + 1)


def replication(out):
    """Square-root call with a digital bonus: the strip with and without the digital, and the price error."""
    m = bs_market(S, T, R, Y, SIG)
    exact = power_call(S, 10.0, T, R, Y, SIG, 0.5) + bs.digital_call(S, H_JUMP, T, R, Y, SIG)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))

    x = np.linspace(80.0, 160.0, 2001)
    k = _grid(10.0)
    rep_f = grid_replication(sqrt_call_digital(k), k)
    rep_g = grid_replication(sqrt_call(k), k)
    ax.plot(x, sqrt_call_digital(x), color=INK, lw=1.2, label="payoff $f$")
    ax.plot(x, grid_payoff(rep_f, x), color=GOLD, lw=1.3, ls="--", label="strip of calls")
    ax.plot(x, grid_payoff(rep_g, x) + (x >= H_JUMP), color=COBALT, lw=1.3, ls=":",
            label="strip + digital")
    ax.plot(k[(k >= 80) & (k <= 160)], sqrt_call_digital(k[(k >= 80) & (k <= 160)]), "o", color=INK, ms=2.5)
    ax.set(xlabel="$S_T$", ylabel="payoff", xlim=(80, 160), title="strikes every $h=10$")
    ax.legend(fontsize=7, loc="upper left")

    hs = 20.0 / 2.0 ** np.arange(0, 8)
    e1, e2 = [], []
    for h in hs:
        kk = _grid(h)
        e1.append(grid_price(grid_replication(sqrt_call_digital(kk), kk), m) - exact)
        e2.append(grid_price(grid_replication(sqrt_call(kk), kk), m) + bs.digital_call(S, H_JUMP, T, R, Y, SIG) - exact)
    e1, e2 = np.abs(e1), np.abs(e2)
    g = green(H_JUMP, S, T, R, Y, SIG)
    ax2.loglog(hs, e1, "o-", color=GOLD, ms=3, lw=1.3, label="strip of calls")
    ax2.loglog(hs, e2, "s-", color=COBALT, ms=3, lw=1.3, label="strip + digital")
    ax2.loglog(hs, 0.5 * hs * g, color=GREY, lw=0.9, ls="--", label=r"$\frac{1}{2}h\,G(120)$")
    ax2.loglog(hs, 0.3 * e2[-1] * (hs / hs[-1]) ** 2, color=GREY, lw=0.9, ls=":", label="slope 2")
    ax2.set(xlabel="strike spacing $h$", ylabel="|price error|", title="error against spacing")
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "ch04-replication.pdf")
    plt.close(fig)


def kolmogorov(out, n_paths=200_000, seed=4):
    """One forward solve in a local-volatility model: density and implied volatilities, against Monte Carlo."""
    b = lambda t, z: R - Y - 0.5 * skew_local_vol(np.exp(z)) ** 2
    a = lambda t, z: skew_local_vol(np.exp(z))
    x = fd_grid(np.log(S), 8 * 0.35, 801)
    h = x[1] - x[0]
    G = forward_solve(x, 400, T, b, a, R, n_t=400)
    xi = np.exp(x)
    ST = local_vol_terminal(S, T, R, Y, skew_local_vol, n_steps=250, n_paths=n_paths, seed=seed)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    edges = np.linspace(40.0, 200.0, 81)
    counts, _ = np.histogram(ST, bins=edges)
    dens = np.exp(-R * T) * counts / (ST.size * np.diff(edges))
    ax.bar(edges[:-1], dens, width=np.diff(edges), align="edge", color=COBALT, alpha=0.25, lw=0,
           label="Euler paths")
    sel = (xi > 40) & (xi < 200)
    ax.plot(xi[sel], G[sel] / xi[sel], color=COBALT, lw=1.4, label="forward equation")
    ax.plot(xi[sel], green(xi[sel], S, T, R, Y, SIG), color=GOLD, lw=1.2, ls="--", label="lognormal, $\\sigma=20\\%$")
    ax.set(xlabel="$S_T$", ylabel="discounted density", xlim=(40, 200), title="density of $S_T$")
    ax.legend(fontsize=7)

    Ks = np.linspace(60.0, 160.0, 51)
    iv = [bs.implied_vol(h * np.sum(G * np.maximum(xi - k, 0.0)), S, k, T, R, Y) for k in Ks]
    ax2.plot(Ks, iv, color=COBALT, lw=1.4, label="implied, forward solve")
    Km = np.arange(60.0, 161.0, 10.0)
    pay = np.exp(-R * T) * np.maximum(ST[:, None] - Km, 0.0)
    mean, se = pay.mean(0), pay.std(0, ddof=1) / np.sqrt(ST.size)
    ivm = np.array([bs.implied_vol(m, S, k, T, R, Y) for m, k in zip(mean, Km)])
    lo = np.array([bs.implied_vol(m - 1.96 * s, S, k, T, R, Y) for m, s, k in zip(mean, se, Km)])
    hi = np.array([bs.implied_vol(m + 1.96 * s, S, k, T, R, Y) for m, s, k in zip(mean, se, Km)])
    ax2.errorbar(Km, ivm, yerr=[ivm - lo, hi - ivm], fmt="o", color=INK, ms=2.5, lw=0.8, capsize=1.5,
                 label="implied, Monte Carlo")
    ax2.plot(Ks, skew_local_vol(Ks), color=GOLD, lw=1.2, ls="--", label="local volatility $\\sigma(K)$")
    ax2.set(xlabel="strike $K$", ylabel="volatility", title="all strikes from one solve")
    ax2.legend(fontsize=7)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "ch04-kolmogorov.pdf")
    plt.close(fig)


def main(out):
    use_style()
    replication(out)
    kolmogorov(out)
