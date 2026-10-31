"""Figures for Chapter 16 (Barriers, lookbacks and the Brownian bridge)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d.barriers import (
    barrier_mc_samples,
    barrier_price,
    bgk_price,
    bridge_extreme,
    discrete_barrier_price,
    knockout_density,
    log_density,
    log_drift,
)
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
TC = dict(T=1.0, r=0.01, y=0.02, sigma=0.2)
S0 = K = 100.0
NU = log_drift(0.01, 0.02, 0.2)


def images(out):
    """Image construction of the knock-out density, and simulated survivors."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.25))
    B = 90.0
    b = np.log(B / S0)
    lam = 2 * NU / 0.2**2
    x = np.linspace(-0.75, 0.6, 600)
    src = log_density(x, 1.0, NU, 0.2)
    img = np.exp(lam * b) * log_density(x - 2 * b, 1.0, NU, 0.2)
    ax.plot(x, src, color=GREY, lw=1.2, label="$p_T(x)$")
    ax.plot(x, -img, color=GOLD, lw=1.2, label=r"$-e^{\lambda b}p_T(x-2b)$")
    ax.plot(x, knockout_density(x, b, 1.0, NU, 0.2), color=COBALT, lw=1.8, label="knock-out density")
    ax.axvline(b, color=INK, ls="--", lw=0.8)
    ax.axhline(0, color=GREY, lw=0.5)
    ax.text(b - 0.07, 2.0, "$b$", color=INK, fontsize=8)
    ax.set(xlabel=r"$x=\log(S_T/S_0)$", ylabel="density", title=f"images, $B={B:.0f}$", ylim=(-2.6, 2.4))
    ax.legend(fontsize=6.5, loc="lower right", handlelength=1.6)

    # survivors: X_T exact; continuous monitoring by the exact minimum of the one-step bridge
    rng = np.random.default_rng(16)
    n = 400_000
    Z = rng.standard_normal((n, 12))
    XT = NU * 1.0 + 0.2 * np.sqrt(1 / 12) * Z.sum(axis=1)
    paths = np.cumsum(NU / 12 + 0.2 * np.sqrt(1 / 12) * Z, axis=1)
    mins = bridge_extreme(0.0, XT, 0.2, 1.0, 1.0 - rng.random(n), "min")
    bins = np.linspace(b, 0.6, 46)
    for sel, color, label in ((mins > b, COBALT, "continuous"),
                              (paths.min(axis=1) > b, GOLD, "12 dates")):
        hist, edges = np.histogram(XT[sel], bins=bins)
        ax2.stairs(hist / (n * np.diff(edges)), edges, color=color, lw=1.1, label=label)
    ax2.plot(x[x > b], knockout_density(x[x > b], b, 1.0, NU, 0.2), color=INK, lw=1.0, ls="--",
             label="image formula")
    ax2.axvline(b, color=INK, ls="--", lw=0.8)
    ax2.set(xlabel=r"$x=\log(S_T/S_0)$", title=r"$4\times10^5$ paths", xlim=(-0.75, 0.6))
    ax2.legend(fontsize=6, loc="upper left", handlelength=1.2, title="survivors", title_fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "ch16-images.pdf")
    plt.close(fig)


def discrete(out):
    """Discrete down-and-out calls: exact, continuous and BGK; errors against m."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.25))
    Bs = np.linspace(85, 100, 16)
    ax.plot(Bs, [float(barrier_price(S0, K, B, **TC)) for B in Bs], color=GREY, lw=1.2, label="continuous")
    for m, color in ((4, GREEN), (12, GOLD), (52, COBALT)):
        ex = [discrete_barrier_price(S0, K, B, **TC, m=m) for B in Bs]
        ax.plot(Bs, ex, color=color, label=f"$m={m}$, exact")
        ax.plot(Bs[:-1], [float(bgk_price(S0, K, B, **TC, m=m)) for B in Bs[:-1]], color=color, ls="--",
                lw=1.1, label=f"$m={m}$, BGK")
    ax.set(xlabel="barrier $B$", ylabel="price", title="discrete down-and-out call")
    ax.legend(fontsize=6, ncol=2, loc="lower left", columnspacing=0.8, handlelength=1.6)

    ms = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    cont = float(barrier_price(S0, K, 95, **TC))
    ex = np.array([discrete_barrier_price(S0, K, 95, **TC, m=m) for m in ms])
    bg = np.array([float(bgk_price(S0, K, 95, **TC, m=m)) for m in ms])
    exu = np.array([discrete_barrier_price(S0, K, 115, **TC, m=m, barrier="up-out") for m in ms])
    bgu = np.array([float(bgk_price(S0, K, 115, **TC, m=m, barrier="up-out")) for m in ms])
    ax2.plot(ms, np.abs(cont - ex), color=GREY, marker="o", ms=2.5, label="DOC $B=95$: continuous formula")
    ax2.plot(ms, np.abs(bg - ex), color=COBALT, marker="o", ms=2.5, label="DOC $B=95$: BGK")
    ax2.plot(ms, np.abs(bgu - exu), color=GOLD, marker="o", ms=2.5, label="UOC $B=115$: BGK")
    ax2.plot(ms, 0.6 * np.sqrt(12 / ms), color=INK, ls=":", lw=0.9, label="slope $-1/2$")
    ax2.plot(ms, 0.025 * 12 / ms, color=INK, ls="-.", lw=0.9, label="slope $-1$")
    ax2.set(xscale="log", yscale="log", xlabel="monitoring dates $m$", ylabel="absolute error", ylim=(1e-7, 5),
            title="errors against the exact discrete price")
    ax2.legend(fontsize=5.8, loc="lower left", handlelength=1.8)
    fig.tight_layout()
    fig.savefig(out / "ch16-discrete.pdf")
    plt.close(fig)


def bridge(out):
    """Continuous down-and-out call by simulation: grid monitoring against the bridge."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.25))
    ms = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    n = 100_000
    exact = float(barrier_price(S0, K, 95, **TC))
    res = {k: [] for k in ("grid", "bridge", "bridge-sample")}
    for i, m in enumerate(ms):
        rng = np.random.default_rng([16, i])
        Z = rng.standard_normal((n, m))
        U = 1.0 - rng.random((n, m))
        for mon in res:
            x = barrier_mc_samples(S0, K, 95, **TC, m=m, Z=Z, U=U, monitor=mon)
            res[mon].append((x.mean(), x.std(ddof=1)))
    for mon, color, label in (("grid", GOLD, "grid monitoring"), ("bridge", COBALT, "bridge: weights $\\prod(1-p_j)$"),
                              ("bridge-sample", GREEN, "bridge: sampled minima")):
        a = np.array(res[mon])
        ax.errorbar(ms, a[:, 0], yerr=2 * a[:, 1] / np.sqrt(n), color=color, marker="o", ms=2.5, capsize=2,
                    lw=1.1, label=label)
        ax2.plot(ms, a[:, 1], color=color, marker="o", ms=2.5, label=label)
    ax.axhline(exact, color=INK, ls="--", lw=0.8, label=f"exact {exact:.5f}")
    ax.set(xscale="log", xlabel="time steps $m$", ylabel="estimate", ylim=(3.4, 7.8),
           title=r"continuous DOC, $B=95$, $10^5$ paths")
    ax.legend(fontsize=6, loc="upper right")
    ax2.set(xscale="log", xlabel="time steps $m$", ylabel="standard deviation per path",
            title="the price of looking between dates")
    ax2.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ch16-bridge.pdf")
    plt.close(fig)


def main(out):
    use_style()
    images(out)
    discrete(out)
    bridge(out)


if __name__ == "__main__":
    import pathlib

    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
