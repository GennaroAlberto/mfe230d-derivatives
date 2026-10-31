"""Figures for Chapter 5 (Random numbers)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri

from mfe230d import rng as R
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def _edge_on(ax, P, h, color, s=1.5):
    """Scatter d-tuples P seen along their planes h.x = const (normal coordinate horizontal)."""
    h = np.asarray(h, dtype=float)
    n = h / np.linalg.norm(h)
    e = np.cross(n, [1.0, 0.0, 0.0])
    e /= np.linalg.norm(e)
    ax.scatter(P @ n, P @ e, s=s, color=color, lw=0, rasterized=True)
    ax.set_xlabel(r"along $h$, edge-on")
    ax.set_xticks([])
    ax.set_yticks([])


def _running_max(x, y, w=15):
    """Block maxima of y over consecutive blocks of w points (an envelope of an oscillating error)."""
    k = len(y) // w
    return x[:k * w].reshape(k, w).mean(1), y[:k * w].reshape(k, w).max(1)


def lattice(out):
    """Pairs and triples of LCGs lie on few lines and planes (Marsaglia's lattice)."""
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.95))
    # (a) the course's generator: the 16 pairs lie on 4 lines
    x = R.lcg_sequence(17, **R.COURSE_LCG, x0=0).astype(float) / 16
    ax = axes[0]
    for k in (-11, -3, 5, 13):                     # x_{i+1} - x_i in {5, 13, -3, -11} / 16
        g = np.linspace(0, 1, 2)
        ax.plot(g, g + k / 16, color=GREY, lw=0.6, alpha=0.6)
    ax.scatter(x[:-1], x[1:], s=9, color=COBALT, zorder=3)
    ax.set(xlim=(-0.03, 1), ylim=(-0.03, 1), xlabel="$u_i$", ylabel="$u_{i+1}$",
           title=r"(a) $m=16$, pairs", aspect="equal")
    # (b) m = 4096, a = 65: pairs look fine ...
    m, a = 4096, 65
    u = R.lcg_sequence(m + 2, m, a, 1, 0).astype(float) / m
    ax = axes[1]
    ax.scatter(u[:-1], u[1:], s=0.6, color=COBALT, lw=0, rasterized=True)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="$u_i$", ylabel="$u_{i+1}$",
           title=r"(b) $m=2^{12}$, $a=65$, pairs", aspect="equal")
    # (c) ... but its triples lie on at most 4 planes (3 occupied), h = (1, -2, 1)
    P = R.tuples(u, 3)[:m]
    h, _ = R.spectral_test(a, m, 3)
    _edge_on(axes[2], P, h, COBALT, s=0.8)
    axes[2].set_title("(c) same, triples")
    # (d) RANDU: 15 planes, h = (9, -6, 1)
    uR = R.uniforms(10_002, "randu", seed=0)             # 10^4 overlapping triples
    hR, _ = R.spectral_test(R.RANDU["a"], R.RANDU["m"], 3)
    _edge_on(axes[3], R.tuples(uR, 3), hR, GOLD, s=0.5)
    axes[3].set_title("(d) RANDU, triples")
    for ax in axes[:2]:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "ch05-lattice.pdf")
    plt.close(fig)


def diagnostics(out):
    """The homework diagnostics on Box--Muller normals built from PCG32 uniforms."""
    n = 100_000
    u = R.uniforms(n, "pcg32", seed=2024)
    z = R.normals_box_muller(u)
    edges = np.round(np.arange(-4.0, 4.0001, 0.1), 10)
    b = R.binned_counts(z, edges, ndtr)
    mid = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.8))
    ax = axes[0, 0]
    ax.bar(mid, b["counts"], width=0.1, color=COBALT, alpha=0.35, lw=0)
    ax.errorbar(mid, b["expected"], yerr=2 * b["sd"], fmt="none", ecolor=INK, elinewidth=0.6, capsize=0)
    ax.set(xlabel="$z$", ylabel="count per bin", xlim=(-4, 4),
           title=fr"(a) histogram, $n=10^5$; bars: $np_j\pm2\sigma_j$")
    ax = axes[0, 1]
    zz = b["z"]
    out_band = np.abs(zz) > 2
    ax.vlines(mid, 0, zz, color=COBALT, lw=1.2)
    ax.scatter(mid[out_band], zz[out_band], color=GOLD, s=10, zorder=3)
    for y in (-2, 2):
        ax.axhline(y, color=GREY, lw=0.8, ls="--")
    ax.set(xlabel="$z$", ylabel=r"$(O_j-np_j)/\sigma_j$", xlim=(-4, 4),
           title=f"(b) standardised counts: {out_band.sum()} of {len(zz)} beyond $\\pm2$")
    ax = axes[1, 0]
    lags = np.arange(1, 21)
    r1, r2 = R.autocorrelation(z, 20), R.autocorrelation(z**2, 20)
    ax.plot(lags - 0.12, r1, "o", color=COBALT, ms=3.5, label="$z_i$")
    ax.plot(lags + 0.12, r2, "s", color=GOLD, ms=3.5, label="$z_i^2$")
    band = 1.96 / math.sqrt(len(z))
    ax.axhspan(-band, band, color=GREY, alpha=0.15, lw=0)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set(xlabel="lag $k$", ylabel="autocorrelation $r_k$", xticks=[1, 5, 10, 15, 20],
           ylim=(-2.0 * band, 3.0 * band), title=r"(c) autocorrelations; band $\pm1.96/\sqrt{n}$")
    ax.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.95, edgecolor="none",
              handletextpad=0.2, columnspacing=1.0)
    ax = axes[1, 1]
    k = 4000
    ax.scatter(ndtr(z[:k]), ndtr(z[1:k + 1]), s=0.8, color=COBALT, lw=0, rasterized=True)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="$N(z_i)$", ylabel="$N(z_{i+1})$", aspect="equal",
           title="(d) lag-1 pairs mapped back by $N$")
    fig.tight_layout()
    fig.savefig(out / "ch05-diagnostics.pdf")
    plt.close(fig)


def inversion(out):
    """Newton and Halley for the semicircle law; accuracy of normal-quantile approximations."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ax = axes[0]
    u = np.linspace(1e-4, 1 - 1e-4, 6000)
    Hn = R.semicircle_inverse(u, 3, "newton", history=True)
    Hh = R.semicircle_inverse(u, 2, "halley", history=True)
    floor = 1e-16
    for k, color in zip(range(4), (GREY, COBALT, GOLD, GREEN)):
        err = np.abs(R.semicircle_cdf(Hn[k]) - u)
        ax.plot(*_running_max(u, np.maximum(err, floor)), color=color, lw=1.0,
                label="guess $2u-1$" if k == 0 else f"Newton, {k} step{'s' if k > 1 else ''}")
    for k, color in zip((1, 2), (COBALT, GOLD)):
        err = np.abs(R.semicircle_cdf(Hh[k]) - u)
        ax.plot(*_running_max(u, np.maximum(err, floor)), color=color, lw=1.0, ls="--",
                label=f"Halley, {k} step{'s' if k > 1 else ''}")
    ax.set(yscale="log", xlabel="$u$", ylabel="$|F(x_k)-u|$", ylim=(3e-17, 1), xlim=(0, 1),
           title="(a) semicircle law")
    ax.legend(fontsize=6.5, loc="lower center", ncol=2, frameon=True, framealpha=0.9, edgecolor="none")
    ax = axes[1]
    x = np.linspace(-8.2, 8.2, 6001)
    uu = ndtr(x)
    exact = ndtri(uu)
    for f, color, label in ((R.norm_inv_bsm, GOLD, "Beasley–Springer–Moro"),
                            (R.norm_inv_acklam, COBALT, "Acklam"),
                            (R.norm_inv, GREEN, "Acklam + one Halley step")):
        err = np.abs(f(uu) - exact)
        ax.plot(*_running_max(x, np.maximum(err, floor)), color=color, lw=0.9, label=label)
    ax.set(yscale="log", xlabel=r"$x=N^{-1}(u)$", ylabel="absolute error", ylim=(3e-17, 1e-5),
           title=r"(b) normal quantile $N^{-1}$")
    ax.legend(fontsize=6.5, loc="upper center", frameon=True, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(out / "ch05-inversion.pdf")
    plt.close(fig)


def rejection(out):
    """Envelopes for the semicircle law and their acceptance probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    x = np.linspace(-1, 1, 1001)
    for th, color, label in ((0.0, GREY, "flat, $\\pi/4$"),
                             (math.pi / 6, GOLD, "tangent at $\\pm1/2$"),
                             (math.pi / 4, COBALT, "tangent at $\\pm\\sqrt{2}/2$")):
        ax.plot(x, R.semicircle_envelope(x, th), color=color, lw=1.2,
                label=f"{label}: {R.envelope_acceptance(th):.3f}")
    ax.fill_between(x, R.semicircle_pdf(x), color=GREEN, alpha=0.25, lw=0)
    ax.plot(x, R.semicircle_pdf(x), color=INK, lw=1.2)
    ax.set(xlabel="$x$", ylim=(0, 0.68), xlim=(-1.02, 1.02), title="(a) density and envelopes")
    ax.legend(fontsize=6.5, loc="lower center", title="envelope: acceptance", title_fontsize=6.5,
              frameon=True, framealpha=0.9, edgecolor="none")
    ax = axes[1]
    th = np.linspace(0, math.pi / 2, 400)
    ax.plot(th, [R.envelope_acceptance(t) for t in th], color=COBALT, lw=1.4, label="exact")
    marks = (0.0, math.pi / 6, math.pi / 4, math.pi / 3)
    emp = []
    for i, t in enumerate(marks):
        s, tried = R.semicircle_rejection(100_000, t, seed=10 + i)
        emp.append(len(s) / tried)
    ax.plot(marks, emp, "o", color=GOLD, ms=4, label=r"simulated ($10^5$ draws)")
    ax.set(xlabel=r"tangent angle $\vartheta$", ylabel="acceptance probability",
           xticks=[0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2],
           xticklabels=["0", r"$\pi/6$", r"$\pi/4$", r"$\pi/3$", r"$\pi/2$"],
           title="(b) acceptance against the tangent point")
    ax.legend(fontsize=7, loc="lower center")
    fig.tight_layout()
    fig.savefig(out / "ch05-rejection.pdf")
    plt.close(fig)


def normals(out):
    """The polar method, and Box--Muller fed with the consecutive pairs of a small LCG."""
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
    ax = axes[0]
    u = R.uniforms(6000, "pcg64", seed=5).reshape(-1, 2)
    v = 2 * u - 1
    inside = (v**2).sum(1) < 1
    ax.scatter(*v[inside].T, s=0.8, color=COBALT, lw=0, rasterized=True)
    ax.scatter(*v[~inside].T, s=0.8, color=GOLD, lw=0, rasterized=True)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=INK, lw=0.8)
    ax.set(aspect="equal", xlim=(-1, 1), ylim=(-1, 1), xticks=[-1, 0, 1], yticks=[-1, 0, 1],
           title="(a) polar method")
    m, a = 4096, 1229
    x = R.lcg_sequence(m + 1, m, a, 1, 0).astype(float)
    uu = (x[:-1] + 0.5) / m                      # all m values, in (0, 1)
    z = R.normals_box_muller(uu)
    lim = 4.3
    for ax, zz, color, title in ((axes[1], z, GOLD, f"(b) LCG $m=2^{{12}}$"),
                                 (axes[2], R.normals_box_muller(R.uniforms(m, "pcg64", seed=6)),
                                  COBALT, "(c) PCG64")):
        ax.scatter(zz[0::2], zz[1::2], s=1.2, color=color, lw=0, rasterized=True)
        if ax is axes[1]:   # the largest radius sqrt(-2 log u) that uniforms (k + 1/2)/m allow
            rmax = math.sqrt(2 * math.log(2 * m))
            ax.plot(rmax * np.cos(t), rmax * np.sin(t), color=GREY, lw=0.6, ls="--")
        ax.set(aspect="equal", xlim=(-lim, lim), ylim=(-lim, lim), xlabel="$z_{2i}$", ylabel="$z_{2i+1}$",
               title=title)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(out / "ch05-normals.pdf")
    plt.close(fig)


def main(out):
    use_style()
    lattice(out)
    diagnostics(out)
    inversion(out)
    rejection(out)
    normals(out)
