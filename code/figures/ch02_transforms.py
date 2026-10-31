"""Figures for Chapter 2 (the Black--Scholes formula three ways; implied volatility)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import bs
from mfe230d import impliedvol as iv
from mfe230d import transforms as tr
from mfe230d.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]
P = bs.TEST_CASE


def transforms_figure(out):
    """(a) Fourier inversion by the trapezoidal rule: error against 2 pi (a - 1)/h for several dampings a,
    with the nearest-pole estimate S e^{-yT} e^{-2 pi (a-1)/h}. (b) Talbot inversion: error against M."""
    exact = bs.call(**P)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.25))
    for i, (a, col, mk) in enumerate(((1.25, GREY, "s"), (1.5, COBALT, "o"), (2.0, GOLD, "^"), (3.0, GREEN, "D"))):
        x = np.arange(2.0 + 0.8 * i, 35.0, 3.2)
        hs = 2 * np.pi * (a - 1) / x
        err = np.array([abs(float(tr.fourier_call(**P, a=a, h=h)) - exact) for h in hs])
        ax.plot(x, np.maximum(err, 1e-17), mk, color=col, ms=3.5, label=f"$a={a:g}$")
    xx = np.linspace(1.5, 35, 50)
    ax.plot(xx, P["S"] * np.exp(-P["y"] * P["T"]) * np.exp(-xx), color=INK, lw=0.9, ls="--",
            label=r"$Se^{-yT}e^{-2\pi(a-1)/h}$")
    ax.set(yscale="log", xlabel=r"$2\pi(a-1)/h$", ylabel="absolute error", ylim=(1e-17, 1e2),
           title="(a) Fourier, trapezoidal rule")
    ax.legend(fontsize=7, loc="lower left")

    Ms = np.arange(4, 41, 2)
    for k, col, mk in ((80.0, GOLD, "^"), (100.0, COBALT, "o"), (125.0, GREEN, "D")):
        p = dict(P, K=k)
        err = [abs(float(tr.laplace_call(**p, M=M)) - float(bs.call(**p))) for M in Ms]
        bx.plot(Ms, np.maximum(err, 1e-17), mk + "-", color=col, ms=3.5, lw=1.0, label=f"$K={k:g}$")
    bx.set(yscale="log", xlabel="nodes $M$ on Talbot's contour", ylabel="absolute error", ylim=(1e-17, 1e2),
           title="(b) Laplace, fixed Talbot")
    bx.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch02-transforms.pdf")
    plt.close(fig)


def implied_vol_figure(out):
    """(a) c(m, w) for m = -0.5: inflection point, Newton on the price from w* towards two targets, and a
    failing step from w0 = 3. (b) Iterations against the implied total volatility, for three m."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    m = -0.5
    w = np.linspace(1e-3, 3.2, 400)
    ax.plot(w, iv.normalised_call(m, w), color=INK, lw=1.3, label=r"$c(m,w)$")
    ws = float(iv.inflection_point(m))
    ax.plot([ws], [iv.normalised_call(m, ws)], "o", color=INK, ms=4, zorder=5)
    ax.annotate(r"$w^*$", (ws, float(iv.normalised_call(m, ws))), xytext=(ws - 0.3, 0.17), fontsize=8, color=INK)
    for w_true, col, mk in ((0.4, COBALT, "o"), (2.0, GOLD, "s")):
        beta = float(iv.normalised_call(m, w_true))
        path = iv.newton_iterates(m, beta, ws, 4)
        for a, b in zip(path[:-1], path[1:]):
            ca = float(iv.normalised_call(m, a))
            ax.plot([a, b], [ca, beta], color=col, lw=0.8)                 # tangent down to the target level
            ax.plot([b, b], [beta, float(iv.normalised_call(m, b))], color=col, lw=0.6, ls=":")
        ax.plot(path, iv.normalised_call(m, path), mk, color=col, ms=3.5,
                label=f"Newton from $w^*$, target $w={w_true:g}$")
        ax.axhline(beta, color=col, lw=0.5, alpha=0.6)
    beta = float(iv.normalised_call(m, 0.4))
    w1 = iv.newton_iterates(m, beta, 3.0, 1)[1]
    c3 = float(iv.normalised_call(m, 3.0))
    ax.plot([3.0, 0.0], [c3, c3 - 3.0 * float(iv.normalised_vega(m, 3.0))], color=GREEN, lw=0.9, ls="--",
            label=rf"tangent at $w_0=3$: $w_1={w1:.2f}$")
    ax.set(xlim=(0, 3.2), ylim=(-0.05, 0.78), xlabel="total volatility $w$", ylabel="normalised call price",
           title=r"(a) $m=\log(F/K)=-0.5$")
    ax.legend(fontsize=6, loc="upper left")

    wgrid = np.exp(np.linspace(np.log(0.02), np.log(4.0), 60))
    for mm, col in ((-0.05, COBALT), (-0.5, GOLD), (-2.0, GREEN)):
        beta = iv.normalised_call(mm, wgrid)
        ok = beta > 1e-300
        _, it_log = iv.implied_total_vol(mm, beta[ok])
        _, it_mk = iv.newton_price(mm, beta[ok], maxiter=300)
        bx.plot(wgrid[ok], it_log, "-", color=col, lw=1.3, label=f"$m={mm:g}$")
        bx.plot(wgrid[ok], it_mk, "--", color=col, lw=1.1)
    bx.plot([], [], "-", color=GREY, lw=1.3, label="log price, our start")
    bx.plot([], [], "--", color=GREY, lw=1.1, label="price, from $w^*$")
    bx.set(xscale="log", yscale="log", xlabel="implied total volatility $w$", ylabel="Newton iterations",
           title="(b) iterations to convergence", ylim=(2, 400))
    bx.legend(fontsize=6.3, ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch02-implied-vol.pdf")
    plt.close(fig)


def main(out):
    use_style()
    transforms_figure(out)
    implied_vol_figure(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
