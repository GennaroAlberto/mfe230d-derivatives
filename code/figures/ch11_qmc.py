"""Figures for Chapter 11 (Quasi-Monte Carlo).

Every random number comes from numpy's default_rng with the seeds below; ``main`` prints
the numbers that the notes quote.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import qmc
from mfe230d.bs import TEST_CASE
from mfe230d.montecarlo import sine_product
from mfe230d.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]
P = tuple(TEST_CASE[k] for k in ("S", "K", "r", "y", "sigma"))
CONSTRUCTIONS = (("increments", GOLD, "increments"), ("bridge", COBALT, "Brownian bridge"),
                 ("pca", GREEN, "principal components"))

# The laboratory of Section 11.4: weekly-like grid, m = 64 dates over one year.
M_ASIAN = 64
ASIAN_NS = 2 ** np.arange(8, 15)
ASIAN_R = 64
ASIAN_CHUNK = 16
ASIAN_SEED = 11
SMOOTH_D = 5
SMOOTH_NS = 2 ** np.arange(4, 17)
SMOOTH_R = 128
SMOOTH_CHUNK = 32
SMOOTH_SEED = 7
SAW_D = 2
SAW_N = 2**12
SAW_R = 64
SAW_SEED = 8
PICK_N = 2**15
PICK_SEED = 5
COURSE_N = 2**14
COURSE_R = 16
COURSE_SEED = 2026


def points(out):
    """Four projections of 256 points: random, Halton (2, 3), Sobol' (1, 2), Halton (47, 53)."""
    n = 256
    sets = [(np.random.default_rng(0).random((n, 2)), "pseudo-random (seed 0)"),
            (qmc.halton(n, 2), "Halton, bases 2 and 3"),
            (qmc.sobol(n, 2), "Sobol', coordinates 1 and 2"),
            (qmc.halton(n, 16)[:, 14:16], "Halton, bases 47 and 53")]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.9))
    for ax, (Q, title) in zip(axes, sets):
        if "Sobol" in title:
            for v in np.arange(1, 16) / 16:
                ax.axvline(v, color=GREY, lw=0.3, alpha=0.6)
                ax.axhline(v, color=GREY, lw=0.3, alpha=0.6)
        ax.scatter(Q[:, 0], Q[:, 1], s=2.2, color=COBALT if "Halton, bases 47" not in title else GOLD, lw=0)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 1], yticks=[0, 1], aspect="equal")
        ax.grid(False)
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(labelsize=7)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "ch11-points.pdf")
    plt.close(fig)


def convergence(out):
    """Star discrepancy in d = 2, errors for a smooth integrand in d = 5, and n not a power of 2."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.6, 2.45))
    # (a) exact star discrepancy, d = 2
    ns = 2 ** np.arange(3, 12)
    D = {"random": [], "halton": [], "sobol": []}
    rnd = np.random.default_rng(1).random((ns[-1], 2))
    H, S = qmc.halton(ns[-1], 2), qmc.sobol(ns[-1], 2)
    for n in ns:
        D["random"].append(qmc.star_discrepancy(rnd[:n]))
        D["halton"].append(qmc.star_discrepancy(H[:n]))
        D["sobol"].append(qmc.star_discrepancy(S[:n]))
    ax1.plot(ns, D["random"], color=GREY, marker="o", ms=2.5, label="random (seed 1)")
    ax1.plot(ns, D["halton"], color=GOLD, marker="o", ms=2.5, label="Halton")
    ax1.plot(ns, D["sobol"], color=COBALT, marker="o", ms=2.5, label="Sobol'")
    ax1.plot(ns, 0.6 / np.sqrt(ns), color=INK, lw=0.7, ls="--", label=r"$\propto n^{-1/2}$")
    ax1.plot(ns, 0.6 * np.log(ns) / ns, color=INK, lw=0.7, ls=":", label=r"$\propto \log n/n$")
    ax1.set(xscale="log", yscale="log", xlabel="$n$", title=r"star discrepancy, $d=2$")
    ax1.set_xscale("log", base=2)
    ax1.legend(fontsize=5.8, loc="lower left")

    # (b) smooth integrand in d = 5: prod (pi/2) sin(pi x_j), integral 1
    d, ns = SMOOTH_D, SMOOTH_NS
    sig = np.sqrt((np.pi**2 / 8) ** d - 1)
    eH = np.abs(np.cumsum(sine_product(qmc.halton(ns[-1], d)))[ns - 1] / ns - 1)
    eS = np.abs(np.cumsum(sine_product(qmc.sobol(ns[-1], d)))[ns - 1] / ns - 1)
    E = qmc.rqmc_nested(sine_product, ns, d, R=SMOOTH_R, seed=SMOOTH_SEED)
    rmse = np.sqrt(np.mean((E - 1.0) ** 2, axis=0))
    ax2.plot(ns, sig / np.sqrt(ns), color=GREY, lw=1.2, label=r"Monte Carlo, $\sigma/\sqrt{n}$")
    ax2.plot(ns, eH, color=GOLD, marker="o", ms=2.2, lw=1.0, label="Halton")
    ax2.plot(ns, eS, color=COBALT, marker="o", ms=2.2, lw=1.0, label="Sobol'")
    ax2.plot(ns, rmse, color=GREEN, marker="s", ms=2.4, lw=1.4, label="scrambled Sobol' (RMSE)")
    ax2.set(xscale="log", yscale="log", xlabel="$n$", title=r"smooth integrand, $d=5$")
    ax2.set_xscale("log", base=2)
    ax2.legend(fontsize=5.8, loc="lower left")

    # (c) scrambled Sobol', d = 2, every n
    nall = np.arange(1, SAW_N + 1)
    E2 = qmc.rqmc_nested(sine_product, nall, SAW_D, R=SAW_R, seed=SAW_SEED)
    r2 = np.sqrt(np.mean((E2 - 1.0) ** 2, axis=0))
    sel = nall >= 16
    ax3.plot(nall[sel], nall[sel] * r2[sel], color=GREEN, lw=0.6, label="all $n$")
    p2 = 2 ** np.arange(4, 13)
    ax3.plot(p2, p2 * r2[p2 - 1], color=COBALT, marker="o", ms=3, lw=0, label="$n=2^m$")
    ax3.set(xscale="log", yscale="log", xlabel="$n$", ylabel=r"$n\times$RMSE",
            title=r"scrambled Sobol', $d=2$, every $n$")
    ax3.set_xscale("log", base=2)
    ax3.legend(fontsize=6, loc="lower left")
    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=7)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(out / "ch11-convergence.pdf")
    plt.close(fig)
    return dict(D=D, ns_D=ns, smooth=dict(ns=ns, halton=eH, sobol=eS, rmse=rmse, sigma=sig, E=E),
                saw=dict(n=nall, rmse=r2))


def asian_lab(m=M_ASIAN, ns=ASIAN_NS, R=ASIAN_R, seed=ASIAN_SEED):
    """RQMC replicates of the arithmetic Asian call under the three constructions, plus its square.

    Column 3 holds the mean of the squared payoff over each replicate's points; averaged over the
    R n_max scrambled points it estimates E[A^2], hence the Monte Carlo variance sigma^2.
    """
    return qmc.asian_rqmc(m, ns, R, seed, *P, square=True)


def asian_sigma(E):
    """sigma of the arithmetic payoff from the second moment over all the scrambled points."""
    m2, mu = E[:, -1, 3].mean(), E[:, -1, 2].mean()
    return float(np.sqrt(m2 - mu**2))


def variance_factors(E, sigma, chunk=ASIAN_CHUNK):
    """sigma^2 / (n Var), pooled over all replicates (one row per n), and per chunk at the largest n."""
    ns = ASIAN_NS
    pooled = sigma**2 / ns[:, None] / E[:, :, :3].var(axis=0, ddof=1)
    R = E.shape[0]
    chunks = np.array([sigma**2 / ns[-1] / E[c:c + chunk, -1, :3].var(axis=0, ddof=1)
                       for c in range(0, R, chunk)])
    return pooled, chunks


def effective_dimension(out, E=None):
    """RMSE against n for the arithmetic Asian call, and the share of variance of the first k normals."""
    m = M_ASIAN
    t = qmc.asian_grid(m)
    if E is None:
        E = asian_lab()
    sd = E[:, :, :3].std(axis=0, ddof=1)
    sigma = asian_sigma(E)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ns = ASIAN_NS
    ax.plot(ns, sigma / np.sqrt(ns), color=GREY, lw=1.2, label=r"Monte Carlo, $\sigma/\sqrt{n}$")
    for c, (me, color, label) in enumerate(CONSTRUCTIONS):
        ax.plot(ns, sd[:, c], color=color, marker="o", ms=3, label=label)
    ax.set(xscale="log", yscale="log", xlabel="points per replicate $n$",
           ylabel="standard deviation of estimate", title=f"arithmetic Asian call, $m={m}$")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=6.3, loc="lower left", handlelength=1.6)

    share = lambda c: qmc.lognormal_call_share(c, t, *P)  # noqa: E731
    k = np.arange(1, m + 1)
    kk = np.array([1, 2, 4, 8, 16, 32])
    exact, pf = {}, {}
    for me, color, label in CONSTRUCTIONS:
        w = qmc.average_direction(t, me)
        exact[me] = qmc.ridge_shares(w, share)
        ax2.plot(k, exact[me]["truncation"], color=color, lw=1.4, label=f"{label}")
        fA = lambda Z, me=me: qmc.asian_call_payoffs(Z, t, *P, method=me)[0]  # noqa: E731
        pf[me] = qmc.pick_freeze_shares(fA, m, kk, n=PICK_N, seed=PICK_SEED)
        ax2.plot(kk, pf[me], color=color, lw=0, marker="o", ms=3.2, mfc="white")
    ax2.plot([], [], color=INK, lw=0, marker="o", ms=3.2, mfc="white", label="arithmetic (pick-freeze)")
    ax2.set(xscale="log", xlabel="first $k$ normals", ylabel="share of variance", ylim=(0, 1.02),
            title="geometric: exact; arithmetic: pick-freeze")
    ax2.legend(fontsize=6.3, loc="center left", bbox_to_anchor=(0.02, 0.42))
    for a in (ax, ax2):
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch11-asian.pdf")
    plt.close(fig)
    return dict(sd=sd, sigma=sigma, exact=exact, pick_freeze=pf, E=E)


def course_example(n=COURSE_N, R=COURSE_R, seed=COURSE_SEED):
    """The course's monthly Asian (m = 12): RQMC estimates under the three constructions."""
    E = qmc.asian_rqmc(12, [n], R, seed, *P, geometric=True)[:, 0, :]
    return dict(mean=E.mean(axis=0), se=E.std(axis=0, ddof=1) / np.sqrt(R),
                geometric_exact=qmc.geometric_asian_call(qmc.asian_grid(12), *P))


def main(out):
    use_style()
    points(out)
    conv = convergence(out)
    lab = effective_dimension(out)
    ns = SMOOTH_NS
    print("smooth d=5 at n=2^16: MC %.2e Halton %.2e Sobol %.2e scrambled RMSE %.2e" % (
        conv["smooth"]["sigma"] / np.sqrt(ns[-1]), conv["smooth"]["halton"][-1],
        conv["smooth"]["sobol"][-1], conv["smooth"]["rmse"][-1]))
    sel = ns >= 2**8
    E5 = conv["smooth"]["E"]
    slope = lambda r: np.polyfit(np.log(ns[sel]), np.log(r[sel]), 1)[0]  # noqa: E731
    cs = [slope(np.sqrt(np.mean((E5[c:c + SMOOTH_CHUNK] - 1) ** 2, axis=0))) for c in range(0, SMOOTH_R, SMOOTH_CHUNK)]
    print("fitted slope of scrambled RMSE, d=5, 2^8..2^16: %.2f (chunks of %d: %.2f to %.2f)" % (
        slope(conv["smooth"]["rmse"]), SMOOTH_CHUNK, min(cs), max(cs)))
    r2, nall = conv["saw"]["rmse"], conv["saw"]["n"]
    print("sawtooth d=2: n*RMSE at 2^12 %.3f, median over 2^11 < n < 2^12 %.3f" % (
        SAW_N * r2[-1], np.median((nall * r2)[2**11:2**12 - 1])))
    print("D* d=2 n=2048: random %.4f halton %.4f sobol %.4f" % tuple(conv["D"][k][-1] for k in ("random", "halton", "sobol")))
    sigma = lab["sigma"]
    pooled, chunks = variance_factors(lab["E"], sigma)
    print("Asian m=64: payoff sd %.4f (from %d scrambled points)" % (sigma, ASIAN_R * ASIAN_NS[-1]))
    for i, n in enumerate(ASIAN_NS):
        print("  n=%5d pooled variance factors (incr, bridge, pca):" % n, np.round(pooled[i], 1))
    print("  chunks of %d at n=2^14: min" % ASIAN_CHUNK, np.round(chunks.min(axis=0), 0), "max", np.round(chunks.max(axis=0), 0))
    for me, _, _ in CONSTRUCTIONS:
        e = lab["exact"][me]
        print("  shares %s: k=1 %.3f k=2 %.3f k=4 %.3f k=8 %.3f additive %.3f; pick-freeze %s" % (
            me, *e["truncation"][[0, 1, 3, 7]], e["additive"], np.round(lab["pick_freeze"][me], 3)))
    # The course example (m = 12) is recomputed and pinned in code/tests/test_qmc.py.


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
