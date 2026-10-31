"""Figures for Chapter 20 (Trees and finite differences)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import bs
from mfe230d import grids as G
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
PURPLE = "#6250D6"
P = bs.TEST_CASE
S, K, T, r, y, SIG = (P[k] for k in ("S", "K", "T", "r", "y", "sigma"))


def trees_figure(out):
    """(a) n (C_n - C) for CRR at K = 100 and K = 110 with A + B theta(1 - theta); (b) errors against n."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.25))
    ns = np.arange(50, 601)
    for KK, col, mk in ((100.0, COBALT, "o"), (110.0, GOLD, "s")):
        c = float(bs.call(S, KK, T, r, y, SIG))
        e = np.array([n * (G.binomial_european(S, KK, T, r, y, SIG, int(n)) - c) for n in ns])
        th = np.array([G.crr_strike_offset(S, KK, T, SIG, int(n)) for n in ns])
        B = G.crr_oscillation_coefficient(S, KK, T, r, y, SIG)
        nf = np.arange(1000, 2001)
        ef = np.array([n * (G.binomial_european(S, KK, T, r, y, SIG, int(n)) - c) for n in nf])
        tf = np.array([G.crr_strike_offset(S, KK, T, SIG, int(n)) for n in nf])
        A = float(np.mean(ef - B * tf * (1 - tf)))
        ax.plot(ns, e, mk, ms=1.6, color=col, label=f"$K={KK:.0f}$")
        ax.plot(ns, A + B * th * (1 - th), "_", ms=3.0, mew=0.6, color=INK)
    ax.plot([], [], "_", color=INK, ms=6, label=r"$A+B\theta_n(1-\theta_n)$")
    ax.set(xlabel="steps $n$", ylabel=r"$n\,(C_n-C)$", title="(a) CRR error times $n$", ylim=(-2.6, 3.4))
    ax.legend(fontsize=6.5, loc="upper center", ncol=3, markerscale=2.5, handletextpad=0.2, columnspacing=0.8)

    c = float(bs.call(S, K, T, r, y, SIG))
    n_even = np.unique(np.round(np.geomspace(20, 4000, 18) / 4).astype(int) * 4)
    n_odd = n_even + 1
    crr_e = [abs(G.binomial_european(S, K, T, r, y, SIG, int(n)) - c) for n in n_even]
    crr_o = [abs(G.binomial_european(S, K, T, r, y, SIG, int(n)) - c) for n in n_odd]
    bbs = [abs(G.binomial_european(S, K, T, r, y, SIG, int(n), smooth=True) - c) for n in n_even]
    bbsr = [abs(G.bbsr(S, K, T, r, y, SIG, int(n), "call", "european") - c) for n in n_even]
    lr = [abs(G.binomial_european(S, K, T, r, y, SIG, int(n), method="lr") - c) for n in n_odd]
    n_tri = n_even[n_even <= 2000]
    tri = [abs(G.trinomial(S, K, T, r, y, SIG, int(n), "call", lam=np.sqrt(1.5)) - c) for n in n_tri]
    bx.loglog(n_even, crr_e, "o-", ms=2.5, color=COBALT, label="CRR, $n$ even")
    bx.loglog(n_odd, crr_o, "o--", ms=2.5, color=COBALT, mfc="white", label="CRR, $n$ odd")
    bx.loglog(n_even, bbs, "s-", ms=2.5, color=GOLD, label="BBS")
    bx.loglog(n_even, bbsr, "s--", ms=2.5, color=GOLD, mfc="white", label="BBSR")
    bx.loglog(n_odd, lr, "^-", ms=2.5, color=GREEN, label="Leisen--Reimer")
    bx.loglog(n_tri, tri, "d-", ms=2.5, color=PURPLE, label=r"trinomial, $\kappa=\sqrt{3/2}$")
    bx.set(xlabel="steps $n$", ylabel="|price error|", title="(b) European call, test case")
    bx.set_ylim(1e-11, 0.5)
    bx.legend(fontsize=6, loc="lower left", ncol=2, handlelength=1.6, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out / "ch20-trees.pdf")
    plt.close(fig)


def schemes_figure(out):
    """(a) convergence of the four schemes; (b) one Crank--Nicolson step of a spike; (c) CN gamma."""
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(6.8, 2.1))
    c = float(bs.call(S, K, T, r, y, SIG))
    Js = np.array([50, 100, 200, 400, 800, 1600, 3200])
    errs = {"explicit": [], "implicit": [], "CN": [], "CN, Rannacher": []}
    for J in Js:
        h = 12 * SIG / J
        if J <= 1600:
            N_ex = int(np.ceil(T / (0.4 * h * h / (0.5 * SIG**2))))
            errs["explicit"].append(abs(G.fd_price(S, K, T, r, y, SIG, int(J), N_ex, theta=0.0) - c))
        N = max(2, int(round(T * SIG / h)))
        errs["implicit"].append(abs(G.fd_price(S, K, T, r, y, SIG, int(J), N, theta=1.0) - c))
        errs["CN"].append(abs(G.fd_price(S, K, T, r, y, SIG, int(J), N, theta=0.5) - c))
        errs["CN, Rannacher"].append(abs(G.fd_price(S, K, T, r, y, SIG, int(J), N, theta=0.5, rannacher=2) - c))
    for (name, e), col, mk in zip(errs.items(), (GREY, GOLD, COBALT, GREEN), ("o", "s", "^", "d")):
        ax.loglog(Js[:len(e)], e, mk + "-", ms=2.5, color=col, label=name)
    ax.set(xlabel="space intervals $J$", ylabel="|price error|", title="(a) convergence")
    ax.legend(fontsize=6, loc="lower left")

    for lam, col in ((1.0, GREEN), (1.5, GOLD), (4.0, COBALT)):
        m, w = G.cn_kernel(lam, half=40)
        sel = np.abs(m) <= 6
        bx.plot(m[sel], w[sel], "o-", ms=2.5, color=col, label=rf"$\lambda={lam:g}$")
    bx.axhline(0.0, color=INK, lw=0.6)
    bx.set(xlabel="node $m$", ylabel="weight", title="(b) one CN step of a spike")
    bx.legend(fontsize=6.5, loc="upper right")

    J, N = 800, 25
    a = G.fd_greeks(G.fd_solve(S, K, T, r, y, SIG, J, N, theta=0.5))
    b = G.fd_greeks(G.fd_solve(S, K, T, r, y, SIG, J, N, theta=0.5, rannacher=2))
    sel = (a["s"] > 85) & (a["s"] < 115)
    cx.plot(a["s"][sel], a["gamma"][sel], color=COBALT, lw=0.9, label="CN")
    cx.plot(b["s"][sel], b["gamma"][sel], color=GREEN, lw=1.4, label="CN, Rannacher")
    cx.plot(a["s"][sel], bs.greeks(a["s"][sel], K, T, r, y, SIG)["gamma"], ":", color=INK, lw=1.0,
            label="Black--Scholes")
    cx.set(xlabel="$S_0$", ylabel=r"$\Gamma$", title=r"(c) gamma, $J=800$, $N=25$", ylim=(-0.06, 0.1))
    cx.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch20-schemes.pdf")
    plt.close(fig)


def american_figure(out):
    """(a) exercise boundary of the LS put: grid and tree; (b) far-field truncation error against the width."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.25))
    S0, KK, TT, rr, sig = 36.0, 40.0, 1.0, 0.06, 0.20
    o = G.fd_solve(S0, KK, TT, rr, 0.0, sig, 1600, 800, 0.5, "put", "american", rannacher=2, solver="bs",
                   history=True)
    tr = G.binomial(S0, KK, TT, rr, 0.0, sig, 2000, "put", "american", history=True)
    ax.plot(TT - o["tau"], o["boundary"], color=COBALT, label="CN grid, Brennan--Schwartz")
    ax.plot(tr["t"][::10], tr["boundary"][::10], ".", ms=2.0, color=GOLD, label="CRR tree, $n=2000$")
    ax.axhline(KK, color=GREY, lw=0.6, ls="--")
    Lstar = 2 * rr * KK / (2 * rr + sig**2)
    ax.axhline(Lstar, color=GREY, lw=0.6, ls=":")
    ax.text(0.02, Lstar + 0.25, "perpetual $L^*$", fontsize=7, color=INK)
    ax.set(xlabel="$t$", ylabel="$b(t)$", title="(a) exercise boundary, LS put", ylim=(28, 41))
    ax.legend(fontsize=6.5, loc="upper left")

    h = 12 * SIG / 800
    N = 400
    J_wide = 2 * int(round(10 * SIG / h))
    wide = G.fd_solve(S, K, T, r, y, SIG, J_wide, N, theta=0.5, rannacher=2, width=J_wide * h / (2 * SIG))
    v_wide = wide["price"]
    ms = np.arange(1.0, 6.01, 0.25)
    e_asy, e_pay, b_asy, b_pay = [], [], [], []
    taus = np.linspace(1e-4, T, 400)
    for m in ms:
        J = 2 * int(round(m * SIG / h))
        w = J * h / (2 * SIG)
        e_asy.append(abs(G.fd_price(S, K, T, r, y, SIG, J, N, theta=0.5, rannacher=2, width=w) - v_wide))
        e_pay.append(abs(G.fd_price(S, K, T, r, y, SIG, J, N, theta=0.5, rannacher=2, width=w, bc="payoff") - v_wide))
        lo, hi = S * np.exp(-w * SIG), S * np.exp(w * SIG)
        eps_asy = max(float(np.max(bs.put(hi, K, taus, r, y, SIG))), float(np.max(bs.call(lo, K, taus, r, y, SIG))))
        eps_pay = max(float(np.max(np.abs(bs.call(hi, K, taus, r, y, SIG) - (hi - K)))),
                      float(np.max(bs.call(lo, K, taus, r, y, SIG))))
        q = G.exit_probability(np.log(S), np.log(lo), np.log(hi), T, r, y, SIG)
        b_asy.append(eps_asy * q)
        b_pay.append(eps_pay * q)
    e_asy = np.where(np.array(e_asy) > 1e-11, e_asy, np.nan)       # below 1e-11: round-off
    bx.semilogy(ms, e_pay, "s-", ms=2.5, color=GOLD, label="payoff at the ends")
    bx.semilogy(ms, b_pay, "--", color=GOLD, lw=0.9, label=r"bound $\varepsilon\,\mathbb{Q}(\vartheta\leq T)$")
    bx.semilogy(ms, e_asy, "o-", ms=2.5, color=COBALT, label="asymptotic values")
    bx.semilogy(ms, b_asy, "--", color=COBALT, lw=0.9, label="bound")
    bx.set(xlabel=r"half-width $m$ (units of $\sigma\sqrt{T}$)", ylabel="truncation error at $S_0$",
           title="(b) far field, test-case call", ylim=(1e-13, 10))
    bx.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch20-american.pdf")
    plt.close(fig)


def main(out):
    use_style()
    trees_figure(out)
    schemes_figure(out)
    american_figure(out)


if __name__ == "__main__":
    import pathlib

    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
