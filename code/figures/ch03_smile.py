"""Figures for Chapter 3 (Arbitrage in the strike and the smile)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfe230d import bs
from mfe230d import smile as sm
from mfe230d.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
PURPLE = "#6250D6"
TC = bs.TEST_CASE
S, T, r, y = TC["S"], TC["T"], TC["r"], TC["y"]
SIG = 0.2                       # flat volatility of the lognormal law
P_ATOM, A_ATOM = 0.2, 120.0     # takeover at 120 with probability 0.2
COURSE = dict(S=3900.0, T=1.0, r=0.03, y=0.013, sigma_atm=0.25, xi=0.28)


def ssvi_call(K, T=T):
    return sm.smile_call(K, S, T, r, y, sm.ssvi_smile(K, S, T, r, y)[0])


def strike(out):
    """Q(S_T > K) from the right slope of C, and e^{rT} times second differences, for three laws."""
    K = np.linspace(40.0, 180.0, 1401)
    disc = np.exp(r * T)
    laws = (
        ("lognormal, $\\sigma=20\\%$", COBALT, lambda k: bs.call(S, k, T, r, y, SIG)),
        ("SSVI smile", GOLD, ssvi_call),
        ("lognormal + atom at 120", GREEN, lambda k: sm.atom_call(k, S, T, r, y, SIG, P_ATOM, A_ATOM)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.55))
    eps = 1e-6
    for lab, color, f in laws:
        tail = -disc * (f(K + eps) - f(K)) / eps          # right derivative: Q(S_T > K)
        axes[0].plot(K, tail, color=color, lw=1.3, label=lab)
    h = 1.0
    Kh = np.arange(40.0, 180.0 + 1e-9, h)
    for lab, color, f in laws:
        axes[1].plot(Kh, disc * sm.second_difference(f, Kh, h), color=color, lw=1.2, label=lab)
    axes[0].set(xlabel="strike $K$", ylabel="$-e^{rT}\\,\\partial^+C(K)$", title="$\\mathbb{Q}(S_T>K)$",
                ylim=(-0.03, 1.03), xlim=(40, 180))
    axes[1].set(xlabel="strike $K$", ylabel="$e^{rT}\\,\\Delta_h^2C(K)$", title="second differences, $h=1$",
                xlim=(40, 180), ylim=(0, 0.215))
    axes[1].annotate("spike $\\approx p/h=0.2$", xy=(120, 0.2), xytext=(135, 0.17), fontsize=7.5, color=INK,
                     arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "ch03-strike.pdf")
    plt.close(fig)


def smile(out):
    """SSVI smiles for three maturities; the density at T = 1 and its estimates from rounded quotes."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for Tt, color in ((0.25, COBALT), (1.0, GOLD), (2.0, GREEN)):
        F = bs.forward(S, Tt, r, y)
        k = np.linspace(-0.6, 0.4, 401)
        sig = sm.ssvi_smile(F * np.exp(k), S, Tt, r, y)[0]
        axes[0].plot(k, 100 * sig, color=color, lw=1.4, label=f"$T={Tt:g}$")
    axes[0].set(xlabel="log-moneyness $\\log(K/F_T)$", ylabel="implied volatility (%)", title="SSVI smiles")
    axes[0].legend(fontsize=7.5)

    K = np.linspace(40.0, 180.0, 1401)
    disc = np.exp(r * T)
    dens = disc * sm.smile_density(K, S, T, r, y, *sm.ssvi_smile(K, S, T, r, y))
    logn = bs.norm_pdf(bs.d_pm(S, K, T, r, y, SIG)[1]) / (K * SIG * np.sqrt(T))
    axes[1].plot(K, logn, color=GOLD, lw=1.3, label="lognormal, $\\sigma=20\\%$")
    axes[1].plot(K, dens, color=COBALT, lw=1.5, label="SSVI, exact")
    rounded = lambda k: np.round(ssvi_call(k), 2)
    for h, color, mk in ((1.0, GREEN, "."), (5.0, PURPLE, "o")):
        Kh = np.arange(60.0, 140.0 + 1e-9, h)
        axes[1].plot(Kh, disc * sm.second_difference(rounded, Kh, h), ls="none", marker=mk, ms=2.6 if h == 1 else 3.2,
                     color=color, label=f"rounded quotes, $h={h:g}$")
    axes[1].set(xlabel="strike $K$", ylabel="density of $S_T$", title="Breeden–Litzenberger at $T=1$",
                xlim=(40, 180), ylim=(-0.005, 0.056))
    axes[1].legend(fontsize=6.5, loc="upper right", markerscale=1.8)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "ch03-smile.pdf")
    plt.close(fig)


def binary(out):
    """The course's smile: digital calls and Arrow-Debreu densities, flat volatility against the smile."""
    c = COURSE
    Sc, Tc, rc, yc = c["S"], c["T"], c["r"], c["y"]
    K = np.linspace(2600.0, 5200.0, 521)
    sig, ds, d2 = sm.course_smile(K, Sc, c["sigma_atm"], c["xi"])
    dig = sm.smile_digital(K, Sc, Tc, rc, yc, sig, ds)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axes[0].plot(K, bs.digital_call(Sc, K, Tc, rc, yc, c["sigma_atm"]), color=GOLD, lw=1.3, label="flat $25\\%$")
    axes[0].plot(K, dig["bs"], color=GREY, lw=1.1, ls="--", label="$e^{-rT}N(d_-)$ at $\\sigma(K)$")
    axes[0].plot(K, dig["price"], color=COBALT, lw=1.5, label="smile, $-C'(K)$")
    axes[0].set(xlabel="strike $K$", ylabel="price", title="digital call", xlim=(2600, 5200))
    axes[0].legend(fontsize=7)
    K2 = np.linspace(1500.0, 7000.0, 1101)
    sig2, ds2, d22 = sm.course_smile(K2, Sc, c["sigma_atm"], c["xi"])
    flat = np.exp(-rc * Tc) * bs.norm_pdf(bs.d_pm(Sc, K2, Tc, rc, yc, c["sigma_atm"])[1]) / (K2 * c["sigma_atm"] * np.sqrt(Tc))
    axes[1].plot(K2, 1e4 * flat, color=GOLD, lw=1.3, label="flat $25\\%$")
    axes[1].plot(K2, 1e4 * sm.smile_density(K2, Sc, Tc, rc, yc, sig2, ds2, d22), color=COBALT, lw=1.5, label="smile")
    axes[1].set(xlabel="strike $K$", ylabel="$10^4\\times C''(K)$", title="Arrow–Debreu density", xlim=(1500, 7000))
    axes[1].legend(fontsize=7)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "ch03-binary.pdf")
    plt.close(fig)


def main(out):
    use_style()
    strike(out)
    smile(out)
    binary(out)
