"""The Black--Scholes call by transform methods (Chapter 2).

The pricing equation is reduced to the heat equation u_s = u_xixi / 2 by

    V(t, S) = K e^{-r tau} u(s, xi),   tau = T - t,   s = sigma^2 tau,   xi = log(S/K) + nu tau,

with nu = r - y - sigma^2/2 and initial value u(0, xi) = (e^xi - 1)^+. This module solves the heat
problem in two ways and maps the answer back:

* Fourier: the payoff grows like e^xi, so its transform exists only at complex frequencies
  z = omega - i a with a > 1 (damping by e^{-a xi}); there it equals -1/(z (z + i)), and the solution
  is the inverse transform of that times e^{-s z^2 / 2}. ``fourier_heat_call`` evaluates the inverse
  transform by the trapezoidal rule.
* Laplace: in the heat time s the Green's function has the transform e^{-q|xi|}/q, q = sqrt(2 lambda),
  and the call the closed form ``laplace_heat_call``. ``talbot`` inverts it numerically on Talbot's
  contour (fixed-Talbot rule of Abate and Valko, 2004).

Both are checked against the closed form ``bs.call``. Every function broadcasts over its arguments.
"""
from __future__ import annotations

import numpy as np

from mfe230d import bs


# ---------------------------------------------------------------------------------------------
# The heat-equation reduction
# ---------------------------------------------------------------------------------------------

def heat_variables(S, K, T, r, y, sigma):
    """Heat time s = sigma^2 T and position xi = log(S/K) + nu T, nu = r - y - sigma^2/2."""
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    nu = r - y - 0.5 * sigma**2
    return sigma**2 * T, np.log(S / K) + nu * T


def from_heat(u, K, T, r):
    """Map a heat solution back to a price: V = K e^{-rT} u."""
    return np.asarray(K, dtype=float) * np.exp(-r * np.asarray(T, dtype=float)) * u


def heat_call_exact(s, xi):
    """Exact heat solution for the call payoff: E[(e^{xi + sqrt(s) Z} - 1)^+]
    = e^{xi + s/2} N((xi + s)/sqrt(s)) - N(xi/sqrt(s)), with the payoff at s = 0."""
    s, xi = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(xi, dtype=float))
    scalar = s.ndim == 0
    out = np.array(np.maximum(np.expm1(xi), 0.0), ndmin=1)
    s, xi = np.atleast_1d(s), np.atleast_1d(xi)
    pos = s > 0
    rs = np.sqrt(s[pos])
    x = xi[pos]
    out[pos] = np.exp(x + 0.5 * s[pos]) * bs.norm_cdf((x + s[pos]) / rs) - bs.norm_cdf(x / rs)
    return out[0] if scalar else out


# ---------------------------------------------------------------------------------------------
# Fourier
# ---------------------------------------------------------------------------------------------

def call_payoff_transform(z):
    """Generalised Fourier transform of (e^xi - 1)^+, int e^{-i z xi} (e^xi - 1)^+ dxi,
    valid for Im z < -1: equals -1/(z (z + i))."""
    z = np.asarray(z, dtype=complex)
    return -1.0 / (z * (z + 1j))


def heat_kernel_transform(z, s):
    """Fourier transform of the heat kernel k(s, .), entire in z: exp(-s z^2 / 2)."""
    return np.exp(-0.5 * np.asarray(s) * np.asarray(z, dtype=complex) ** 2)


def fourier_heat_call(s, xi, a=1.5, h=0.05, omega_max=None):
    """Heat solution of the call by Fourier inversion along Im z = -a (a > 1):

        u(s, xi) = e^{a xi} / pi * int_0^inf Re[ e^{i omega xi} uhat(omega - i a) e^{-s (omega - i a)^2 / 2} ] d omega,

    computed with the trapezoidal rule of step ``h`` on [0, omega_max]. By default omega_max makes the
    Gaussian factor smaller than e^{-40}. Requires s > 0.
    """
    if a <= 1:
        raise ValueError("the damping a must exceed 1 (the payoff grows like e^xi)")
    s, xi = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(xi, dtype=float))
    if np.any(s <= 0):
        raise ValueError("Fourier inversion needs s > 0; at s = 0 the solution is the payoff")
    if omega_max is None:
        omega_max = np.sqrt(80.0 / s.min()) + 10.0
    omega = np.arange(0.0, omega_max + h, h)
    w = np.full(omega.shape, h)
    w[0] = 0.5 * h                                    # trapezoid on [0, omega_max]
    z = omega - 1j * a
    uhat = call_payoff_transform(z)[None, :]
    ss, xx = s.ravel(), xi.ravel()
    u = np.empty(ss.shape)
    step = max(1, 2_000_000 // omega.size)            # rows per block, to bound memory
    for i in range(0, ss.size, step):
        sl = slice(i, i + step)
        g = uhat * np.exp(1j * np.outer(xx[sl], omega)) * heat_kernel_transform(z[None, :], ss[sl, None])
        u[sl] = np.exp(a * xx[sl]) / np.pi * (g.real @ w)
    return u.reshape(s.shape)


def fourier_call(S, K, T, r, y, sigma, a=1.5, h=0.05, omega_max=None):
    """Black--Scholes call by Fourier inversion of the heat-kernel solution."""
    s, xi = heat_variables(S, K, T, r, y, sigma)
    return from_heat(fourier_heat_call(s, xi, a, h, omega_max), K, T, r)


# ---------------------------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------------------------

def laplace_heat_kernel(lam, xi):
    """Laplace transform in s of the heat kernel k(s, xi): e^{-q |xi|}/q, q = sqrt(2 lambda)
    (principal branch), for lambda off the half-line (-inf, 0]."""
    q = np.sqrt(2.0 * np.asarray(lam, dtype=complex))
    return np.exp(-q * np.abs(xi)) / q


def laplace_heat_call(lam, xi):
    """Laplace transform in s of the heat solution of the call, for Re lambda > 1/2, continued
    analytically to the plane cut along (-inf, 0] and punctured at lambda = 1/2:

        e^{q xi} / (q^2 (q - 1))                              for xi <= 0,
        e^xi / (lambda - 1/2) - 1/lambda + e^{-q xi} / (q^2 (q + 1))   for xi >= 0,

    with q = sqrt(2 lambda).
    """
    lam = np.asarray(lam, dtype=complex)
    xi = np.asarray(xi, dtype=float)
    q = np.sqrt(2.0 * lam)
    left = np.exp(q * np.minimum(xi, 0.0)) / (q**2 * (q - 1.0))
    right = (np.exp(xi) / (lam - 0.5) - 1.0 / lam
             + np.exp(-q * np.maximum(xi, 0.0)) / (q**2 * (q + 1.0)))
    return np.where(xi <= 0, left, right)


def talbot(F, t, M=20):
    """Invert a Laplace transform F at times t > 0 on Talbot's contour (fixed Talbot, Abate--Valko):

        f(t) ~ (rho/M) [ F(rho) e^{rho t}/2 + sum_{k=1}^{M-1} Re( e^{t lam_k} F(lam_k) (1 + i beta_k) ) ],

    lam_k = rho theta_k (cot theta_k + i), theta_k = k pi / M, beta_k = theta_k + (theta_k cot theta_k - 1) cot theta_k,
    rho = 2M/(5t). F must be analytic off the negative real half-line (singularities at points <= 0) and
    accept complex arrays; it is called with lam of shape t.shape + (M,).
    """
    t = np.asarray(t, dtype=float)
    if np.any(t <= 0):
        raise ValueError("Talbot inversion needs t > 0")
    rho = 2.0 * M / (5.0 * t)
    theta = np.arange(1, M) * np.pi / M
    cot = 1.0 / np.tan(theta)
    lam = rho[..., None] * theta * (cot + 1j)
    beta = theta + (theta * cot - 1.0) * cot
    terms = np.exp(t[..., None] * lam) * F(lam) * (1.0 + 1j * beta)
    first = 0.5 * np.exp(rho * t) * F(rho[..., None] + 0j)[..., 0].real
    return rho / M * (first + terms.real.sum(axis=-1))


def laplace_call(S, K, T, r, y, sigma, M=20):
    """Black--Scholes call by numerical Laplace inversion in the heat time.

    The transform has a pole at lambda = 1/2 (u grows like e^{s/2}); we invert the shifted transform
    lambda -> ubar(lambda + 1/2), whose singularities all lie on (-inf, 0], and multiply by e^{s/2}.
    """
    s, xi = heat_variables(S, K, T, r, y, sigma)
    s, xi = np.broadcast_arrays(s, xi)
    shifted = lambda lam: laplace_heat_call(lam + 0.5, xi[..., None])
    u = np.exp(0.5 * s) * talbot(shifted, s, M)
    return from_heat(u, K, T, r)
