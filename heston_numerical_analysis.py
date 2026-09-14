# =============================================================
# Heston Stochastic Volatility Model
# Numerical implementation for Section 6.4
# =============================================================

import math
import numpy as np
from scipy.integrate import quad
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter

# -------------------------------------------------------------
# 1. Model parameters
# -------------------------------------------------------------

S0 = 100
K = 100
r = 0.04
q = 0.0

v0 = 0.04
vbar_star = 0.04
kappa_star = 2.0
zeta = 0.3
rho = -0.7

T = 1

# -------------------------------------------------------------
# 2. Heston auxiliary quantities
# -------------------------------------------------------------

def heston_auxiliary(phi, j, kappa_star, zeta, rho):
    if j == 1:
        u = 0.5
        b = kappa_star - rho * zeta
    elif j == 2:
        u = -0.5
        b = kappa_star
    else:
        raise ValueError("j must be either 1 or 2")

    gamma = 0.5 * zeta**2
    alpha = 1j * u * phi - 0.5 * phi**2
    beta = b - rho * zeta * 1j * phi

    return u, b, gamma, alpha, beta

# -------------------------------------------------------------
# 3. Riccati quantities
# -------------------------------------------------------------

def heston_riccati(phi, j, kappa_star, zeta, rho):
    u, b, gamma, alpha, beta = heston_auxiliary(phi, j, kappa_star, zeta, rho)

    d = np.sqrt(beta**2 - 4 * gamma * alpha)

    # Choose the square-root with non-negative real part
    if np.real(d) < 0:
        d = -d

    r_plus = (beta + d) / zeta**2
    r_minus = (beta - d) / zeta**2
    g = r_minus / r_plus

    return d, r_plus, r_minus, g

# -------------------------------------------------------------
# 4. Closed-form C_j and D_j
# -------------------------------------------------------------
def heston_CD(phi, j, tau, r, vbar_star, kappa_star, zeta, rho):

    d, r_plus, r_minus, g = heston_riccati(phi, j, kappa_star, zeta, rho)

    exp_term = np.exp(-d * tau)

    D = r_minus * (1 - exp_term) / (1 - g * exp_term)

    C = ((1j * r * phi / vbar_star) * tau + kappa_star
         * (r_minus * tau - (2 / zeta**2)
         * np.log((1 - g * exp_term) / (1 - g))))

    return C, D

# -------------------------------------------------------------
# 5. Fourier integrand
# -------------------------------------------------------------

def heston_integrand(phi, j, S0, K, tau, r, v0, vbar_star, kappa_star, zeta, rho):

    C, D = heston_CD(phi, j, tau, r, vbar_star, kappa_star, zeta, rho)

    x = np.log(S0)

    numerator = np.exp(C * vbar_star + D * v0
        + 1j * phi * x
        - 1j * phi * np.log(K))

    integrand = np.real(numerator / (1j * phi))

    return integrand

# -------------------------------------------------------------
# 6. Numerical evaluation of P1 and P2
# -------------------------------------------------------------

def heston_probability(j, S0, K, tau, r, v0, vbar_star, kappa_star, zeta, rho, upper_bound=100):

    integral, error = quad(heston_integrand, 1e-8, upper_bound,
        args=(j, S0, K, tau, r, v0, vbar_star, kappa_star, zeta, rho),
        limit=200)

    P = 0.5 + integral / np.pi

    return P, error

# -------------------------------------------------------------
# 7. Heston European call price
# -------------------------------------------------------------

def heston_price(S0, K, tau, r, v0, vbar_star, kappa_star, zeta, rho, upper_bound=100):

    P1, error1 = heston_probability(1, S0, K, tau, r, v0, vbar_star, kappa_star, zeta, rho, upper_bound)
    P2, error2 = heston_probability(2, S0, K, tau, r, v0, vbar_star, kappa_star, zeta, rho, upper_bound)

    price = S0 * P1 - K * np.exp(-r * tau) * P2

    return price, P1, P2, error1, error2

#7.1 Obtain price using the default upper bound = 100

price, P1, P2, error1, error2 = heston_price(S0, K, T, r, v0, vbar_star, kappa_star, zeta, rho)

price, P1, P2, error1, error2 = heston_price(S0, K, T, r, v0, vbar_star, kappa_star, zeta, rho)


print(f"1. Heston European call price")
print(f"Heston call price: {price:.6f}")
print(f"P1: {P1:.6f}")
print(f"P2: {P2:.6f}")
print(f"Quadrature error for P1: {error1:.3e}")
print(f"Quadrature error for P2: {error2:.3e}")
print(f"--------------------------")

# -------------------------------------------------------------
# 8. Fourier integration convergence
# -------------------------------------------------------------
#This explains the upper bound of the integral; integral 0 to 100 is a nice numerical approximation of int 0 to inf
#So by about 75–100, the price has stabilized

bounds = [25, 50, 75, 100, 150, 200]
print(f"2. Fourier integration convergence")
for bound in bounds:
    price_bound, _, _, _, _ = heston_price(
        S0, K, T, r, v0, vbar_star, kappa_star, zeta, rho, bound)
    print(bound, price_bound)

print(f"--------------------------")

# -------------------------------------------------------------
# 9. Black--Scholes sanity check
# -------------------------------------------------------------

def bs_call_price(S0, K, tau, r, q, sigma):

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    price = S0 * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    return price

sigma0 = np.sqrt(v0)

bs_price = bs_call_price(S0, K, T, r, q, sigma0)

print(f"3. Black-Scholes Comparison (Sanity Check)")
print("Heston price:", price)
print("Black--Scholes price:", bs_price)
print("Difference:", price - bs_price)
print(f"--------------------------")

# -------------------------------------------------------------
# 10. Implied volatility by bisection
# -------------------------------------------------------------

def implied_vol(market_price, S0, K, r, q, T,
                sigma_low=1e-8, sigma_high=2.0, tol=1e-8,
                max_iter=200, expand_max=10):

    lower_bound = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper_bound = S0 * np.exp(-q * T)

    if not (lower_bound <= market_price <= upper_bound):
        return np.nan

    def pricing_error(sigma):
        return bs_call_price(S0, K, T, r, q, sigma) - market_price

    low = sigma_low
    high = sigma_high

    for _ in range(expand_max):
        if pricing_error(low) * pricing_error(high) <= 0:
            break
        high *= 2

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        error_mid = pricing_error(mid)

        if abs(error_mid) < tol:
            return mid

        if pricing_error(low) * error_mid <= 0:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)

# -------------------------------------------------------------
# 11. Correlation sensitivity
# -------------------------------------------------------------

rho_values = [0.0, -0.3, -0.7]

curve_styles = {
    0.0: {"color": "#4C78A8", "linestyle": "--", "linewidth": 1.4, "label": r"$\rho=0.0$"},
    -0.3: {"color": "0.45", "linestyle": "-.", "linewidth": 1.4, "label": r"$\rho=-0.3$"},
    -0.7: {"color": "black", "linestyle": "-", "linewidth": 1.6, "label": r"$\rho=-0.7$"},
}

F0 = S0 * np.exp((r - q) * T)

log_moneyness = np.linspace(-0.55, 0.55, 80)
K_values = F0 * np.exp(log_moneyness)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(8.5, 5.0))

rho_results = []

for rho_i in rho_values:

    implied_vols_rho = []

    for K_i in K_values:

        price_i, _, _, _, _ = heston_price(
            S0, K_i, T, r, v0, vbar_star, kappa_star, zeta, rho_i, upper_bound=100
        )

        iv_i = implied_vol(price_i, S0, K_i, r, q, T)
        implied_vols_rho.append(iv_i)

        rho_results.append({
            "rho": rho_i,
            "K": K_i,
            "log_moneyness": np.log(K_i / F0),
            "Heston_price": price_i,
            "implied_volatility": iv_i
        })

    ax.plot(log_moneyness, implied_vols_rho, **curve_styles[rho_i])

ax.axvline(0, color="0.5", linestyle=(0, (6, 3)), linewidth=1.0,
           label="At-the-money forward")

ax.set_xlabel(r"Log-forward moneyness $\log(K/F_0)$", fontsize=12)
ax.set_ylabel("Implied volatility", fontsize=12)

ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.tick_params(axis="both", labelsize=10)

ax.grid(axis="y", linewidth=0.5, alpha=0.35)

ax.legend(frameon=False, loc="upper right", fontsize=10)

ax.set_xlim(-0.55, 0.55)

fig.tight_layout()

# -------------------------------------------------------------
# 12. Save numerical results
# -------------------------------------------------------------

rho_results = pd.DataFrame(rho_results)

results_filename = "heston_rho_sensitivity_results.csv"
rho_results.to_csv(results_filename, index=False)

print(f"Saved Heston correlation-sensitivity results to: {results_filename}")

# -------------------------------------------------------------
# 13. Save figure
# -------------------------------------------------------------

figure_filename = "heston_rho_sensitivity.png"

fig.savefig(figure_filename, dpi=400, bbox_inches="tight")

print(f"Saved figure to: {figure_filename}")

plt.show()