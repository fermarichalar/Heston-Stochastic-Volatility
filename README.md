# Heston Stochastic Volatility Model

Implementation and analysis of the **Heston stochastic volatility model** for pricing European options.

The project explores how stochastic variance and stock–variance correlation affect option prices and the shape of the implied-volatility curve.

A mathematical derivation and numerical implementation are provided alongside the project.

---

## Overview

This repository implements the **Heston model** for European option pricing.

Unlike Black–Scholes, where volatility is assumed to be constant, Heston allows variance to evolve stochastically through a mean-reverting **CIR process**.

The project covers:

- Heston stock and variance dynamics
- risk-neutral pricing
- derivation of the Heston pricing PDE
- log-price transformation
- Fourier-transform methods
- exponential-affine solutions
- Riccati equations
- numerical evaluation of the Heston pricing formula
- implied-volatility smile and skew analysis

---

## Mathematical Framework

Under the risk-neutral measure, the Heston model is

```math
dS_t = rS_t\,dt + \sqrt{v_t}S_t\,dW_t^{S,\mathbb{Q}}.
```

```math
dv_t
=
\kappa^*(\bar v^*-v_t)\,dt
+
\zeta\sqrt{v_t}\,dW_t^{v,\mathbb{Q}}.
```

with correlated Brownian motions

```math
d\langle W^{S,\mathbb{Q}},W^{v,\mathbb{Q}}\rangle_t
=
\rho\,dt.
```

Here:

- $v_t$ is instantaneous variance
- $\kappa^*$ is the risk-neutral mean-reversion speed
- $\bar v^*$ is the long-run variance level
- $\zeta$ is the volatility of variance
- $\rho$ is the correlation between stock and variance shocks

The square-root variance process preserves non-negative variance and introduces mean reversion.

---

## Pricing PDE

For a European option with price $C(t,S,v)$, the Heston pricing PDE is

```math
C_t
+\frac12 vS^2C_{SS}
+\rho\zeta vS C_{Sv}
+\frac12\zeta^2vC_{vv}
+rSC_S-rC
+\kappa^*(\bar v^*-v)C_v
=0.
```

The stock–variance correlation parameter enters directly through the mixed derivative term

```math
\rho\zeta vS C_{Sv}.
```

---

## Fourier Pricing Method

The pricing problem is transformed using log-price

```math
x=\ln S.
```

The European call price is decomposed as

```math
C(t,S,v)
=
SP_1(t,x,v)
-
Ke^{-r\tau}P_2(t,x,v),
\qquad
\tau=T-t.
```

The resulting equations for $P_1$ and $P_2$ are transformed into Fourier space.

Using an **exponential-affine ansatz**, the transformed pricing problem reduces to a system involving:

- a Riccati ODE for $D_j$
- a linear ODE for $C_j$

These equations admit closed-form solutions, leaving only a one-dimensional Fourier integral to evaluate numerically.

The resulting Heston call price is therefore **semi-analytic**.

---

## Numerical Implementation

The baseline numerical experiment uses

```math
S_0=100,\qquad
K=100,\qquad
r=0.04,\qquad
T=1.
```

with variance parameters

```math
v_0=0.04,\qquad
\bar v^*=0.04,\qquad
\kappa^*=2,\qquad
\zeta=0.3,\qquad
\rho=-0.7.
```

The numerical Fourier integration gives

```math
P_1 \approx 0.674107,
\qquad
P_2 \approx 0.599492,
```

and therefore

```math
C_{\mathrm{Heston}}
\approx
9.812195.
```

For comparison, Black–Scholes using the initial volatility

```math
\sigma_0=\sqrt{v_0}=20\%
```

gives

```math
C_{\mathrm{BS}}
\approx
9.925054.
```

---

## Fourier Integral Convergence

The Heston pricing formula contains an integral over

```math
[0,\infty).
```

Numerically, this is replaced by a finite upper integration bound $U$.

The implementation checks convergence using

```math
U\in\{25,50,75,100,150,200\}.
```

The estimated option price is stable to approximately six decimal places by

```math
U\approx75,
```

so $U=100$ is used for the remaining numerical calculations.

---

## Implied-Volatility Analysis

The project also studies how **stock–variance correlation** affects the implied-volatility curve.

All parameters are held fixed while

```math
\rho\in\{0,-0.3,-0.7\}.
```

For each value of $\rho$:

1. Heston call prices are computed across a range of strikes
2. Black–Scholes implied volatility is recovered numerically
3. implied volatility is plotted against log-forward moneyness

```math
\log\left(\frac{K}{F_0}\right),
\qquad
F_0=S_0e^{(r-q)T}.
```

---

## Results

The numerical experiment shows that:

- stochastic variance generates curvature in the implied-volatility curve
- when $\rho=0$, the smile is approximately symmetric
- increasingly negative correlation makes the smile progressively more asymmetric
- lower-strike implied volatility increases as correlation becomes more negative
- higher-strike implied volatility decreases
- negative stock–variance correlation therefore generates the downward skew commonly observed in equity options

The strongest skew in the experiment occurs for

```math
\rho=-0.7.
```

---

## Implementation Features

- Heston European call pricing
- CIR stochastic variance process
- Fourier-transform pricing
- closed-form Riccati coefficients
- numerical quadrature
- Fourier-integral convergence analysis
- Black–Scholes benchmark
- implied-volatility inversion
- correlation sensitivity analysis
- implied-volatility smile and skew visualisation

---

## References

- Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). *A Theory of the Term Structure of Interest Rates*. Econometrica.
- Duffie, D., Pan, J., & Singleton, K. J. (2000). *Transform Analysis and Asset Pricing for Affine Jump-Diffusions*. Econometrica.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
- Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*. The Review of Financial Studies.
- Hull, J., & White, A. (1987). *The Pricing of Options on Assets with Stochastic Volatilities*. The Journal of Finance.
