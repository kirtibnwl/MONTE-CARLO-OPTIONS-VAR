# Monte Carlo Option Pricing & Value at Risk (VaR) Calculator

> A complete quantitative finance simulation engine built from scratch using Geometric Brownian Motion (GBM) to price European options and compute portfolio risk metrics — fully validated against Black-Scholes analytical solutions.

**Author:** Kirti Beniwal  
**Affiliation:** Department of Applied Mathematics, Delhi Technological University (DTU), Delhi, India  
**Contact:** kirtibnwl1912@gmail.com

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [The Mathematics](#the-mathematics)
3. [Project Architecture](#project-architecture)
4. [Quickstart](#quickstart)
5. [Configuration Guide](#configuration-guide)
6. [Output Files](#output-files)
7. [Module-by-Module Explanation](#module-by-module-explanation)
8. [Key Results Explained](#key-results-explained)
9. [Bugs Fixed](#bugs-fixed)
10. [Dependencies](#dependencies)
11. [References](#references)

---

## What This Project Does

This project simulates **50,000 possible futures** for a stock or index price and uses them for two critical financial computations:

### Purpose 1 — European Option Pricing

An **option** gives the buyer the *right (but not obligation)* to buy or sell a stock at a pre-agreed price (the **strike price K**) on a future date.

**The question:** How much should you pay for this right *today*?

**The Monte Carlo answer:**
1. Simulate 50,000 possible stock price paths using GBM
2. For each simulated future, compute what the option would pay
3. Average all payoffs and discount them back to today

The **Black-Scholes analytical formula** is also computed to validate that the simulation is correct. If Monte Carlo ≈ Black-Scholes → the implementation is working.

### Purpose 2 — Value at Risk (VaR)

**VaR answers:** *"What is the maximum I could lose on a bad day?"*

Specifically: *"On 95% of trading days, my loss will NOT exceed ₹X."*

VaR is required by **Basel III banking regulations** — every bank and financial institution must compute it daily to determine how much capital to hold against trading risk.

---

## The Mathematics

### 1. Geometric Brownian Motion (GBM)

Stock prices do not move in straight lines. They drift upward on average but fluctuate randomly around that drift. GBM models both:

```
dS = μ · S · dt  +  σ · S · dW
     ───────────     ───────────
       Drift            Random
      (growth)       (Wiener process)
```

**Discrete simulation formula** — what we actually code, derived from Itô's Lemma:

```
S(t + Δt) = S(t) × exp[ (μ − σ²/2) · Δt  +  σ · √Δt · Z ]
```

| Symbol | Meaning | Example value |
|--------|---------|---------------|
| `S(t)` | Stock price at time t | ₹19,000 (Nifty 50) |
| `μ` | Annual expected return (drift) | 0.12 = 12% per year |
| `σ` | Annual volatility | 0.18 = 18% per year |
| `Δt` | One time-step in years | 1/252 ≈ one trading day |
| `Z` | Random number ~ N(0,1) | Drawn fresh each step |

**Why `(μ − σ²/2)` and not just `μ`?**

This is the **Itô correction**. When we take the log of a lognormal variable, Jensen's inequality means:

```
E[log(X)] ≠ log(E[X])
```

The `−σ²/2` term corrects this bias so the simulated expected growth rate matches `μ` in the original price space. It arises directly from Itô's Lemma applied to `log(S)`.

---

### 2. Option Pricing Formulas

#### Call Option
Gives the right to **BUY** a stock at strike K at expiry.

```
Call Payoff   = max(S_T − K, 0)
MC Call Price = exp(−rT) × mean[ max(S_T − K, 0) ]
```

If `S_T > K` → profit = `S_T − K` (buy cheap at K, worth S_T in the market)  
If `S_T ≤ K` → walk away, payoff = 0

#### Put Option
Gives the right to **SELL** a stock at strike K at expiry.

```
Put Payoff   = max(K − S_T, 0)
MC Put Price = exp(−rT) × mean[ max(K − S_T, 0) ]
```

#### Black-Scholes Analytical Formula (for validation)

Black and Scholes (1973, Nobel Prize) derived a closed-form solution under the same GBM assumptions:

```
C = S0 · N(d1) − K · e^{−rT} · N(d2)
P = K · e^{−rT} · N(−d2) − S0 · N(−d1)

where:
  d1 = [ log(S0/K) + (r + σ²/2) · T ] / (σ · √T)
  d2 = d1 − σ · √T
  N(·) = Cumulative Standard Normal CDF (Φ in probability notation)
```

**Interpretation:**
- `S0 · N(d1)` = expected value of the stock if the option is exercised
- `K · e^{−rT} · N(d2)` = present value of the strike × probability of exercise

#### Put-Call Parity (sanity check)
```
C − P = S0 − K · e^{−rT}
```
This algebraic identity always holds for European options. We check it numerically to verify the simulation.

---

### 3. Option Greeks

Greeks measure the sensitivity of option price to each input parameter. They are the core tool of options risk management.

| Greek | Formula | What it measures |
|-------|---------|-----------------|
| **Delta Δ** | `∂C/∂S = N(d1)` | Price change per ₹1 move in underlying |
| **Gamma Γ** | `∂²C/∂S² = N'(d1) / (S·σ·√T)` | Rate of change of Delta (curvature) |
| **Vega ν** | `∂C/∂σ = S·N'(d1)·√T` | Price change per 1% change in volatility |
| **Theta Θ** | `∂C/∂t` (negative) | Daily time decay — options lose value as expiry approaches |
| **Rho ρ** | `∂C/∂r = K·T·e^{−rT}·N(d2)` | Price change per 1% change in interest rate |

Note: `N'(x)` is the standard Normal PDF = `(1/√2π) · exp(−x²/2)`

---

### 4. Value at Risk (VaR)

**Mathematical definition:**  
VaR at confidence level α is the loss threshold V such that:
```
P(Loss > V) = 1 − α
```

For α = 0.95: the loss exceeds V on only 5% of days.

**Three methods implemented and compared:**

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| **Monte Carlo VaR** | `percentile(losses, 95)` | Captures actual simulated distribution | Slow |
| **Parametric VaR** | `−(μ_PnL + z_{0.05} · σ_PnL)` | Very fast, closed form | Assumes Normal distribution → underestimates fat tails |
| **Historical VaR** | Percentile of actual past returns | Uses real market data | Limited to scenarios that have already occurred |

**Conditional VaR (CVaR) / Expected Shortfall:**
```
CVaR_α = E[ Loss | Loss > VaR_α ]
```
The average loss given that you are already in the worst (1−α) of scenarios.  
- CVaR ≥ VaR always  
- CVaR is a **coherent risk measure** (VaR is not)  
- Preferred by regulators under Basel III / FRTB

---

## Project Architecture

```
monte-carlo-options-var/
│
├── main.py                    ← Entry point: CONFIG + full 4-step pipeline
│
├── src/
│   ├── __init__.py            ← Package exports
│   ├── gbm_simulator.py       ← Simulate GBM stock price paths
│   ├── option_pricing.py      ← MC + Black-Scholes pricing + Greeks
│   ├── var_calculator.py      ← VaR / CVaR / parametric / historical
│   └── visualise.py           ← All 4 charts
│
├── outputs/                   ← Auto-created on first run
│   ├── gbm_paths.png
│   ├── option_payoffs.png
│   ├── var_dashboard.png
│   ├── convergence.png
│   └── results_summary.txt
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Data Flow

```
CONFIG (S0, mu, sigma, K, T, r, n_sim)
           │
           ▼
  simulate_gbm_paths()
  → paths: shape (n_steps+1, n_sim)
           │
     ┌─────┴──────────────────┐
     │                        │
     ▼                        ▼
get_terminal_prices()    compute_daily_returns_pnl()
S_T: shape (n_sim,)      daily_pnl: shape (n_sim,)
     │                        │
     ▼                        ▼
price_european_call_mc() compute_var_cvar()
price_european_put_mc()  compute_var_parametric()
black_scholes_call/put() print_var_report()
compute_greeks()
     │                        │
     └──────────┬─────────────┘
                │
                ▼
        visualise.py (4 charts)
                │
                ▼
          outputs/ folder
```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/monte-carlo-options-var.git
cd monte-carlo-options-var
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python main.py
```

All output files are written to the `outputs/` folder automatically. Runtime is approximately 30–60 seconds for 50,000 simulations.

---

## Configuration Guide

All parameters are in the `CONFIG` dictionary at the top of `main.py`. You do not need to edit any other file.

```python
CONFIG = {
    # ── Underlying Asset ──────────────────────────────────────────────────────
    "S0"            : 19000,   # Current price.  19000 = Nifty 50 index level.
    "mu"            : 0.12,    # Annual drift.   0.12 = 12% expected return/year.
    "sigma"         : 0.18,    # Annual vol.     0.18 = 18% volatility/year.

    # ── Option Parameters ─────────────────────────────────────────────────────
    "K"             : 19500,   # Strike price.   K > S0 → out-of-the-money call.
    "T"             : 0.5,     # Expiry (years). 0.5 = 6 months.
    "r"             : 0.065,   # Risk-free rate. 0.065 = 6.5% (RBI repo rate proxy).

    # ── Simulation ────────────────────────────────────────────────────────────
    "n_simulations" : 50000,   # Paths to simulate. More = more accurate, slower.
    "n_steps"       : 126,     # Time steps.  Should equal T × 252 trading days/year.

    # ── VaR ───────────────────────────────────────────────────────────────────
    "position_value"       : 1000000,             # Portfolio size in ₹.
    "var_confidence_levels": [0.90, 0.95, 0.99],  # Confidence levels for VaR.
}
```

**Common configuration examples:**

| Use Case | Parameters to change |
|----------|---------------------|
| Reliance Industries stock | `S0=2500`, `K=2600`, `sigma=0.22` |
| 3-month option | `T=0.25`, `n_steps=63` |
| Higher accuracy | `n_simulations=100000` |
| US stock (USD) | `S0=150`, `K=155`, `r=0.053` |
| Higher volatility market | `sigma=0.28` |

---

## Output Files

All files are saved to `outputs/` automatically.

### `gbm_paths.png` — Simulated Price Paths

Two-panel chart:
- **Left panel**: Fan of 200 randomly selected price paths out of 50,000.
  - Blue paths = end above K (call option is profitable at expiry)
  - Red paths = end below K (call option expires worthless)
  - Black thick line = median path
  - Grey shaded band = 5th to 95th percentile range (90% of all outcomes)
  - Orange dashed line = strike price K
- **Right panel**: Histogram of terminal prices S_T.
  - Shows the characteristic log-normal distribution (right-skewed bell curve)
  - Green shaded region = in-the-money area where call pays off

### `option_payoffs.png` — Payoff Distributions

Two-panel chart (call left, put right):
- Histogram of individual option payoffs across all 50,000 simulations
- Key insight: most options expire worthless (payoff = 0) — a large spike at zero
- When profitable, payoffs can be very large → long right tail
- The Monte Carlo price = discounted average of the entire payoff distribution

### `var_dashboard.png` — VaR Risk Chart

The primary risk management output. Two-panel chart:
- **Left panel**: Full P&L distribution with vertical lines showing VaR cutoffs at 90%, 95%, 99%
  - Red shaded region = extreme loss scenarios (worst 1%)
- **Right panel**: Zoomed-in view of the loss tail
  - Dashed lines = VaR thresholds
  - Dotted lines = CVaR (Expected Shortfall) — always to the right of (worse than) VaR

### `convergence.png` — Monte Carlo Accuracy vs Number of Simulations

Shows the Law of Large Numbers in action:
- X-axis (log scale): number of simulations from 10 to 50,000
- Y-axis: Monte Carlo call price estimate
- Red dashed line: Black-Scholes exact price (target)
- Green bands: ±1% and ±0.1% error regions
- As simulations increase, the estimate converges to the true price

**Key insight:** Error ∝ 1/√n. Doubling accuracy requires 4× more simulations.

### `results_summary.txt` — Numerical Results

Complete text file with:
- All input parameters
- Monte Carlo vs Black-Scholes prices with % error
- All five Greeks
- VaR and CVaR at each confidence level using all three methods

---

## Module-by-Module Explanation

### `src/gbm_simulator.py`

#### `simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_simulations)`

Generates all price paths in three fully vectorised NumPy steps:

```python
# Step 1: Draw all random shocks at once (very fast — no Python loop)
Z = np.random.standard_normal((n_steps, n_simulations))
# Shape: (126, 50000) — one random number per (day, simulation)

# Step 2: Convert to log returns using the GBM discretisation
log_returns = (mu - 0.5 * sigma**2) * dt  +  sigma * np.sqrt(dt) * Z
#              ────────────────────────       ─────────────────────────
#                  Deterministic drift           Random diffusion term

# Step 3: Cumulative sum gives total log return from t=0 to each step
#         exp() converts log returns back to price ratios
#         Multiply by S0 to get actual price levels
cumulative_log_returns = np.cumsum(log_returns, axis=0)
paths = np.vstack([
    np.full(n_simulations, S0),         # Row 0: all paths start at S0
    S0 * np.exp(cumulative_log_returns) # Rows 1…n: evolved prices
])
# Final shape: (n_steps+1, n_simulations)
```

**Why use cumsum + exp instead of a for loop?**  
NumPy vectorisation is 100–1000× faster than a Python loop. With 50,000 paths × 126 steps = 6.3 million calculations, this matters a lot.

#### `get_terminal_prices(paths)`
Returns `paths[-1, :]` — the last row, which is the final price at expiry for each simulation. For European options, only the terminal price matters (not the path).

#### `compute_path_statistics(paths, S0)`
Computes summary statistics (mean, std, percentiles, % paths above S0) to understand the distribution of simulated outcomes.

---

### `src/option_pricing.py`

#### `price_european_call_mc(S_T, K, r, T)`

```python
payoffs      = np.maximum(S_T - K, 0)                     # max(S_T - K, 0)
discount     = np.exp(-r * T)                              # e^{-rT}
price        = discount * np.mean(payoffs)                 # Discounted avg
std_error    = discount * np.std(payoffs) / np.sqrt(len(payoffs))  # Precision
```

The `std_error` quantifies simulation precision. With 50,000 paths, the 95% confidence interval is `price ± 1.96 × std_error`.

**Why do we discount?**  
The payoff is received at time T in the future. We need its value in today's money. `exp(−rT)` is the present value factor — the same principle as discounted cash flow in finance.

#### `black_scholes_call(S0, K, r, sigma, T)`

```python
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
C  = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
```

This is the **exact analytical solution** under GBM assumptions. We use it as ground truth to validate the MC simulation.

#### `compute_greeks(S0, K, r, sigma, T)`

All five Greeks computed analytically from the Black-Scholes formula:

```python
N_prime_d1 = norm.pdf(d1)                     # Standard Normal PDF at d1

call_delta  = norm.cdf(d1)                     # Δ_call = N(d1)
put_delta   = norm.cdf(d1) - 1                 # Δ_put  = N(d1) - 1
gamma       = N_prime_d1 / (S0*sigma*np.sqrt(T))  # Γ (same for call & put)
vega        = S0 * N_prime_d1 * np.sqrt(T) / 100  # ν per 1% vol change
call_theta  = (-(S0*N_prime_d1*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
put_theta   = (-(S0*N_prime_d1*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
call_rho    = K*T*np.exp(-r*T)*norm.cdf(d2)   / 100   # ρ per 1% rate change
put_rho     = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
```

---

### `src/var_calculator.py`

#### `compute_daily_returns_pnl(paths, S0, position_value)`

```python
S_1            = paths[1, :]               # First-step price = "tomorrow's price"
daily_log_return = np.log(S_1 / S0)        # Log return (standard in finance)
daily_pnl      = position_value * daily_log_return  # ₹ P&L
```

**Why log returns (not arithmetic returns)?**
- Log returns are time-additive: `r_{total} = r_1 + r_2 + … + r_n`
- Consistent with GBM model (which uses log returns internally)
- Standard convention in financial risk management

#### `compute_var_cvar(pnl, confidence_level=0.95)`

```python
losses     = -pnl                                         # Loss = -P&L (positive = bad)
var        = np.percentile(losses, confidence_level*100)  # 95th percentile of losses
tail_losses = losses[losses >= var]                        # Scenarios worse than VaR
cvar       = np.mean(tail_losses)                          # Average of worst 5%
```

#### `compute_var_parametric(pnl, confidence_level=0.95)`

```python
z   = norm.ppf(1 - confidence_level)    # norm.ppf(0.05) = -1.645 for 95% VaR
var = -(mean_pnl + z * std_pnl)         # = 1.645σ - μ  (loss amount)
```

Assumes P&L ~ Normal distribution. Underestimates risk vs Monte Carlo because financial returns have heavier tails than Normal.

---

### `src/visualise.py`

Four chart functions, each saving a PNG to `outputs/`. Global matplotlib settings applied once at module load:
```python
plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
})
```

---

## Key Results Explained

With default parameters (`S0=19000`, `K=19500`, `T=0.5 yr`, `σ=18%`, `r=6.5%`, 50,000 paths):

```
OPTION PRICING
  Monte Carlo CALL  ≈  794    ←  less than 0.5% error vs Black-Scholes
  Monte Carlo PUT   ≈  1,089  ←  less than 0.5% error vs Black-Scholes

GREEKS
  Delta  (call) ≈  0.43   → call price rises ₹0.43 for every ₹1 rise in Nifty
  Delta  (put)  ≈ -0.57   → put price falls ₹0.57 for every ₹1 rise in Nifty
  Gamma         ≈  0.0001 → Delta changes by 0.0001 per ₹1 move
  Vega          ≈  47.5   → option price changes ₹47.5 for 1% change in volatility
  Theta  (call) ≈ -3.5    → call loses ₹3.50 per day due to time decay

VALUE AT RISK  (₹10,00,000 portfolio)
  90% VaR  ≈ ₹13,500    CVaR ≈ ₹18,200
  95% VaR  ≈ ₹17,200    CVaR ≈ ₹22,800
  99% VaR  ≈ ₹24,600    CVaR ≈ ₹30,100
```

**Why is the call price low relative to index level 19,000?**  
The option is **out-of-the-money**: the strike K=19,500 is 2.6% above the current price. The stock must rise 2.6% just to break even at expiry. Most simulated paths end below K, so most options expire worthless — this is correctly reflected in the low average payoff.

**Why is MC VaR higher than Parametric VaR?**  
The parametric method assumes P&L is Normally distributed (thin tails). The GBM simulation produces returns that are lognormally distributed with slightly fatter tails — so the 5% worst outcomes are worse than Normal predicts. This difference is the **model risk** of using the Normal approximation.

---

## Bugs Fixed

Two visualisation bugs were found and corrected in `src/visualise.py`:

### Bug 1 — Wrong legend handle for "5th–95th band" in `plot_gbm_paths`

**Root cause:** `matplotlib.pyplot.fill_between()` creates a `PolyCollection` object internally, which does **not** appear in `ax.get_lines()`. The original code used `ax.get_lines()[n_display+1]` expecting the grey percentile band, but that index actually returned the orange K-axhline. The legend was silently showing the wrong artist (orange line) labelled as "5th–95th band".

**Fix:** Replaced the incorrect `ax.get_lines()` call with an explicit `mpatches.Patch`:

```python
# Before (bug):
ax.get_lines()[n_display+1]   # returns K-axhline, not the fill_between band

# After (correct):
band_patch = mpatches.Patch(color="gray", alpha=0.3, label="5th–95th percentile band")
```

### Bug 2 — Fragile x-boundary in `fill_betweenx` in `plot_var_dashboard`

**Root cause:** The original code used `ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else daily_pnl.min()`. Since P&L always has a negative minimum (losses exist), `ax.get_xlim()[0]` is never zero — so the `!= 0` guard was always false, and `ax.get_xlim()[0]` was always used. This could include matplotlib's auto-padding offset, causing the shaded region to start slightly to the right of the actual data minimum and miss the most extreme loss scenarios.

**Fix:** Always use the actual data minimum directly:

```python
# Before (fragile):
ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else daily_pnl.min()

# After (correct):
daily_pnl.min()
```

---

## Dependencies

| Package | Min Version | Purpose |
|---------|------------|---------|
| `numpy` | 1.26 | Vectorised simulation, array math, random number generation |
| `scipy` | 1.12 | `norm.cdf`, `norm.pdf`, `norm.ppf` for Black-Scholes and Greeks |
| `matplotlib` | 3.8 | All visualisations |
| `pandas` | 2.0 | Data handling utilities |

Install:
```bash
pip install -r requirements.txt
```

---

## Connection to Mathematical Research

This project connects directly to the author's research background:

- The GBM discretisation `S(t+Δt) = S(t)·exp[...]` is a **numerical time-stepping scheme** — the same class of methods used in PDE solvers.
- The Black-Scholes equation:
  ```
  ∂V/∂t  +  (1/2)σ²S²·∂²V/∂S²  +  rS·∂V/∂S  −  rV  =  0
  ```
  is a **convection-diffusion PDE** — the same class as the singularly perturbed boundary value problems studied in [Beniwal et al., *Physica Scripta* 2025](https://doi.org/10.1088/1402-4896/adb703), where the Bernstein Collocation Method was applied.
- Monte Carlo convergence analysis (error ∝ 1/√n) parallels the convergence order analysis in numerical PDE methods.

---

## References

1. **Black, F. & Scholes, M. (1973).** "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654.
2. **Merton, R.C. (1973).** "Theory of Rational Option Pricing." *Bell Journal of Economics*, 4(1), 141–183.
3. **Glasserman, P. (2003).** *Monte Carlo Methods in Financial Engineering.* Springer.
4. **Hull, J.C. (2021).** *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
5. **Basel Committee on Banking Supervision (2019).** *Minimum Capital Requirements for Market Risk.* Bank for International Settlements.
6. **Itô, K. (1951).** "On Stochastic Differential Equations." *Memoirs of the American Mathematical Society*, 4, 1–51.

---

## License

MIT License — free to use, modify, and distribute with attribution.
