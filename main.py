# =============================================================================
#
#   MONTE CARLO OPTION PRICING AND VALUE AT RISK CALCULATOR
#   Using Geometric Brownian Motion (GBM)
#
#   AUTHOR    : Kirti Beniwal
#   AFFILIATION: Department of Applied Mathematics,
#                Delhi Technological University, Delhi, India
#   CONTACT   : kirtibnwl1912@gmail.com
#
#   WHAT THIS PROJECT DOES (Plain English):
#   ────────────────────────────────────────
#
#   This project models how stock prices move using a mathematical model
#   called Geometric Brownian Motion (GBM), and uses it for two purposes:
#
#   PURPOSE 1 — OPTION PRICING:
#   ─────────────────────────────
#   An option gives you the RIGHT (but not obligation) to buy or sell a
#   stock at a pre-agreed price (called the strike price) at a future date.
#
#   How much should you PAY for this right today?
#   Monte Carlo answers this by simulating 10,000 possible futures,
#   computing the option payoff in each future, and averaging them.
#
#   We also compute the Black-Scholes analytical price to VALIDATE
#   that our simulation is working correctly.
#
#   PURPOSE 2 — VALUE AT RISK (VaR):
#   ──────────────────────────────────
#   VaR answers: "What is the maximum I could lose on a bad day?"
#   Specifically: "On 95% of trading days, my loss will not exceed ₹X"
#
#   This is required by banking regulations (Basel III) for every
#   financial institution holding trading positions.
#   Every risk management team at a bank computes VaR daily.
#
#   THE MATHEMATICS:
#   ─────────────────
#   GBM Equation: dS = μS dt + σS dW
#
#   Discrete form (one step forward):
#     S(t+dt) = S(t) × exp[(μ − σ²/2)dt + σ√dt × Z]
#     where Z ~ N(0,1) is a standard Normal random number
#
#   Call option payoff: max(S_T − K, 0)
#   Put  option payoff: max(K − S_T, 0)
#   Option price = exp(−rT) × E[payoff]   (risk-neutral pricing)
#
#   HOW TO RUN:
#   ────────────
#   pip install -r requirements.txt
#   python main.py
#
#   OUTPUT FILES (in outputs/ folder):
#   ───────────────────────────────────
#   • gbm_paths.png         — Fan of simulated price paths
#   • option_payoffs.png    — Call and put payoff distributions
#   • var_dashboard.png     — VaR chart (main risk output)
#   • convergence.png       — MC accuracy vs number of simulations
#   • results_summary.txt   — All numerical results in one file
#
#   CONFIGURATION:
#   ───────────────
#   Change the values in CONFIG below to model different stocks/options.
#   All explanations of each parameter are in the comments.
#
# =============================================================================

import sys
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)   # Set seed for reproducibility (same results each run)

sys.path.insert(0, str(Path(__file__).parent))

from src.gbm_simulator  import simulate_gbm_paths, get_terminal_prices, compute_path_statistics
from src.option_pricing import (price_european_call_mc, price_european_put_mc,
                                 black_scholes_call, black_scholes_put,
                                 compute_greeks, print_pricing_comparison)
from src.var_calculator import (compute_portfolio_pnl, compute_daily_returns_pnl,
                                 compute_var_cvar, compute_var_parametric,
                                 print_var_report)
from src.visualise      import (plot_gbm_paths, plot_option_payoffs,
                                 plot_var_dashboard, plot_mc_convergence)


# =============================================================================
#   CONFIGURATION — Edit these values to model your own stock/option
# =============================================================================

CONFIG = {
    # ── Stock / Underlying Asset ───────────────────────────────────────────────
    "S0"           : 19000,   # Current stock/index price
                              # e.g. 19000 = Nifty 50 current level
                              # or 2500 = Reliance Industries price in ₹

    "mu"           : 0.12,    # Annual expected return (drift)
                              # e.g. 0.12 = 12% per year
                              # For Nifty 50: historical average ≈ 12-14% per year
                              # This is the μ (mu) in the GBM equation

    "sigma"        : 0.18,    # Annual volatility (standard deviation of returns)
                              # e.g. 0.18 = 18% per year
                              # For Nifty 50: typical range is 15-25% per year
                              # Higher sigma → wider fan of paths → higher option price
                              # This is the σ (sigma) in the GBM equation

    # ── Option Parameters ─────────────────────────────────────────────────────
    "K"            : 19500,   # Strike price (agreed buy/sell price)
                              # K > S0 → out-of-the-money call (cheaper)
                              # K = S0 → at-the-money call
                              # K < S0 → in-the-money call (more expensive)

    "T"            : 0.5,     # Time to expiry in YEARS
                              # e.g. 0.5 = 6 months, 0.25 = 3 months, 1.0 = 1 year

    "r"            : 0.065,   # Annual risk-free interest rate
                              # e.g. 0.065 = 6.5% per year
                              # Use RBI repo rate or Indian T-bill yield as proxy

    # ── Simulation Parameters ─────────────────────────────────────────────────
    "n_simulations": 50000,   # Number of Monte Carlo paths to simulate
                              # More = more accurate but slower
                              # 10,000 → ~1% error, 50,000 → ~0.4% error
                              # 100,000 → ~0.3% error (diminishing returns)

    "n_steps"      : 126,     # Number of daily time steps
                              # 126 = approx. 6 months of trading days (252/2)
                              # Should match T: n_steps ≈ T × 252

    # ── VaR Parameters ────────────────────────────────────────────────────────
    "position_value"     : 1000000,   # Portfolio value in ₹ (₹10 lakh = ₹10,00,000)
    "var_confidence_levels": [0.90, 0.95, 0.99],  # 90%, 95%, 99% confidence
}


# =============================================================================
#   MAIN PIPELINE
# =============================================================================

def main():
    """
    Run the complete Monte Carlo option pricing and VaR pipeline.

    STEPS:
    1. Simulate stock price paths using GBM
    2. Price European call and put options
    3. Compute Greeks (sensitivities)
    4. Compute Value at Risk and Conditional VaR
    5. Generate all visualisations
    6. Save results summary
    """

    print("\n" + "=" * 65)
    print("  MONTE CARLO OPTION PRICING AND VaR CALCULATOR")
    print("  Using Geometric Brownian Motion")
    print()
    print("  Author     : Kirti Beniwal")
    print("  Affiliation: Delhi Technological University")
    print("=" * 65)

    # Unpack config for convenience
    S0            = CONFIG["S0"]
    mu            = CONFIG["mu"]
    sigma         = CONFIG["sigma"]
    K             = CONFIG["K"]
    T             = CONFIG["T"]
    r             = CONFIG["r"]
    n_sim         = CONFIG["n_simulations"]
    n_steps       = CONFIG["n_steps"]
    pos_val       = CONFIG["position_value"]
    conf_levels   = CONFIG["var_confidence_levels"]

    print(f"\n  PARAMETERS:")
    print(f"  Current price (S0)   = {S0:,.0f}")
    print(f"  Strike price (K)     = {K:,.0f}")
    print(f"  Time to expiry (T)   = {T} years ({T*252:.0f} trading days)")
    print(f"  Annual drift (μ)     = {mu*100:.1f}%")
    print(f"  Annual volatility (σ)= {sigma*100:.1f}%")
    print(f"  Risk-free rate (r)   = {r*100:.1f}%")
    print(f"  Simulations          = {n_sim:,}")
    print(f"  Portfolio value      = ₹{pos_val:,.0f}")

    # ═════════════════════════════════════════════════════════════════════════
    #   STEP 1: SIMULATE GBM PRICE PATHS
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n\n[STEP 1 of 4]  Simulating {n_sim:,} GBM Price Paths")
    print("─" * 45)
    print(f"  Simulating {n_sim:,} possible futures...")
    print(f"  Each path = {n_steps} daily time steps over {T} years")
    print(f"  Daily time step (dt) = {T/n_steps:.6f} years ≈ 1 trading day")

    paths, dt = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_sim)

    # Get terminal prices (final price for each simulation)
    S_T = get_terminal_prices(paths)

    # Print path statistics
    stats = compute_path_statistics(paths, S0)
    print(f"\n  Path Statistics:")
    for key, val in stats.items():
        if "─" not in str(key):
            print(f"    {key:<35}: {val}")

    print(f"\n  Simulations complete: {paths.shape[1]:,} paths × {paths.shape[0]} steps")

    # ═════════════════════════════════════════════════════════════════════════
    #   STEP 2: PRICE OPTIONS
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n\n[STEP 2 of 4]  Pricing European Options")
    print("─" * 45)
    print(f"  Computing call and put prices using:")
    print(f"  (a) Monte Carlo simulation  — average of simulated payoffs")
    print(f"  (b) Black-Scholes formula   — analytical exact solution")
    print(f"  Comparison validates that our simulation is working correctly.")

    # Monte Carlo pricing
    mc_call, mc_call_err, call_payoffs = price_european_call_mc(S_T, K, r, T)
    mc_put,  mc_put_err,  put_payoffs  = price_european_put_mc(S_T,  K, r, T)

    # Black-Scholes pricing (analytical)
    bs_call, d1, d2 = black_scholes_call(S0, K, r, sigma, T)
    bs_put,  _,  _  = black_scholes_put(S0,  K, r, sigma, T)

    # Print comparison
    greeks = print_pricing_comparison(
        S0, K, r, sigma, T,
        mc_call, mc_call_err,
        mc_put,  mc_put_err
    )

    # ═════════════════════════════════════════════════════════════════════════
    #   STEP 3: COMPUTE VALUE AT RISK
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n\n[STEP 3 of 4]  Computing Value at Risk (VaR)")
    print("─" * 45)
    print(f"  Position size: ₹{pos_val:,.0f}")
    print(f"  Using first step of each path as 'next day's price'")
    print(f"  Daily P&L = position_value × daily_log_return")

    # Compute daily P&L from simulated paths
    daily_pnl = compute_daily_returns_pnl(paths, S0, pos_val)

    # Print VaR report
    print_var_report(daily_pnl, position_value=pos_val,
                     confidence_levels=conf_levels)

    # ═════════════════════════════════════════════════════════════════════════
    #   STEP 4: GENERATE ALL PLOTS
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n\n[STEP 4 of 4]  Generating Visualisations")
    print("─" * 45)

    print("\n  Plot 1/4: GBM price paths...")
    plot_gbm_paths(paths, S0, K, T)

    print("\n  Plot 2/4: Option payoff distributions...")
    plot_option_payoffs(call_payoffs, put_payoffs, K,
                        mc_call, mc_put, bs_call, bs_put)

    print("\n  Plot 3/4: VaR dashboard...")
    plot_var_dashboard(daily_pnl, pos_val, conf_levels)

    print("\n  Plot 4/4: Monte Carlo convergence...")
    plot_mc_convergence(paths, K, r, T, bs_call)

    # ═════════════════════════════════════════════════════════════════════════
    #   SAVE RESULTS SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    save_results_summary(S0, K, T, r, sigma, mu, n_sim,
                         mc_call, mc_call_err, mc_put, mc_put_err,
                         bs_call, bs_put, greeks,
                         daily_pnl, pos_val, conf_levels)

    # ═════════════════════════════════════════════════════════════════════════
    #   FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print()
    print("  Output files in outputs/ folder:")
    print("  ├── gbm_paths.png          ← Simulated price paths")
    print("  ├── option_payoffs.png     ← Payoff distributions")
    print("  ├── var_dashboard.png      ← VaR risk chart")
    print("  ├── convergence.png        ← MC accuracy vs n_simulations")
    print("  └── results_summary.txt    ← All numerical results")
    print()
    print("  KEY RESULTS:")
    print(f"  Call option price (MC)    = {mc_call:.4f}")
    print(f"  Call option price (BS)    = {bs_call:.4f}")
    print(f"  Error vs Black-Scholes    = {abs(mc_call-bs_call)/bs_call*100:.3f}%")
    var_95, cvar_95 = compute_var_cvar(daily_pnl, 0.95)
    print(f"  95% Daily VaR             = ₹{var_95:,.0f}")
    print(f"  95% Daily CVaR            = ₹{cvar_95:,.0f}")
    print()
    print("  INTERVIEW LINE:")
    print("  'I built a Monte Carlo option pricing engine from scratch using")
    print("   GBM simulation — no Black-Scholes shortcut — to price European")
    print(f"  options with {n_sim:,} paths, achieving {abs(mc_call-bs_call)/bs_call*100:.2f}% error")
    print("   vs the analytical Black-Scholes price. I also computed VaR and")
    print("   CVaR using three methods: Monte Carlo, parametric, and historical.'")
    print("=" * 65)


def save_results_summary(S0, K, T, r, sigma, mu, n_sim,
                         mc_call, mc_call_err, mc_put, mc_put_err,
                         bs_call, bs_put, greeks,
                         daily_pnl, pos_val, conf_levels):
    """Save all numerical results to a text file."""

    from src.var_calculator import compute_var_cvar, compute_var_parametric

    lines = []
    lines.append("=" * 60)
    lines.append("MONTE CARLO OPTION PRICING AND VaR — RESULTS SUMMARY")
    lines.append("Author: Kirti Beniwal | DTU Delhi")
    lines.append("=" * 60)
    lines.append("")
    lines.append("INPUT PARAMETERS")
    lines.append(f"  Current price (S0) = {S0:,.2f}")
    lines.append(f"  Strike price (K)   = {K:,.2f}")
    lines.append(f"  Expiry (T)         = {T} years")
    lines.append(f"  Drift (mu)         = {mu*100:.1f}%")
    lines.append(f"  Volatility (sigma) = {sigma*100:.1f}%")
    lines.append(f"  Risk-free rate (r) = {r*100:.1f}%")
    lines.append(f"  N simulations      = {n_sim:,}")
    lines.append("")
    lines.append("OPTION PRICING RESULTS")
    lines.append(f"  Monte Carlo CALL   = {mc_call:.6f}  (±{1.96*mc_call_err:.6f} at 95% CI)")
    lines.append(f"  Black-Scholes CALL = {bs_call:.6f}")
    lines.append(f"  Error              = {abs(mc_call-bs_call)/bs_call*100:.4f}%")
    lines.append(f"  Monte Carlo PUT    = {mc_put:.6f}  (±{1.96*mc_put_err:.6f} at 95% CI)")
    lines.append(f"  Black-Scholes PUT  = {bs_put:.6f}")
    lines.append(f"  Error              = {abs(mc_put-bs_put)/bs_put*100:.4f}%")
    lines.append("")
    lines.append("OPTION GREEKS")
    for k, v in greeks.items():
        if k not in ["d1", "d2"]:
            lines.append(f"  {k:<20} = {v}")
    lines.append("")
    lines.append(f"VALUE AT RISK  |  Portfolio: Rs {pos_val:,.0f}")
    for cl in conf_levels:
        var, cvar = compute_var_cvar(daily_pnl, cl)
        pvar = compute_var_parametric(daily_pnl, cl)
        lines.append(f"  {cl*100:.0f}% MC VaR    = Rs {var:,.0f}")
        lines.append(f"  {cl*100:.0f}% MC CVaR   = Rs {cvar:,.0f}")
        lines.append(f"  {cl*100:.0f}% Param VaR = Rs {pvar:,.0f}")
        lines.append("")

    with open("outputs/results_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Saved: outputs/results_summary.txt")


if __name__ == "__main__":
    main()
