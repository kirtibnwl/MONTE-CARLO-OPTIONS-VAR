# =============================================================================
#  FILE: var_calculator.py
#  PURPOSE: Compute Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall)
#           using Monte Carlo simulation
#
#  ─────────────────────────────────────────────────────────────────────────────
#  WHAT IS VALUE AT RISK (VaR)?
#  ─────────────────────────────────────────────────────────────────────────────
#
#  VaR answers ONE question:
#  "What is the maximum loss I will NOT exceed on X% of days?"
#
#  Example: "95% daily VaR = ₹5,000" means:
#  → On 95 out of 100 trading days, my loss will be LESS than ₹5,000
#  → On 5 out of 100 trading days, my loss COULD BE MORE than ₹5,000
#
#  MATHEMATICALLY:
#  VaR at confidence level α (e.g. α = 0.95) is the value V such that:
#    P(Loss > V) = 1 − α = 0.05
#  Or equivalently:
#    P(Loss ≤ V) = α = 0.95
#
#  VaR is the (1-α) percentile of the LOSS distribution.
#  For α = 0.95: VaR = 5th percentile of loss = -5th percentile of P&L
#
#  WHAT IS CONDITIONAL VaR (CVaR)?
#  ─────────────────────────────────
#  CVaR (also called Expected Shortfall or ES) answers:
#  "Given that I AM in the worst 5% of days, what is my EXPECTED loss?"
#
#  CVaR ≥ VaR always (CVaR is always worse than VaR)
#  CVaR is a "coherent risk measure" — it better captures tail risk.
#
#  Example:
#  95% VaR  = ₹5,000   (I will not lose more than ₹5,000 on 95% of days)
#  95% CVaR = ₹8,000   (But on the WORST 5% of days, I expect to lose ₹8,000)
#
#  WHY DO BANKS USE VaR?
#  ─────────────────────
#  Under Basel III regulations (global banking rules including RBI guidelines),
#  banks MUST compute VaR to determine how much capital to set aside.
#  Higher VaR → more capital required → bank is safer but less profitable.
#  This is why risk management teams at Barclays, BNP Paribas, etc. need VaR.
#
#  THREE METHODS FOR VaR (we compare all three):
#  ────────────────────────────────────────────────
#  1. Historical VaR: Use actual past returns. Simple but limited by history.
#  2. Parametric VaR (Normal): Assume Normal distribution. Fast but fat-tail problem.
#  3. Monte Carlo VaR: Simulate many scenarios. Most flexible, most accurate.
#
#  AUTHOR: Kirti Beniwal
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_portfolio_pnl(paths, S0, position_size=1):
    """
    Compute the portfolio Profit & Loss (P&L) from simulated price paths.

    P&L = (Final Price − Initial Price) × number of shares held

    Parameters
    ----------
    paths          : array (n_steps + 1, n_simulations)  Simulated price paths
    S0             : float  Starting price
    position_size  : int    Number of shares held (default = 1)

    Returns
    -------
    pnl : array of shape (n_simulations,)
          P&L for each simulation: positive = profit, negative = loss
    """
    # Terminal prices (final row of the paths matrix)
    S_T = paths[-1, :]

    # P&L = (price change) × number of shares
    pnl = (S_T - S0) * position_size

    return pnl


def compute_daily_returns_pnl(paths, S0, position_value=100000):
    """
    Compute single-day P&L by looking at first-step price changes.

    WHY ONE DAY?
    ─────────────
    Banks typically compute DAILY VaR:
    "How much can I lose in ONE day?"

    We use the first step of each simulated path as the "next day's price",
    then compute P&L = position_value × (next_day_return).

    Parameters
    ----------
    paths          : array   Simulated price paths
    S0             : float   Starting price
    position_value : float   Total value of position in ₹ (e.g. ₹1,00,000)

    Returns
    -------
    daily_pnl : array of shape (n_simulations,)
                Daily P&L for each simulation
    """
    # First-step price (one day forward)
    S_1 = paths[1, :]

    # Daily log return
    daily_log_return = np.log(S_1 / S0)

    # P&L = position_value × return (approximate for small returns)
    daily_pnl = position_value * daily_log_return

    return daily_pnl


def compute_var_cvar(pnl, confidence_level=0.95):
    """
    Compute VaR and CVaR at a given confidence level from P&L distribution.

    HOW TO READ THE RESULT:
    ────────────────────────
    VaR is a LOSS value (positive number represents a loss).
    If VaR = 3500, it means: "95% of the time, I will NOT lose more than ₹3,500"

    CVaR is the average loss in the WORST (1 − confidence_level) of scenarios.
    If CVaR = 5200, it means: "On the 5% worst days, I expect to lose ₹5,200 on average"

    Parameters
    ----------
    pnl              : array   P&L values (negative = loss, positive = profit)
    confidence_level : float   e.g. 0.95 for 95% confidence

    Returns
    -------
    var  : float   Value at Risk (positive number = loss amount)
    cvar : float   Conditional VaR / Expected Shortfall
    """

    # LOSSES = negative of P&L
    # We work with losses so VaR is a positive number (easier to interpret)
    losses = -pnl

    # VaR = percentile of loss distribution
    # For 95% confidence: VaR = 95th percentile of losses
    # i.e. 95% of losses are BELOW this value
    # The percentile value = 100 × (1 - confidence_level)? No:
    # For 95% confidence → VaR = 95th percentile of the LOSS distribution
    var = float(np.percentile(losses, confidence_level * 100))

    # CVaR = average of losses that EXCEED the VaR threshold
    # i.e. average of the worst (1 - confidence_level) scenarios
    tail_losses = losses[losses >= var]

    if len(tail_losses) == 0:
        cvar = var  # Fallback if no tail losses (shouldn't happen with many sims)
    else:
        cvar = float(np.mean(tail_losses))

    return var, cvar


def compute_var_parametric(pnl, confidence_level=0.95):
    """
    Compute VaR assuming the P&L follows a Normal distribution.

    WHY PARAMETRIC?
    ────────────────
    If we assume P&L ~ N(μ, σ²), we can use the Normal distribution formula:
    VaR = μ − z × σ
    where z = Normal quantile for the confidence level (e.g. z = 1.645 for 95%)

    This is fast but UNDERESTIMATES risk because of fat tails.
    (Remember: financial returns have excess kurtosis → Normal underestimates crashes)

    Parameters
    ----------
    pnl              : array   P&L values
    confidence_level : float   e.g. 0.95

    Returns
    -------
    var : float   Parametric VaR
    """

    mean_pnl = np.mean(pnl)
    std_pnl  = np.std(pnl)

    # z-score for the given confidence level
    # norm.ppf(0.05) = -1.645 (the 5th percentile of N(0,1))
    # norm.ppf(0.01) = -2.326 (the 1st percentile of N(0,1))
    z = norm.ppf(1 - confidence_level)

    # VaR = -(mean - z*std) because we flip sign (loss = positive)
    var = -(mean_pnl + z * std_pnl)

    return float(var)


def compute_historical_var(historical_returns, position_value, confidence_level=0.95):
    """
    Compute Historical VaR using actual past returns.

    HOW IT WORKS:
    ──────────────
    1. Take the last N days of actual Nifty 50 returns
    2. Convert returns to P&L: P&L = position_value × return
    3. Find the (1-confidence_level) percentile of the P&L distribution
    4. That is the Historical VaR

    NO ASSUMPTIONS about distribution — uses actual data.
    But limited to scenarios that have actually occurred in history.

    Parameters
    ----------
    historical_returns : array   Actual past log returns
    position_value     : float   Position size in ₹
    confidence_level   : float   e.g. 0.95

    Returns
    -------
    var, cvar : floats
    """

    historical_pnl = position_value * historical_returns
    var, cvar = compute_var_cvar(historical_pnl, confidence_level)

    return var, cvar


def print_var_report(mc_pnl, hist_returns=None, position_value=100000,
                     confidence_levels=[0.90, 0.95, 0.99]):
    """
    Print a comprehensive VaR report comparing multiple methods and confidence levels.

    Parameters
    ----------
    mc_pnl            : array  Monte Carlo daily P&L
    hist_returns      : array  Historical returns (optional, for comparison)
    position_value    : float  Portfolio value in ₹
    confidence_levels : list   Confidence levels to compute (e.g. [0.90, 0.95, 0.99])
    """

    print("\n" + "=" * 65)
    print("  VALUE AT RISK (VaR) REPORT")
    print(f"  Portfolio size: ₹{position_value:,.0f}")
    print("=" * 65)

    print(f"\n  {'Confidence':>12}  {'MC VaR':>12}  {'MC CVaR':>12}  "
          f"{'Parametric':>12}  {'Hist VaR':>10}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}")

    for cl in confidence_levels:
        mc_var,   mc_cvar   = compute_var_cvar(mc_pnl, cl)
        param_var           = compute_var_parametric(mc_pnl, cl)

        hist_var_str = "N/A"
        if hist_returns is not None:
            h_var, _ = compute_historical_var(hist_returns, position_value, cl)
            hist_var_str = f"₹{h_var:>8,.0f}"

        print(f"  {cl*100:.0f}%  {'':<8}  ₹{mc_var:>10,.0f}  ₹{mc_cvar:>10,.0f}  "
              f"₹{param_var:>10,.0f}  {hist_var_str:>10}")

    print(f"\n  READING THE TABLE:")
    mc_var_95, mc_cvar_95 = compute_var_cvar(mc_pnl, 0.95)
    print(f"  95% MC VaR  = ₹{mc_var_95:,.0f}")
    print(f"    → On 95% of days, your loss will be LESS than ₹{mc_var_95:,.0f}")
    print(f"    → On 5% of days (≈ 13 trading days per year), loss could exceed this")
    print(f"  95% MC CVaR = ₹{mc_cvar_95:,.0f}")
    print(f"    → On those worst 5% of days, your EXPECTED loss = ₹{mc_cvar_95:,.0f}")
    print(f"    → CVaR > VaR because it's the average of the WORST outcomes")
    print(f"\n  WHY MC VaR > PARAMETRIC VaR:")
    param_var_95 = compute_var_parametric(mc_pnl, 0.95)
    print(f"  Parametric assumes Normal distribution (thin tails)")
    print(f"  MC simulation captures actual distribution (fat tails)")
    print(f"  Difference: ₹{mc_var_95 - param_var_95:,.0f}")
    print(f"  → Parametric UNDERESTIMATES tail risk by this amount")
    print("=" * 65)
