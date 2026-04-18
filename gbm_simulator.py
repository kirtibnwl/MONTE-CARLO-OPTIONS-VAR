# =============================================================================
#  FILE: gbm_simulator.py
#  PURPOSE: Simulate stock price paths using Geometric Brownian Motion (GBM)
#
#  ─────────────────────────────────────────────────────────────────────────────
#  WHAT IS GEOMETRIC BROWNIAN MOTION? (Plain English)
#  ─────────────────────────────────────────────────────────────────────────────
#
#  Stock prices do NOT move in straight lines. They move randomly.
#  But the randomness has a structure:
#    • A general upward drift (stocks tend to grow over time)
#    • Random fluctuations around that drift
#
#  GBM captures BOTH of these:
#    dS = μ·S·dt  +  σ·S·dW
#     ↑              ↑
#  Drift term     Random term
#  (average       (Wiener process —
#   growth)        pure randomness)
#
#  THE DISCRETE FORMULA (what we actually code):
#  ──────────────────────────────────────────────
#  To simulate one day forward from price S:
#
#    S(t + Δt) = S(t) × exp( (μ − σ²/2) × Δt  +  σ × √Δt × Z )
#
#  WHERE:
#    S(t)      = stock price today
#    μ (mu)    = annual drift / expected return (e.g. 0.12 = 12% per year)
#    σ (sigma) = annual volatility (e.g. 0.20 = 20% per year)
#    Δt        = time step (for daily: Δt = 1/252 because 252 trading days/year)
#    Z         = random number drawn from N(0,1) standard Normal distribution
#    exp(·)    = the exponential function (same as e^x in mathematics)
#
#  WHY (μ − σ²/2) and not just μ?
#  ─────────────────────────────────
#  This is the Itô correction term.
#  When we work with log prices (which is what exp does), Jensen's inequality
#  means the average of log(X) ≠ log(average of X).
#  The (−σ²/2) term corrects for this bias.
#  It comes directly from Itô's Lemma — which you likely know from your
#  stochastic differential equations background.
#
#  WHY MONTE CARLO?
#  ─────────────────
#  Instead of solving the PDE analytically (Black-Scholes formula),
#  we simulate THOUSANDS of possible futures.
#  For each simulated future, we ask: "What is the stock price at expiry?"
#  Then we compute the option payoff for each future.
#  The option price = average payoff × discount factor.
#
#  AUTHOR: Kirti Beniwal
# =============================================================================

import numpy as np
import pandas as pd


def simulate_gbm_paths(
        S0,           # Current stock price (e.g. 19,000 for Nifty 50)
        mu,           # Annual drift/return (e.g. 0.12 = 12% per year)
        sigma,        # Annual volatility (e.g. 0.18 = 18% per year)
        T,            # Time to expiry in YEARS (e.g. 0.5 = 6 months)
        n_steps,      # Number of time steps (e.g. 126 for 6 months of trading days)
        n_simulations # Number of price paths to simulate (e.g. 10,000)
):
    """
    Simulate multiple stock price paths using Geometric Brownian Motion.

    WHAT THIS FUNCTION RETURNS:
    ────────────────────────────
    A 2D array of shape (n_steps + 1, n_simulations).
    • Each COLUMN is one complete simulated price path (one possible future)
    • Each ROW is one time step (e.g. one trading day)
    • Row 0 = today (all values = S0)
    • Last row = simulated price at expiry

    EXAMPLE:
    If n_steps = 126 and n_simulations = 10,000:
    → We simulate 10,000 possible 6-month futures for the stock
    → Each future has 126 daily price points

    Parameters
    ----------
    S0           : float  Current stock price
    mu           : float  Annual expected return (decimal, e.g. 0.12 for 12%)
    sigma        : float  Annual volatility (decimal, e.g. 0.20 for 20%)
    T            : float  Time to expiry in years
    n_steps      : int    Number of time steps
    n_simulations: int    Number of simulation paths

    Returns
    -------
    paths : numpy array of shape (n_steps + 1, n_simulations)
            Simulated price paths. Column i = one possible future.
    dt    : float   Size of each time step in years
    """

    # dt = size of each time step in years
    # For daily steps: dt = T / n_steps = 0.5 / 126 ≈ 0.00397 years = 1 trading day
    dt = T / n_steps

    # ── Generate Random Numbers ───────────────────────────────────────────────
    # We need one random number per (time step, simulation path)
    # Z ~ N(0,1): standard Normal distribution
    # np.random.standard_normal((rows, cols)) → matrix of N(0,1) numbers
    Z = np.random.standard_normal((n_steps, n_simulations))

    # ── Compute Log Returns for Each Step ─────────────────────────────────────
    # The log return at each step follows this formula (from Itô's Lemma):
    #   log(S(t+dt) / S(t)) = (μ − σ²/2) × dt  +  σ × √dt × Z
    #
    # Breaking it down:
    #   (mu - 0.5 * sigma**2) * dt  = the DETERMINISTIC part (drift correction)
    #   sigma * np.sqrt(dt) * Z     = the RANDOM part (volatility × random noise)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    # Shape: (n_steps, n_simulations)

    # ── Build Price Paths ─────────────────────────────────────────────────────
    # Start with all paths at S0 (today's price)
    # Then apply cumulative log returns step by step
    #
    # np.vstack stacks row on top — we add S0 as the first row
    # np.cumsum computes running sum of log returns
    # np.exp converts log returns back to price levels
    #
    # If log_returns = [r1, r2, r3, ...], then:
    # cumsum = [r1, r1+r2, r1+r2+r3, ...]
    # exp(cumsum) = [S1/S0, S2/S0, S3/S0, ...]
    # S0 × exp(cumsum) = [S1, S2, S3, ...]
    cumulative_log_returns = np.cumsum(log_returns, axis=0)

    # First row: all paths start at S0
    # Remaining rows: S0 × exp(cumulative log returns)
    paths = np.vstack([
        np.full(n_simulations, S0),            # Row 0: all paths = S0 (today)
        S0 * np.exp(cumulative_log_returns)    # Rows 1 to n_steps: evolved prices
    ])
    # Final shape: (n_steps + 1, n_simulations)

    return paths, dt


def get_terminal_prices(paths):
    """
    Extract just the final price (at expiry) from all simulated paths.

    WHY WE NEED THIS:
    ──────────────────
    For European option pricing, we only care about the FINAL price,
    not the path taken to get there.
    (European options can only be exercised at expiry, not before)

    Parameters
    ----------
    paths : array of shape (n_steps + 1, n_simulations)

    Returns
    -------
    S_T : array of shape (n_simulations,)
          Final stock price for each simulation
    """
    return paths[-1, :]   # Last row = final prices at expiry


def compute_path_statistics(paths, S0):
    """
    Compute summary statistics about the simulated price paths.

    This helps us understand the distribution of simulated futures.

    Parameters
    ----------
    paths : array of shape (n_steps + 1, n_simulations)
    S0    : float  Starting price

    Returns
    -------
    stats : dict with summary statistics
    """
    S_T = get_terminal_prices(paths)

    stats = {
        "Starting price (S0)"         : round(float(S0), 2),
        "Simulated paths"             : paths.shape[1],
        "Time steps per path"         : paths.shape[0] - 1,
        "─" * 25                      : "─" * 12,
        "Mean final price"            : round(float(S_T.mean()), 2),
        "Median final price"          : round(float(np.median(S_T)), 2),
        "Std dev of final price"      : round(float(S_T.std()), 2),
        "─ " * 12                     : "─" * 12,
        "5th percentile final price"  : round(float(np.percentile(S_T, 5)),  2),
        "25th percentile final price" : round(float(np.percentile(S_T, 25)), 2),
        "75th percentile final price" : round(float(np.percentile(S_T, 75)), 2),
        "95th percentile final price" : round(float(np.percentile(S_T, 95)), 2),
        "─  " * 8                     : "─" * 12,
        "Min final price"             : round(float(S_T.min()), 2),
        "Max final price"             : round(float(S_T.max()), 2),
        "% paths above S0"           : f"{(S_T > S0).mean() * 100:.1f}%",
    }

    return stats
