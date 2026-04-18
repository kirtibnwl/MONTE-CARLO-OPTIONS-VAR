# =============================================================================
#  FILE: option_pricing.py
#  PURPOSE: Price European call and put options using Monte Carlo simulation
#           and the Black-Scholes analytical formula. Compare both.
#
#  ─────────────────────────────────────────────────────────────────────────────
#  OPTION TYPES — EXPLAINED FROM SCRATCH
#  ─────────────────────────────────────────────────────────────────────────────
#
#  CALL OPTION:
#  ─────────────
#  Gives you the RIGHT (but not obligation) to BUY a stock at price K
#  at the expiry date.
#
#  Why would you buy a CALL?
#  You expect the stock to RISE above K.
#  If stock rises to S_T > K:
#      Payoff = S_T − K   (you buy cheap at K, it's worth S_T)
#  If stock falls below K:
#      Payoff = 0         (you walk away — no obligation)
#
#  CALL PAYOFF FORMULA: max(S_T − K, 0)
#  This is also written as (S_T − K)⁺ in mathematics.
#
#  PUT OPTION:
#  ────────────
#  Gives you the RIGHT (but not obligation) to SELL a stock at price K
#  at the expiry date.
#
#  Why would you buy a PUT?
#  You expect the stock to FALL below K.
#  If stock falls to S_T < K:
#      Payoff = K − S_T   (you sell at high price K when market gives only S_T)
#  If stock rises above K:
#      Payoff = 0         (you walk away — no obligation)
#
#  PUT PAYOFF FORMULA: max(K − S_T, 0)
#  This is also written as (K − S_T)⁺ in mathematics.
#
#  KEY TERMS:
#  ───────────
#  K (Strike price)  = Agreed price at which you can buy/sell
#  S_T (Spot price)  = Actual stock price at expiry date
#  T (Maturity)      = Time to expiry in years (e.g. 0.5 = 6 months)
#  r (Risk-free rate)= Return on a risk-free investment (e.g. Indian T-bill rate)
#  σ (Sigma)         = Volatility of the stock (annualised standard deviation)
#
#  IN-THE-MONEY, AT-THE-MONEY, OUT-OF-THE-MONEY:
#  ───────────────────────────────────────────────
#  Call option:
#    In-the-money    (ITM): S0 > K  → currently profitable to exercise
#    At-the-money    (ATM): S0 = K  → break-even point
#    Out-of-the-money(OTM): S0 < K  → not currently profitable
#
#  MONTE CARLO PRICING:
#  ─────────────────────
#  1. Simulate n_simulations paths of the stock price up to time T
#  2. For each path, compute the payoff at expiry
#     Call: payoff_i = max(S_T_i − K, 0)
#     Put:  payoff_i = max(K − S_T_i, 0)
#  3. Average all payoffs
#  4. Discount back to today: price = mean(payoffs) × exp(−r × T)
#     WHY DISCOUNT? Because ₹100 today is worth more than ₹100 in 6 months.
#     The discount factor exp(−rT) converts future money to today's value.
#
#  BLACK-SCHOLES ANALYTICAL FORMULA:
#  ───────────────────────────────────
#  Black and Scholes (1973, Nobel Prize) derived an exact formula.
#  We use it to VALIDATE our Monte Carlo result.
#  If Monte Carlo ≈ Black-Scholes → our simulation is working correctly.
#
#  Call = S0 × N(d1) − K × exp(−rT) × N(d2)
#  Put  = K × exp(−rT) × N(−d2) − S0 × N(−d1)
#
#  where:
#    d1 = [log(S0/K) + (r + σ²/2) × T] / (σ × √T)
#    d2 = d1 − σ × √T
#    N(·) = cumulative Normal distribution function (Φ in your notation)
#
#  AUTHOR: Kirti Beniwal
# =============================================================================

import numpy as np
from scipy.stats import norm   # For Normal distribution N(d1), N(d2)


# ── Monte Carlo Option Pricing ─────────────────────────────────────────────────

def price_european_call_mc(S_T, K, r, T):
    """
    Price a European CALL option using Monte Carlo simulation.

    WHAT THIS DOES:
    ────────────────
    Step 1: Compute payoff for each simulated terminal price S_T
            payoff = max(S_T − K, 0) for each simulation
            If S_T > K → you profit (buy at K, worth S_T)
            If S_T ≤ K → payoff = 0 (you walk away)

    Step 2: Average all payoffs
            average_payoff = mean of all max(S_T_i − K, 0)

    Step 3: Discount to today
            price = average_payoff × exp(−r × T)
            WHY: The payoff is received at time T in the future.
                 We need its value TODAY, so we discount it.

    Parameters
    ----------
    S_T : numpy array   Terminal stock prices (final column of GBM paths)
    K   : float         Strike price (agreed buy price)
    r   : float         Annual risk-free interest rate (e.g. 0.065 = 6.5%)
    T   : float         Time to expiry in years

    Returns
    -------
    price    : float    Monte Carlo call option price
    std_error: float    Standard error of the estimate
                        (tells us how precise our Monte Carlo estimate is)
    payoffs  : array    Individual payoffs (useful for VaR analysis)
    """

    # Step 1: Compute call payoff for each simulated path
    # np.maximum(a, 0) returns a if a > 0, else 0 — exactly max(S_T - K, 0)
    payoffs = np.maximum(S_T - K, 0)
    # Example: if S_T = [21000, 18000, 22500, 17000] and K = 19000:
    # payoffs = [2000, 0, 3500, 0]

    # Step 2: Discount factor = exp(−r × T)
    # This converts a future value to present value
    # e.g. r=0.065, T=0.5: discount = exp(-0.065 × 0.5) ≈ 0.968
    discount_factor = np.exp(-r * T)

    # Step 3: Option price = average discounted payoff
    price = discount_factor * np.mean(payoffs)

    # Standard error = tells us the precision of our Monte Carlo estimate
    # More simulations → smaller standard error → more precise estimate
    # Formula: σ_payoff / √n   (like the CLT confidence interval)
    std_error = discount_factor * np.std(payoffs) / np.sqrt(len(payoffs))

    return price, std_error, payoffs


def price_european_put_mc(S_T, K, r, T):
    """
    Price a European PUT option using Monte Carlo simulation.

    PUT PAYOFF = max(K − S_T, 0)
    If stock falls below K → you profit (sell at high price K)
    If stock rises above K → payoff = 0 (no need to sell at low price)

    Parameters
    ----------
    S_T : numpy array   Terminal stock prices
    K   : float         Strike price
    r   : float         Annual risk-free rate
    T   : float         Time to expiry in years

    Returns
    -------
    price    : float    Monte Carlo put option price
    std_error: float    Standard error
    payoffs  : array    Individual payoffs
    """

    # Put payoff = max(K - S_T, 0)
    payoffs = np.maximum(K - S_T, 0)

    discount_factor = np.exp(-r * T)
    price     = discount_factor * np.mean(payoffs)
    std_error = discount_factor * np.std(payoffs) / np.sqrt(len(payoffs))

    return price, std_error, payoffs


# ── Black-Scholes Analytical Formula ──────────────────────────────────────────

def black_scholes_call(S0, K, r, sigma, T):
    """
    Compute European CALL price using the Black-Scholes formula (1973).

    This gives the EXACT theoretical price (under GBM assumptions).
    We use this to VALIDATE our Monte Carlo result.

    THE FORMULA:
    ─────────────
    C = S0 × N(d1) − K × exp(−rT) × N(d2)

    WHERE:
    d1 = [log(S0/K) + (r + σ²/2) × T] / (σ × √T)
    d2 = d1 − σ × √T
    N(x) = Φ(x) = cumulative standard Normal CDF

    INTUITION:
    S0 × N(d1)         → expected stock value if option is exercised
    K × exp(−rT) × N(d2) → present value of strike price × probability of exercise

    Parameters
    ----------
    S0    : float  Current stock price
    K     : float  Strike price
    r     : float  Annual risk-free rate
    sigma : float  Annual volatility
    T     : float  Time to expiry in years

    Returns
    -------
    call_price : float   Black-Scholes call price
    d1, d2     : floats  Intermediate values (used for Greeks)
    """

    # d1 formula
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    # d2 formula
    d2 = d1 - sigma * np.sqrt(T)

    # N(d1) and N(d2): cumulative Normal probabilities
    # norm.cdf(x) = P(Z ≤ x) where Z ~ N(0,1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    # Black-Scholes call formula
    call_price = S0 * N_d1 - K * np.exp(-r * T) * N_d2

    return call_price, d1, d2


def black_scholes_put(S0, K, r, sigma, T):
    """
    Compute European PUT price using Black-Scholes formula.

    FORMULA:
    P = K × exp(−rT) × N(−d2) − S0 × N(−d1)

    Note: N(−x) = 1 − N(x)

    PUT-CALL PARITY (important relationship):
    C − P = S0 − K × exp(−rT)
    This always holds for European options — we use it to verify our results.

    Parameters
    ----------
    S0    : float  Current stock price
    K     : float  Strike price
    r     : float  Annual risk-free rate
    sigma : float  Annual volatility
    T     : float  Time to expiry in years

    Returns
    -------
    put_price : float
    d1, d2    : floats
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # N(-d1) and N(-d2) for put formula
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)

    put_price = K * np.exp(-r * T) * N_neg_d2 - S0 * N_neg_d1

    return put_price, d1, d2


# ── Option Greeks ──────────────────────────────────────────────────────────────

def compute_greeks(S0, K, r, sigma, T):
    """
    Compute the Option Greeks — sensitivities of option price to parameters.

    WHAT ARE GREEKS?
    ─────────────────
    Greeks measure HOW MUCH the option price changes when each input changes.
    They are essential for risk management at banks and trading desks.

    DELTA (Δ):
    ─────────────
    Delta = ∂(Option Price) / ∂(Stock Price)
    = how much option price changes for a ₹1 change in stock price

    For a CALL:  Delta = N(d1)     → always between 0 and 1
    For a PUT:   Delta = N(d1) − 1 → always between −1 and 0

    Example: Call with Delta = 0.6
    → If Nifty rises by ₹100, call price rises by ₹60
    → Delta = 0.5 means option is "at the money" (50-50 chance of expiring in-profit)

    GAMMA (Γ):
    ─────────────
    Gamma = ∂²(Option Price) / ∂(Stock Price)²  = ∂(Delta) / ∂(Stock Price)
    = how much Delta itself changes for a ₹1 change in stock price
    = the "acceleration" of the option's value

    For both calls and puts: Gamma = N'(d1) / (S0 × σ × √T)
    where N'(x) = standard Normal PDF = (1/√2π) × exp(−x²/2)

    VEGA (ν):
    ──────────
    Vega = ∂(Option Price) / ∂(Volatility)
    = how much option price changes for a 1% change in volatility

    For both calls and puts: Vega = S0 × N'(d1) × √T

    Positive Vega for both calls and puts:
    Higher volatility → higher option price (more chance of large moves)

    THETA (Θ):
    ──────────
    Theta = ∂(Option Price) / ∂(Time)
    = how much option price changes as one day passes
    Usually negative: options lose value as they approach expiry ("time decay")

    RHO (ρ):
    ─────────
    Rho = ∂(Option Price) / ∂(Risk-free rate)
    = how much option price changes for a 1% change in interest rate

    Parameters
    ----------
    S0    : float  Current stock price
    K     : float  Strike price
    r     : float  Annual risk-free rate
    sigma : float  Annual volatility
    T     : float  Time to expiry

    Returns
    -------
    greeks : dict with all Greek values
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Normal PDF at d1: N'(d1) = φ(d1)
    N_prime_d1 = norm.pdf(d1)

    # ── Delta ────────────────────────────────────────────────────────────────
    call_delta = norm.cdf(d1)          # N(d1): between 0 and 1
    put_delta  = norm.cdf(d1) - 1     # N(d1) - 1: between -1 and 0

    # ── Gamma (same for call and put) ─────────────────────────────────────────
    gamma = N_prime_d1 / (S0 * sigma * np.sqrt(T))

    # ── Vega (same for call and put, reported per 1% change in vol) ───────────
    vega = S0 * N_prime_d1 * np.sqrt(T) / 100   # divide by 100 for 1% vol change

    # ── Theta (per calendar day) ──────────────────────────────────────────────
    # Full formula for call theta
    call_theta = (
        - (S0 * N_prime_d1 * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    ) / 365   # convert from annual to daily

    put_theta = (
        - (S0 * N_prime_d1 * sigma) / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    ) / 365

    # ── Rho (per 1% change in interest rate) ──────────────────────────────────
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2)  / 100
    put_rho  = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    greeks = {
        "call_delta" : round(call_delta, 4),
        "put_delta"  : round(put_delta,  4),
        "gamma"      : round(gamma,      6),
        "vega"       : round(vega,       4),
        "call_theta" : round(call_theta, 4),
        "put_theta"  : round(put_theta,  4),
        "call_rho"   : round(call_rho,   4),
        "put_rho"    : round(put_rho,    4),
        "d1"         : round(d1,         4),
        "d2"         : round(d2,         4),
    }

    return greeks


def print_pricing_comparison(S0, K, r, sigma, T,
                              mc_call, mc_call_err,
                              mc_put,  mc_put_err):
    """
    Print a formatted comparison of Monte Carlo vs Black-Scholes prices.

    Parameters
    ----------
    S0, K, r, sigma, T : floats  Option parameters
    mc_call, mc_call_err: floats MC call price and standard error
    mc_put, mc_put_err  : floats MC put price and standard error
    """

    bs_call, d1, d2 = black_scholes_call(S0, K, r, sigma, T)
    bs_put, _, _    = black_scholes_put(S0, K, r, sigma, T)
    greeks = compute_greeks(S0, K, r, sigma, T)

    # Determine moneyness
    if S0 > K * 1.02:
        moneyness = "In-the-money (ITM)"
    elif S0 < K * 0.98:
        moneyness = "Out-of-the-money (OTM)"
    else:
        moneyness = "At-the-money (ATM)"

    print("\n" + "=" * 65)
    print("  OPTION PRICING RESULTS")
    print("=" * 65)

    print(f"\n  INPUT PARAMETERS:")
    print(f"  Current stock price (S0)  = {S0:,.2f}")
    print(f"  Strike price (K)          = {K:,.2f}")
    print(f"  Time to expiry (T)        = {T:.2f} years ({T*252:.0f} trading days)")
    print(f"  Volatility (σ)            = {sigma*100:.1f}% per year")
    print(f"  Risk-free rate (r)        = {r*100:.1f}% per year")
    print(f"  Moneyness                 = {moneyness}")

    print(f"\n  PRICING COMPARISON: Monte Carlo vs Black-Scholes")
    print(f"  {'─'*60}")
    print(f"  {'':35} {'Monte Carlo':>12}  {'Black-Scholes':>13}")
    print(f"  {'─'*60}")
    print(f"  {'CALL option price':35} {mc_call:>12.4f}  {bs_call:>13.4f}")
    print(f"  {'  95% CI: ±':35} {1.96*mc_call_err:>12.4f}  {'(exact)':>13}")
    error_call = abs(mc_call - bs_call) / bs_call * 100
    print(f"  {'  Error vs Black-Scholes':35} {error_call:>11.3f}%")
    print(f"  {'─'*60}")
    print(f"  {'PUT option price':35} {mc_put:>12.4f}  {bs_put:>13.4f}")
    print(f"  {'  95% CI: ±':35} {1.96*mc_put_err:>12.4f}  {'(exact)':>13}")
    error_put = abs(mc_put - bs_put) / bs_put * 100
    print(f"  {'  Error vs Black-Scholes':35} {error_put:>11.3f}%")

    print(f"\n  VALIDATION — Put-Call Parity Check:")
    print(f"  C − P should equal S0 − K × exp(−rT)")
    parity_lhs = mc_call - mc_put
    parity_rhs = S0 - K * np.exp(-r * T)
    print(f"  MC:  C − P = {parity_lhs:.4f}")
    print(f"  Theory:     {parity_rhs:.4f}")
    print(f"  Difference: {abs(parity_lhs - parity_rhs):.4f}  "
          f"{'✓ (< 1% error)' if abs(parity_lhs-parity_rhs)/abs(parity_rhs) < 0.01 else '✗'}")

    print(f"\n  OPTION GREEKS (Black-Scholes):")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Delta (call) = {greeks['call_delta']:.4f}")
    print(f"    → Call price rises by {greeks['call_delta']:.2f} for every ₹1 rise in stock")
    print(f"  Delta (put)  = {greeks['put_delta']:.4f}")
    print(f"    → Put price falls by {abs(greeks['put_delta']):.2f} for every ₹1 rise in stock")
    print(f"  Gamma        = {greeks['gamma']:.6f}")
    print(f"    → Delta changes by {greeks['gamma']:.4f} for every ₹1 change in stock")
    print(f"  Vega         = {greeks['vega']:.4f}")
    print(f"    → Option price changes by {greeks['vega']:.2f} for every 1% change in vol")
    print(f"  Theta (call) = {greeks['call_theta']:.4f}")
    print(f"    → Call loses {abs(greeks['call_theta']):.2f} value per day (time decay)")
    print(f"  Theta (put)  = {greeks['put_theta']:.4f}")
    print(f"    → Put loses {abs(greeks['put_theta']):.2f} value per day (time decay)")
    print("=" * 65)

    return greeks
