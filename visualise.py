# =============================================================================
#  FILE: visualise.py
#  PURPOSE: Create all charts for the Monte Carlo Options & VaR project
#
#  PLOTS PRODUCED:
#  ─────────────────
#  1. gbm_paths.png          — Simulated stock price paths (fan of futures)
#  2. option_payoffs.png     — Call and put payoff distributions
#  3. var_dashboard.png      — VaR visualisation (main risk chart)
#  4. greeks_sensitivity.png — How option price changes with stock price
#  5. convergence.png        — Monte Carlo accuracy vs number of simulations
#
#  AUTHOR: Kirti Beniwal
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "figure.dpi"      : 140,
})


# ── Plot 1: Simulated GBM Price Paths ─────────────────────────────────────────

def plot_gbm_paths(paths, S0, K, T, n_display=200,
                   save_path="outputs/gbm_paths.png"):
    """
    Visualise simulated stock price paths.

    WHAT THIS SHOWS:
    ─────────────────
    • Each thin line = one possible future of the stock price
    • The fan shape shows the RANGE of possible outcomes
    • Wider fan = more uncertainty (higher volatility)
    • The strike price K is shown as a horizontal dashed line
    • Paths above K at expiry → call option is profitable (in the money)
    • Paths below K at expiry → call option expires worthless (out of money)
    """

    S_T       = paths[-1, :]
    n_steps   = paths.shape[0] - 1
    time_axis = np.linspace(0, T, n_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Price paths ────────────────────────────────────────────────────
    ax = axes[0]
    # Plot a random subset of paths (plotting all 10,000 would be slow)
    idx = np.random.choice(paths.shape[1], size=min(n_display, paths.shape[1]),
                           replace=False)

    # Colour paths by whether they end above or below the strike
    for i in idx:
        color  = "#1565C0" if paths[-1, i] > K else "#C62828"
        alpha  = 0.12
        ax.plot(time_axis, paths[:, i], color=color, lw=0.5, alpha=alpha)

    # Highlight median path
    median_path = np.median(paths, axis=1)
    ax.plot(time_axis, median_path, color="black", lw=2,
            label="Median path", zorder=5)

    # 5th and 95th percentile bands
    p5  = np.percentile(paths, 5,  axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    ax.fill_between(time_axis, p5, p95, alpha=0.15, color="gray",
                    label="5th–95th percentile")

    # Strike price line
    ax.axhline(K, color="orange", lw=2, linestyle="--",
               label=f"Strike price K = {K:,.0f}", zorder=4)
    ax.axhline(S0, color="green", lw=1.5, linestyle=":",
               label=f"Start price S0 = {S0:,.0f}", zorder=4)

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Simulated GBM Price Paths\n"
                 f"(showing {n_display} of {paths.shape[1]:,} total paths)")
    # Colour legend
    # BUG FIX: fill_between creates a PolyCollection (not a Line2D), so it does
    # NOT appear in ax.get_lines(). Using ax.get_lines()[n_display+1] was
    # incorrectly picking up the orange K-axhline as the "5th–95th band" handle.
    # Fix: use a grey Patch to represent the band instead.
    blue_patch = mpatches.Patch(color="#1565C0", alpha=0.5, label="Ends above K (call profitable)")
    red_patch  = mpatches.Patch(color="#C62828", alpha=0.5, label="Ends below K (call worthless)")
    band_patch = mpatches.Patch(color="gray",    alpha=0.3, label="5th–95th percentile band")
    ax.legend(handles=[blue_patch, red_patch,
                       ax.get_lines()[n_display],
                       band_patch,
                       plt.Line2D([0],[0], color="orange", lw=2, linestyle="--"),
                       plt.Line2D([0],[0], color="green",  lw=1.5, linestyle=":")],
              labels=["Ends above K (call profitable)", "Ends below K (call worthless)",
                      "Median path", "5th–95th band",
                      f"Strike K = {K:,.0f}", f"Start S0 = {S0:,.0f}"],
              fontsize=8, loc="upper left")

    # ── Right: Distribution of terminal prices ────────────────────────────────
    ax2 = axes[1]
    ax2.hist(S_T, bins=80, color="#1565C0", alpha=0.7, edgecolor="white",
             linewidth=0.3, density=True, label="Simulated final prices")

    # Overlay Normal distribution for comparison
    mu_approx   = np.mean(S_T)
    std_approx  = np.std(S_T)
    x_range     = np.linspace(S_T.min(), S_T.max(), 300)
    normal_pdf  = norm.pdf(x_range, mu_approx, std_approx)
    ax2.plot(x_range, normal_pdf, color="black", lw=1.5,
             linestyle="--", label="Normal distribution (for comparison)")

    ax2.axvline(K,  color="orange", lw=2, linestyle="--",
                label=f"Strike K = {K:,.0f}")
    ax2.axvline(S0, color="green",  lw=1.5, linestyle=":",
                label=f"Start S0 = {S0:,.0f}")
    ax2.axvline(np.mean(S_T), color="purple", lw=1.5,
                label=f"Mean S_T = {np.mean(S_T):,.0f}")

    # Shade in-the-money region
    itm = x_range[x_range > K]
    if len(itm) > 0:
        ax2.fill_between(itm, norm.pdf(itm, mu_approx, std_approx),
                         alpha=0.2, color="green", label="In-the-money region")

    ax2.set_xlabel("Terminal Stock Price S_T")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Distribution of Terminal Stock Prices\n"
                  "Call payoff = shaded green region (S_T > K)")
    ax2.legend(fontsize=8)

    fig.suptitle("Geometric Brownian Motion — Simulated Price Paths",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"  Saved: {save_path}")


# ── Plot 2: Option Payoff Distributions ───────────────────────────────────────

def plot_option_payoffs(call_payoffs, put_payoffs, K, mc_call, mc_put,
                        bs_call, bs_put, save_path="outputs/option_payoffs.png"):
    """
    Visualise the distribution of call and put payoffs.

    WHAT THIS SHOWS:
    ─────────────────
    • MOST options expire worthless (payoff = 0) → large bar at zero
    • When profitable, the payoff can be large (long right tail for calls)
    • Monte Carlo price = discounted average of these payoffs
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Call payoffs ──────────────────────────────────────────────────────────
    ax = axes[0]
    # Separate zero payoffs from positive payoffs
    zero_call     = (call_payoffs == 0).sum()
    positive_call = call_payoffs[call_payoffs > 0]
    pct_zero      = zero_call / len(call_payoffs) * 100

    ax.hist(positive_call, bins=60, color="#1565C0", alpha=0.75,
            edgecolor="white", linewidth=0.3,
            label=f"Positive payoffs: {100-pct_zero:.1f}% of simulations")
    ax.axvline(0, color="gray", lw=0.5, linestyle="--")

    # Add text annotation for zero-payoff percentage
    ax.text(0.02, 0.90,
            f"{pct_zero:.1f}% of options\nexpire WORTHLESS\n(payoff = 0)",
            transform=ax.transAxes, fontsize=10, color="#C62828",
            va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.axvline(np.mean(call_payoffs), color="orange", lw=2,
               label=f"Mean payoff = {np.mean(call_payoffs):.2f}")
    ax.set_xlabel("Call Payoff = max(S_T − K, 0)")
    ax.set_ylabel("Number of simulations")
    ax.set_title(f"CALL Option Payoff Distribution\n"
                 f"MC Price = {mc_call:.4f}  |  BS Price = {bs_call:.4f}")
    ax.legend(fontsize=9)

    # ── Put payoffs ───────────────────────────────────────────────────────────
    ax2 = axes[1]
    zero_put     = (put_payoffs == 0).sum()
    positive_put = put_payoffs[put_payoffs > 0]
    pct_zero_put = zero_put / len(put_payoffs) * 100

    ax2.hist(positive_put, bins=60, color="#C62828", alpha=0.75,
             edgecolor="white", linewidth=0.3,
             label=f"Positive payoffs: {100-pct_zero_put:.1f}% of simulations")
    ax2.text(0.02, 0.90,
             f"{pct_zero_put:.1f}% of options\nexpire WORTHLESS\n(payoff = 0)",
             transform=ax2.transAxes, fontsize=10, color="#1565C0",
             va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax2.axvline(np.mean(put_payoffs), color="orange", lw=2,
                label=f"Mean payoff = {np.mean(put_payoffs):.2f}")
    ax2.set_xlabel("Put Payoff = max(K − S_T, 0)")
    ax2.set_ylabel("Number of simulations")
    ax2.set_title(f"PUT Option Payoff Distribution\n"
                  f"MC Price = {mc_put:.4f}  |  BS Price = {bs_put:.4f}")
    ax2.legend(fontsize=9)

    fig.suptitle("Option Payoff Distributions — Monte Carlo vs Black-Scholes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"  Saved: {save_path}")


# ── Plot 3: VaR Dashboard ─────────────────────────────────────────────────────

def plot_var_dashboard(daily_pnl, position_value,
                       confidence_levels=[0.90, 0.95, 0.99],
                       save_path="outputs/var_dashboard.png"):
    """
    Main VaR visualisation — the most important risk management chart.

    WHAT THIS SHOWS:
    ─────────────────
    Left panel: Full P&L distribution with VaR cutoff lines
    Right panel: Loss tail (zoomed in on the worst scenarios)

    Reading the chart:
    • Each bar = how many simulations had that P&L outcome
    • Red dotted lines = VaR cutoffs at different confidence levels
    • Everything to the LEFT of a red line = losses exceeding that VaR
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    losses = -daily_pnl   # Flip sign: losses = positive number

    # ── Left: Full P&L Distribution ──────────────────────────────────────────
    ax = axes[0]
    ax.hist(daily_pnl, bins=100, color="#1565C0", alpha=0.7,
            edgecolor="white", linewidth=0.2, density=True,
            label="Simulated daily P&L")

    colors_cl = ["#FF8F00", "#E65100", "#B71C1C"]
    var_values = []

    for cl, color in zip(confidence_levels, colors_cl):
        var = float(np.percentile(losses, cl * 100))
        var_values.append(var)
        ax.axvline(-var, color=color, lw=2.5, linestyle="--",
                   label=f"{cl*100:.0f}% VaR = ₹{var:,.0f}")

    ax.axvline(0, color="black", lw=0.8, linestyle="-", alpha=0.5)
    # BUG FIX: ax.get_xlim()[0] may include matplotlib padding and the != 0
    # guard is logically inverted (P&L min is always negative, never 0).
    # Use daily_pnl.min() directly for the left boundary — always correct.
    ax.fill_betweenx([0, ax.get_ylim()[1]],
                     daily_pnl.min(),
                     -var_values[-1],
                     alpha=0.1, color="#B71C1C", label="Extreme loss region (1%)")

    ax.set_xlabel("Daily P&L (₹)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"P&L Distribution  |  Position: ₹{position_value:,.0f}\n"
                 f"Dotted lines = VaR at different confidence levels")
    ax.legend(fontsize=9)

    # ── Right: Loss Tail (Zoomed) ─────────────────────────────────────────────
    ax2 = axes[1]
    tail_cutoff = np.percentile(losses, 85)   # Show worst 15%
    tail_losses = losses[losses >= tail_cutoff]

    ax2.hist(tail_losses, bins=50, color="#C62828", alpha=0.75,
             edgecolor="white", linewidth=0.3, density=True,
             label="Loss tail (worst 15% of scenarios)")

    for cl, color, var in zip(confidence_levels, colors_cl, var_values):
        cvar = float(np.mean(losses[losses >= var]))
        ax2.axvline(var, color=color, lw=2.5, linestyle="--",
                    label=f"{cl*100:.0f}% VaR = ₹{var:,.0f}")
        ax2.axvline(cvar, color=color, lw=1.5, linestyle=":",
                    label=f"{cl*100:.0f}% CVaR = ₹{cvar:,.0f}")

    ax2.set_xlabel("Loss (₹)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Loss Tail Distribution (Zoomed)\n"
                  "Solid = VaR cutoff  |  Dotted = CVaR (average tail loss)")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Value at Risk (VaR) Dashboard — Monte Carlo Simulation\n"
                 f"Portfolio: ₹{position_value:,.0f}  |  {len(daily_pnl):,} simulations",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"  Saved: {save_path}")


# ── Plot 4: Convergence of Monte Carlo ────────────────────────────────────────

def plot_mc_convergence(paths, K, r, T, bs_call_price,
                        save_path="outputs/convergence.png"):
    """
    Show how Monte Carlo option price converges to Black-Scholes as n increases.

    WHY THIS MATTERS:
    ──────────────────
    Monte Carlo is a statistical method — more simulations = more accurate.
    This plot shows the "Law of Large Numbers" in action:
    As n → ∞, the Monte Carlo estimate → true Black-Scholes price.

    For a portfolio manager: this tells you HOW MANY simulations are enough.
    Typically 10,000 gives < 1% error. 100,000 gives < 0.1% error.
    """

    print(f"  Computing convergence analysis...")

    S_T = paths[-1, :]
    discount_factor = np.exp(-r * T)
    call_payoffs    = np.maximum(S_T - K, 0)

    # Compute running average of MC price as we add more simulations
    n_points     = 100
    sim_counts   = np.logspace(1, np.log10(len(S_T)), n_points).astype(int)
    sim_counts   = np.unique(sim_counts)
    mc_estimates = []

    for n in sim_counts:
        estimate = discount_factor * np.mean(call_payoffs[:n])
        mc_estimates.append(estimate)

    mc_estimates = np.array(mc_estimates)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.semilogx(sim_counts, mc_estimates, color="#1565C0", lw=1.5,
                label="Monte Carlo estimate")
    ax.axhline(bs_call_price, color="#C62828", lw=2, linestyle="--",
               label=f"Black-Scholes exact price = {bs_call_price:.4f}")

    # Error bands: ±1% and ±0.1%
    ax.axhspan(bs_call_price * 0.99, bs_call_price * 1.01,
               alpha=0.1, color="green", label="±1% error band")
    ax.axhspan(bs_call_price * 0.999, bs_call_price * 1.001,
               alpha=0.2, color="green", label="±0.1% error band")

    ax.set_xlabel("Number of Monte Carlo simulations (log scale)")
    ax.set_ylabel("Estimated Call Option Price")
    ax.set_title("Monte Carlo Convergence to Black-Scholes Price\n"
                 "More simulations → more accurate → closer to the red line")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"  Saved: {save_path}")
