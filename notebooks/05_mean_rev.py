import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/carbon_prices.csv")
df["Date"] = pd.to_datetime(df["Date"])     # Convert 'Date' column to datetime
df = df.sort_values(by="Date", ascending = True)    # Sort by date in ascending order

df["Log_Returns"] = np.log(df["Price"] / df["Price"].shift(1))  # Calculate log returns. log return = log(price_t/price_t-1)
df = df.dropna()  # Remove rows with NaN values (first row will have NaN log return since there is no previous price to compare it to)

# Random walk simulation. Given today's price, what are the possible prices in 1 year (252 trading days)?

rng = np.random.default_rng(50) # gives same random numbers each run so figures and numbers don't change

n_days = 252  # Number of trading days in a year
dt = 1/n_days  # Time increment 
n_paths = 10000 

mu = np.mean(df["Log_Returns"]) * n_days  # Annualized mean of log returns
sigma = np.std(df["Log_Returns"]) * np.sqrt(n_days) # Annualized standard deviation of log returns

Z = rng.normal(0,1, size = (n_days, n_paths))  # Generate 2D array of random numbers from normal dist. T be used to generate random paths

# Geometric Brownian Motion formula: S(t) = S(0) * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
# log returns: ln(S(t)/S(0)) = (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z
# Log returns are additive so to get the price at day T, we can take the cumulative sum of T log returns
# and exponenitate
#               S(T) = S(0) * exp(cumulative_log_returns)

S_0 = df["Price"].iloc[-1]    # Last observed price. ie. Price on 01/07/2026
daily_log_returns = (mu - 0.5 * sigma**2) * dt + (sigma * np.sqrt(dt) * Z)    # 2D array of daily log returns for each path (252, 10000)
cumulative_log_returns = np.cumsum(daily_log_returns, axis=0) 
gbm_paths = S_0 * np.exp(cumulative_log_returns) # Note: changed price_paths --> gbm_paths to avoid confusion between GBM and OU

# At this point, we have a 2D array (252,10000), but each first day is different, so we should add the same starting price to each path

gbm_paths = np.vstack([np.full(n_paths, S_0), gbm_paths]) # Add the starting price S_0 to the first row (vstack adds row to the top of gbm_paths)

# =======================================================================================================================
# We want a model that pulls prices back to a mean when they get too high or low, as this will reflect real-life better
# This mimics how a spring in harmonic motion would act
#       --> the spring's motion will always converge back to its equilibrium position
#       --> the further the extension of the spring, the greater the force that pulls it back ==> the further the price moves from the mean,
#                                                                                                  the faster it will be pulled back
# BUT we also must account for 'noise', which is random
#
# The Ornstein-Uhlenbeck (OU) process:
#
#     dS = theta * (mu - S) * dt  +  sigma * dW
#          \____________________/    \________/
#            restoring force          noise
#               /DRIFT                /SHOCK
# 
# S = stock price at time t, theta = rate of mean reversion, mu = mean value, sigma = volatility, dW = Brownian motion/ Wiener Process/ noise
#
# We need to estimate these parameters:
#
# S(t+1) - S(t) = theta*mu*dt   -   theta*dt*S(t)   +   sigma*sqrt(dt)*Z 
#
# This looks just like a simple linear regression y = a + b*x + error, where y = daily price change and x = yesterday's price
#
# S(t+1) - S(t) = theta*mu*dt   -   theta*dt*S(t)   +   sigma*sqrt(dt)*Z 
# \___________/   \_________/       \_________/         \_____________/
#       |              |                 |                   |
#       y          a(intercept)    b(gradient) * S(t)      error term (ε)
#
#       Therefore:       theta = - b / dt
#                        mu = a / (theta * dt) = - a / b
#                        sigma = std(residual) / sqrt(dt)      <-- we say residual rather than error (ε) because it is an estimate of error
# 
# Note: a NEGATIVE b ==> mean reversion ==> OU model is appropratie
#               --> this is because theta = - b / dt. dt>0 so if b<0 then theta > 0 which means thata negative change is expected tomorrow.
#               --> is b > 0, this predicts momentum in prices, not mean reversion. So the OU model would not be appropriate

 

import scipy

dS = np.diff(df["Price"])  # y: daily price change
S_lag = df["Price"][:-1] # x: the price the day before the price change. Doesn't include last entry since there is no dS for that.

from scipy.stats import linregress
fit = linregress(S_lag,dS)
b = fit.slope
a = fit.intercept
theta = - b / dt
mu_ou = - a / b
residuals = dS - (a + (b * S_lag))
sigma_ou = residuals.std(ddof = 2) / np.sqrt(dt) # degrees of freedom = 2 since we estimated 2 parameters, a and b
half_life = np.log(2) / theta

print("\nOU parameters estimated from data:")
print(f"  regression slope b = {b:.6f}   (must be negative for mean reversion)")
print(f"  theta (reversion speed) = {theta:.4f} /yr")
print(f"  mu    (long-run mean)   = {mu_ou:.2f} EUR")
print(f"  sigma (volatility)      = {sigma_ou:.4f} EUR/sqrt(yr)")
print(f"  half-life               = {half_life:.2f} yr "
      f"({half_life * 252:.0f} trading days)")
 
if b >= 0:
    print("  WARNING: positive slope ==> the data does not mean-revert. "
          "Interpret the OU results tentatively - this does not seem to be the correct model.")
 


#--------------------SIMULATING OU PATHS--------------------

Z_ou = rng.normal(0,1, (n_days, n_paths))
ou_paths = np.zeros((n_days + 1, n_paths))
ou_paths[0,:] = S_0

for t in range(n_days):
    drift = theta * (mu_ou - ou_paths[t,:]) * dt
    shock = sigma_ou * np.sqrt(dt) * Z_ou[t,:]
    ou_paths[t+1,:] = ou_paths[t,:] + drift + shock

# The OU process is additive so it can go negative, but carbon allowances can, of course, not be negative
frac_negative_ou = np.mean(ou_paths.min(axis=0)< 0)
print(f"\nFraction of OU paths that go negative at some point: {frac_negative_ou:.2%}")

#-------------------------CHECKING SIMULATION AGAINST ANALYTIC SOLUTION-----------------------------------------
# Expected value over time: E[S_t]   = mu + (S0 - mu) * exp(-theta*t)
#           --> As t --> ∞ the exponential term tends to zero so the price is pulled back to the long-term mean, mu
#
# Variance over time: Var[S_t] = ( sigma^2 / (2*theta) ) * (1 - exp(-2*theta*t))
#           --> As t --> ∞ Var[S_t] = sigma^2 / (2*theta) ==> the variance becomes stationary (there is no time dependency)
#           --> CONTRAST TO GBM in which variance grows limitlessly 

time_years = n_days * dt
analytic_mean = mu_ou + (S_0 - mu_ou)*np.exp(-theta * time_years)
analytic_std = np.sqrt((sigma_ou**2 / (2 * theta) )* (1 - np.exp(-2 * theta * time_years)))

print("\nChecking simulation vs analytic Ornstein-Uhlenbeck processes:")
print(f"          mean      : {ou_paths[-1,:].mean():10.2f} vs {analytic_mean:10.2f}")
print(f"  standard deviation:   {ou_paths[-1].std():8.2f}  vs  {analytic_std:8.2f}")
print(f"  long-term (t -> ∞) standard deviation is {sigma_ou / np.sqrt(2 * theta):.2f}")

# Fan charts of GBM (made previously in notebooks/03_monte_carlo.py and dispayed in figures/03_monte_calo_fan.png) vs OU

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
 
for ax, paths, name, colour in [
    (axes[0], gbm_paths, 'Geometric Brownian Motion', 'blue'),
    (axes[1], ou_paths, 'Mean-Reverting (Ornstein-Uhlenbeck)', 'orange'),
]:
    ax.plot(paths[:, :100], color=colour, alpha=0.1, linewidth=0.8)
    ax.plot(paths.mean(axis=1), color='crimson', linewidth=1.5, label='Mean path')
    ax.plot(np.percentile(paths, 5, axis=1), 'black', linewidth=1, label='5th / 95th percentile')
    ax.plot(np.percentile(paths, 95, axis=1), 'black', linewidth=1)
    ax.set_title(name)
    ax.set_xlabel('Trading day')
    ax.grid(True, alpha=0.3)
 
axes[0].set_ylabel('Price (EUR)')
axes[1].axhline(mu_ou, color='green', linestyle=':', linewidth=2,
                label=f'Long-run mean = {mu_ou:.1f}')
axes[0].legend(loc='upper left')
axes[1].legend(loc='upper left')
 
fig.suptitle(f'{n_paths:,} simulated one-year carbon price paths', fontsize=13)
plt.tight_layout()
plt.savefig('figures/05_gbm_vs_meanreverting.png', dpi=150)
plt.show()
 
# Fan chart conclusions: the OU fan chart flattens since the long-term standard deviation eventually becaomes constant as t --> ∞. 
# The GBM chart continues to spread out since variance is boundless

from functions import confidence_level_analysis

gbm_pnl = gbm_paths[-1, :] - S_0
ou_pnl = ou_paths[-1, :] - S_0
 
confidence_level = [0.90, 0.95, 0.99]
gbm_results = confidence_level_analysis(gbm_pnl, confidence_level)
ou_results = confidence_level_analysis(ou_pnl, confidence_level)
 
print("\n" + "=" * 70)
print("RISK METRIC COMPARISON - GBM vs Ornstein-Uhlenbeck, 1 year horizon")
print("=" * 70)
 
for cl in confidence_level:
    gbm_cl, ou_cl = gbm_results[cl], ou_results[cl]
    print(f"\n{cl*100:.0f}% confidence:")
    print(f"  {'':<6}{'GBM':>22}{'Mean-Reverting':>22}")
    print(f"  {'VaR':<6}{gbm_cl['VaR']:>15.2f} +/- {gbm_cl['VaR_Error']:<4.2f}{ou_cl['VaR']:>15.2f} +/- {ou_cl['ES_Error']:<4.2f}")
    print(f"  {'ES':<6}{gbm_cl['ES']:>15.2f} +/- {gbm_cl['ES_Error']:<4.2f}{ou_cl['ES']:>15.2f} +/- {ou_cl['ES_Error']:<4.2f}")
 
print("\n" + "-" * 70)
print(f"  {'':<6}{'GBM':>22}{'Mean-Reverting':>22}")
print(f"{'Mean final price':<16}{gbm_paths[-1].mean():>15.2f}{ou_paths[-1].mean():>22.2f}")
print(f"{'Std dev final price':10}{gbm_paths[-1].std():>12.2f}{ou_paths[-1].std():>22.2f}")
print(f"{'P(loss)':<15}{np.mean(gbm_pnl < 0)*100:>15.1f}%{np.mean(ou_pnl < 0)*100:>22.1f}%") # proportion of 10,000 simulations that end below starting price
print("-" * 70)

# Overlaid P&L distributions

 
plt.figure(figsize=(12, 6))
bins = np.linspace(min(gbm_pnl.min(), ou_pnl.min()),
                   max(gbm_pnl.max(), ou_pnl.max()), 100)
 
plt.hist(gbm_pnl, bins=bins, density=True, alpha=0.55,
         color='steelblue', label='GBM')
plt.hist(ou_pnl, bins=bins, density=True, alpha=0.55,
         color='darkorange', label='Mean-reverting')
plt.axvline( x = -gbm_results[0.95]["VaR"], color='steelblue', linestyle='--', linewidth=2,
            label=f'GBM 95% VaR = {gbm_results[0.95]["VaR"]:.1f}')
plt.axvline(x = -ou_results[0.95]["VaR"], color='darkorange', linestyle='--', linewidth=2,
            label=f'OU 95% VaR = {ou_results[0.95]["VaR"]:.1f}')
plt.axvline(0, color='black', linewidth=1)
 
plt.title('One-year P&L distribution: GBM vs mean-reverting model')
plt.xlabel('Profit / loss per allowance (EUR)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('figures/05_pnl_comparison.png', dpi=150)
plt.show()


# -------------------READING THE P&L DISTRIBUTION CHART--------------------------
#
# SHAPE: the two distributions are not just different widths, they are different SHAPES.
#   skewness:  GBM +1.13   OU +0.01   (0 = symmetric)
#   GBM is LOGNORMAL - a price can rise without limit but cannot fall below zero, so the
#   right tail runs out to +200 while the left tail is bounded at -S_0 (about -80).
#   OU is NORMAL - the restoring force acts equally on both sides, so it is symmetric.
#   This is visible immediately: the blue histogram has a long right tail, the orange
#   one is a near-perfect bell.
#
# WHY GBM's MEAN IS HIGHER THAN ITS MODE:
#   GBM:  mean 78.99 > median 74.41 > mode ~68     (classic lognormal ordering)
#   OU:   mean 75.44 = median 75.44               (symmetric)
#   Both distributions PEAK to the LEFT of zero, i.e. the single most likely outcome is
#   a small loss under either model. GBM's higher mean comes entirely from the rare large
#   gains in its right tail, not from the typical path.
#
# PEAK HEIGHT: OU peaks at ~0.046 density, GBM at ~0.016 - a ratio of ~2.9, matching the
#   ratio of their standard deviations (27.64 / 8.65 = 3.2). Both areas must integrate to
#   1, so a narrower distribution must be proportionally taller. The tall orange spike IS
#   the variance ceiling sigma^2/(2*theta), seen from a different angle.
#
# THE HONEST COST OF MEAN REVERSION - it truncates the UPSIDE too:
#       P(final price > 90 EUR):   GBM 28.8%   OU  4.7%
#       P(final price > 100 EUR):  GBM 19.4%   OU  0.2%
#       best path in 10,000:       GBM 279.12  OU  107.25
#   OU does not simply remove downside risk; it removes large moves in BOTH directions.
#   Lower risk and lower reward. Anyone holding allowances as a speculative bet on rising
#   carbon prices should note that the OU model prices that bet as nearly worthless.
#
# THE MODELS ONLY DISAGREE IN THE TAILS:
#   Between roughly -20 and +10 EUR the two histograms overlap closely. For typical
#   outcomes, model choice barely matters. Risk management is entirely concerned with
#   the tails - which is exactly where the two models diverge most.
#
# LINK TO THE CLEAN ENERGY QUESTION (Day 5):
#   Renewables are commonly said to out-compete gas generation somewhere around
#   100 EUR/tonne. Asking each model how likely that is:
#       P(price ever reaches 100 EUR during the year):   GBM 41.3%   OU 6.3%
#   The model choice changes the policy answer by a factor of ~7. This is the strongest
#   argument in the whole project for why model selection is not a technical detail:
#   the same data, the same simulation engine, and two defensible models give completely
#   different answers to "will carbon prices support the energy transition this year?"


# ----------------------CONCLUSIONS----------------------------
#
# HEADLINE: switching from GBM to a mean-reverting model roughly HALVES the estimated risk.
#
#                       GBM          OU        ratio
#       90% VaR        31.32       15.10        2.07
#       95% VaR        36.69       18.26        2.01
#       99% VaR        45.36       24.34        1.86
#       95% ES         42.06       22.00        1.91
#
#   This is NOT sampling noise. At 95% the gap is 36.69 - 18.26 = 18.43 EUR, while the
#   bootstrap errors are 0.32 and 0.22 (combined ~0.39). The gap is ~47 times the error.
#   Model choice matters roughly 50x more than simulation noise here.
#
# WHY: the standard deviation of the final price falls from 27.64 (GBM) to 8.65 (OU),
#   a factor of 3.2. This is exactly the maths:
#       GBM  variance grows as sigma^2 * t, without limit.
#       OU   variance saturates at sigma^2 / (2*theta) = 25.2662 / sqrt(2*4.1016) = 8.82
#   The simulated 8.65 matches that 8.82 ceiling. The restoring force stops the
#   distribution spreading, and VaR is essentially a multiple of the spread.
#
# THE INTERESTING RESULT - higher P(loss) but LOWER VaR:
#   P(loss) rises from 57.6% (GBM) to 68.4% (OU), yet OU's VaR is half GBM's.
#   These are not contradictory. mu_ou = 75.38 is about 4.6 EUR BELOW the starting price,
#   so OU pulls nearly every path slightly below where it began -> more paths end in loss.
#   But the same restoring force prevents large excursions -> those losses are small.
#
#   In one sentence: MEAN REVERSION CONVERTS A SMALL CHANCE OF A LARGE LOSS
#   INTO A LARGE CHANCE OF A SMALL LOSS.
#
#   This matters practically: a trader worried about ruin should prefer the OU picture;
#   a trader judged on how often they finish down should prefer the GBM picture.
#
# TAIL SHAPE (ES/VaR ratio at 95%): GBM 42.06/36.69 = 1.15, OU 22.00/18.26 = 1.20.
#   OU's tail is proportionally slightly deeper because the OU final distribution is
#   NORMAL, whereas GBM's is LOGNORMAL and therefore left-skewed with a compressed
#   lower tail (a price can never go below zero). So GBM's tail is thin in shape but
#   enormous in absolute size. Absolute size is what dominates the risk numbers.
#
# VALIDATION:
#   Simulated mean 75.44 vs analytic 75.45 - excellent agreement.
#   Simulated std  8.65  vs analytic 8.82  - about 2% low, roughly 3 Monte Carlo
#   standard errors. Probably sampling, but worth re-running with a different seed
#   to confirm it does not persist.
#
#   No path ever went below zero (0.00%). That is expected rather than lucky: zero sits
#   75.38 / 8.82 = 8.5 stationary standard deviations below the mean. OU's theoretical
#   weakness (it permits negative prices) is harmless at these parameter values.
#
# HALF-LIFE = 43 trading days (~2 months). A deviation from the long-run mean loses half
#   its size in about two months. Economically plausible for a market where the Market
#   Stability Reserve adjusts allowance supply and emitters change abatement behaviour.
#   CAVEAT: theta is estimated imprecisely from ~4 years of data. scipy already gives the
#   standard error - add these two lines to quantify it:
#       theta_se = fit.stderr / dt          # standard error on theta
#       t_stat   = fit.slope / fit.stderr   # compare to DICKEY-FULLER 5% crit = -2.86,
#                                           # NOT -1.96: under a unit root the statistic
#                                           # is not normally distributed
#
# LIMITATIONS:
#   1. mu_ou = 75.38 is treated as CONSTANT, but the EU ETS cap tightens every year, so
#      the true long-run mean is itself drifting upward. Plain OU cannot capture this.
#      This is the strongest criticism of the model as applied here.
#   2. theta has wide confidence bounds; the VaR difference is robust in direction but
#      not precise in magnitude.
#   3. Constant volatility - no volatility clustering, no jumps on policy announcements.
#   4. Normal shocks, but the Day 1 histogram showed fatter tails than normal.
#   5. Only ~4 years of data (2022-2026), covering one large run-up and fall.
#
# VERDICT: OU captures the short-run pull-back behaviour that GBM misses, and the
#   evidence (negative regression slope, 43-day half-life, variance ceiling matching
#   theory) supports it. But neither model handles the policy-driven upward drift in the
#   cap. The most realistic description is probably mean reversion around a RISING mean,
#   which would sit between these two answers.