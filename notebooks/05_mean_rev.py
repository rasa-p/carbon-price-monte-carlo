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
print(f"  long-run (t -> ∞) standard deviation is {sigma_ou / np.sqrt(2 * theta):.2f}")

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
 
 