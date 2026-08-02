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
price_paths = S_0 * np.exp(cumulative_log_returns)

# At this point, we have a 2D array (252,10000), but each first day is different, so we should add the same starting price to each path

price_paths = np.vstack([np.full(n_paths, S_0), price_paths]) # Add the starting price S_0 to the first row (vstack adds row to the top of price_paths)

# Profit and Loss, Value at Risk and Expected Shortfall calculations
#     P&L  = S_T - S0        -----> positive = profit, negative = loss
#     Loss = S0 - S_T = -P&L   ---> positive = loss   (the "loss convention")

# Let L = -P&L be the loss. For confidence level a (e.g. 0.95):
 
#        VaR_a = -quantile_({1-a}(P&L))              --> the (1-a)-quantile of P&L, with sign flipped since VaR measures loss.
#      ES_a = E( L | L >= VaR_a )  --> the expected loss given given that the loss>= VaR_a. Average loss inside the tail.

final_prices = price_paths[-1,:]
pnl = final_prices - S_0
confidence_level=[0.90, 0.95, 0.99]


def calculate_var_es(pnl_samples, confidence_level):
    """
    Calculate VaR and ES for a given set of profit and loss samples.
    
    Parameters:
    pnl_samples (array): Array of profit and loss samples.
    confidence_level (float): Confidence level for VaR and ES calculation (eg. 0.95 for 95% confidence).
    
    Returns:
    tuple: VaR and ES values.
    """
   
    var = -np.quantile(pnl_samples, 1-confidence_level)

    es = -np.mean(pnl_samples[pnl_samples <= -var])

    return var, es


def bootstrap_var_es(pnl_samples, confidence_level, n_bootstrap=1000):
    """
    Bootstrap VaR and ES to estimate error.
    
    Parameters:
    pnl_samples (array): Array of profit and loss samples.
    confidence_level (float): Confidence level for VaR and ES calculation (eg. 0.95 for 95% confidence).
    n_bootstrap (int): Number of bootstrap samples.
    
    Returns:
    tuple: Arrays of bootstrapped VaR and ES values.
    """
    var_bootstrap, es_bootstrap = np.zeros(n_bootstrap), np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        resampled_pnls = np.random.choice(pnl_samples, size = len(pnl_samples), replace = True)
        var_bootstrap[i], es_bootstrap[i] = calculate_var_es(resampled_pnls, confidence_level)

    return np.std(var_bootstrap), np.std(es_bootstrap)


def confidence_level_analysis(pnl_samples, confidence_level, n_bootstrap=1000):
    """
    Analyze VaR and ES for different confidence levels and estimate error.
    
    Parameters:
    pnl_samples (array): Array of profit and loss samples.
    confidence_level (list): List of confidence levels.
    n_bootstrap (int): Number of bootstrap samples.
    
    Returns:
    dict: Dictionary containing VaR, ES, and their errors for each confidence level.
    """
    results = {}
    for cl in confidence_level:
        var, es = calculate_var_es(pnl_samples, cl)
        var_error, es_error = bootstrap_var_es(pnl_samples, cl, n_bootstrap)
        results[cl] = {
            "VaR": var,
            "ES": es,
            "VaR_Error": var_error,
            "ES_Error": es_error
        }
    return results


# Printing results


print("\nValue at Risk (VaR) = loss you exceed only (1 - confidence) of the time.")
print("Expected Shortfall (ES)  = average loss on the occasions you DO exceed it. Therefore, ES >= VaR.\n")
print("-" * 50)
results = confidence_level_analysis(pnl, confidence_level, n_bootstrap=1000)
for cl, metrics in results.items():
    print(f"Confidence Level: {cl*100:.0f}%")
    print(f"  VaR: {metrics['VaR']:.2f} EUR ± {metrics['VaR_Error']:.2f}")
    print(f"  ES: {metrics['ES']:.2f} EUR ± {metrics['ES_Error']:.2f}\n")
    print("-" * 50, "\n")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Checking against the analytical solution 
# ----------------------------------------------------------------------------------------------------------------------------------------
# Under GBM: S(T) = S(0) * exp((mu - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z
#                              |_______________________|  |______________|
#                                          |                     |
#                                          m                     s
# 
#                           VaR_a = S(0) - q_1-a(S(T))
#                               ln(S(T)/S(0)) ~ N(m,s^2) 
#                   So, q_1-a(S(T)) = S(0) * exp(m + s * q_1-a(Z))
#                   VaR_a = S(0) - S(0) * exp(m + s * q_1-a(Z)) 
#                         = S(0) * (1 - exp(m + s * q_1-a(Z)))

import statistics as stats
from statistics import NormalDist

T = 1 # Note: We are using T in the formula rather than dT because m and s represent the annualised mean and sd, and mu and sigma are already annualised (see above).
m = (mu - 0.5 * sigma**2) * T
s = sigma * np.sqrt(T)


print("CHECKING simulated VaR against analytical VaR under GBM assumptions:\n")

def analytical_var_es(S0, m, s, confidence_level):
    """
    Calculate analytical VaR and ES for a given confidence level under GBM.
    
    Parameters:
    S0 (float): Initial price.
    m (float): Mean of log returns.
    s (float): Standard deviation of log returns.
    confidence_level (float): Confidence level for VaR and ES calculation (eg. 0.95 for 95% confidence).
    
    Returns:
    tuple: Analytical VaR and ES values.
    """
    comparisons = {}
    for cl in confidence_level:
        z = NormalDist().inv_cdf(1-cl)  # Quantile for standard normal distribution

        var_analytical = S0 * (1 - np.exp(m + (s * z)))
        var_simulated = results[cl]["VaR"]  # Get the VaR from the simulation results
        sd = results[cl]["VaR_Error"]  # Get the VaR error from the bootstrap results

        n_sd = abs(var_analytical - var_simulated)/sd
        verdict = "OK" if n_sd < 3 else "CHECK ME" # If n_sd > 3, var_analytical and var_simulated are significantly different, which is cause for concern.

        comparisons[cl] = {
            "Simulated VaR": var_simulated,
            "Analytical VaR": var_analytical,
            "No. of standard deviations apart": n_sd,
            "Verdict": verdict
        }
        

    return comparisons


comparisons = analytical_var_es(S_0, m, s, confidence_level)
for cl, metrics in comparisons.items():
    print(f"Confidence Level: {cl*100:.0f}%")
    print(f"  Simulated VaR: {metrics['Simulated VaR']:.2f} EUR")
    print(f"  Analytical VaR: {metrics['Analytical VaR']:.2f} EUR")
    print(f"  No. of standard deviations apart: {metrics['No. of standard deviations apart']:.1f}")
    print(f"  Verdict: {metrics['Verdict']}")
    print("-" * 50)


# Plotting histogram of simulated P&L with VaR and ES lines

plt.style.use("seaborn-v0_8-whitegrid")

var_95, es_95 = results[0.95]["VaR"], results[0.95]["ES"]
fig,ax = plt.subplots(figsize=(12,6))
counts, bins, patches = plt.hist(pnl, bins = 100, density = True,color = "blue", label = "Simulated P&L", alpha=0.7)

# Shading 95% VaR tail
tail = bins[:-1] <= -var_95
ax.bar(bins[:-1][tail], counts[tail], width= np.diff(bins)[tail], align = "edge",  color = "crimson", alpha = 0.7, label = "95% VaR Tail") 
ax.axvline(0, color = "black")
ax.axvline(x = -var_95, color = "red", linestyle = "--", label = f"95% VaR = {var_95:.2f} EUR")
ax.axvline(x = -es_95, color = "orange", linestyle = "--", label = f"95% ES = {es_95:.2f} EUR")
ax.set_title(f"Profit & Loss Distribution with VaR and Expected Shortfall\n"
          f"EU carbon allowance, {n_days}-day horizon, {n_paths} GBM paths",
          fontsize=13)
ax.set_xlabel("1 Year Profit & Loss (EUR)")
ax.set_ylabel("Probability Density")

min_x = pnl.min()
max_x = pnl.max()


current_ticks = list(plt.xticks()[0]) # list converts the numpy array to a list so we can append it
new_ticks = current_ticks + [min_x, max_x]
plt.xticks(new_ticks, [f"{x:.2f}" for x in new_ticks])

ax.legend()

plt.tight_layout()
plt.savefig("figures/04_pnl_95%_var_es_histogram.png")
plt.show() 


# Finally, let's look at how VaR changes with horizon length (holding period) - currently we have only looked at a 1-year horizon
# We can compare our actual simulated VaR growth with the theoretical VaR growth under GBM, which says that it grows with sqrt(T)
#
# We can understand this sqrt(T) factor by considering the variance of the sum of T independent random variables
#
#               Var(X_1 + X_2 + ... + X_T) = Var(X_1) + Var(X_2) + ... + Var(X_T) = T * Var(X_i)
#               SD(Total movement) = sqrt(T) * SD(X_i)            <---- AKA volatility scales with sqrt(T)
#               Value at Risk is a function of volatility (VaR = S0 * (1 - exp(m + s * q_1-a(Z))), where s is the volatility or we could even go back to VaR_a = mu + sigma * q_1-a(Z))               
#               Therefore, VaR also scales with sqrt(T).



horizons = np.arange(1, n_days + 1)
pnl_by_day = price_paths[1:,:] - S_0 # Excluding the first row since it's the starting price. Shape (252,10000) 
var_by_day = -np.quantile(pnl_by_day, 0.05, axis = 1) # shape (252,10000)


fig, ax = plt.subplots(figsize=(8,4))
ax.plot(horizons, var_by_day, color = "red", label = "95% VaR") # actual VaR growth form simulation
ax.plot(horizons, var_by_day[0] * np.sqrt(horizons), color="blue", label="1-day VaR ×√t scaling") #VaR from day 1 scaled by sqrt(t) to compare with the actual VaR growth
ax.set_title(f"95% Value at Risk (VaR) by Holding Period\n"
            f"EU carbon allowance, {n_paths} GBM paths", fontsize=13)
ax.set_xlabel("Holding Period (days)")
ax.set_ylabel("95% VaR (EUR)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/04_var_by_holding_period.png")
plt.show()




# ----------------------CONCLUSIONS----------------------------
# PROFIT & LOSS HISTOGRAM
#      - Right-skewed distribution, with a shorter tail to the left (losses) and a longer tail to the right (profits). 
#                --> indicates that while there is a higher probability of small profits, there is also a significant risk of large losses.
#      - Cannot lose > 62.53 EUR - cap on losses
#      - At 95% confidence, VaR = 36.69 EUR, ES = 42.06 EUR. This is not too bad - it tells us that in the worst 5% of cases, we can expect to on average lose
#        5.37 EUR more than the VaR threshold (14.6% of VaR). This is attributed to a thin and bounded tail
#      - The bulk of the distribution sits below zero, so more than half the simulated paths end in a loss. 
#                 --> note that this is based on the trend in data from 2022-2026, so not necessarily representative of the long-term trend in carbon prices
#      - The maximum profit from the histogram is 197.3 EUR - but note that there is no actual cap on the profit, this is just the maximum profit from the 10000 simulated paths.
#
# HOLDING PERIOD CHART
#     - The √t scaling holds well for the first ~30 days, after which it begins to overstate the risk
#     - The √t scaling grows boundlessly, while actual VaR growth is bounded by a maximum loss - in this case 62.53 EUR
#                --> this explains why it overstates the risk after ~30 days - it doesn't account for the bounded nature of the distribution.
#
# NEXT SECTION:
# In using Geometric Brownian Motion on EU carbon allowance data, we have failed to account for the pull-back behaviour of carbon prices, due to real-life factors.
# In reality, carbon prices are stabilised by regulation and eco-friendly efforts by businesses. When prices spike, emitters adjust and when they crash, regulators
# tighten the supply of permits. These are mean-reversion techniques, which mean less volatility in the market, and lower VaR and ES than the model predicts.
#
# NEXT STEPS --------> Mean Reversion Model