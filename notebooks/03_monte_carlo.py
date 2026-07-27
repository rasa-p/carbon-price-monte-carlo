import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/carbon_prices.csv")
df["Date"] = pd.to_datetime(df["Date"])     # Convert 'Date' column to datetime
df = df.sort_values(by="Date", ascending = True)    # Sort by date in ascending order

df["Log_Returns"] = np.log(df["Price"] / df["Price"].shift(1))  # Calculate log returns. log return = log(price_t/price_t-1)
df = df.dropna()  # Remove rows with NaN values (first row will have NaN log return since there is no previous price to compare it to)

# Random walk simulation. Given today's price, what are the possible prices in 1 year (252 trading days)?


n_days = 252  # Number of trading days in a year
dt = 1/n_days  # Time increment 
n_paths = 10000 

mu = np.mean(df["Log_Returns"]) * n_days  # Annualized mean of log returns
sigma = np.std(df["Log_Returns"]) * np.sqrt(n_days) # Annualized standard deviation of log returns

Z = np.random.normal(0,1, size = (n_days, n_paths))  # Generate 2D array of random numbers from normal dist.

# Geometric Brownian Motion formula: S(t) = S(0) * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
# log returns: ln(S(t)/S(0)) = (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z
# Log returns are additive so to get the price at day T, we can take the cumulative sum of T log returns
# and exponenitate
#               S(T) = S(0) * exp(cumulative_log_returns)

S_0 = df["Price"].iloc[-1]    # Last observed price
daily_log_returns = (mu - 0.5 * sigma**2) * dt + (sigma * np.sqrt(dt) * Z)    # 2D array of daily log returns for each path (252, 10000)
cumulative_log_returns = np.cumsum(daily_log_returns, axis=0) 
S_T_paths = S_0 * np.exp(cumulative_log_returns)

# At this point, we have a 2D array (252,10000), but each first day is different, so we should add 
# the same starting price to each path 

S_T_paths = np.vstack([np.full(n_paths, S_0), S_T_paths]) # Add the starting price S_0 to the first row (vstack adds row to the top of S_T_paths)

#Plotting the similuated paths

for i in range(100):    # only plot 100 paths to avoid cluttering 
    plt.plot(S_T_paths[:,i], color= "blue", alpha=0.1)
plt.plot()

# Showing the mean and 5th and 95th percentiles of the simulated paths, therby giving a 90% confidence interval for the future prices
plt.plot(np.mean(S_T_paths, axis=1), color="red", linewidth=1, label="Mean Paths")
plt.plot(np.percentile(S_T_paths, 5, axis=1), color="black", linewidth=1, label='5th percentile')
plt.plot(np.percentile(S_T_paths, 95, axis=1), color="black", linewidth=1, label='95th percentile')

plt.xlabel("Time")
plt.ylabel("Price (EUR)")
plt.title("Monte Carlo Simulation of EU Carbon Prices")
plt.legend()
plt.savefig("figures/04_monte_carlo_fan.png", dpi=100)
plt.show()

# -----Conclusions-----
# Since the data from 2022-2026 has a mean of -0.0002, we would expect the simulations' means so be roughly flat, as we see
# in the plot. 
# The 5th and 95th pencentiles determine there is a 90% confidence interval that future prices will be roughly between 70 and 120 EUR.
# We could increase the reliability of this prediction by increasing the number of paths simulated
