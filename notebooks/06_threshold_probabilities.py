# 06 — Threshold probabilities: what does each model say about reaching a given price?
#
# Sections 03-05 asked "how much could I lose?" (VaR / ES).
# "How likely is the price to REACH a given level?" 
# Showing that the two models, fitted to the SAME data with the SAME seed over the SAME horizon, disagree about that probability by up to almost two 
# orders of magnitude (factor of ~77), while the sampling error on each individual estimate is well under one percentage point.

import pandas as pd
import numpy as np
from scipy.stats import linregress

df = pd.read_csv("data/carbon_prices.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date", ascending=True)
df["Log_Returns"] = np.log(df["Price"] / df["Price"].shift(1))
df = df.dropna()

rng = np.random.default_rng(50)
n_days = 252
dt = 1 / n_days
n_paths = 10000

# ---------------- GBM (same construction as 03/04/05) ----------------
mu = np.mean(df["Log_Returns"]) * n_days
sigma = np.std(df["Log_Returns"]) * np.sqrt(n_days)
Z = rng.normal(0, 1, size=(n_days, n_paths))

S_0 = df["Price"].iloc[-1]
daily_log_returns = (mu - 0.5 * sigma**2) * dt + (sigma * np.sqrt(dt) * Z)
gbm_paths = S_0 * np.exp(np.cumsum(daily_log_returns, axis=0))
gbm_paths = np.vstack([np.full(n_paths, S_0), gbm_paths])

# ---------------- OU (same construction as 05) ----------------
dS = np.diff(df["Price"])
S_lag = df["Price"][:-1]
fit = linregress(S_lag, dS)
b, a = fit.slope, fit.intercept
theta = -b / dt
mu_ou = -a / b
residuals = dS - (a + b * S_lag)
sigma_ou = residuals.std(ddof=2) / np.sqrt(dt)

Z_ou = rng.normal(0, 1, (n_days, n_paths))
ou_paths = np.zeros((n_days + 1, n_paths))
ou_paths[0, :] = S_0
for t in range(n_days):
    drift = theta * (mu_ou - ou_paths[t, :]) * dt
    shock = sigma_ou * np.sqrt(dt) * Z_ou[t, :]
    ou_paths[t + 1, :] = ou_paths[t, :] + drift + shock

# ---------------- Threshold probabilities ----------------
# A probability estimated from n independent paths is a binomial proportion, so its standard error is sqrt(p(1-p)/n). 
# With n = 10,000 this is at most 0.5 pp — the sampling error is tiny compared to the model-to-model gap.

def prob_and_error(mask):
    p = mask.mean()
    return p, np.sqrt(p * (1 - p) / len(mask))

thresholds = [90, 100, 110, 120, 150]

print(f"\nStarting price S_0 = {S_0:.2f} EUR, {n_paths:,} paths, {n_days}-day horizon\n")
print("PROBABILITY THE PRICE ENDS THE YEAR ABOVE A THRESHOLD")
print("-" * 68)
print(f"{'Threshold':>12}{'GBM':>20}{'Mean-reverting':>22}")
for K in thresholds:
    pg, eg = prob_and_error(gbm_paths[-1, :] > K)
    po, eo = prob_and_error(ou_paths[-1, :] > K)
    print(f"{'EUR ' + str(K):>12}{pg*100:>13.2f}% +/-{eg*100:<4.2f}{po*100:>14.2f}% +/-{eo*100:<4.2f}")

print("\nPROBABILITY THE PRICE TOUCHES A THRESHOLD AT ANY POINT IN THE YEAR")
print("-" * 68)
print(f"{'Threshold':>12}{'GBM':>20}{'Mean-reverting':>22}")
for K in thresholds:
    pg, eg = prob_and_error(gbm_paths.max(axis=0) > K)
    po, eo = prob_and_error(ou_paths.max(axis=0) > K)
    print(f"{'EUR ' + str(K):>12}{pg*100:>13.2f}% +/-{eg*100:<4.2f}{po*100:>14.2f}% +/-{eo*100:<4.2f}")


# ----------------------------- CONCLUSIONS -----------------------------
# At EUR 100, the two models give:
#
#       GBM: 19.2% +/- 0.4        Mean-reverting: 0.25% +/- 0.05
#
# That is a factor of ~77 between two models that were fitted to identical data.
# The sampling errors are ~0.4 pp and ~0.05 pp, so the gap is not noise.
#
# WHY: under GBM the variance of the log price grows without bound as sigma^2 * t, so the distribution keeps spreading. 
# Under OU the variance saturates at sigma^2 / (2*theta), giving a stationary standard deviation of 8.82 EUR around a long-run mean of 75.38 EUR. 
# EUR 100 sits ~2.8 stationary standard deviations above that mean, and EUR 120 sits ~5 — this is why the OU probability collapses to zero.
#
# The answer to "How likely is the carbon price to reach level X" is essentially determined by whether the price has a long-run anchor. The data
# over 2022-2026 can't settle this. The OU regression slope is negative (b = -0.0163), so mean reversion is present in the sample, but theta is 
# estimated from only 4 years of data and mu is held constant even though the EU ETS cap tightens every year. Rising cap = rising mu, which would 
# put EUR 100 back in  reach - plain OU, as I have used here, cannot represent this.
#
# So we can conclude that the dominant uncertainty here is MODEL uncertainty, not statistical uncertainty, and that any single headline number 
# quoted is constrained by the model and its limited scope of the data.
