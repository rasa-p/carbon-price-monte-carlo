# carbon-price-monte-carlo

Monte Carlo simulation of EU carbon allowance (EUA) prices with risk metrics (Value at Risk, Expected Shortfall).
Comparison of Geometric Brownian Motion with a mean-reverting (Ornstein–Uhlenbeck) model.

**Question:** *How much of the answer to "where will the carbon price go?" comes from the data, and how much comes from the model you chose?*

Data: ~4 years of daily EUA settlement prices (July 2022 – July 2026, 1,025 observations). 10,000 simulated paths, 252-day horizon, starting price €79.54.

## Contents

| Script | What it does |
|---|---|
| `01_explore.py` | Load and plot the price series |
| `02_log_returns.py` | Log-return distribution vs. a normal fit |
| `03_monte_carlo.py` | GBM Monte Carlo, fan chart |
| `04_var_es.py` | VaR and Expected Shortfall, bootstrapped errors, analytical cross-check, √t scaling |
| `05_mean_rev.py` | Ornstein–Uhlenbeck model fitted by regression; GBM vs. OU risk comparison |
| `06_threshold_probabilities.py` | Probability of reaching a given price level under each model |

## Conclusion

Fitted to the same data over the same horizon, the two models disagree about the one-year risk of holding an allowance by a factor of two — 95% VaR of €36.69 ± 0.33 under GBM against €18.26 ± 0.21 under mean reversion, a gap of 47 standard errors — and about the probability of the price ending the year above €100 by a factor of 77 (19.2% vs. 0.25%). The mechanism is a single structural assumption: GBM lets variance grow without bound as σ²t, while the OU process saturates it at σ²/(2θ), pinning the distribution to a stationary standard deviation of €8.82 about a long-run mean of €75.38. Because the bootstrap and binomial errors on every one of those figures are under half a percentage point, the disagreement cannot be attributed to sampling noise — it is model uncertainty, and here it dominates statistical uncertainty by roughly two orders of magnitude.

The data cannot arbitrate between the two. Mean reversion is clearly present in-sample (regression slope b = −0.0163, half-life 43 trading days), but θ is estimated from only four years and μ is held fixed even though the EU ETS cap tightens annually — a rising cap means a rising μ, which is precisely the mechanism that would put higher prices back within reach and which plain OU cannot represent.

So the useful output of this project is not a price forecast. It is a demonstration that a headline number like "there is a 19% chance of €100 carbon" is close to meaningless when quoted without the model that produced it, and that the first question to ask of any such figure is not how many paths were simulated but whether the process was assumed to have a long-run anchor. A secondary result points the same way: mean reversion raises the probability of ending the year down (68.4% vs. 57.6%) while halving the size of the tail, so the choice of model does not simply scale risk up or down — it changes its shape, converting a small chance of a large loss into a large chance of a small one.

## Author

Rasa Pedram — MSci Natural Sciences, UCL
