
Summary of findings from `notebooks/05_mean_rev.py`. Starting price S₀ ≈ 79.55 EUR, 10,000 simulated paths, 252-day (one year) horizon.

# DATA FINDINGS (see figures for more)

OU parameters estimated from data:
  regression slope b = -0.016276   (must be negative for mean reversion)
  theta (reversion speed) = 4.1016 /yr
  mu    (long-run mean)   = 75.38 EUR
  sigma (volatility)      = 25.2662 EUR/sqrt(yr)
  half-life               = 0.17 yr (43 trading days)

Fraction of OU paths that go negative at some point: 0.00%              

Checking simulation vs analytic Ornstein-Uhlenbeck processes:
          mean      :      75.44   vs      75.45
  standard deviation:       8.65   vs      8.82
  long-term (t -> ∞) standard deviation is 8.82

======================================================================
RISK METRIC COMPARISON - GBM vs Ornstein-Uhlenbeck, 1 year horizon
======================================================================

90% confidence:
                         GBM        Mean-Reverting
  VaR             31.32 +/- 0.28          15.10 +/- 0.18
  ES              37.97 +/- 0.26          19.26 +/- 0.18

95% confidence:
                           GBM        Mean-Reverting
  VaR             36.69 +/- 0.33          18.26 +/- 0.21
  ES              42.06 +/- 0.30          22.00 +/- 0.21

99% confidence:
                           GBM        Mean-Reverting
  VaR             45.36 +/- 0.48          24.34 +/- 0.39
  ES              49.35 +/- 0.54          27.33 +/- 0.39

----------------------------------------------------------------------
                           GBM        Mean-Reverting
Mean final price          78.95                 75.44
Std dev final price       27.64                  8.65
P(loss)                   57.6%                  68.4%
----------------------------------------------------------------------

The risk-metric comparison shows that replacing Geometric Brownian Motion with an Ornstein-Uhlenbeck (mean-reverting) model roughly halves the estimated risk of holding EU carbon allowances. We can see this from the ratios of GBM:OU VaR and ES:

Confidence	|   GBM VaR	      |    OU VaR	 |  ratio  |    GBM ES	  |     OU ES	 |  ratio
------------|-----------------|--------------|---------|--------------|--------------|--------
    90%	    |    31.32 ± 0.28 | 15.10 ± 0.18 | 2.07    | 37.97 ± 0.26 | 19.26 ± 0.18 | 1.97
    95%	    |    36.69 ± 0.33 | 18.26 ± 0.21 | 2.01	   | 42.06 ± 0.30 | 22.00 ± 0.21 | 1.91
    99%	    |   45.36 ± 0.48  | 24.34 ± 0.39 | 1.86	   | 49.35 ± 0.54 | 27.33 ± 0.39 | 1.81

Difference between VaR estimates at 95% confidence: 36.69 - 18.26 = 18.43 EUR
Standard error: σ_C ​= sqrt((σ_A)^2 ​+ (σ_B)^2)
                    ​= sqrt( 0.33^2 + 0.21^2)
                    ​= 0.39
The gap, 18.43 EUR = 47 Standard Errors   <--- this is large enough to prove that the difference in Values at Risk is due to the modelling                       differences and not just randomness or noise    

----------------------------------------------------------------------------------------------------------------------------
                           GBM                    OU
Std dev final price       27.64                  8.65

The standard deviation of the final price falls from 27.64 EUR (GBM) to 8.65 EUR (OU) — a factor of 3.2.

This is exactly what the maths predicts:
    - Under GBM, variance limitlessly grows as σ²t 
    - Under OU, variance saturates at σ²/(2θ), giving a standard deviation of 25.2662 / √(2 × 4.1016) = 8.82 EUR.

The simulated 8.65 sits just below that 8.82 ceiling, confirming the process has essentially reached its stationary distribution within one year.

The fan charts show the same thing geometrically - the GBM 5th–95th percentile band keeps widening across the full 252 days, reaching roughly 43–130 EUR. The OU band opens for about 40 days and then runs flat at roughly 62–89 EUR. The restoring force stops the distribution spreading, and since VaR is essentially a multiple of the spread, the risk estimate collapses with it.

----------------------------------------------------------------------------------------------------------------------------
                           GBM                    OU
P(loss)                   57.6%                  68.4%
Mean (EUR)                -0.6                   -4.1

Slightly counterintuitively, we see that the OU model predicts that a higher proportion of paths will result in loss (10.8% more likely).
The fitted long-run mean μ = 75.38 EUR sits 4.1 EUR below the starting price, so the restoring force pulls nearly every path slightly below where it began, ie. more paths finish in loss. But that same force prevents large excursions, so those losses are small and tightly bounded.
So, the higher probability of loss and more negative mean, aren't necessaily bad things because the SIZE of loss matters too.

MAIN MESSAGE: Mean reversion converts a small chance of a large loss --> large chance of a small loss.

----------------------------------------------------------------------------------------------------------------------------

Parameter  |     Estimate	   |        Interpretation
-----------|-------------------|-------------------------------
    θ	   |    4.1016 /yr	   |    speed of reversion
    μ	   |    75.38 EUR	   |    long-run mean level
    σ	   |    25.2662 EUR/√yr|        volatility
half-life  |    43 trading days|    time for a deviation to halve

The regression of daily price change on price level gives a slope of b = −0.016276
    --> A negative slope is the signature of mean reversion: a high price today implies an expected fall tomorrow.
A 43 day half-life means a deviation from the long-run mean loses half its size in about two months.

Fraction of OU paths that go negative at some point: 0.00%   
    --> OU is additive so it can produce negative prices
    --> It did not happen here, why? --> zero sits 75.38 / 8.82 ≈ 8.5 stationary standard deviations below the mean, so it's very unlikely for this to happen.

----------------------------------------------------------------------------------------------------------------------------

LIMITATIONS of the OU model

1. μ is treated as constant. The EU ETS cap tightens every year, so the true long-run mean is actually going upwards. Plain OU cannot represent this. 
2. θ is imprecisely estimated. Roughly four years of daily data constrain the mean-reversion speed only loosely. 
3. Constant volatility is an unrealistic feature of real carbon markets
4. Normally distributed shocks, although the Day 1 histogram showed fatter tails than a normal distribution.
5. I used a short sample (~4 years), covering one large run-up and subsequent fall (see figures/01_price_plot.png)

