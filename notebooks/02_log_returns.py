import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/carbon_prices.csv")
df["Date"] = pd.to_datetime(df["Date"])     # Convert 'Date' column to datetime
df = df.sort_values(by="Date", ascending = True)    # Sort by date in ascending order

df["Log_Returns"] = np.log(df["Price"] / df["Price"].shift(1))  # Calculate log returns. log return = log(price_t/price_t-1)
df = df.dropna()  # Remove rows with NaN values (first row will have NaN log return since there is no previosu price to compare it to)

# Plotting the log returns histogram

plt.hist(df["Log_Returns"], bins = 50, label = "Log Returns")
plt.xlabel("Log Returns")
plt.ylabel("Frequency")
plt.title("EU Carbon Log Returns")
plt.savefig("figures/02_log_returns_histogram.png")


#Normal distribution overlay with normalised log returns histogram

mean = np.mean(df["Log_Returns"])
std = np.std(df["Log_Returns"])

plt.figure()
x = np.linspace(df["Log_Returns"].min(), df["Log_Returns"].max(), 100)
normal_pdf = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean)/std)**2)
plt.plot(x, normal_pdf, color = "red", label = "Normal Distribution, mean = {:.5f}, std = {:.5f}".format(mean, std))
plt.hist(df["Log_Returns"], bins = 50, density = True, label = "Actual Log Returns")
plt.xlabel("Log Returns")
plt.ylabel("Density")
plt.title("EU Carbon Log Returns with Normal Distribution Overlay")
plt.legend()
plt.savefig("figures/02_normal_overlay_log_returns.png")

plt.show()


# ----Conclusions----
# The log returns are approximately normally distributed, with the histogram showing a slightly taller and narrower
# peak, and fatter tails than the normal distrubution. This indicates that extreme price changes are more likely 
# to occur than expected by the normal distrubution.
#
# The distribution is largely symmetric around zero, suggesting a stable market and no strong drift in prices.
#
# Standard deviation 2.13%, indicating moderate-to-high daily volatility in prices.
#
#