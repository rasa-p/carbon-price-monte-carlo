import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/carbon_prices.csv")

df['Date'] = pd.to_datetime(df['Date'])     # Convert 'Date' column to datetime
df = df.sort_values(by="Date", ascending = True)    # Sort by date in ascending order

print(df.head())

print(df[pd.isnull(df['Price'])])       # Print rows where 'Price' is null. 

# DataFrame contains 1026 rows. There are no null values in the 'Price' column, so we can proceed with the analysis.

plt.plot(df['Date'], df['Price'])
plt.title("EU Carbon Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price (EUR)")
plt.savefig("figures/01_price_plot.png")
plt.show()