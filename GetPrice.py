import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Obtain MSFT 
msft = yf.Ticker("MSFT")
print(msft)

# get historical market data, here max is 5 years.
stock_prices = msft.history(period="max")
print(stock_prices)

# Get the close price for every day 
close_prices = stock_prices.iloc[:,3].values
dates = np.arange(len(close_prices)) # make an array for the days to plot

# moving average calculation
MA = np.convolve(close_prices, np.ones(10), 'valid') / 10
#print(MA)

# Plot 
plt.plot(dates,close_prices)
plt.plot(dates[8:-1],MA)
plt.title('Microsoft Price')
plt.xlabel('Day')
plt.ylabel('Price($)')
plt.legend('Original Stock Price, 10-day MA')
plt.show()








