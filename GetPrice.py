import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

msft = yf.Ticker("MSFT")
print(msft)
"""
returns
<yfinance.Ticker object at 0x1a1715e898>
"""

# get stock info
a = msft.institutional_holders

"""
returns:
{
 'quoteType': 'EQUITY',
 'quoteSourceName': 'Nasdaq Real Time Price',
 'currency': 'USD',
 'shortName': 'Microsoft Corporation',
 'exchangeTimezoneName': 'America/New_York',
  ...
 'symbol': 'MSFT'
}
"""

# get historical market data, here max is 5 years.
stock_prices = msft.history(period="max")
print(stock_prices)



close_prices = stock_prices.iloc[:,3].values
dates = np.arange(len(close_prices))

MA = np.convolve(close_prices, np.ones(10), 'valid') / 10
#print(MA)

plt.plot(dates,close_prices)
plt.plot(dates[8:-1],MA)
plt.title('Microsoft Price')
plt.xlabel('Day')
plt.ylabel('Price($)')
plt.legend('Original Stock Price, 10-day MA')
plt.show()








