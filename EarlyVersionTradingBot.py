import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

# Obtain Stock Data
stock = yf.Ticker("TD")

# get historical market data, here max is 5 years.
stock_prices = stock.history(period="max")

# Get the close price for every day 
close_prices = stock_prices.iloc[:,3].values #obtain the close price from column 3
volume = stock_prices.iloc[:,4].values # obtain the volume in trades from column 4
dates = np.arange(len(close_prices)) # make an array for the days to plot

# moving average calculation
#MA = np.convolve(close_prices, np.ones(10), 'valid') / 10
#print(MA)

# Plot 
#plt.plot(dates,close_prices)
#plt.plot(dates[8:-1],MA)
#plt.title('Microsoft Price')
#plt.xlabel('Day')
#plt.ylabel('Price($)')
#plt.legend('Original Stock Price, 10-day MA')
#plt.show()

print('')
print('Beginning Optimal Window Sizing Process')
k = 0
window_size_start = 40
window_size_end = 61
accuracy = np.zeros((window_size_end-window_size_start))
for window_size in range(window_size_start,window_size_end):

# partition the total prices into n-block sizes
    n = window_size
    split_prices = np.array_split(close_prices, int(np.floor(len(close_prices)/n)))
    split_volumes = np.array_split(volume, int(np.floor(len(close_prices)/n)))

    # Obtain the Class Labels
    # if the next window end price > current window end price set label as 1 or buy at previous window
    labels = np.zeros((int(np.floor(len(close_prices)/n))-1,))
    for i in range(1,len(labels)+1):
        if split_prices[i][-1] > split_prices[i-1][-1]:
            labels[i-1] = 1

    # Obtain Statistical Features 
    from scipy import stats
    #print(stats.describe(split_prices[3]))
    features = np.zeros((len(labels),5))
    for i in range(len(labels)):
        features[i,0] = stats.describe(split_prices[i]).mean
        features[i,1] = stats.describe(split_prices[i]).variance
        features[i,2] = stats.describe(split_prices[i]).skewness
        features[i,3] = stats.describe(split_prices[i]).kurtosis
        features[i,4] = stats.describe(split_volumes[i]).mean

    # Perform Various ML Algorithms
    from PerformVariousML import PerformAnalysis
    accuracy[k],model_name = PerformAnalysis(features,labels,0.2,1)
    print('Window Size:', n, ' Max Accuracy of:', accuracy[k], ' from', model_name)
    k=k+1












