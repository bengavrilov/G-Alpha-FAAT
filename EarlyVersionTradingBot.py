import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

# Obtain MSFT 
msft = yf.Ticker("MSFT")
#print('Microsoft Information:')
#print(msft)
#print('')

# get historical market data, here max is 5 years.
stock_prices = msft.history(period="max")
#print('Microsoft Stock Prices:')
#print(stock_prices)
#print('')

# Get the close price for every day 
close_prices = stock_prices.iloc[:,3].values
dates = np.arange(len(close_prices)) # make an array for the days to plot

# moving average calculation
MA = np.convolve(close_prices, np.ones(10), 'valid') / 10
#print(MA)

# Plot 
#plt.plot(dates,close_prices)
#plt.plot(dates[8:-1],MA)
plt.title('Microsoft Price')
plt.xlabel('Day')
plt.ylabel('Price($)')
plt.legend('Original Stock Price, 10-day MA')
#plt.show()

# partition the total prices into n-block sizes
n = 30
split_prices = np.array_split(close_prices, int(np.floor(len(close_prices)/30)))
print('The Window Seperated Stock Prices:')
print(split_prices)
print('')

# Obtain the Class Labels
# if the next window end price > current window end price set label as 1 or buy at previous window
labels = np.zeros((int(np.floor(len(close_prices)/30))-1,))
for i in range(1,len(labels)+1):
    if split_prices[i][-1] > split_prices[i-1][-1]:
       labels[i-1] = 1
print('The Labels are:')
print(labels)
print('')

# Obtain Statistical Features 
from scipy import stats
print(stats.describe(split_prices[3]))
features = np.zeros((len(labels),4))
for i in range(len(labels)):
    features[i,0] = stats.describe(split_prices[i]).mean
    features[i,1] = stats.describe(split_prices[i]).variance
    features[i,2] = stats.describe(split_prices[i]).skewness
    features[i,3] = stats.describe(split_prices[i]).kurtosis
#print('Features:')
#print(features)

# Feature Scaling
from sklearn.preprocessing import StandardScaler #Standardization transforms each column to have a value between -3 and +3 (gaussian normalization)
sc = StandardScaler()
features = sc.fit_transform(features) 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split #split data
x_train, x_test, y_train, y_test = train_test_split(features,labels,test_size = 0.2)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
acc_score = accuracy_score(y_test,y_pred)
print(acc_score)




a = 5







