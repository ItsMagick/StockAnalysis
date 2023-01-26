import math
from datetime import date
from sklearn.preprocessing import MinMaxScaler      #### scicit-learn to normalize data ####
from keras.layers import LSTM, Dense                ####LSTM as model to train time sensitive data and make reliable data analysis in this training
from keras.models import Sequential
import pandas as pd
import yfinance as yf                               #### Yahoo Finance Api ####
import numpy as np
import matplotlib.pyplot as mplt


#---------------- preamble set tickers ----------------
tickerG = 'GOOGL'
tickerA = 'AAPL'
tickerM = 'MSFT'
tickerT = 'TSM'
start_date = '2020-01-01'
end_date = date.today().strftime("%Y-%m-%d")
dataG = yf.download(tickerG, start_date, end_date)
dataA = yf.download(tickerA, start_date, end_date)
dataM = yf.download(tickerM, start_date, end_date)
dataT = yf.download(tickerT, start_date, end_date)
#--------------- uncomment if u need csv initially ------------------
#dataG.to_csv("GOOGL.csv")
#dataG.to_csv("APPL.csv")
#dataG.to_csv("MSFT.csv")
#dataG.to_csv("TSM.csv")

#print(dataG)
#print(yf.Ticker(tickerT).info.keys())
#market_price = tickerG['regularMarketPrice']
#previous_close_price = tickerG['regularMarketPreviousClose']
#print('Ticker: GOOGL')
#print('Market Price: ', market_price)
#print('Previous Close Price: ', previous_close_price)
##df = pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/AAPL_data.csv')
#df.head()

#mplt.figure(figsize=(16,8))
#mplt.title('Close Price History')
#mplt.plot(dataG['Close'])
#mplt.xlabel('Date', fontsize = 18)
#mplt.ylabel('Close Price USD($)', fontsize=18)
#mplt.show()
#---------------- prescale for training ----------------
data_filtered = dataG.filter(['Close'])
data_set = data_filtered.values

training_data_len = math.ceil(len(data_set) * .8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_training_data = scaler.fit_transform(data_set)

#---------------- training ----------------

train_data = scaled_training_data[0:training_data_len, :]

x_train = [] #### training features // independent data ####
y_train = [] #### target data // dependent data ####

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#--------------- reshaping data ---------------
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss= 'mean_squared_error')

#--------------- train Model ----------------
model.fit(x_train, y_train, batch_size=1, epochs=1)

#--------------- verification ---------------

verification = scaled_training_data[training_data_len - 60:, :]
x_test = []
y_test = data_set[training_data_len:, :]
for i in range(60, len(verification)):
    x_test.append(verification[i-60:i, 0])

x_test = np.array(x_test)

#---------------- reshape------------------

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


#----------------- predictions -----------------


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


#-------------- evaluate model ----------------

rm = np.sqrt(np.mean(predictions - y_test)**2)
