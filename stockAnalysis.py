import pandas as pd
import yfinance as yf    #### Yahoo Finance Api ####
import numpy as np
import matplotlib.pyplot as mplt

#preamble set tickers
#tickerG = yf.Ticker('GOOGL').info
#tickerA = yf.Ticker('APPL').info
#tickerM = yf.Ticker('MSFT').info
#tickerT = yf.Ticker('TSM').info
start_date = '2020-01-01'
end_date = '2023-01-01'
dataG = yf.download('GOOGL', start_date, end_date)
#dataA = yf.download(tickerA, start_date, end_date)
#dataM = yf.download(tickerM, start_date, end_date)
#dataT = yf.download(tickerT, start_date, end_date)


print(dataG)
#print(tickerT.keys())
#market_price = tickerG['regularMarketPrice']
#previous_close_price = tickerG['regularMarketPreviousClose']
#print('Ticker: GOOGL')
#print('Market Price: ', market_price)
#print('Previous Close Price: ', previous_close_price)
##df = pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/AAPL_data.csv')
#df.head()