# Make sure to install these libraries before running the program:
# !pip install yfinance tensorflow backtrader matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Step 1: Data Collection (using yfinance)

ticker = 'SPY'
# Define periods for training and backtesting
train_start_date = '2013-01-01'
train_end_date = '2020-12-31'  

# Backtest on unseen recent data
backtest_start_date = '2015-01-01'
backtest_end_date = '2024-07-01'

# Download historical stock data for training (LSTM)
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Fetch data for training the LSTM (2013-2020)
stock_data_train = download_stock_data(ticker, train_start_date, train_end_date)

# Fetch data for backtesting the strategy (2022-2024)
stock_data_backtest = download_stock_data(ticker, backtest_start_date, backtest_end_date)

# Step 2: Preprocessing and Feature Engineering (Training Data for LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_train = scaler.fit_transform(stock_data_train['Close'].values.reshape(-1, 1))

# Create sequences for LSTM (using look_back window)
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# Set the look_back window (e.g., 60 days)
look_back = 60
X, y = create_sequences(scaled_data_train, look_back)

# Reshape X to be [samples, time steps, features] for LSTM input
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Step 3: Building and Training the LSTM Model

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # Output layer (predicting 1 value)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Split the data into training and testing sets (80/20 split for LSTM training)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the LSTM model
with open('training_log.txt', 'w') as f:
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2, callbacks=[tf.keras.callbacks.CSVLogger('log.csv')])

# Step 4: Make Predictions and Evaluate Model

# Predicting stock prices using test data
predicted_stock_price = model.predict(X_test, verbose=0)

# Inverse transform to get the actual price from the scaled prediction
predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))
real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))

# Prepare the dates for the plot from the original training data
dates = stock_data_train.index[-len(y_test):]  # Get dates for test set

# Plot the real vs predicted stock prices with proper time formatting
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.plot(dates, real_stock_price, color='red', label='Real Stock Price')
plt.plot(dates, predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Format the x-axis to show years or months
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Show every month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Month + year format
plt.gcf().autofmt_xdate()

plt.show()

# Step 5: Backtesting using Backtrader

# Define a custom Backtrader strategy using LSTM and RSI for decision-making
class LSTM_RSI_Strategy(bt.Strategy):
    params = (('lstm_threshold', 0.05),)  # LSTM threshold (e.g., 5%)

    def __init__(self, lstm_model):
        self.dataclose = self.datas[0].close  # Reference to closing prices
        self.lstm_model = lstm_model  # Pass the trained LSTM model
        self.look_back = 60  # The look-back period for input sequences

        # RSI Indicator
        self.rsi = bt.indicators.RelativeStrengthIndex(self.datas[0], period=14)

    def next(self):
        if len(self.dataclose) >= self.look_back:
            # Get the last 60 closing prices
            recent_prices = np.array(self.dataclose.get(size=self.look_back)).reshape(-1, 1)
            recent_prices_scaled = scaler.transform(recent_prices)  # Scale the data

            # Reshape for LSTM input [samples, time steps, features]
            X_input = np.reshape(recent_prices_scaled, (1, self.look_back, 1))

            # Make prediction for the next price
            predicted_price_scaled = self.lstm_model.predict(X_input, verbose=0)
            predicted_price = scaler.inverse_transform(predicted_price_scaled)

            # Buy/Sell logic based on predictions and RSI
            current_price = self.dataclose[0]
            price_diff = (predicted_price - current_price) / current_price  # Price difference as percentage

            # Buy if LSTM predicts upward movement and RSI shows oversold (below 40)
            if price_diff > self.params.lstm_threshold and self.rsi < 40:
                #if not self.position:  # Not in the market
                portfolio_value = self.broker.getvalue()  # Get total portfolio value
                allocation = 0.1  # Invest x% of portfolio
                invest_amount = portfolio_value * allocation  # Amount to invest
                current_price = self.dataclose[0]
                shares_to_buy = int(invest_amount / current_price)  # Calculate number of shares
                self.buy(size=shares_to_buy)  # Buy shares based on x% portfolio allocation

            # Sell if LSTM predicts downward movement and RSI shows overbought (above 70)
            elif price_diff < -self.params.lstm_threshold and self.rsi > 70:
                if self.position:  # Already in the market
                    self.sell()  # Sell if conditions hold
                # elif not self.position:
                #     shares_to_short = 100
                #     self.sell(size=shares_to_short)  # Sell shares short

# Prepare data for backtesting
class PandasData(bt.feeds.PandasData):
    pass

# Initialize the backtesting engine
cerebro = bt.Cerebro()

# Add stock data to backtesting engine
data_feed = PandasData(dataname=stock_data_backtest)
cerebro.adddata(data_feed)

# Add the strategy
cerebro.addstrategy(LSTM_RSI_Strategy, lstm_model=model)

# Set initial cash for backtesting
cerebro.broker.setcash(10000)

# Run the backtest
initial_value = cerebro.broker.getvalue()
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
final_value = cerebro.broker.getvalue()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
percentage_change = ((final_value - initial_value) / initial_value) * 100
print('Percentage Change: %.2f%%' % percentage_change)

# Plot the results
cerebro.plot()