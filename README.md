# LSTM + RSI Stock Price Prediction Algorithm
## Overview
This algorithm is designed to predict stock prices using a Long Short-Term Memory (LSTM) neural network, combined with the Relative Strength Index (RSI) for decision-making. It aims to predict future stock prices based on historical price data and determine whether to buy, sell, or hold the stock using momentum indicators like RSI.

The model takes 60 days of historical stock prices as input to predict the price for the next day. Trading decisions are based on these predictions and the momentum of the stock, identified using the RSI indicator.

## Algorithm Workflow
### 1. Data Collection
- The algorithm retrieves historical stock data using yfinance, a Python library that interfaces with Yahoo Finance to download stock data.
- For this project, the stock data for SPY (S&P 500 ETF) is downloaded for a specified time period.
### 2. Data Preprocessing
- The raw stock price data (closing prices) is scaled between 0 and 1 using MinMaxScaler from the scikit-learn library. Scaling ensures that the LSTM model can process the data efficiently without issues caused by large numerical values.
- After scaling, the data is transformed into sequences of 60 time steps. Each sequence represents the stock's closing prices over 60 days, which the model will use to predict the price on the 61st day.
### 3. LSTM Model Architecture
- The model is built using TensorFlow's Keras API. It consists of two LSTM layers followed by a dense layer that outputs the predicted stock price.
  - First LSTM Layer: 50 neurons, that outputs the entire sequence to pass the sequence to the next layer.
  - Second LSTM Layer: 50 neurons, outputs the final hidden state, which is the model's final understanding of the 60 day input data.
  - Dense Layer: A fully connected output layer with 1 neuron to predict the stock price for the next day.
  - The model is compiled using the Adam optimizer and the Mean Squared Error (MSE) loss function.
    - Adam Optimizer: Adaptive Moment Estimation, a widely used optimizer that adjusts the learning rate based on the gradients of the loss function.
    - Mean Squared Error (MSE): A regression-based loss function that measures the difference between the predicted and actual values.
### 5. Training the LSTM Model
- The dataset is split into training (80%) and testing (20%) sets. The model is trained on the training data and validated using the test data.
- The model is trained for 10 epochs with a batch size of 32.
### 6. Predicting Stock Prices
- After training, the LSTM model is used to predict the stock prices for the test set (the last 20% of the dataset).
- The predicted values are inverse transformed from the scaled format back to the original price scale.
### 7. Relative Strength Index (RSI) Calculation
- RSI is used as a momentum indicator to help determine whether the stock is overbought or oversold:
  - RSI > 70: Overbought (potential sell signal).
  - RSI < 30: Oversold (potential buy signal).
  - The RSI is calculated over a 14-day window to identify trends in stock momentum.
### 8. Decision-Making Using LSTM and RSI
- Buying Logic: The algorithm buys stock (10% of current portfolio value) when the LSTM model predicts an upward price movement for the next day (above 5%) and the RSI indicates an oversold condition (below 40).
- Selling Logic: The algorithm sells stock (100 shares) when the LSTM model predicts a price drop for the next day (below 5%) and the RSI indicates an overbought condition (above 70).
### 9. Backtesting
- The algorithm uses Backtrader, a Python library for backtesting trading strategies, to simulate the stock trading strategy over a defined historical period.
- It starts with an initial cash value of $10,000 and simulates buying and selling based on the LSTM + RSI strategy.
- The algorithm tracks the portfolio value and calculates the percentage change in the portfolio after backtesting.

## Results
The algorithm was backtested on SPY (S&P 500 ETF) stock data from January 1, 2015, to July 1, 2024. The results are approximate, as every time the program is run, the model may learn slightly differently due to the stochastic nature of the training process.

### Starting Portfolio Value: $10,000.00
### Final Portfolio Value: $24,540.60
### Percentage Change: +145.41%


## Model Predictions vs SPY Actual Price
![SPY Model Predictions](https://github.com/user-attachments/assets/93fd9ac5-b3ee-4808-9950-c460b58883f6)

## Backtesting Results
![Algorithm Backtesting SPY](https://github.com/user-attachments/assets/b61612d7-c152-4aac-9d91-9a83ac41db4f)


