import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LassoCV
from scipy.optimize import minimize

def download_and_calculate_returns(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close'].pct_change(fill_method=None).dropna()

def optimize_portfolio(returns):
    X = returns[:-1]
    y = returns.shift(-1).iloc[:-1].mean(axis=1)
    lasso = LassoCV(cv=5)
    lasso.fit(X, y)
    initial_weights = np.abs(lasso.coef_ / np.sum(np.abs(lasso.coef_)))

    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(returns.cov(), weights))

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0, 1) for _ in range(len(returns.columns))]
    result = minimize(portfolio_variance, initial_weights, bounds=bounds, constraints=constraints, method='SLSQP')
    return result.x

def calculate_ticker_returns(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = {}
    for ticker in tickers:
        start_price = data[ticker].iloc[0]
        end_price = data[ticker].iloc[-1]
        returns[ticker] = (end_price - start_price) / start_price
    return returns

def backtest_portfolio(tickers, start_date, end_date, weights):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    portfolio_return = 0
    for ticker, weight in zip(tickers, weights):
        start_price = data[ticker].iloc[0]
        end_price = data[ticker].iloc[-1]
        if weight > 0.01:
            print(ticker, start_price)
            print(ticker, end_price)
            ticker_return = (end_price - start_price) / start_price
            print(ticker, ticker_return)
            portfolio_return += weight * (ticker_return * 100)
    return portfolio_return

# Define tickers and date ranges
tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JNJ', 'JPM',
    'V', 'UNH', 'HD', 'MA', 'PFE', 'KO', 'PEP', 'DIS', 'INTC',
    'VZ', 'T', 'CVX', 'XOM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BA',
    'MMM', 'GE', 'F', 'GM', 'IBM', 'ORCL', 'CRM', 'ADBE', 'NFLX', 'PYPL',
    'SQ', 'UBER', 'LYFT', 'ZM', 'SNAP', 'AXP', 'SBUX', 'MCD', 'NKE', 'SPY'
]

training_start_date = '2020-05-10'
training_end_date = '2023-12-31'
testing_start_date = '2024-01-05'
testing_end_date = '2024-08-07'

# Training phase
training_returns = download_and_calculate_returns(tickers, training_start_date, training_end_date)
optimal_weights = optimize_portfolio(training_returns)

# Testing phase
test_result = backtest_portfolio(tickers, testing_start_date, testing_end_date, optimal_weights)

# Filter and print weights greater than 0.01 with ticker names
print("Optimal Weights for Tickers with > 0.01 Weight:")
for ticker, weight in zip(tickers, optimal_weights):
    if weight > 0.01:
        print(f"{ticker}: {weight:.4f}")

print("Backtest Result: Cumulative return at the end of the period:", test_result)

