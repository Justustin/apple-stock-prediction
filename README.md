# 📈 Stock Price Prediction with LSTM Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <img src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch Logo" width="250px">
</div>

## 🚀 Project Overview

This project implements a sophisticated **Long Short-Term Memory (LSTM)** neural network to predict Apple (AAPL) stock prices using PyTorch. The model analyzes historical stock data patterns to forecast future price movements, demonstrating the power of deep learning for financial time series analysis.

> *"Prediction is very difficult, especially if it's about the future."* - Niels Bohr

## ⭐ Key Features

- **Historical Data Analysis**: Fetches and processes Apple's stock data using `yfinance`
- **Advanced Data Preprocessing**: Applies normalization and sequence creation techniques
- **Deep Learning Architecture**: Implements multi-layer LSTM neural network with PyTorch
- **Visualization**: Presents model predictions vs. actual stock prices
- **Time Series Forecasting**: Predicts future stock price movements

## 🔧 Technical Implementation

### Data Acquisition & Processing

```python
# Fetch Apple stock data
ticker_symbol = 'AAPL'
ticker_data = yf.Ticker(ticker_symbol)
apple_data = ticker_data.history(start='2014-01-01', end=datetime.now())

# Normalize the data
def min_max_scalar(tensor):
    max_val, _ = torch.max(tensor, dim=0)
    min_val, _ = torch.min(tensor, dim=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    
    scaled_tensor = (tensor - min_val) / range_val
    return scaled_tensor, max_val, min_val
```

### Model Architecture

The LSTM model consists of multiple layers designed to capture complex temporal patterns:

| Layer | Description | Size |
|-------|-------------|------|
| LSTM 1 | First LSTM layer with dropout | 1 → 128 |
| LSTM 2 | Second LSTM layer | 128 → 64 |
| Dense 1 | First fully connected layer | 64 → 25 |
| Dense 2 | Output layer | 25 → 1 |

```python
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, dropout=0.2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dense1 = nn.Linear(64,25)
        self.dense2 = nn.Linear(25,1)
    
    def forward(self, x):
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x = x[:,-1,:]
        x = self.dense1(x)
        x = self.dense2(x)
        return x
```

## 📊 Results

The model demonstrates promising results in capturing the general trends and patterns in Apple's stock movements. The visualization below shows the actual stock prices alongside the model's predictions:

![Stock Prediction Visualization](https://example.com/stock_prediction.png)

## 🔍 Future Enhancements

- [ ] **Feature Engineering**: Add technical indicators (RSI, MACD, etc.)
- [ ] **Architecture Tuning**: Experiment with bidirectional LSTMs and attention mechanisms
- [ ] **Multi-Stock Analysis**: Expand to predict multiple correlated stocks
- [ ] **Sentiment Integration**: Incorporate news and social media sentiment
- [ ] **Ensemble Methods**: Combine predictions from multiple model architectures

## ⚠️ Disclaimer

**This project is for educational and research purposes only.** The predictions made by this model should not be used for actual trading decisions. Stock market investments involve substantial risk, and predictions based solely on historical data cannot guarantee future performance.

## 📦 Requirements

```
pytorch >= 1.9.0
yfinance >= 0.1.63
numpy >= 1.19.5
pandas >= 1.3.0
matplotlib >= 3.4.2
seaborn >= 0.11.1
```

## 📄 License

This project is available under the MIT License. See the LICENSE file for more details.

---

<div align="center">
  <p><i>If you found this project helpful, consider giving it a ⭐ on GitHub!</i></p>
</div> 
