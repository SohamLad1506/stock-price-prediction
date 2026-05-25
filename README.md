📈 Stock Price Prediction — End-to-End Data Science Project

Predicting next-day closing prices for AAPL (Apple Inc.) using Linear Regression and Random Forest, with full EDA and feature engineering.


🗂️ Project Structure
stock-price-prediction/
├── stock_prediction.py     # Main script — data → features → models → plots
├── eda_plots.png           # EDA visualisations (price, volume, heatmap)
├── model_evaluation.png    # Predicted vs actual + feature importances
└── README.md

🔍 Overview
ItemDetailAssetAAPL (Apple Inc.)Period2019 – 2024Data sourceYahoo Finance via yfinanceTaskPredict next-day closing priceModelsLinear Regression · Random ForestSplit80% train / 20% test (chronological)

⚙️ Features Engineered
FeatureDescriptionMA7 / MA307-day & 30-day rolling mean of CloseDaily_ReturnDaily percentage change of CloseVolatility_1414-day rolling std of daily returnsLag1 / Lag2 / Lag5Lagged closing prices (1, 2, 5 days)TargetNext day's closing price (label)

📊 Results
ModelRMSEMAEMAPELinear Regression2.311.791.21% ✅Random Forest34.2522.7212.90%
EDA
Show Image
Model Evaluation
Show Image

🚀 Quick Start
bash# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction

# 2. Install dependencies
pip install yfinance pandas numpy matplotlib seaborn scikit-learn

# 3. Run
python stock_prediction.py

💡 Key Findings

Lag-1 (yesterday's close) is the single most predictive feature — consistent with stock prices following a near-random walk.
Linear Regression significantly outperforms Random Forest because RF cannot extrapolate beyond price ranges seen in training (e.g. $40 in 2019 vs $180+ in 2022).
Volume shows periodic spikes coinciding with earnings announcements and macro events.

⚠️ Limitations

No external signals (earnings, macro indicators, news sentiment).
Linear Regression's low MAPE largely reflects the near-unit-root nature of prices ("tomorrow ≈ today"), not genuine predictive power.
Random Forest needs percentage-based (return) targets or normalisation to handle different price regimes.

🔭 Future Work

 Predict daily returns instead of raw prices
 Add LSTM / Transformer model on return sequences
 Incorporate sentiment analysis from financial news
 Add macro features (VIX, interest rates, S&P 500)


🛠️ Tech Stack
Python 3 · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · yfinance

📄 License
MIT — free to use and adapt.
