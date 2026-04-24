# sports-nlp-forecasting
Sports, Sentiment, and Stock Returns
A Student Research Project in Business Analytics

A machine learning pipeline that predicts next-day stock returns for sports-sector companies by combining traditional price/technical features with NLP sentiment signals from four data sources. Built and evaluated on Nike (NKE) and DraftKings (DKNG) over the 2020–2023 period.

---

## Results at a Glance

| Metric | NKE (Nike) | DKNG (DraftKings) |
|---|---|---|
| **Sharpe Ratio** | 1.258 | 2.208 |
| **Win Rate** | 60.4% | 63.3% |
| **Total Return** | +27.96% | +69.75% |
| **Max Drawdown** | 11.27% | 21.06% |
| **Buy & Hold Benchmark** | −9.1% | +95.9 |
| **Total Trades** | 49 | 49 |
| **Transaction Fees** | 0.1% per trade | 0.1% per trade |

Both models significantly outperform their buy-and-hold benchmarks over the 2023 test period.

---

## Project Overview

This project answers the question: **can NLP sentiment signals from news, earnings filings, and search trends improve stock return predictions beyond what price data alone can achieve?**

The pipeline combines:
- **7 price/technical features** (momentum, volatility, intraday range, volume)
- **11 NLP sentiment features** from four sources (GDELT, SEC EDGAR, Google Trends, sports-keyword-filtered headlines)

A **Random Forest model** is trained via the [Darts](https://github.com/unit8co/darts) time series library and evaluated through walk-forward historical forecasting. Backtesting is performed using [VectorBT](https://vectorbt.pro/) with realistic transaction costs.

---

## NLP Signal Sources

| Source | Signal | Features |
|---|---|---|
| **GDELT API** | General news headlines → FinBERT | `news_sentiment`, `news_ema5`, `news_momentum` |
| **GDELT (filtered)** | Sports-keyword headlines → FinBERT | `sports_sentiment`, `sports_ema5`, `sports_momentum` |
| **SEC EDGAR** | 10-Q / 10-K MD&A text → FinBERT | `earnings_sentiment`, `earnings_ema5` |
| **Google Trends** | Sports-related search interest | `trends_score`, `trends_ema5`, `trends_momentum` |

[FinBERT](https://huggingface.co/ProsusAI/finbert) (a BERT model fine-tuned on financial text) scores each text source as positive, negative, or neutral with a confidence score. EMA-smoothed and momentum variants are computed for each raw signal.

---

## Model Architecture

```
Yahoo Finance (OHLCV)
    └── 7 technical features

GDELT API → FinBERT
    └── news_sentiment / ema5 / momentum
    └── sports_sentiment / ema5 / momentum

SEC EDGAR → FinBERT
    └── earnings_sentiment / ema5

Google Trends (pytrends)
    └── trends_score / ema5 / momentum

         ↓  All 18 features  ↓
      StandardScaler
         ↓
      Darts SKLearnModel
      (RandomForest, lags=10,
       covariate lags: -1,-2,-3,-5,-10,-20)
         ↓
      Walk-forward historical forecasts
         ↓
      Signal threshold: ±0.5σ
         ↓
      VectorBT backtest (fees=0.001)
         ↓
      Performance metrics
```

---

## Hyperparameter Tuning

A 27-combination grid search across `n_estimators` [200, 400, 600], `max_depth` [4, 6, 8], and `min_samples_leaf` [5, 10, 20] was run for each ticker, optimised by Sharpe ratio with 0.1% fees applied throughout.

| Ticker | n_estimators | max_depth | min_leaf | Sharpe | Win Rate | Return |
|---|---|---|---|---|---|---|
| NKE | 200 | 8 | 10 | 1.993 | 60.4% | +16.12% |
| DKNG | 600 | 8 | 20 | 2.208 | 63.3% | +69.75% |

The different optimal `n_estimators` between tickers reflects their different volatility profiles — DKNG's higher price volatility benefits from more trees to average out noise.

---

## Ablation Study

A `FwdReturn_5d` feature (5-day rolling return sum) was tested and found to hurt performance across every metric. It was permanently removed from the pipeline.

| Metric | With FwdReturn_5d | Without | Change |
|---|---|---|---|
| Sharpe Ratio | 0.479 | 0.826 | **+0.347** |
| Win Rate | 50.9% | 56.6% | **+5.7pp** |
| Total Return | +5.48% | +11.48% | **+6.0pp** |
| Max Drawdown | 17.02% | 16.60% | −0.42pp |

---

## Key Findings

- **NLP signals improve performance** — removing them reduces Sharpe ratio and win rate on both tickers compared to a price-only baseline
- **DraftKings benefits more from NLP** — Google Trends momentum ranks 7th in DKNG feature importance vs negligible for NKE, confirming that sports-specific signals are more informative for companies directly tied to sports events
- **Earnings sentiment EMA5 is the strongest NLP feature for NKE** — the 5-day smoothed trend in management language tone from 10-Q/10-K filings carries durable predictive signal across weeks
- **Optimal hyperparameters differ by ticker** — DKNG requires more trees (600 vs 200) to handle its higher price volatility
- **Walk-forward backtesting with realistic fees confirms robust out-of-sample results** on both tickers

---

## Repository Structure

```
├── darts_forecasting_nlp_v5_NKE.ipynb    # Full pipeline for Nike (NKE)
├── darts_forecasting_nlp_v5_DKNG.ipynb   # Full pipeline for DraftKings (DKNG)
└── README.md
```

Each notebook is self-contained and ticker-agnostic. Changing `TICKER` and `COMPANY` in the config cell is all that is needed to run a new stock.

---

## How to Run

### Requirements
- Google Colab (recommended) or a local Python environment
- Google Drive (for caching — optional but strongly recommended)

### Steps

1. Open either notebook in Google Colab
2. Mount Google Drive when prompted (caches data to avoid re-fetching)
3. Update `EDGAR_EMAIL` in the config cell with your email address (required by SEC EDGAR)
4. Run all cells in order

The first run fetches all data from GDELT, SEC EDGAR, and Google Trends and caches results as `.parquet` files in Google Drive. Subsequent runs load from cache and complete in minutes.

### Cell-by-Cell Overview

| Cell | Description |
|---|---|
| `cell_install` | Installs dependencies (Colab only) |
| `cell_drive_mount` | Mounts Google Drive for caching |
| `cell_config` | **Edit this cell to change ticker** |
| `cell_imports` | Loads libraries and FinBERT model |
| `cell_data` | Downloads price data, engineers technical features |
| `cell_gdelt` | Fetches general news sentiment (GDELT + FinBERT) |
| `cell_sports` | Filters sports-specific headlines from GDELT cache |
| `cell_trends` | Fetches Google Trends search interest |
| `cell_earnings` | Downloads and scores SEC EDGAR filings |
| `cell_merge` | Joins all features, applies StandardScaler |
| `cell_timeseries` | Wraps data in Darts TimeSeries objects |
| `cell_model` | Trains Random Forest, generates walk-forward forecasts |
| `cell_hyperparam` | 27-combination hyperparameter grid search |
| `cell_nlp_comparison` | NLP vs price-only comparison (measures NLP contribution) |
| `cell_importance` | Feature importance chart |
| `cell_signals` | Converts forecasts to long/short trading signals |
| `cell_backtest` | VectorBT backtest with 0.1% fees |
| `cell_plot` | Equity curve visualisation |
| `cell_sentiment_plot` | Sentiment signals over time |
| `cell_sentiment_returns` | Sentiment vs stock price twin-axis chart |

---

## Dependencies

```
numpy>=1.26,<2.0
darts
vectorbt
transformers
pytorch-lightning
pytrends
sec-edgar-downloader
yfinance
scikit-learn
pandas
matplotlib
```

All dependencies are installed automatically by `cell_install` when running in Google Colab.

---

## Limitations

- Results are based on one test period (March–December 2023). A longer evaluation window would provide more robust evidence.
- GDELT news coverage was sparse for NKE (~23% of trading days had scored headlines). DKNG likely has better coverage given its direct connection to sports events.
- Google Trends returns weekly data, which is forward-filled to daily frequency.
- 0.1% per-trade fees are applied as a realistic bid-ask spread estimate, but no position sizing or slippage modelling is included.

---

## Technologies

| Tool | Role |
|---|---|
| [Darts](https://github.com/unit8co/darts) | Time series ML framework, SKLearnModel wrapper |
| [VectorBT](https://vectorbt.pro/) | Vectorized backtesting and portfolio simulation |
| [FinBERT](https://huggingface.co/ProsusAI/finbert) | Financial sentiment classification |
| [GDELT](https://www.gdeltproject.org/) | Global news headlines API |
| [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) | 10-Q / 10-K earnings filings |
| [pytrends](https://github.com/GeneralMills/pytrends) | Google Trends unofficial API |
| [yfinance](https://github.com/ranaroussi/yfinance) | Yahoo Finance price data |
| scikit-learn | RandomForestRegressor |
| Google Colab | Cloud compute environment |

---

*DATA 491 · Undergraduate Research Day · Business and Data Science*
