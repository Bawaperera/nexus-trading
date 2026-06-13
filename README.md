# NEXUS - AI Trading System

> An end-to-end algorithmic trading system using XGBoost, real-time news sentiment, and professional risk management. Built for educational purposes and paper trading.

---

## What it does

NEXUS watches the crypto market 24/7, reads live news, and decides whether to BUY, SELL, or HOLD — automatically.

```
Binance WebSocket  →  102 features  →  XGBoost model  →  Sentiment check  →  Risk engine  →  Trade
live price stream     engineered        61.7% BUY?        Fear/Greed: 12      size + SL/TP    paper or live
```

**Every 1-hour candle close triggers this full pipeline in under 1 second.**

---

## Key features

- **Real-time data** — Binance WebSocket for tick-by-tick BTC/ETH/SOL prices
- **102 engineered features** — RSI, MACD, ATR, Bollinger Bands, volume, market structure, time, lags
- **XGBoost ML model** — trained with 5-fold walk-forward validation (no lookahead bias)
- **Live news sentiment** — CoinDesk, CoinTelegraph RSS + Fear & Greed Index
- **Signal fusion** — 80% model + 20% sentiment, with sentiment veto on extreme readings
- **Professional risk engine** — 1% risk per trade, ATR-based stop loss, Kelly criterion, daily loss limit
- **Paper trading mode** — trade with fake money on real market data before going live
- **Backtest engine** — simulate 5 years of trades with full metrics (Sharpe, drawdown, profit factor)
- **React dashboard** — live equity curve, signal output, trade journal

---

## Backtest results (5 years BTC daily, out-of-sample)

| Metric | Value | Target for live |
|---|---|---|
| Total return | +7.98% | — |
| Win rate | 41.8% | > 50% |
| Profit factor | 1.11× | > 1.5× |
| Max drawdown | -9.92% | < 15% |
| Sharpe ratio | 0.256 | > 1.0 |
| Trades | 122 | — |

> ⚠️ These results are on daily BTC data. Hourly Binance data (live mode) performs better due to more signal per unit time.

---

## Project structure

```
nexus/
├── data/
│   ├── data_engine.py        # Fetch OHLCV from Binance or yfinance
│   ├── feature_engineer.py   # Compute 102 technical + structural features
│   ├── realtime_stream.py    # Binance WebSocket live price stream
│   └── news_collector.py     # RSS news + Fear & Greed sentiment scoring
│
├── models/
│   └── model_trainer.py      # XGBoost training + walk-forward validation
│
├── backtest/
│   └── backtester.py         # Historical trade simulation + performance metrics
│
├── signals/
│   └── signal_engine.py      # Fuse model output + sentiment → BUY/SELL/HOLD
│
├── risk/
│   └── risk_engine.py        # Position sizing, stop loss, take profit, circuit breakers
│
├── execution/
│   └── paper_trader.py       # Track open/closed positions + CSV journal
│
├── dashboard/
│   └── NEXUSDashboard.jsx    # React dashboard (equity curve, signals, journal)
│
├── config.py                 # Central config (reads from .env)
├── orchestrator.py           # Main loop — ties all 6 layers together
├── .env.example              # API key template
└── requirements.txt
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/nexus-trading.git
cd nexus-trading
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Open .env and add your keys
```

| Key | Required | Where to get it |
|---|---|---|
| `BINANCE_API_KEY` | For live/paper trading | [Binance API Management](https://www.binance.com/en/my/settings/api-management) |
| `BINANCE_API_SECRET` | For live/paper trading | Same page |
| `CRYPTOPANIC_TOKEN` | Optional (better news) | [CryptoPanic API](https://cryptopanic.com/developers/api/) |

> **Binance API permissions:** Enable "Read" and "Enable Spot & Margin Trading" only. Never enable withdrawals.

### 3. Run the backtest first

```bash
python -c "
from data.data_engine import DataEngine
from data.feature_engineer import FeatureEngineer
from models.model_trainer import ModelTrainer
from backtest.backtester import Backtester

engine = DataEngine(source='stock')
df = engine.fetch('BTC-USD', days=1825)

fe = FeatureEngineer()
feat = fe.build(df)
X, y = fe.get_targets(feat)

trainer = ModelTrainer(n_splits=5)
results = trainer.walk_forward_validate(X, y)
model = trainer.train_final(X, y)
trainer.save(model, 'nexus_xgb.pkl')

bt = Backtester(initial_capital=1000)
bt_results = bt.run(feat.loc[results['oof_predictions'].index],
                    results['oof_predictions'],
                    results['oof_probabilities'])
bt.print_report(bt_results)
"
```

### 4. Start paper trading (safe — no real money)

```bash
python orchestrator.py
```

This connects to Binance WebSocket, runs inference on every candle close, and logs all trades to `logs/trade_journal.csv`. **Only switch `TRADING_MODE=live` after 30+ profitable paper trading days.**

---

## How the signal works

```
1. Candle closes on Binance
2. Feature engineer computes 102 features from rolling buffer
3. XGBoost predicts: BUY 61.7% / SELL 29.8% / HOLD 8.5%
4. News collector reads: Fear & Greed = 12 (Extreme Fear)
5. Signal engine: sentiment is -0.27 → HOLD (confidence too low, sentiment vetoes BUY)
6. Risk engine: skips trade (confidence < 58% threshold)
7. Waits for next candle
```

The system never trades blindly. Both the model and sentiment must agree before entering.

---

## Risk rules (non-negotiable)

```
✅ Max 1% account risk per trade
✅ ATR-based stop loss (adapts to current volatility)
✅ 2:1 minimum reward:risk ratio
✅ Max 3 simultaneous open positions
✅ Daily loss limit: -5% stops all trading for the day
✅ Confidence threshold: model must be > 58% sure
❌ Never risk more than 2% per trade
❌ Never trade live without 30+ profitable paper days
```

---

## Roadmap

- [x] Phase 1 — Data pipeline + 102 features
- [x] Phase 2 — XGBoost model + walk-forward validation
- [x] Phase 3 — Backtest engine + paper trader
- [x] Phase 4 — Real-time news + sentiment engine
- [x] Phase 5 — React dashboard
- [ ] Phase 6 — Hourly Binance data + regime filter
- [ ] Phase 7 — LSTM magnitude model (complement XGBoost direction model)
- [ ] Phase 8 — On-chain data (whale flows, funding rates, exchange inflows)
- [ ] Phase 9 — Telegram alerts + mobile notifications
- [ ] Phase 10 — Live trading (after proven paper track record)

---

## Disclaimer

This project is for **educational purposes only**. Algorithmic trading involves significant financial risk. Past backtest performance does not guarantee future results. Never invest money you cannot afford to lose. Always start with paper trading.

---

## Built with

Python · XGBoost · pandas · ta · CCXT · websockets · feedparser · React · Recharts

---

*Built by [Bawantha Perera](https://github.com/Bawaperera) — CS undergraduate at IIT Sri Lanka*
