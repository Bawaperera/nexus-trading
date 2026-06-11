"""
NEXUS — Phase 1 Quick Test
Tests the Data Engine, Feature Engineer, and Risk Engine
using yfinance (no API key needed) with Apple stock.
"""

import sys
import os
sys.path.insert(0, "/home/claude/nexus")

import pandas as pd
import numpy as np

# ─── Test 1: Data Engine (yfinance — no API key needed) ───────────────────────
print("\n" + "="*60)
print("TEST 1: DATA ENGINE")
print("="*60)

from data.data_engine import DataEngine

engine = DataEngine(source="stock")
df = engine.fetch("AAPL", days=400)

print(f"\n✅ Fetched {len(df)} bars")
print(f"   Date range: {df.index[0].date()} → {df.index[-1].date()}")
print(f"   Columns: {list(df.columns)}")
print(f"\n   Latest 3 rows:")
print(df.tail(3)[["open", "high", "low", "close", "volume"]].to_string())

# ─── Test 2: Feature Engineering ──────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: FEATURE ENGINEERING")
print("="*60)

from data.feature_engineer import FeatureEngineer

fe = FeatureEngineer(target_horizon=1, target_pct=0.002)
features_df = fe.build(df)

feature_cols = [c for c in features_df.columns if c not in ["open", "high", "low", "close", "volume", "symbol", "timeframe"]]
print(f"\n✅ {len(feature_cols)} features engineered from {len(features_df)} valid rows")
print(f"\n   Feature categories:")

categories = {
    "Price action":      [c for c in feature_cols if any(x in c for x in ["return_", "body", "shadow", "gap", "hl_"])],
    "Volume":            [c for c in feature_cols if any(x in c for x in ["vol", "vwap", "ad_"])],
    "Trend":             [c for c in feature_cols if any(x in c for x in ["ema", "macd", "adx", "ichi"])],
    "Momentum":          [c for c in feature_cols if any(x in c for x in ["rsi", "stoch", "roc", "cci", "williams"])],
    "Volatility":        [c for c in feature_cols if any(x in c for x in ["atr", "bb_", "realized"])],
    "Market structure":  [c for c in feature_cols if any(x in c for x in ["swing", "higher", "bull_", "dist_from", "regime", "ema50"])],
    "Support/Resistance":[c for c in feature_cols if any(x in c for x in ["round_num", "52w"])],
    "Time":              [c for c in feature_cols if any(x in c for x in ["hour", "day_", "month", "session", "weekend", "sin", "cos"])],
    "Lags":              [c for c in feature_cols if "lag" in c],
}

for cat, cols in categories.items():
    if cols:
        print(f"   {cat:22s} [{len(cols):2d}]: {', '.join(cols[:4])}{'...' if len(cols) > 4 else ''}")

X, y = fe.get_targets(features_df)
print(f"\n   Target distribution: {y.value_counts().to_dict()}")
print(f"   Feature matrix shape: {X.shape}")

# ─── Test 3: Risk Engine ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3: RISK ENGINE")
print("="*60)

from risk.risk_engine import RiskEngine

risk = RiskEngine(account_size=1000, risk_pct=0.01, rr_ratio=2.0, max_positions=3)

# Simulate a trade signal
latest_close = float(df["close"].iloc[-1])
latest_atr   = float(features_df["atr_14"].iloc[-1])

print(f"\n   Simulating trade on AAPL:")
print(f"   Entry: ${latest_close:.2f} | ATR: ${latest_atr:.2f}")

order = risk.calculate_order(
    symbol="AAPL",
    signal=1,
    confidence=0.72,
    entry_price=latest_close,
    atr=latest_atr
)

print(f"\n   Order result:")
for k, v in order.items():
    if k not in ["symbol"]:
        print(f"   {k:30s}: {v}")

# Simulate a winning trade
risk.open_positions = 1
risk.update_pnl(+15.50, "WIN")
risk.update_pnl(-8.20, "LOSS")
risk.update_pnl(+22.10, "WIN")

stats = risk.get_stats()
print(f"\n   Performance stats:")
for k, v in stats.items():
    print(f"   {k:30s}: {v}")

print("\n" + "="*60)
print("✅ ALL PHASE 1 TESTS PASSED — NEXUS Foundation is ready")
print("="*60)
print("\nNext steps:")
print("  Phase 2 → Train XGBoost model + backtest on 2 years of data")
print("  Phase 3 → Add Binance crypto data (BTC/USDT)")
print("  Phase 4 → Paper trading bot")
print()
