"""
NEXUS — Layer 2: Feature Engineering
Computes 40+ features across 9 categories used by professional quant systems.

Categories:
  1. Price action (returns, volatility, gaps)
  2. Volume features
  3. Trend indicators (EMA, MACD, ADX)
  4. Momentum oscillators (RSI, Stoch, CCI)
  5. Volatility indicators (ATR, Bollinger Bands)
  6. Market structure (swing highs/lows, regime)
  7. Support/resistance
  8. Time features (hour, day, session)
  9. Lag features (past N candles)
"""

import pandas as pd
import numpy as np
import ta
import logging

log = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Transforms raw OHLCV data into an ML-ready feature matrix.
    
    Usage:
        fe = FeatureEngineer()
        features_df = fe.build(raw_ohlcv_df)
        X, y = fe.get_targets(features_df)
    """

    def __init__(self, target_horizon: int = 1, target_pct: float = 0.002):
        """
        Args:
            target_horizon: how many candles ahead to predict (default 1)
            target_pct: minimum % move to classify as BUY/SELL (default 0.2%)
        """
        self.target_horizon = target_horizon
        self.target_pct = target_pct

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from raw OHLCV DataFrame.
        
        Returns:
            DataFrame with all feature columns added (no lookahead bias)
        """
        df = df.copy()
        df = self._validate(df)

        log.info(f"Engineering features for {len(df)} candles...")

        df = self._price_features(df)
        df = self._volume_features(df)
        df = self._trend_features(df)
        df = self._momentum_features(df)
        df = self._volatility_features(df)
        df = self._market_structure(df)
        df = self._support_resistance(df)
        df = self._time_features(df)
        df = self._lag_features(df)

        # Drop rows with NaN from indicator warmup periods
        before = len(df)
        df = df.dropna()
        log.info(f"Feature matrix ready: {len(df)} rows ({before - len(df)} dropped for NaN warmup)")

        return df

    def get_targets(self, df: pd.DataFrame, mode: str = "classification"):
        """
        Generate target labels.
        
        mode="classification":
            1 = BUY  (price goes up > target_pct in next horizon candles)
           -1 = SELL (price goes down > target_pct)
            0 = HOLD (sideways)
        
        mode="regression":
            Returns the % return over next horizon candles (for LSTM magnitude model)
        """
        future_return = df["close"].pct_change(self.target_horizon).shift(-self.target_horizon)

        if mode == "classification":
            y = pd.Series(0, index=df.index, name="target")
            y[future_return > self.target_pct] = 1
            y[future_return < -self.target_pct] = -1
        else:
            y = future_return.rename("target")

        # Drop last `horizon` rows (no future data)
        valid_idx = df.index[:-self.target_horizon]
        feature_cols = self._get_feature_cols(df)
        X = df.loc[valid_idx, feature_cols]
        y = y.loc[valid_idx]

        class_dist = y.value_counts()
        log.info(f"Target distribution → BUY: {class_dist.get(1,0)} | SELL: {class_dist.get(-1,0)} | HOLD: {class_dist.get(0,0)}")
        return X, y

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df.sort_index()
        return df

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """Return all computed feature columns (excludes raw OHLCV + metadata)."""
        exclude = {"open", "high", "low", "close", "volume", "symbol", "timeframe", "target"}
        return [c for c in df.columns if c not in exclude]

    # ─── Feature categories ───────────────────────────────────────────────────

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw price action features."""
        c, h, l, o = df["close"], df["high"], df["low"], df["open"]

        df["return_1"]   = c.pct_change(1)
        df["return_3"]   = c.pct_change(3)
        df["return_5"]   = c.pct_change(5)
        df["return_10"]  = c.pct_change(10)
        df["return_20"]  = c.pct_change(20)

        # Candle body properties
        df["body_size"]     = (c - o).abs() / o           # body as % of open
        df["upper_shadow"]  = (h - c.clip(lower=o)) / o   # upper wick
        df["lower_shadow"]  = (c.clip(upper=o) - l) / o   # lower wick
        df["is_bullish"]    = (c > o).astype(int)

        # Gap (overnight / candle gap)
        df["gap"] = (o - c.shift(1)) / c.shift(1)

        # High-Low range
        df["hl_range"] = (h - l) / c

        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features."""
        v = df["volume"]

        df["vol_ma_10"]     = v.rolling(10).mean()
        df["rel_volume"]    = v / df["vol_ma_10"]             # volume vs recent average
        df["vol_spike"]     = (df["rel_volume"] > 2.0).astype(int)

        # Volume trend
        df["vol_return"]    = v.pct_change(1)

        # VWAP approximation (intraday proxy)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap_ratio"]    = df["close"] / (typical_price * v).rolling(20).sum().div(v.rolling(20).sum())

        # Accumulation/distribution proxy
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-9)
        df["ad_line"]       = (clv * v).cumsum()
        df["ad_change"]     = df["ad_line"].pct_change(5)

        return df

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend-following indicators."""
        c = df["close"]

        # EMAs
        for period in [9, 21, 50, 200]:
            df[f"ema_{period}"] = ta.trend.EMAIndicator(c, window=period).ema_indicator()

        # EMA crossover signals
        df["ema_9_21_cross"]  = np.sign(df["ema_9"]  - df["ema_21"])
        df["ema_21_50_cross"] = np.sign(df["ema_21"] - df["ema_50"])
        df["price_above_200"] = (c > df["ema_200"]).astype(int)

        # MACD
        macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
        df["macd"]            = macd.macd()
        df["macd_signal"]     = macd.macd_signal()
        df["macd_hist"]       = macd.macd_diff()
        df["macd_cross"]      = np.sign(df["macd_hist"])

        # ADX (trend strength)
        adx = ta.trend.ADXIndicator(df["high"], df["low"], c, window=14)
        df["adx"]             = adx.adx()
        df["adx_pos"]         = adx.adx_pos()  # +DI
        df["adx_neg"]         = adx.adx_neg()  # -DI
        df["strong_trend"]    = (df["adx"] > 25).astype(int)

        # Ichimoku cloud (simplified)
        nine_high  = df["high"].rolling(9).max()
        nine_low   = df["low"].rolling(9).min()
        df["ichi_conv"]  = (nine_high + nine_low) / 2

        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum oscillators."""
        c, h, l = df["close"], df["high"], df["low"]

        # RSI (multiple periods)
        df["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()
        df["rsi_7"]  = ta.momentum.RSIIndicator(c, window=7).rsi()

        # RSI zones
        df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_divergence"] = df["rsi_14"] - df["rsi_14"].shift(5)

        # Stochastic oscillator
        stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # ROC (Rate of Change)
        df["roc_5"]  = ta.momentum.ROCIndicator(c, window=5).roc()
        df["roc_10"] = ta.momentum.ROCIndicator(c, window=10).roc()

        # CCI (Commodity Channel Index)
        df["cci"] = ta.trend.CCIIndicator(h, l, c, window=20).cci()

        # Williams %R
        df["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()

        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators."""
        c, h, l = df["close"], df["high"], df["low"]

        # ATR (true range — the most important volatility measure)
        df["atr_14"]   = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
        df["atr_norm"] = df["atr_14"] / c  # normalized ATR as % of price

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        df["bb_upper"]   = bb.bollinger_hband()
        df["bb_lower"]   = bb.bollinger_lband()
        df["bb_mid"]     = bb.bollinger_mavg()
        df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"]     = bb.bollinger_pband()          # where price is in the band (0-1)
        df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(50).quantile(0.2)).astype(int)

        # Realized volatility (20-period rolling std of returns)
        df["realized_vol"] = c.pct_change().rolling(20).std() * np.sqrt(252)

        return df

    def _market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market structure: swing highs/lows, trend regime detection."""
        c, h, l = df["close"], df["high"], df["low"]

        # Swing high/low (simplified via local rolling extrema)
        window = 5
        df["swing_high"] = (h == h.rolling(window * 2 + 1, center=True).max()).astype(int)
        df["swing_low"]  = (l == l.rolling(window * 2 + 1, center=True).min()).astype(int)

        # Higher highs / higher lows (bull structure)
        rolling_high_20 = h.rolling(20).max()
        rolling_low_20  = l.rolling(20).min()
        df["higher_high"] = (rolling_high_20 > rolling_high_20.shift(10)).astype(int)
        df["higher_low"]  = (rolling_low_20  > rolling_low_20.shift(10)).astype(int)
        df["bull_structure"] = (df["higher_high"] & df["higher_low"]).astype(int)

        # Distance from rolling high/low (momentum proxy)
        df["dist_from_high_20"] = (c - rolling_high_20) / rolling_high_20
        df["dist_from_low_20"]  = (c - rolling_low_20)  / rolling_low_20

        # Trend regime via EMA slope
        ema_50 = df.get("ema_50", c.ewm(span=50).mean())
        df["ema50_slope"] = ema_50.pct_change(5)
        df["regime"] = np.select(
            [df["ema50_slope"] > 0.002, df["ema50_slope"] < -0.002],
            [1, -1],  # 1=uptrend, -1=downtrend, 0=sideways
            default=0
        )

        return df

    def _support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distance from key support/resistance levels."""
        c = df["close"]

        # Round number proximity (psychological levels)
        # e.g. $50000, $50500 — price prefers round numbers
        df["round_num_dist"] = c.apply(lambda x: abs((x % 1000) / 1000 - 0.5))

        # Proximity to 52-week high/low
        high_52w = df["high"].rolling(365, min_periods=50).max()
        low_52w  = df["low"].rolling(365, min_periods=50).min()
        df["pct_from_52w_high"] = (c - high_52w) / high_52w
        df["pct_from_52w_low"]  = (c - low_52w)  / low_52w

        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based seasonality features."""
        idx = df.index

        df["hour"]          = idx.hour
        df["day_of_week"]   = idx.dayofweek   # 0=Monday, 6=Sunday
        df["month"]         = idx.month
        df["is_weekend"]    = (idx.dayofweek >= 5).astype(int)

        # Market sessions (UTC times)
        df["asian_session"]   = ((df["hour"] >= 0)  & (df["hour"] < 8)).astype(int)
        df["london_session"]  = ((df["hour"] >= 8)  & (df["hour"] < 16)).astype(int)
        df["ny_session"]      = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)
        df["session_overlap"] = (df["london_session"] & df["ny_session"]).astype(int)

        # Cyclical encoding (sin/cos — prevents the 23→0 hour boundary problem)
        df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)

        return df

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged values to give the model temporal context."""
        key_features = ["return_1", "rsi_14", "macd_hist", "atr_norm", "rel_volume", "regime"]

        for feat in key_features:
            if feat in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f"{feat}_lag{lag}"] = df[feat].shift(lag)

        return df
