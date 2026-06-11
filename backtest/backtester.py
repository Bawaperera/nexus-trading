"""
NEXUS — Phase 3: Backtester
Simulates trades on out-of-sample predictions with full risk management.

Trade simulation rules:
  - Enter at NEXT candle's open (no lookahead)
  - Stop loss:   entry ± (ATR × 1.5)  — automatically calculated
  - Take profit: entry ± (ATR × 3.0)  — 2:1 reward/risk
  - Max hold:    20 candles (prevent zombie trades)
  - Intrabar SL/TP check: uses high/low to detect hits within a candle
    If both SL and TP hit in same candle → assume SL hit first (conservative)

Key metrics reported:
  Total return, Win rate, Profit factor, Max drawdown,
  Sharpe ratio, Calmar ratio, Expectancy, # trades
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_bar:    int
    entry_time:   object
    entry_price:  float
    direction:    int          # 1=LONG, -1=SHORT
    stop_loss:    float
    take_profit:  float
    size_usd:     float
    confidence:   float
    exit_bar:     Optional[int]   = None
    exit_time:    object          = None
    exit_price:   float           = 0.0
    exit_reason:  str             = ""   # "TP", "SL", "MAX_HOLD", "END"
    pnl_usd:      float           = 0.0
    pnl_pct:      float           = 0.0
    bars_held:    int             = 0


class Backtester:
    """
    Simulates trades on historical price data given model signals.

    Usage:
        bt = Backtester(initial_capital=1000, risk_pct=0.01, rr_ratio=2.0)
        results = bt.run(features_df, oof_predictions, oof_probabilities)
        bt.print_report(results)
    """

    def __init__(
        self,
        initial_capital: float = 1000,
        risk_pct: float = 0.01,
        rr_ratio: float = 2.0,
        atr_sl_multiplier: float = 1.5,
        max_hold_bars: int = 20,
        min_confidence: float = 0.55,
        commission_pct: float = 0.001,  # 0.1% Binance spot taker fee
    ):
        self.initial_capital   = initial_capital
        self.risk_pct          = risk_pct
        self.rr_ratio          = rr_ratio
        self.atr_sl_multiplier = atr_sl_multiplier
        self.max_hold_bars     = max_hold_bars
        self.min_confidence    = min_confidence
        self.commission_pct    = commission_pct

        log.info(
            f"Backtester initialized | Capital: ${initial_capital:,.0f} | "
            f"Risk: {risk_pct*100:.1f}%/trade | R:R {rr_ratio:.1f}:1 | "
            f"Commission: {commission_pct*100:.2f}%"
        )

    def run(
        self,
        features_df: pd.DataFrame,
        signals: pd.Series,
        probabilities: pd.DataFrame = None,
    ) -> dict:
        """
        Run the backtest simulation.

        Args:
            features_df:  Full feature DataFrame (must contain ATR, close, high, low)
            signals:      Series of predicted signals (-1, 0, 1) aligned with features_df
            probabilities: Optional DataFrame with BUY/SELL/HOLD probabilities

        Returns:
            dict with trades list, equity_curve, and all performance metrics
        """
        # Align data
        df       = features_df.copy()
        df       = df.loc[signals.index]   # Only rows with OOF predictions

        trades      = []
        equity      = self.initial_capital
        equity_curve = [equity]
        timestamps  = [df.index[0]]

        in_trade     = False
        current_trade: Optional[Trade] = None

        log.info(f"\nBacktesting {len(df)} bars | {signals.value_counts().to_dict()}")

        for i in range(len(df)):
            bar  = df.iloc[i]
            time = df.index[i]

            high  = float(bar["high"])
            low   = float(bar["low"])
            close = float(bar["close"])
            atr   = float(bar.get("atr_14", close * 0.02))

            # ── Manage open trade ─────────────────────────────────────────
            if in_trade and current_trade:
                sl = current_trade.stop_loss
                tp = current_trade.take_profit
                d  = current_trade.direction
                bars_held = i - current_trade.entry_bar

                exit_price  = None
                exit_reason = None

                if d == 1:  # LONG
                    if low <= sl and high >= tp:    # Both hit in same candle → conservative: SL
                        exit_price, exit_reason = sl, "SL"
                    elif low <= sl:
                        exit_price, exit_reason = sl, "SL"
                    elif high >= tp:
                        exit_price, exit_reason = tp, "TP"
                else:       # SHORT
                    if high >= sl and low <= tp:    # Both → SL
                        exit_price, exit_reason = sl, "SL"
                    elif high >= sl:
                        exit_price, exit_reason = sl, "SL"
                    elif low <= tp:
                        exit_price, exit_reason = tp, "TP"

                if exit_price is None and bars_held >= self.max_hold_bars:
                    exit_price, exit_reason = close, "MAX_HOLD"

                if exit_price is not None:
                    # P&L calculation
                    raw_pnl_pct = d * (exit_price - current_trade.entry_price) / current_trade.entry_price
                    commission  = current_trade.size_usd * self.commission_pct * 2  # entry + exit
                    pnl_usd     = (raw_pnl_pct * current_trade.size_usd) - commission

                    current_trade.exit_bar    = i
                    current_trade.exit_time   = time
                    current_trade.exit_price  = exit_price
                    current_trade.exit_reason = exit_reason
                    current_trade.pnl_usd     = round(pnl_usd, 4)
                    current_trade.pnl_pct     = round(raw_pnl_pct * 100, 4)
                    current_trade.bars_held   = bars_held

                    equity += pnl_usd
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None

            equity_curve.append(equity)
            timestamps.append(time)

            # ── Check for new signal (only enter if not already in trade) ─
            if not in_trade and i < len(df) - 1:
                sig = int(signals.iloc[i])
                if sig == 0:
                    continue

                # Confidence filter
                conf = 0.5
                if probabilities is not None:
                    proba_row = probabilities.iloc[i]
                    if sig == 1:
                        conf = float(proba_row.get("BUY", 0.5))
                    else:
                        conf = float(proba_row.get("SELL", 0.5))

                if conf < self.min_confidence:
                    continue

                # Enter at NEXT bar's open
                next_bar = df.iloc[i + 1]
                entry_price = float(next_bar["open"])
                entry_atr   = float(next_bar.get("atr_14", entry_price * 0.02))

                sl_dist  = entry_atr * self.atr_sl_multiplier
                risk_amt = equity * self.risk_pct
                sl_pct   = sl_dist / entry_price
                size_usd = min(risk_amt / sl_pct, equity * 0.25)  # Max 25% of account

                if sig == 1:  # BUY
                    sl = entry_price - sl_dist
                    tp = entry_price + (sl_dist * self.rr_ratio)
                else:         # SELL / SHORT
                    sl = entry_price + sl_dist
                    tp = entry_price - (sl_dist * self.rr_ratio)

                current_trade = Trade(
                    entry_bar   = i + 1,
                    entry_time  = df.index[i + 1],
                    entry_price = entry_price,
                    direction   = sig,
                    stop_loss   = sl,
                    take_profit = tp,
                    size_usd    = size_usd,
                    confidence  = conf,
                )
                in_trade = True

        # Close any open trade at end of data
        if in_trade and current_trade:
            last  = df.iloc[-1]
            close = float(last["close"])
            d     = current_trade.direction
            raw_pnl_pct = d * (close - current_trade.entry_price) / current_trade.entry_price
            commission  = current_trade.size_usd * self.commission_pct * 2
            pnl_usd     = (raw_pnl_pct * current_trade.size_usd) - commission

            current_trade.exit_bar    = len(df) - 1
            current_trade.exit_time   = df.index[-1]
            current_trade.exit_price  = close
            current_trade.exit_reason = "END"
            current_trade.pnl_usd     = round(pnl_usd, 4)
            current_trade.pnl_pct     = round(raw_pnl_pct * 100, 4)
            current_trade.bars_held   = len(df) - 1 - current_trade.entry_bar

            equity += pnl_usd
            trades.append(current_trade)

        equity_series = pd.Series(equity_curve, index=timestamps[:len(equity_curve)])
        metrics       = self._compute_metrics(trades, equity_series)
        return {
            "trades":       trades,
            "equity_curve": equity_series,
            "metrics":      metrics,
        }

    def print_report(self, results: dict):
        """Print a clean backtest report."""
        m = results["metrics"]
        trades = results["trades"]

        log.info(f"\n{'='*55}")
        log.info("  NEXUS BACKTEST REPORT")
        log.info(f"{'='*55}")
        log.info(f"  Period            : {results['equity_curve'].index[0].date()} → {results['equity_curve'].index[-1].date()}")
        log.info(f"  Total Trades      : {m['total_trades']}")
        log.info(f"  Win Rate          : {m['win_rate']*100:.1f}%")
        log.info(f"  Profit Factor     : {m['profit_factor']:.2f}x")
        log.info(f"  Total Return      : {m['total_return_pct']:+.2f}%")
        log.info(f"  Annualized Return : {m['annualized_return_pct']:+.2f}%")
        log.info(f"  Max Drawdown      : {m['max_drawdown_pct']:.2f}%")
        log.info(f"  Sharpe Ratio      : {m['sharpe_ratio']:.3f}")
        log.info(f"  Calmar Ratio      : {m['calmar_ratio']:.3f}")
        log.info(f"  Expectancy        : ${m['expectancy_usd']:+.2f} per trade")
        log.info(f"  Avg Win           : ${m['avg_win_usd']:+.2f}")
        log.info(f"  Avg Loss          : ${m['avg_loss_usd']:-.2f}")
        log.info(f"  Avg Bars Held     : {m['avg_bars_held']:.1f}")
        log.info(f"  Starting Capital  : ${self.initial_capital:,.2f}")
        log.info(f"  Final Capital     : ${self.initial_capital + m['total_pnl_usd']:,.2f}")
        log.info(f"{'='*55}")

        # Trade breakdown
        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        log.info(f"  Exit reasons: {reasons}")

        # Verdict
        log.info("")
        if m["profit_factor"] > 1.5 and m["win_rate"] > 0.50 and m["max_drawdown_pct"] > -30:
            log.info("  🟢 VERDICT: Strategy is PROFITABLE — proceed to paper trading")
        elif m["profit_factor"] > 1.0:
            log.info("  🟡 VERDICT: Marginally profitable — needs improvement before live trading")
        else:
            log.info("  🔴 VERDICT: Unprofitable — DO NOT trade live. Retrain model.")

    # ─── Metrics ──────────────────────────────────────────────────────────────

    def _compute_metrics(self, trades: list, equity_curve: pd.Series) -> dict:
        if not trades:
            return {"total_trades": 0}

        wins   = [t.pnl_usd for t in trades if t.pnl_usd > 0]
        losses = [t.pnl_usd for t in trades if t.pnl_usd <= 0]
        all_pnl = [t.pnl_usd for t in trades]

        total_pnl      = sum(all_pnl)
        total_return   = total_pnl / self.initial_capital * 100

        # Annualize (approximate — assumes daily bars)
        n_bars = len(equity_curve)
        years  = n_bars / 252
        ann_return = ((1 + total_return / 100) ** (1 / max(years, 0.1)) - 1) * 100

        # Max drawdown
        peak = equity_curve.cummax()
        dd   = (equity_curve - peak) / peak * 100
        max_dd = float(dd.min())

        # Sharpe ratio (daily returns)
        daily_returns = equity_curve.pct_change().dropna()
        sharpe = float(daily_returns.mean() / (daily_returns.std() + 1e-9) * np.sqrt(252))

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        profit_factor = sum(w for w in wins) / abs(sum(l for l in losses)) if losses else float("inf")
        avg_win   = np.mean(wins) if wins else 0
        avg_loss  = abs(np.mean(losses)) if losses else 0
        win_rate  = len(wins) / len(trades)
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        avg_bars  = np.mean([t.bars_held for t in trades])

        return {
            "total_trades":          len(trades),
            "winning_trades":        len(wins),
            "losing_trades":         len(losses),
            "win_rate":              round(win_rate, 4),
            "profit_factor":         round(profit_factor, 4),
            "total_pnl_usd":         round(total_pnl, 2),
            "total_return_pct":      round(total_return, 4),
            "annualized_return_pct": round(ann_return, 4),
            "max_drawdown_pct":      round(max_dd, 4),
            "sharpe_ratio":          round(sharpe, 4),
            "calmar_ratio":          round(calmar, 4),
            "expectancy_usd":        round(expectancy, 4),
            "avg_win_usd":           round(avg_win, 4),
            "avg_loss_usd":          round(avg_loss, 4),
            "avg_bars_held":         round(avg_bars, 2),
        }
