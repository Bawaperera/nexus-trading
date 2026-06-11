"""
NEXUS — Layer 4: Risk Engine
Professional risk management: position sizing, stop loss, take profit.

Rules enforced:
- Max 1-2% account risk per trade
- Kelly Criterion for optimal position sizing
- Dynamic ATR-based stop loss
- 2:1 minimum reward/risk ratio
- Max concurrent positions limit
- Daily loss limit (circuit breaker)
"""

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class RiskEngine:
    """
    Calculates safe position sizes and enforces risk management rules.
    
    This is the layer that separates professional traders from gamblers.
    Even a 50% win rate becomes profitable with proper risk management.
    
    Usage:
        risk = RiskEngine(account_size=1000, risk_pct=0.01)
        order = risk.calculate_order("BTC/USDT", signal=1, confidence=0.72,
                                      entry_price=65000, atr=800)
        if order["approved"]:
            place_trade(order)
    """

    def __init__(
        self,
        account_size: float,
        risk_pct: float = 0.01,       # 1% risk per trade (NEVER change this as beginner)
        max_risk_pct: float = 0.02,   # Hard ceiling: 2%
        rr_ratio: float = 2.0,        # Minimum reward:risk ratio
        max_positions: int = 3,       # Max open trades at once
        daily_loss_limit: float = 0.05,  # Stop trading if -5% in one day
        use_kelly: bool = False,      # Kelly criterion (advanced, use only after 100+ backtested trades)
    ):
        self.account_size = account_size
        self.risk_pct = min(risk_pct, max_risk_pct)
        self.rr_ratio = rr_ratio
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        self.use_kelly = use_kelly

        # State tracking
        self.open_positions = 0
        self.daily_pnl = 0.0
        self.trade_history = []

        log.info(
            f"RiskEngine initialized | Account: ${account_size:,.2f} | "
            f"Risk/trade: {risk_pct*100:.1f}% | Max positions: {max_positions}"
        )

    def calculate_order(
        self,
        symbol: str,
        signal: int,              # 1=BUY, -1=SELL/SHORT
        confidence: float,        # model probability (0.5 to 1.0)
        entry_price: float,
        atr: float,               # Average True Range for dynamic SL
        atr_multiplier: float = 1.5,
    ) -> dict:
        """
        Calculate a complete trade order with all risk parameters.

        Returns a dict with:
            approved: bool
            reject_reason: str (if not approved)
            symbol, direction, entry_price
            stop_loss, take_profit
            position_size_usd, position_size_units
            risk_amount_usd
            confidence, reward_risk_ratio
        """
        order = {
            "symbol": symbol,
            "direction": "BUY" if signal == 1 else "SELL",
            "entry_price": entry_price,
            "confidence": confidence,
            "approved": False,
            "reject_reason": None,
        }

        # ── Gate 1: Circuit breakers ──────────────────────────────────────
        if self.open_positions >= self.max_positions:
            order["reject_reason"] = f"Max positions reached ({self.max_positions})"
            log.warning(f"REJECTED {symbol}: {order['reject_reason']}")
            return order

        if self.daily_pnl <= -(self.account_size * self.daily_loss_limit):
            order["reject_reason"] = f"Daily loss limit hit ({self.daily_loss_limit*100:.0f}%)"
            log.warning(f"REJECTED {symbol}: {order['reject_reason']} — STOP TRADING TODAY")
            return order

        if confidence < 0.55:
            order["reject_reason"] = f"Confidence too low ({confidence:.0%})"
            log.info(f"SKIP {symbol}: {order['reject_reason']}")
            return order

        # ── Stop loss (ATR-based) ─────────────────────────────────────────
        sl_distance = atr * atr_multiplier
        if signal == 1:  # BUY
            stop_loss   = entry_price - sl_distance
            take_profit = entry_price + (sl_distance * self.rr_ratio)
        else:            # SELL / SHORT
            stop_loss   = entry_price + sl_distance
            take_profit = entry_price - (sl_distance * self.rr_ratio)

        # ── Reward:risk validation ────────────────────────────────────────
        actual_rr = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        if actual_rr < self.rr_ratio:
            order["reject_reason"] = f"R:R ratio too low ({actual_rr:.2f} < {self.rr_ratio})"
            return order

        # ── Position sizing ───────────────────────────────────────────────
        base_risk_pct = self.risk_pct
        if self.use_kelly and len(self.trade_history) >= 30:
            base_risk_pct = self._kelly_fraction()

        # Scale position by confidence (higher confidence = larger position)
        # But never exceed 2% risk regardless of confidence
        confidence_scalar = min(1.0, max(0.5, (confidence - 0.5) * 2))
        effective_risk_pct = min(base_risk_pct * confidence_scalar, 0.02)

        risk_amount_usd    = self.account_size * effective_risk_pct
        sl_pct             = abs(entry_price - stop_loss) / entry_price
        position_size_usd  = risk_amount_usd / sl_pct
        position_size_usd  = min(position_size_usd, self.account_size * 0.25)  # Max 25% of account per trade
        position_size_units = position_size_usd / entry_price

        order.update({
            "approved": True,
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "sl_distance": round(sl_distance, 4),
            "sl_pct": round(sl_pct * 100, 3),
            "position_size_usd": round(position_size_usd, 2),
            "position_size_units": round(position_size_units, 6),
            "risk_amount_usd": round(risk_amount_usd, 2),
            "effective_risk_pct": round(effective_risk_pct * 100, 3),
            "reward_risk_ratio": round(actual_rr, 2),
            "atr": atr,
        })

        log.info(
            f"ORDER APPROVED [{symbol}] {order['direction']} @ ${entry_price:,.2f} | "
            f"SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f} | "
            f"Size: ${position_size_usd:,.2f} | Risk: ${risk_amount_usd:.2f} "
            f"({effective_risk_pct*100:.2f}%) | R:R {actual_rr:.2f}"
        )
        return order

    def update_pnl(self, pnl_usd: float, outcome: str):
        """Update account state after a trade closes."""
        self.daily_pnl += pnl_usd
        self.account_size += pnl_usd
        self.open_positions = max(0, self.open_positions - 1)

        self.trade_history.append({
            "pnl": pnl_usd,
            "outcome": outcome,  # "WIN" or "LOSS"
            "account_after": self.account_size,
        })

        log.info(f"Trade closed: {outcome} | P&L: ${pnl_usd:+.2f} | Account: ${self.account_size:,.2f}")

        if self.account_size > 0:
            stats = self.get_stats()
            log.info(f"Stats | Win rate: {stats['win_rate']:.0%} | Profit factor: {stats['profit_factor']:.2f}")

    def reset_daily(self):
        """Call this at the start of each trading day."""
        self.daily_pnl = 0.0
        log.info("Daily PnL reset")

    def get_stats(self) -> dict:
        """Calculate performance statistics from trade history."""
        if not self.trade_history:
            return {}

        wins   = [t["pnl"] for t in self.trade_history if t["outcome"] == "WIN"]
        losses = [t["pnl"] for t in self.trade_history if t["outcome"] == "LOSS"]

        total_trades = len(self.trade_history)
        win_rate     = len(wins) / total_trades if total_trades > 0 else 0
        avg_win      = np.mean(wins) if wins else 0
        avg_loss     = abs(np.mean(losses)) if losses else 0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")

        # Max drawdown
        cumulative = pd.Series([t["account_after"] for t in self.trade_history])
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "expectancy_usd": round(win_rate * avg_win - (1 - win_rate) * avg_loss, 2),
            "account_size": round(self.account_size, 2),
        }

    def _kelly_fraction(self) -> float:
        """
        Kelly Criterion: f* = (bp - q) / b
        where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        
        We use half-Kelly for safety (common professional practice).
        """
        wins   = [t["pnl"] for t in self.trade_history if t["outcome"] == "WIN"]
        losses = [t["pnl"] for t in self.trade_history if t["outcome"] == "LOSS"]

        if not wins or not losses:
            return self.risk_pct

        p = len(wins) / len(self.trade_history)
        q = 1 - p
        b = abs(np.mean(wins) / np.mean(losses))

        kelly = (b * p - q) / b
        half_kelly = kelly / 2  # Half-Kelly is standard — full Kelly is too aggressive

        # Clamp between 0.5% and 2%
        return max(0.005, min(half_kelly, 0.02))
