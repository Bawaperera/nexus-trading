"""
NEXUS — Phase 4: Paper Trader
Manages open/closed positions in paper (simulated) trading mode.

Paper trading is mandatory before going live.
Run this for at least 30 days and check:
  ✅ Win rate > 50%
  ✅ Profit factor > 1.3
  ✅ Max drawdown < 15%
  ✅ Consistent with backtest results (no overfitting)
Only then move to live trading.
"""

import csv, os, logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Position:
    id:           str
    symbol:       str
    direction:    str   # "LONG" or "SHORT"
    entry_time:   datetime
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    size_usd:     float
    confidence:   float
    sentiment:    float
    status:       str = "OPEN"   # "OPEN", "CLOSED_TP", "CLOSED_SL", "CLOSED_MANUAL"
    exit_time:    Optional[datetime] = None
    exit_price:   float = 0.0
    pnl_usd:      float = 0.0
    pnl_pct:      float = 0.0
    peak_price:   float = 0.0   # for trailing stop (Phase 5)

    @property
    def is_open(self):
        return self.status == "OPEN"


class PaperTrader:
    """
    Tracks open and closed positions in paper trading mode.
    Writes every trade to a journal CSV for performance analysis.

    Usage:
        pt = PaperTrader(journal_path="logs/paper_journal.csv")
        pos = pt.open_position(order, signal, sentiment)
        # ... on next candle:
        pt.update_positions(current_price, current_high, current_low)
    """

    def __init__(
        self,
        initial_capital: float = 1000,
        journal_path: str = "logs/paper_journal.csv",
        commission_pct: float = 0.001,
    ):
        self.capital         = initial_capital
        self.initial_capital = initial_capital
        self.journal_path    = journal_path
        self.commission_pct  = commission_pct

        self.open_positions: dict[str, Position] = {}
        self.closed_positions: list[Position]    = []
        self._trade_counter = 0

        os.makedirs(os.path.dirname(journal_path), exist_ok=True)
        log.info(f"PaperTrader ready | Capital: ${initial_capital:,.2f} | Journal: {journal_path}")

    # ─── Position management ──────────────────────────────────────────────────

    def open_position(self, order: dict, signal=None, sentiment=None) -> Optional[Position]:
        """Open a new paper position from a risk engine order."""
        if not order.get("approved"):
            return None

        self._trade_counter += 1
        pos_id = f"PAPER-{self._trade_counter:04d}"

        pos = Position(
            id          = pos_id,
            symbol      = order["symbol"],
            direction   = order["direction"],
            entry_time  = datetime.now(tz=timezone.utc),
            entry_price = order["entry_price"],
            stop_loss   = order["stop_loss"],
            take_profit = order["take_profit"],
            size_usd    = order["position_size_usd"],
            confidence  = order.get("effective_risk_pct", 0),
            sentiment   = getattr(sentiment, "composite_score", 0) if sentiment else 0,
            peak_price  = order["entry_price"],
        )

        # Deduct commission on entry
        commission = pos.size_usd * self.commission_pct
        self.capital -= commission

        self.open_positions[pos_id] = pos
        log.info(
            f"📋 PAPER OPEN  [{pos_id}] {pos.direction} {pos.symbol} "
            f"@ ${pos.entry_price:,.2f} | SL: ${pos.stop_loss:,.2f} | "
            f"TP: ${pos.take_profit:,.2f} | Size: ${pos.size_usd:,.2f}"
        )
        return pos

    def update_positions(self, current_price: float, high: float, low: float):
        """
        Check all open positions against current candle's high/low.
        Closes positions that hit SL or TP.
        Call this on every candle close.
        """
        to_close = []

        for pos_id, pos in self.open_positions.items():
            sl, tp, d = pos.stop_loss, pos.take_profit, pos.direction

            if d == "LONG":
                if low <= sl and high >= tp:
                    to_close.append((pos_id, sl, "CLOSED_SL"))  # Conservative: SL first
                elif low <= sl:
                    to_close.append((pos_id, sl, "CLOSED_SL"))
                elif high >= tp:
                    to_close.append((pos_id, tp, "CLOSED_TP"))
            else:  # SHORT
                if high >= sl and low <= tp:
                    to_close.append((pos_id, sl, "CLOSED_SL"))
                elif high >= sl:
                    to_close.append((pos_id, sl, "CLOSED_SL"))
                elif low <= tp:
                    to_close.append((pos_id, tp, "CLOSED_TP"))

            # Update peak price (for future trailing stop)
            if d == "LONG" and current_price > pos.peak_price:
                pos.peak_price = current_price
            elif d == "SHORT" and current_price < pos.peak_price:
                pos.peak_price = current_price

        for pos_id, exit_price, reason in to_close:
            self._close_position(pos_id, exit_price, reason)

    def close_position_manual(self, pos_id: str, price: float):
        """Manually close a position (e.g. end of session)."""
        self._close_position(pos_id, price, "CLOSED_MANUAL")

    def close_all(self, current_price: float):
        """Close all open positions at current price."""
        for pos_id in list(self.open_positions.keys()):
            self._close_position(pos_id, current_price, "CLOSED_MANUAL")

    # ─── Stats & reporting ────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Return current paper trading performance summary."""
        closed = self.closed_positions
        if not closed:
            return {
                "capital":        self.capital,
                "total_trades":   0,
                "open_positions": len(self.open_positions),
            }

        wins   = [p.pnl_usd for p in closed if p.pnl_usd > 0]
        losses = [p.pnl_usd for p in closed if p.pnl_usd <= 0]
        total_pnl  = sum(p.pnl_usd for p in closed)
        win_rate   = len(wins) / len(closed) if closed else 0
        pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
        avg_win  = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0

        return {
            "capital":             round(self.capital, 2),
            "initial_capital":     self.initial_capital,
            "total_pnl":           round(total_pnl, 2),
            "return_pct":          round(total_pnl / self.initial_capital * 100, 3),
            "total_trades":        len(closed),
            "open_positions":      len(self.open_positions),
            "win_rate":            round(win_rate * 100, 1),
            "profit_factor":       round(pf, 3),
            "avg_win_usd":         round(avg_win, 2),
            "avg_loss_usd":        round(avg_loss, 2),
            "expectancy_usd":      round(win_rate * avg_win - (1 - win_rate) * avg_loss, 2),
            "tp_exits":            sum(1 for p in closed if p.status == "CLOSED_TP"),
            "sl_exits":            sum(1 for p in closed if p.status == "CLOSED_SL"),
        }

    def print_summary(self):
        s = self.get_summary()
        log.info(f"\n{'─'*50}")
        log.info(f"  PAPER TRADING SUMMARY")
        log.info(f"{'─'*50}")
        for k, v in s.items():
            log.info(f"  {k:25s}: {v}")
        log.info(f"{'─'*50}")

    def get_open_positions_display(self, current_price: float) -> list:
        """Returns open positions with live unrealized P&L."""
        result = []
        for pos in self.open_positions.values():
            d = 1 if pos.direction == "LONG" else -1
            unreal_pct = d * (current_price - pos.entry_price) / pos.entry_price * 100
            unreal_usd = unreal_pct / 100 * pos.size_usd
            result.append({
                "id":          pos.id,
                "symbol":      pos.symbol,
                "direction":   pos.direction,
                "entry":       pos.entry_price,
                "stop_loss":   pos.stop_loss,
                "take_profit": pos.take_profit,
                "unrealized_pct": round(unreal_pct, 3),
                "unrealized_usd": round(unreal_usd, 2),
            })
        return result

    # ─── Private ──────────────────────────────────────────────────────────────

    def _close_position(self, pos_id: str, exit_price: float, reason: str):
        pos = self.open_positions.pop(pos_id, None)
        if not pos:
            return

        d   = 1 if pos.direction == "LONG" else -1
        raw_pnl_pct = d * (exit_price - pos.entry_price) / pos.entry_price
        commission  = pos.size_usd * self.commission_pct
        pnl_usd     = (raw_pnl_pct * pos.size_usd) - commission

        pos.exit_time  = datetime.now(tz=timezone.utc)
        pos.exit_price = exit_price
        pos.status     = reason
        pos.pnl_usd    = round(pnl_usd, 4)
        pos.pnl_pct    = round(raw_pnl_pct * 100, 4)

        self.capital += pnl_usd + pos.size_usd  # return size + profit
        # Note: size was deducted implicitly when opened (simplified — not tracking margin)
        self.capital -= pos.size_usd  # Actually just track P&L on capital, not full size
        self.capital += pnl_usd

        self.closed_positions.append(pos)
        self._write_journal(pos)

        icon = "✅" if pnl_usd > 0 else "❌"
        log.info(
            f"{icon} PAPER CLOSE [{pos_id}] {reason} | "
            f"Exit: ${exit_price:,.2f} | "
            f"P&L: ${pnl_usd:+.2f} ({raw_pnl_pct*100:+.2f}%) | "
            f"Capital: ${self.capital:,.2f}"
        )

    def _write_journal(self, pos: Position):
        write_header = not os.path.exists(self.journal_path)
        row = {
            "id":           pos.id,
            "symbol":       pos.symbol,
            "direction":    pos.direction,
            "entry_time":   pos.entry_time.isoformat() if pos.entry_time else "",
            "entry_price":  pos.entry_price,
            "stop_loss":    pos.stop_loss,
            "take_profit":  pos.take_profit,
            "size_usd":     pos.size_usd,
            "exit_time":    pos.exit_time.isoformat() if pos.exit_time else "",
            "exit_price":   pos.exit_price,
            "status":       pos.status,
            "pnl_usd":      pos.pnl_usd,
            "pnl_pct":      pos.pnl_pct,
            "sentiment":    pos.sentiment,
        }
        with open(self.journal_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
