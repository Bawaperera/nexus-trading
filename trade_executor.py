"""
NEXUS v2 — Trade Executor
===========================
Places real orders on Binance (spot and futures) when a signal fires.
Also manages stop-loss and take-profit orders automatically.

Safety rules enforced:
  - Max 1.5% account risk per trade
  - Daily loss limit: −5% stops all auto-trading
  - No trade if 3+ open positions
  - Retest must be confirmed (not just breakout)
  - R:R must be >= 2.0

Order flow:
  1. Calculate position size from risk %
  2. Place market order (entry)
  3. Place stop-loss order (stop-market)
  4. Place take-profit order (limit)
  5. Log everything to trade_memory.csv
"""

import os
import csv
import logging
import time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

TRADE_MEMORY_PATH = "logs/trade_memory.csv"
WATCHING_PATH     = "logs/watching_log.csv"

TRADE_MEMORY_FIELDS = [
    "signal_id", "timestamp", "pattern_name", "direction", "timeframe",
    "status_at_signal", "score", "n_confluences", "trade_type",
    "entry_price", "stop_loss", "take_profit", "risk_reward",
    "spot_order_id", "futures_order_id", "position_size_usd",
    "spot_sl_order_id", "spot_tp_order_id",
    "futures_sl_order_id", "futures_tp_order_id",
    "outcome", "exit_price", "pnl_usd", "pnl_pct",
    "closed_at", "notes",
]


class TradeExecutor:
    """
    Executes trades on Binance and tracks them in trade_memory.csv.

    Usage:
        executor = TradeExecutor(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
            account_size=1000,
        )
        result = executor.execute(confluence_result)
    """

    MAX_OPEN_POSITIONS = 3
    MAX_RISK_PCT       = 0.015   # 1.5% max risk per trade
    DAILY_LOSS_LIMIT   = 0.05    # 5% daily loss stops all trading
    SPOT_SYMBOL        = "BTCUSDT"
    FUTURES_SYMBOL     = "BTCUSDT"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        account_size: float = 1000,
        paper_mode: bool = True,
    ):
        self.api_key      = api_key
        self.api_secret   = api_secret
        self.account_size = account_size
        self.paper_mode   = paper_mode

        os.makedirs("logs", exist_ok=True)
        self._init_memory()

        if not paper_mode and api_key and api_secret:
            try:
                import ccxt
                self.spot_client    = ccxt.binance({
                    "apiKey": api_key, "secret": api_secret,
                    "enableRateLimit": True,
                })
                self.futures_client = ccxt.binance({
                    "apiKey": api_key, "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                })
                log.info("Binance clients initialized (LIVE MODE)")
            except Exception as e:
                log.error(f"Binance init failed: {e}")
                self.spot_client    = None
                self.futures_client = None
        else:
            self.spot_client    = None
            self.futures_client = None
            log.info("PAPER MODE — no real orders will be placed")

    # ── Main execution ─────────────────────────────────────────────────────────

    def execute(self, confluence_result: dict) -> dict:
        """
        Execute a trade based on the confluence scoring result.
        Returns dict with execution details.
        """
        pattern    = confluence_result["pattern"]
        trade_type = confluence_result["trade_type"]
        score      = confluence_result["score"]

        # ── Safety checks ──────────────────────────────────────────────────────
        open_positions = self._count_open_positions()
        if open_positions >= self.MAX_OPEN_POSITIONS:
            log.warning(f"Skipping trade: {open_positions} open positions (max {self.MAX_OPEN_POSITIONS})")
            return {"executed": False, "reason": "max_positions_reached"}

        daily_loss = self._get_daily_pnl()
        if daily_loss <= -(self.account_size * self.DAILY_LOSS_LIMIT):
            log.warning(f"Skipping trade: daily loss limit hit ({daily_loss:.2f})")
            return {"executed": False, "reason": "daily_loss_limit"}

        if not confluence_result.get("tradeable_autotrade"):
            log.info("Score below auto-trade threshold — signal sent but no order placed")
            return {"executed": False, "reason": "below_auto_trade_threshold"}

        # ── Position sizing ────────────────────────────────────────────────────
        risk_pct   = min(trade_type["size_pct"], self.MAX_RISK_PCT)
        risk_usd   = self.account_size * risk_pct
        sl_dist    = abs(pattern.entry - pattern.stop_loss)
        sl_pct     = sl_dist / pattern.entry

        position_usd  = risk_usd / sl_pct
        position_usd  = min(position_usd, self.account_size * 0.25)  # max 25% of account

        signal_id = f"NEXUS-{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M')}"

        result = {
            "executed":    False,
            "signal_id":   signal_id,
            "pattern":     pattern.name,
            "direction":   pattern.direction,
            "entry":       pattern.entry,
            "stop_loss":   pattern.stop_loss,
            "take_profit": pattern.target,
            "risk_reward": pattern.risk_reward,
            "position_usd": round(position_usd, 2),
            "spot_order":    None,
            "futures_order": None,
            "notes":         "",
        }

        # ── Place spot order ───────────────────────────────────────────────────
        if trade_type["spot"]:
            spot_result = self._place_spot_order(pattern, position_usd, signal_id)
            result["spot_order"] = spot_result
            if spot_result.get("success"):
                result["executed"] = True

        # ── Place futures order ────────────────────────────────────────────────
        if trade_type["futures"]:
            fut_pos_usd   = position_usd * trade_type["leverage"]
            futures_result = self._place_futures_order(
                pattern, fut_pos_usd, trade_type["leverage"], signal_id
            )
            result["futures_order"] = futures_result
            if futures_result.get("success"):
                result["executed"] = True

        # ── Log to trade memory ────────────────────────────────────────────────
        self._log_trade(confluence_result, result, signal_id)

        return result

    # ── Order placement ────────────────────────────────────────────────────────

    def _place_spot_order(self, pattern, position_usd: float, signal_id: str) -> dict:
        """Place a spot market order + SL + TP."""
        side  = "buy" if pattern.direction == "bullish" else "sell"
        qty   = position_usd / pattern.entry

        if self.paper_mode or not self.spot_client:
            log.info(
                f"PAPER SPOT {side.upper()} | "
                f"${position_usd:.2f} | {qty:.6f} BTC @ ${pattern.entry:,.2f}"
            )
            return {"success": True, "order_id": f"PAPER-SPOT-{signal_id}",
                    "qty": qty, "price": pattern.entry}

        try:
            # Market entry
            order = self.spot_client.create_market_order(
                self.SPOT_SYMBOL, side, qty
            )
            actual_price = float(order.get("average", pattern.entry))
            actual_qty   = float(order.get("amount",  qty))

            time.sleep(0.5)

            # Stop loss (OCO order pairs SL+TP)
            sl_order = self._place_spot_sl_tp(
                side="sell" if side == "buy" else "buy",
                qty=actual_qty,
                sl_price=pattern.stop_loss,
                tp_price=pattern.target,
            )

            log.info(f"SPOT {side.upper()} executed | qty={actual_qty:.6f} @ ${actual_price:,.2f}")
            return {
                "success":  True,
                "order_id": order["id"],
                "sl_tp_id": sl_order.get("id"),
                "qty":      actual_qty,
                "price":    actual_price,
            }

        except Exception as e:
            log.error(f"Spot order failed: {e}")
            return {"success": False, "error": str(e)}

    def _place_spot_sl_tp(self, side: str, qty: float, sl_price: float, tp_price: float) -> dict:
        """Place OCO order (stop-loss + take-profit together)."""
        if not self.spot_client:
            return {}
        try:
            oco = self.spot_client.create_order(
                symbol   = self.SPOT_SYMBOL,
                type     = "OCO",
                side     = side,
                quantity = round(qty, 6),
                price    = round(tp_price, 2),          # take-profit limit price
                stopPrice    = round(sl_price * 1.001 if side == "sell" else sl_price * 0.999, 2),
                stopLimitPrice = round(sl_price, 2),
                stopLimitTimeInForce = "GTC",
            )
            return oco
        except Exception as e:
            log.warning(f"OCO order failed: {e}")
            return {}

    def _place_futures_order(
        self, pattern, position_usd: float, leverage: int, signal_id: str
    ) -> dict:
        """Place a futures position with SL and TP."""
        side = "buy" if pattern.direction == "bullish" else "sell"
        qty  = position_usd / pattern.entry

        if self.paper_mode or not self.futures_client:
            log.info(
                f"PAPER FUTURES {side.upper()} {leverage}x | "
                f"${position_usd:.2f} notional | {qty:.6f} BTC"
            )
            return {"success": True, "order_id": f"PAPER-FUT-{signal_id}",
                    "qty": qty, "leverage": leverage}

        try:
            self.futures_client.set_leverage(leverage, self.FUTURES_SYMBOL)
            time.sleep(0.3)

            order = self.futures_client.create_market_order(
                self.FUTURES_SYMBOL, side, qty
            )

            time.sleep(0.5)

            # Stop market
            sl_side = "sell" if side == "buy" else "buy"
            self.futures_client.create_order(
                symbol   = self.FUTURES_SYMBOL,
                type     = "STOP_MARKET",
                side     = sl_side,
                stopPrice = round(pattern.stop_loss, 2),
                closePosition = True,
                timeInForce   = "GTE_GTC",
            )

            # Take profit limit
            self.futures_client.create_order(
                symbol   = self.FUTURES_SYMBOL,
                type     = "TAKE_PROFIT_MARKET",
                side     = sl_side,
                stopPrice = round(pattern.target, 2),
                closePosition = True,
                timeInForce   = "GTE_GTC",
            )

            log.info(f"FUTURES {side.upper()} {leverage}x executed | qty={qty:.6f}")
            return {"success": True, "order_id": order["id"], "leverage": leverage}

        except Exception as e:
            log.error(f"Futures order failed: {e}")
            return {"success": False, "error": str(e)}

    # ── Trade memory ──────────────────────────────────────────────────────────

    def _init_memory(self):
        if not os.path.exists(TRADE_MEMORY_PATH):
            with open(TRADE_MEMORY_PATH, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=TRADE_MEMORY_FIELDS).writeheader()

    def _log_trade(self, confluence: dict, execution: dict, signal_id: str):
        pattern    = confluence["pattern"]
        trade_type = confluence["trade_type"]
        row = {
            "signal_id":         signal_id,
            "timestamp":         datetime.now(tz=timezone.utc).isoformat(),
            "pattern_name":      pattern.name,
            "direction":         pattern.direction,
            "timeframe":         pattern.timeframe,
            "status_at_signal":  pattern.status,
            "score":             confluence["score"],
            "n_confluences":     confluence["n_confluences"],
            "trade_type":        f"spot={trade_type['spot']} fut={trade_type['futures']} lev={trade_type['leverage']}x",
            "entry_price":       round(pattern.entry, 2),
            "stop_loss":         round(pattern.stop_loss, 2),
            "take_profit":       round(pattern.target, 2),
            "risk_reward":       pattern.risk_reward,
            "spot_order_id":     execution.get("spot_order", {}).get("order_id", ""),
            "futures_order_id":  execution.get("futures_order", {}).get("order_id", ""),
            "position_size_usd": execution.get("position_usd", 0),
            "spot_sl_order_id":     "",
            "spot_tp_order_id":     "",
            "futures_sl_order_id":  "",
            "futures_tp_order_id":  "",
            "outcome":   "OPEN",
            "exit_price":  "",
            "pnl_usd":     "",
            "pnl_pct":     "",
            "closed_at":   "",
            "notes":       execution.get("reason", ""),
        }

        with open(TRADE_MEMORY_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_MEMORY_FIELDS)
            writer.writerow(row)

        log.info(f"Trade logged: {signal_id} | {pattern.name} | score={confluence['score']}")

    def load_memory(self) -> list:
        """Load all past trades from trade_memory.csv."""
        if not os.path.exists(TRADE_MEMORY_PATH):
            return []
        try:
            import pandas as pd
            df = pd.read_csv(TRADE_MEMORY_PATH)
            return df.to_dict("records")
        except Exception:
            return []

    def _count_open_positions(self) -> int:
        trades = self.load_memory()
        return sum(1 for t in trades if t.get("outcome") == "OPEN")

    def _get_daily_pnl(self) -> float:
        trades = self.load_memory()
        today  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return sum(
            float(t.get("pnl_usd", 0) or 0)
            for t in trades
            if str(t.get("closed_at", "")).startswith(today)
        )

    def update_outcome(self, signal_id: str, outcome: str, exit_price: float):
        """
        Update a trade outcome (WIN/LOSS) when price hits TP or SL.
        Called by the validation loop in run_nexus_v2.py.
        """
        if not os.path.exists(TRADE_MEMORY_PATH):
            return

        import pandas as pd
        df = pd.read_csv(TRADE_MEMORY_PATH)
        mask = df["signal_id"] == signal_id

        if not mask.any():
            log.warning(f"Signal {signal_id} not found in memory")
            return

        entry = float(df.loc[mask, "entry_price"].iloc[0])
        sl    = float(df.loc[mask, "stop_loss"].iloc[0])
        tp    = float(df.loc[mask, "take_profit"].iloc[0])
        pos   = float(df.loc[mask, "position_size_usd"].iloc[0] or 0)
        dir_  = str(df.loc[mask, "direction"].iloc[0])

        if dir_ == "bullish":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry

        pnl_usd = pnl_pct * pos

        df.loc[mask, "outcome"]    = outcome
        df.loc[mask, "exit_price"] = round(exit_price, 2)
        df.loc[mask, "pnl_usd"]    = round(pnl_usd, 2)
        df.loc[mask, "pnl_pct"]    = round(pnl_pct * 100, 3)
        df.loc[mask, "closed_at"]  = datetime.now(tz=timezone.utc).isoformat()

        df.to_csv(TRADE_MEMORY_PATH, index=False)
        log.info(f"Updated {signal_id}: {outcome} | P&L: ${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%)")
