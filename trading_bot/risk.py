from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .portfolio import Portfolio


@dataclass
class TradeIntent:
    symbol: str
    side: str
    quantity: float
    price_limit: float


class RiskManager:
    def __init__(
        self,
        portfolio: Portfolio,
        max_capital: float,
        max_position_per_symbol_pct: float,
        max_daily_loss_pct: float,
    ):
        self.portfolio = portfolio
        self.max_capital = max_capital
        self.max_position_per_symbol_pct = max_position_per_symbol_pct
        self.max_daily_loss_pct = max_daily_loss_pct

    def _current_day_pnl(self, now: datetime) -> float:
        day = now.date()
        pnl = self.portfolio.daily_pnl.get(day)
        return (pnl.realized + pnl.unrealized) if pnl else 0.0

    def is_trade_allowed(self, intent: TradeIntent, now: datetime) -> bool:
        day_pnl = self._current_day_pnl(now)
        if day_pnl < -self.max_capital * self.max_daily_loss_pct / 100:
            return False

        if intent.side.upper() == "BUY":
            pos = self.portfolio.positions.get(intent.symbol)
            existing_qty = pos.quantity if pos else 0
            projected_value = (existing_qty + intent.quantity) * intent.price_limit

            symbol_cap_limit = self.max_capital * self.max_position_per_symbol_pct / 100
            if projected_value > symbol_cap_limit:
                return False

        return True

    def max_buy_quantity_for_price(self, symbol: str, price: float) -> float:
        pos = self.portfolio.positions.get(symbol)
        existing_qty = pos.quantity if pos else 0
        symbol_cap_limit = self.max_capital * self.max_position_per_symbol_pct / 100
        existing_value = existing_qty * price
        remaining_value = max(symbol_cap_limit - existing_value, 0)
        if price <= 0 or remaining_value <= 0:
            return 0.0
        return remaining_value / price
