from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
from datetime import date, datetime


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float


@dataclass
class DailyPnl:
    day: date
    realized: float = 0.0
    unrealized: float = 0.0


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_pnl: Dict[date, DailyPnl] = field(default_factory=dict)

    def update_on_fill(self, symbol: str, qty: float, price: float):
        cost = qty * price
        self.cash -= cost

        pos = self.positions.get(symbol)
        if pos is None:
            if qty != 0:
                self.positions[symbol] = Position(symbol, qty, price)
            return

        new_qty = pos.quantity + qty
        if new_qty == 0:
            del self.positions[symbol]
        else:
            pos.avg_price = (pos.avg_price * pos.quantity + cost) / new_qty
            pos.quantity = new_qty

    def record_realized_pnl(self, ts: datetime, pnl: float):
        d = ts.date()
        dp = self.daily_pnl.get(d)
        if dp is None:
            dp = DailyPnl(day=d)
            self.daily_pnl[d] = dp
        dp.realized += pnl
