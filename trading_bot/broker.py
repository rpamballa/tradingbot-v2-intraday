from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from .risk import TradeIntent
from .portfolio import Portfolio
from .config import BrokerConfig


@dataclass
class OrderResult:
    symbol: str
    side: str
    quantity: float
    avg_price: float
    status: str


class Broker(Protocol):
    def submit_order(self, intent: TradeIntent) -> OrderResult:
        ...


class PaperBroker:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    def submit_order(self, intent: TradeIntent) -> OrderResult:
        qty = intent.quantity if intent.side.upper() == "BUY" else -intent.quantity
        self.portfolio.update_on_fill(intent.symbol.upper(), qty, intent.price_limit)
        return OrderResult(
            symbol=intent.symbol.upper(),
            side=intent.side.upper(),
            quantity=abs(qty),
            avg_price=intent.price_limit,
            status="FILLED",
        )


class AlpacaBroker:
    def __init__(self, portfolio: Portfolio, config: BrokerConfig, paper: bool = True):
        self.portfolio = portfolio
        self.config = config
        self.trading_client = TradingClient(
            api_key=config.api_key,
            secret_key=config.api_secret,
            paper=paper,
        )

    def submit_order(self, intent: TradeIntent) -> OrderResult:
        symbol = intent.symbol.upper()
        side = OrderSide.BUY if intent.side.upper() == "BUY" else OrderSide.SELL
        qty = intent.quantity

        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(order_req)
        updated = self.trading_client.get_order_by_id(order.id)

        status = updated.status
        filled_qty = float(updated.filled_qty or 0)
        avg_price = float(updated.filled_avg_price or intent.price_limit)

        if filled_qty > 0:
            signed = filled_qty if side == OrderSide.BUY else -filled_qty
            self.portfolio.update_on_fill(symbol, signed, avg_price)

        return OrderResult(
            symbol=symbol,
            side=intent.side.upper(),
            quantity=filled_qty,
            avg_price=avg_price,
            status=status.upper(),
        )
