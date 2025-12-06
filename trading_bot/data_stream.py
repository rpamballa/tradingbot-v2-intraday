# trading_bot/data_stream.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Callable

from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar as AlpacaBar

from .config import BrokerConfig
from .data_feed import Bar  # reuse Bar model


class AlpacaStreamingDataFeed:
    """
    WebSocket-based bar stream using Alpaca's StockDataStream.
    Yields Bar objects via a callback.
    """

    def __init__(self, broker_config: BrokerConfig, timeframe: str = "1Min"):
        self.broker_config = broker_config
        self.timeframe = timeframe
        self.stream = StockDataStream(
            broker_config.api_key,
            broker_config.api_secret,
        )

    def run(self, symbols: List[str], on_bar: Callable[[Bar], None]) -> None:
        symbols = [s.upper() for s in symbols]

        async def _bar_handler(bar: AlpacaBar):
            b = Bar(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )
            on_bar(b)

        # Subscribe to minute bars
        for sym in symbols:
            self.stream.subscribe_bars(_bar_handler, sym)

        self.stream.run()  # blocking; wrap in process manager in prod
