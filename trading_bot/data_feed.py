from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AsyncIterable, Dict, List
import asyncio

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream

from .config import BrokerConfig


@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class DummyDataFeed:
    async def stream_bars(self, symbols: List[str]) -> AsyncIterable[Bar]:
        if False:
            yield  # pragma: no cover
        raise RuntimeError("DummyDataFeed cannot stream bars")


class AlpacaPollingDataFeed:
    def __init__(
        self,
        broker_config: BrokerConfig,
        timeframe: str = "1Min",
        poll_interval_seconds: int = 60,
    ):
        self.client = StockHistoricalDataClient(
            api_key=broker_config.api_key,
            secret_key=broker_config.api_secret,
        )
        self.timeframe = self._parse_timeframe(timeframe)
        self.poll_interval_seconds = poll_interval_seconds
        self.last_bar_time: Dict[str, datetime] = {}

    def _parse_timeframe(self, tf: str) -> TimeFrame:
        mapping = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame.FiveMinutes,
            "15Min": TimeFrame.FifteenMinutes,
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        return mapping.get(tf, TimeFrame.Minute)

    async def stream_bars(self, symbols: List[str]) -> AsyncIterable[Bar]:
        symbols = [s.upper() for s in symbols]
        while True:
            now = datetime.now(timezone.utc)
            start = now - timedelta(minutes=10)

            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=self.timeframe,
                start=start,
                end=now,
            )
            resp = self.client.get_stock_bars(req)

            for symbol in symbols:
                sym = symbol.upper()
                raw_bars = resp.get(sym, [])
                for b in raw_bars:
                    ts = b.timestamp
                    last_ts = self.last_bar_time.get(sym)
                    if last_ts is not None and ts <= last_ts:
                        continue

                    bar = Bar(
                        symbol=sym,
                        timestamp=ts,
                        open=float(b.open),
                        high=float(b.high),
                        low=float(b.low),
                        close=float(b.close),
                        volume=float(b.volume),
                    )
                    self.last_bar_time[sym] = ts
                    yield bar

            await asyncio.sleep(self.poll_interval_seconds)


class AlpacaStreamDataFeed:
    def __init__(self, broker_config: BrokerConfig, timeframe: str = "1Min"):
        self._api_key = broker_config.api_key
        self._api_secret = broker_config.api_secret
        self._stream = StockDataStream(self._api_key, self._api_secret)
        self._timeframe = timeframe

    async def stream_bars(self, symbols: List[str]) -> AsyncIterable[Bar]:
        symbols = [s.upper() for s in symbols]
        queue: asyncio.Queue[Bar] = asyncio.Queue()

        async def on_bar(bar):
            await queue.put(
                Bar(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                )
            )

        for sym in symbols:
            self._stream.subscribe_bars(on_bar, sym)

        async with self._stream:
            while True:
                bar = await queue.get()
                yield bar
