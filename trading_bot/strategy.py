from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict

from .data_feed import Bar


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    symbol: str
    signal: SignalType
    price: float
    strategy_name: str | None = None


@dataclass
class SmaState:
    short_window: int
    long_window: int
    short_prices: Deque[float]
    long_prices: Deque[float]
    short_sma: float | None = None
    long_sma: float | None = None
    last_signal: SignalType = SignalType.HOLD


class SmaCrossoverStrategy:
    def __init__(
        self,
        short_window: int,
        long_window: int,
        name: str = "sma_crossover",
        min_sma_diff_pct: float = 0.0,
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.name = name
        self.short_window = short_window
        self.long_window = long_window
        self.min_sma_diff_pct = min_sma_diff_pct
        self.state: Dict[str, SmaState] = {}

    def _get_state(self, symbol: str) -> SmaState:
        if symbol not in self.state:
            self.state[symbol] = SmaState(
                short_window=self.short_window,
                long_window=self.long_window,
                short_prices=deque(maxlen=self.short_window),
                long_prices=deque(maxlen=self.long_window),
            )
        return self.state[symbol]

    def on_bar(self, bar: Bar) -> Signal:
        s = self._get_state(bar.symbol)
        s.short_prices.append(bar.close)
        s.long_prices.append(bar.close)

        if len(s.long_prices) < s.long_window:
            return Signal(bar.symbol, SignalType.HOLD, bar.close, strategy_name=self.name)

        s.short_sma = sum(s.short_prices) / len(s.short_prices)
        s.long_sma = sum(s.long_prices) / len(s.long_prices)

        if s.long_sma == 0:
            return Signal(bar.symbol, SignalType.HOLD, bar.close, strategy_name=self.name)

        diff_pct = (s.short_sma / s.long_sma - 1.0) * 100.0

        if diff_pct > self.min_sma_diff_pct and s.last_signal != SignalType.BUY:
            s.last_signal = SignalType.BUY
            return Signal(bar.symbol, SignalType.BUY, bar.close, strategy_name=self.name)

        if diff_pct < -self.min_sma_diff_pct and s.last_signal != SignalType.SELL:
            s.last_signal = SignalType.SELL
            return Signal(bar.symbol, SignalType.SELL, bar.close, strategy_name=self.name)

        return Signal(bar.symbol, SignalType.HOLD, bar.close, strategy_name=self.name)


@dataclass
class RsiState:
    lookback: int
    closes: Deque[float]
    last_signal: SignalType = SignalType.HOLD


class MeanReversionRsiStrategy:
    def __init__(
        self,
        lookback: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        name: str = "mean_reversion_rsi",
    ):
        self.name = name
        self.lookback = lookback
        self.oversold = oversold
        self.overbought = overbought
        self.state: Dict[str, RsiState] = {}

    def _get_state(self, symbol: str) -> RsiState:
        if symbol not in self.state:
            self.state[symbol] = RsiState(
                lookback=self.lookback,
                closes=deque(maxlen=self.lookback + 1),
            )
        return self.state[symbol]

    @staticmethod
    def _compute_rsi(closes: Deque[float]) -> float:
        if len(closes) < 2:
            return 50.0
        gains = 0.0
        losses = 0.0
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            if diff > 0:
                gains += diff
            elif diff < 0:
                losses -= diff

        avg_gain = gains / (len(closes) - 1)
        avg_loss = losses / (len(closes) - 1)
        if avg_loss == 0:
            return 100.0
        if avg_gain == 0:
            return 0.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def on_bar(self, bar: Bar) -> Signal:
        s = self._get_state(bar.symbol)
        s.closes.append(bar.close)

        if len(s.closes) < self.lookback + 1:
            return Signal(bar.symbol, SignalType.HOLD, bar.close, strategy_name=self.name)

        rsi = self._compute_rsi(s.closes)

        if rsi < self.oversold and s.last_signal != SignalType.BUY:
            s.last_signal = SignalType.BUY
            return Signal(bar.symbol, SignalType.BUY, bar.close, strategy_name=self.name)

        if rsi > self.overbought and s.last_signal != SignalType.SELL:
            s.last_signal = SignalType.SELL
            return Signal(bar.symbol, SignalType.SELL, bar.close, strategy_name=self.name)

        return Signal(bar.symbol, SignalType.HOLD, bar.close, strategy_name=self.name)
