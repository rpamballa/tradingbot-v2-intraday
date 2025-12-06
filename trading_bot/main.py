from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Tuple

from .config import load_config, StrategyDefConfig
from .data_feed import AlpacaStreamDataFeed, DummyDataFeed
from .strategy import (
    SmaCrossoverStrategy,
    MeanReversionRsiStrategy,
    SignalType,
)
from .portfolio import Portfolio
from .risk import RiskManager, TradeIntent
from .broker import AlpacaBroker, PaperBroker


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


async def main_async():
    config = load_config()
    setup_logging(config.logging.level)
    log = logging.getLogger(__name__)

    log.info("Starting trading bot in %s mode", config.environment)

    portfolio = Portfolio(cash=config.trading.max_capital)
    risk = RiskManager(
        portfolio=portfolio,
        max_capital=config.trading.max_capital,
        max_position_per_symbol_pct=config.trading.max_position_per_symbol_pct,
        max_daily_loss_pct=config.trading.max_daily_loss_pct,
    )

    safety = config.safety
    if safety is None:
        class _Dummy:
            global_enabled = True
            allow_new_trades = True
            max_total_open_positions = 999_999
            atr_lookback = 14
            atr_stop_loss_mult = 1.5
            atr_take_profit_mult = 3.0
        safety = _Dummy()

    if not safety.global_enabled:
        log.warning("Global safety.global_enabled is FALSE; bot will not trade.")
        return

    atr_lookback = safety.atr_lookback
    atr_stop_mult = safety.atr_stop_loss_mult
    atr_take_mult = safety.atr_take_profit_mult

    bar_history: Dict[str, Deque] = {}
    sl_tp_levels: Dict[Tuple[str, str], Dict[str, float]] = {}

    def update_bar_history(symbol: str, bar) -> None:
        if symbol not in bar_history:
            bar_history[symbol] = deque(maxlen=atr_lookback + 1)
        bar_history[symbol].append(bar)

    def compute_atr(symbol: str) -> float:
        hist = bar_history.get(symbol)
        if not hist or len(hist) < 2:
            return 0.0

        bars = list(hist)
        trs = []
        for i in range(1, len(bars)):
            prev_close = bars[i - 1].close
            cur = bars[i]
            true_range = max(
                cur.high - cur.low,
                abs(cur.high - prev_close),
                abs(cur.low - prev_close),
            )
            trs.append(true_range)

        if not trs:
            return 0.0
        use_n = min(len(trs), atr_lookback)
        return sum(trs[-use_n:]) / use_n

    use_alpaca = config.broker.name.lower() == "alpaca"
    if use_alpaca:
        paper = (config.environment.lower() == "paper")
        broker = AlpacaBroker(portfolio=portfolio, config=config.broker, paper=paper)
        data_feed = AlpacaStreamDataFeed(
            broker_config=config.broker,
            timeframe=config.trading.bar_interval,
        )
    else:
        broker = PaperBroker(portfolio=portfolio)
        data_feed = DummyDataFeed()

    ms_cfg = config.multi_strategy
    strategies_by_name: dict[str, object] = {}
    symbol_to_strategies: dict[str, list[str]] = {}
    all_symbols: set[str] = set()

    if ms_cfg and ms_cfg.enabled and ms_cfg.strategies:
        for sdef in ms_cfg.strategies:
            if not sdef.enabled:
                continue
            stype = sdef.type.lower()
            if stype == "sma_crossover":
                sw = sdef.params.get("short_window", 20)
                lw = sdef.params.get("long_window", 50)
                min_diff = sdef.params.get("min_sma_diff_pct", 0.0)
                strat = SmaCrossoverStrategy(
                    short_window=sw,
                    long_window=lw,
                    name=sdef.name,
                    min_sma_diff_pct=min_diff,
                )
            elif stype == "mean_reversion_rsi":
                lookback = sdef.params.get("lookback", 14)
                oversold = sdef.params.get("oversold", 30.0)
                overbought = sdef.params.get("overbought", 70.0)
                strat = MeanReversionRsiStrategy(
                    lookback=lookback,
                    oversold=oversold,
                    overbought=overbought,
                    name=sdef.name,
                )
            else:
                raise ValueError(f"Unknown strategy type: {sdef.type}")

            strategies_by_name[sdef.name] = strat
            for sym in sdef.symbols:
                sym_u = sym.upper()
                all_symbols.add(sym_u)
                symbol_to_strategies.setdefault(sym_u, []).append(sdef.name)
    else:
        if not config.strategy:
            raise ValueError("No strategy config found and multi_strategy not enabled")
        strat = SmaCrossoverStrategy(
            short_window=config.strategy.short_window,
            long_window=config.strategy.long_window,
            name="single_sma",
        )
        strategies_by_name["single_sma"] = strat
        for sym in config.trading.symbols:
            sym_u = sym.upper()
            all_symbols.add(sym_u)
            symbol_to_strategies.setdefault(sym_u, []).append("single_sma")

    if not all_symbols:
        raise ValueError("No symbols configured for trading")

    log.info("Trading symbols: %s", sorted(all_symbols))

    open_trades: dict[tuple[str, str], bool] = {}

    def total_open_positions() -> int:
        return sum(1 for is_open in open_trades.values() if is_open)

    async for bar in data_feed.stream_bars(sorted(all_symbols)):
        symbol = bar.symbol.upper()
        update_bar_history(symbol, bar)

        for (sym, strat_name), is_open in list(open_trades.items()):
            if not is_open or sym != symbol:
                continue

            levels = sl_tp_levels.get((sym, strat_name))
            if not levels:
                continue

            stop = levels.get("stop")
            take = levels.get("take")

            hit_stop = stop is not None and bar.low <= stop
            hit_take = take is not None and bar.high >= take

            if hit_stop or hit_take:
                exit_price = stop if hit_stop else take
                side = "SELL"

                qty = 0.0
                pos = portfolio.positions.get(symbol)
                if pos:
                    qty = pos.quantity

                if qty <= 0:
                    continue

                intent = TradeIntent(
                    symbol=symbol,
                    side=side,
                    quantity=round(qty, 4),
                    price_limit=exit_price,
                )

                now = datetime.utcnow()
                if not risk.is_trade_allowed(intent, now):
                    log.warning("ATR exit blocked by risk manager: %s", intent)
                    continue

                result = broker.submit_order(intent)
                log.info(
                    "ATR %s EXIT %s qty=%.4f @ %.2f status=%s (hit %s)",
                    strat_name,
                    symbol,
                    result.quantity,
                    result.avg_price,
                    result.status,
                    "STOP" if hit_stop else "TARGET",
                )

                key = (symbol, strat_name)
                open_trades[key] = False
                sl_tp_levels.pop(key, None)

        strat_names = symbol_to_strategies.get(symbol, [])
        if not strat_names:
            continue

        for strat_name in strat_names:
            ms_defs = ms_cfg.strategies if (ms_cfg and ms_cfg.enabled and ms_cfg.strategies) else []
            sdef: StrategyDefConfig | None = None
            if ms_defs:
                sdef = next((sd for sd in ms_defs if sd.name == strat_name), None)
                if sdef is None or not sdef.enabled:
                    continue
                risk_per_trade_pct = sdef.params.get("risk_per_trade_pct", 2.0)
                max_open_trades = sdef.params.get("max_open_trades", 999_999)
            else:
                risk_per_trade_pct = 2.0
                max_open_trades = 999_999

            strat = strategies_by_name[strat_name]
            signal = strat.on_bar(bar)
            if signal.signal == SignalType.HOLD:
                continue

            side = "BUY" if signal.signal == SignalType.BUY else "SELL"

            if side == "BUY" and not safety.allow_new_trades:
                log.info(
                    "Global allow_new_trades is FALSE; skipping BUY for %s (%s).",
                    strat_name,
                    symbol,
                )
                continue

            if side == "BUY":
                open_for_strategy = sum(
                    1 for (sym, sname), is_open in open_trades.items()
                    if sname == strat_name and is_open
                )
                if open_for_strategy >= max_open_trades:
                    log.info(
                        "Skipping BUY for %s (%s): per-strategy max_open_trades=%d reached",
                        strat_name,
                        symbol,
                        max_open_trades,
                    )
                    continue

                if total_open_positions() >= safety.max_total_open_positions:
                    log.warning(
                        "Skipping BUY for %s (%s): global max_total_open_positions=%d reached",
                        strat_name,
                        symbol,
                        safety.max_total_open_positions,
                    )
                    continue

            current_price = signal.price

            if side == "BUY":
                allocation_dollars = config.trading.max_capital * (risk_per_trade_pct / 100.0)
                raw_qty = allocation_dollars / current_price if current_price > 0 else 0.0
                max_qty_allowed = risk.max_buy_quantity_for_price(symbol, current_price)
                qty = min(raw_qty, max_qty_allowed)
            else:
                pos = portfolio.positions.get(symbol)
                qty = pos.quantity if pos else 0.0

            if qty <= 0:
                continue

            intent = TradeIntent(
                symbol=symbol,
                side=side,
                quantity=round(qty, 4),
                price_limit=current_price,
            )

            now = datetime.utcnow()
            if not risk.is_trade_allowed(intent, now):
                log.warning("Trade blocked by risk manager: %s", intent)
                continue

            result = broker.submit_order(intent)
            log.info(
                "Strategy %s executed %s %s qty=%.4f @ %.2f status=%s",
                strat_name,
                side,
                symbol,
                result.quantity,
                result.avg_price,
                result.status,
            )

            key = (symbol, strat_name)
            if side == "BUY" and result.quantity > 0:
                open_trades[key] = True

                atr = compute_atr(symbol)
                if atr > 0:
                    entry_price = result.avg_price
                    stop_price = entry_price - atr_stop_mult * atr
                    take_price = entry_price + atr_take_mult * atr
                    sl_tp_levels[key] = {"stop": stop_price, "take": take_price}
                    log.info(
                        "ATR levels for %s %s: entry=%.2f stop=%.2f take=%.2f (ATR=%.2f)",
                        strat_name,
                        symbol,
                        entry_price,
                        stop_price,
                        take_price,
                        atr,
                    )
            elif side == "SELL" and result.quantity > 0:
                open_trades[key] = False
                sl_tp_levels.pop(key, None)

        log.debug(
            "Portfolio cash=%.2f positions=%s",
            portfolio.cash,
            {s: (p.quantity, p.avg_price) for s, p in portfolio.positions.items()},
        )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
