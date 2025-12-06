from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, timezone

import pandas as pd
import matplotlib.pyplot as plt

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .config import load_config, BrokerConfig, StrategyDefConfig
from .data_feed import Bar
from .strategy import (
    SmaCrossoverStrategy,
    MeanReversionRsiStrategy,
    SignalType,
)
from .portfolio import Portfolio
from .risk import RiskManager, TradeIntent


def fetch_alpaca_bars(
    symbols: List[str],
    start: datetime,
    end: datetime,
    broker_config: BrokerConfig,
    timeframe: str = "1Day",
) -> Dict[str, List[Bar]]:
    client = StockHistoricalDataClient(
        api_key=broker_config.api_key,
        secret_key=broker_config.api_secret,
    )

    tf_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame.FiveMinutes,
        "15Min": TimeFrame.FifteenMinutes,
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
    tf = tf_map.get(timeframe, TimeFrame.Day)

    req = StockBarsRequest(
        symbol_or_symbols=[s.upper() for s in symbols],
        timeframe=tf,
        start=start,
        end=end,
    )

    resp = client.get_stock_bars(req)
    bars_by_symbol: Dict[str, List[Bar]] = {}

    for symbol in symbols:
        sym = symbol.upper()
        raw_bars = resp.get(sym, [])
        bars: List[Bar] = []
        for b in raw_bars:
            bars.append(
                Bar(
                    symbol=sym,
                    timestamp=b.timestamp,
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume),
                )
            )
        bars.sort(key=lambda x: x.timestamp)
        bars_by_symbol[sym] = bars

    return bars_by_symbol


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    bars_held: int
    strategy_name: str


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]


def run_multi_strategy_backtest(
    bars_by_symbol: Dict[str, List[Bar]],
    initial_capital: float,
    max_position_per_symbol_pct: float,
    max_daily_loss_pct: float,
    strategy_defs: List[StrategyDefConfig],
    slippage_bps: float = 1.0,
    commission_per_trade: float = 0.0,
) -> BacktestResult:
    all_bars: List[Bar] = []
    for sym, bars in bars_by_symbol.items():
        all_bars.extend(bars)
    all_bars.sort(key=lambda b: b.timestamp)
    if not all_bars:
        raise ValueError("No bars provided for backtest")

    portfolio = Portfolio(cash=initial_capital)
    risk_mgr = RiskManager(
        portfolio=portfolio,
        max_capital=initial_capital,
        max_position_per_symbol_pct=max_position_per_symbol_pct,
        max_daily_loss_pct=max_daily_loss_pct,
    )

    strategies_by_name: Dict[str, object] = {}
    symbol_to_strategies: Dict[str, List[str]] = {}

    for sdef in strategy_defs:
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
            symbol_to_strategies.setdefault(sym_u, []).append(sdef.name)

    equity_records: List[dict] = []
    trades: List[TradeRecord] = []
    last_prices: Dict[str, float] = {}
    open_trade_info: Dict[Tuple[str, str], Tuple[datetime, float, int]] = {}

    def compute_equity(ts: datetime, prices: Dict[str, float]) -> Tuple[float, float]:
        positions_value = 0.0
        for sym, pos in portfolio.positions.items():
            price = prices.get(sym, pos.avg_price)
            positions_value += pos.quantity * price
        return portfolio.cash + positions_value, positions_value

    for bar in all_bars:
        symbol = bar.symbol.upper()
        last_prices[symbol] = bar.close
        strat_names = symbol_to_strategies.get(symbol, [])
        if not strat_names:
            equity, pos_val = compute_equity(bar.timestamp, last_prices)
            equity_records.append(
                {"timestamp": bar.timestamp, "equity": equity, "cash": portfolio.cash, "positions_value": pos_val}
            )
            continue

        for strat_name in strat_names:
            sdef = next(sd for sd in strategy_defs if sd.name == strat_name)
            if not sdef.enabled:
                continue

            strat = strategies_by_name[strat_name]
            risk_per_trade_pct = sdef.params.get("risk_per_trade_pct", 2.0)
            max_open_trades = sdef.params.get("max_open_trades", 999999)

            signal = strat.on_bar(bar)
            if signal.signal == SignalType.HOLD:
                key = (symbol, strat_name)
                if key in open_trade_info:
                    et, ep, bh = open_trade_info[key]
                    open_trade_info[key] = (et, ep, bh + 1)
                continue

            open_trades_for_strategy = sum(1 for (sym, sname) in open_trade_info.keys() if sname == strat_name)

            side = "BUY" if signal.signal == SignalType.BUY else "SELL"
            if side == "BUY" and open_trades_for_strategy >= max_open_trades:
                continue

            current_price = signal.price

            if side == "BUY":
                allocation_dollars = initial_capital * (risk_per_trade_pct / 100.0)
                raw_qty = allocation_dollars / current_price if current_price > 0 else 0.0
                max_qty_allowed = risk_mgr.max_buy_quantity_for_price(symbol, current_price)
                qty = min(raw_qty, max_qty_allowed)
            else:
                pos = portfolio.positions.get(symbol)
                qty = pos.quantity if pos else 0.0

            if qty <= 0:
                continue

            if side == "BUY":
                trade_price = current_price * (1.0 + slippage_bps / 10_000.0)
            else:
                trade_price = current_price * (1.0 - slippage_bps / 10_000.0)

            intent = TradeIntent(
                symbol=symbol,
                side=side,
                quantity=qty,
                price_limit=trade_price,
            )

            now = bar.timestamp
            if not risk_mgr.is_trade_allowed(intent, now):
                continue

            signed_qty = qty if side == "BUY" else -qty
            portfolio.update_on_fill(symbol, signed_qty, trade_price)
            portfolio.cash -= commission_per_trade

            key = (symbol, strat_name)
            if side == "SELL":
                entry_time, entry_price, bars_held = open_trade_info.get(
                    key,
                    (now, trade_price, 0),
                )
                pos_qty = qty
                pnl = (trade_price - entry_price) * pos_qty - commission_per_trade
                pnl_pct = (trade_price / entry_price - 1.0) if entry_price > 0 else 0.0

                trades.append(
                    TradeRecord(
                        symbol=symbol,
                        side="LONG",
                        entry_time=entry_time,
                        exit_time=now,
                        quantity=pos_qty,
                        entry_price=entry_price,
                        exit_price=trade_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        bars_held=bars_held,
                        strategy_name=strat_name,
                    )
                )
                portfolio.record_realized_pnl(now, pnl)
                open_trade_info.pop(key, None)
            elif side == "BUY":
                open_trade_info[key] = (now, trade_price, 0)

        equity, pos_val = compute_equity(bar.timestamp, last_prices)
        equity_records.append(
            {"timestamp": bar.timestamp, "equity": equity, "cash": portfolio.cash, "positions_value": pos_val}
        )

    eq_df = pd.DataFrame(equity_records).drop_duplicates(subset=["timestamp"])
    eq_df.set_index("timestamp", inplace=True)
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    metrics = compute_metrics(eq_df, trades_df)
    return BacktestResult(equity_curve=eq_df, trades=trades_df, metrics=metrics)


def compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, float]:
    if equity_df.empty:
        raise ValueError("Equity curve is empty")

    equity = equity_df["equity"]
    initial = equity.iloc[0]
    final = equity.iloc[-1]
    total_return = final / initial - 1.0

    duration_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    years = max(duration_days / 365.25, 1e-6)
    cagr = (final / initial) ** (1.0 / years) - 1.0

    bar_returns = equity.pct_change().fillna(0.0)
    daily_returns = bar_returns.resample("1D").sum()
    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_drawdown = drawdown.min()

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0.0
        avg_win = wins["pnl"].mean() if not wins.empty else 0.0
        avg_loss = losses["pnl"].mean() if not losses.empty else 0.0
        avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    else:
        win_rate = avg_win = avg_loss = avg_rr = 0.0

    return {
        "initial_equity": float(initial),
        "final_equity": float(final),
        "total_return_pct": float(total_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_drawdown_pct": float(max_drawdown * 100.0),
        "sharpe_ratio": float(sharpe),
        "num_trades": float(len(trades_df)),
        "win_rate_pct": float(win_rate * 100.0),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "avg_reward_risk": float(avg_rr),
    }


def plot_dashboard(result: BacktestResult, title: str = "Backtest Dashboard"):
    eq_df = result.equity_curve.copy()
    equity = eq_df["equity"]
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(equity.index, equity.values)
    plt.ylabel("Equity")
    plt.title(title)

    plt.subplot(2, 1, 2)
    plt.plot(drawdown.index, drawdown.values)
    plt.ylabel("Drawdown")
    plt.xlabel("Time")

    plt.tight_layout()
    plt.show()


def main():
    config = load_config()
    initial_capital = config.trading.max_capital
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 2)

    ms_cfg = config.multi_strategy
    if ms_cfg and ms_cfg.enabled and ms_cfg.strategies:
        active_strats = [s for s in ms_cfg.strategies if s.enabled]
        if not active_strats:
            raise ValueError("All strategies are disabled in multi_strategy config")

        symbols = sorted({sym.upper() for s in active_strats for sym in s.symbols})
        print(f"Running multi-strategy backtest on symbols: {symbols}")

        bars_by_symbol = fetch_alpaca_bars(
            symbols=symbols,
            start=start,
            end=end,
            broker_config=config.broker,
            timeframe=config.trading.bar_interval,
        )

        result = run_multi_strategy_backtest(
            bars_by_symbol=bars_by_symbol,
            initial_capital=initial_capital,
            max_position_per_symbol_pct=config.trading.max_position_per_symbol_pct,
            max_daily_loss_pct=config.trading.max_daily_loss_pct,
            strategy_defs=active_strats,
            slippage_bps=1.0,
            commission_per_trade=0.0,
        )

        print("=== Multi-Strategy Backtest Metrics ===")
        for k, v in result.metrics.items():
            print(f"{k}: {v}")

        if not result.trades.empty:
            print("\n=== Trades by Strategy (pnl sum) ===")
            print(result.trades.groupby("strategy_name")["pnl"].sum())

        plot_dashboard(result, title="Multi-Strategy Backtest")
    else:
        raise ValueError("multi_strategy not enabled or empty in config.yaml")


if __name__ == "__main__":
    main()
