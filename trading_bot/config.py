from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class BrokerConfig:
    name: str
    api_key: str
    api_secret: str
    base_url: str


@dataclass
class TradingConfig:
    symbols: List[str]
    bar_interval: str
    max_capital: float
    max_position_per_symbol_pct: float
    max_daily_loss_pct: float


@dataclass
class StrategyConfig:
    type: str
    short_window: int
    long_window: int


@dataclass
class LoggingConfig:
    level: str


@dataclass
class StrategyDefConfig:
    name: str
    type: str
    symbols: List[str]
    params: dict[str, Any]
    enabled: bool = True


@dataclass
class MultiStrategyConfig:
    enabled: bool
    strategies: List[StrategyDefConfig]


@dataclass
class SafetyConfig:
    global_enabled: bool = True
    allow_new_trades: bool = True
    max_total_open_positions: int = 999999
    atr_lookback: int = 14
    atr_stop_loss_mult: float = 1.5
    atr_take_profit_mult: float = 3.0


@dataclass
class AppConfig:
    environment: str
    broker: BrokerConfig
    trading: TradingConfig
    strategy: Optional[StrategyConfig]
    logging: LoggingConfig
    multi_strategy: Optional[MultiStrategyConfig] = None
    safety: Optional[SafetyConfig] = None


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file {path} not found. "
            f"Copy config_example.yaml to config.yaml and update it."
        )

    with path.open() as f:
        raw = yaml.safe_load(f)

    broker = BrokerConfig(**raw["broker"])
    trading = TradingConfig(**raw["trading"])
    logging_cfg = LoggingConfig(**raw["logging"])

    strat_cfg: Optional[StrategyConfig] = None
    if "strategy" in raw:
        strat_cfg = StrategyConfig(**raw["strategy"])

    ms_cfg: Optional[MultiStrategyConfig] = None
    if "multi_strategy" in raw:
        ms_raw = raw["multi_strategy"]
        ms_cfg = MultiStrategyConfig(
            enabled=ms_raw.get("enabled", False),
            strategies=[
                StrategyDefConfig(
                    name=s["name"],
                    type=s["type"],
                    symbols=s["symbols"],
                    params=s.get("params", {}),
                    enabled=s.get("enabled", True),
                )
                for s in ms_raw.get("strategies", [])
            ],
        )

    safety_cfg: Optional[SafetyConfig] = None
    if "safety" in raw:
        s_raw = raw["safety"]
        safety_cfg = SafetyConfig(
            global_enabled=s_raw.get("global_enabled", True),
            allow_new_trades=s_raw.get("allow_new_trades", True),
            max_total_open_positions=s_raw.get("max_total_open_positions", 999999),
            atr_lookback=s_raw.get("atr_lookback", 14),
            atr_stop_loss_mult=s_raw.get("atr_stop_loss_mult", 1.5),
            atr_take_profit_mult=s_raw.get("atr_take_profit_mult", 3.0),
        )

    return AppConfig(
        environment=raw["environment"],
        broker=broker,
        trading=trading,
        strategy=strat_cfg,
        logging=logging_cfg,
        multi_strategy=ms_cfg,
        safety=safety_cfg,
    )
