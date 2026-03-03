"""
Configuration management for the trading ecosystem.
Uses Pydantic for validation and environment variable loading.
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field, validator
from enum import Enum

class TradingMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class DataSource(str, Enum):
    YFINANCE = "yfinance"
    CCXT = "ccxt"
    ALPACA = "alpaca"

class RLAlgorithm(str, Enum):
    DQN = "dqn"
    PPO = "ppo"
    SAC = "sac"

class TradingConfig(BaseSettings):
    """Main configuration class"""
    
    # Trading Parameters
    MODE: TradingMode = TradingMode.PAPER
    DATA_SOURCE: DataSource = DataSource.YFINANCE
    SYMBOLS: List[str] = ["AAPL", "MSFT", "GOOGL"]
    TIMEFRAME: str = "1h"
    INITIAL_BALANCE: float = 100000.0
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio per trade
    COMMISSION_RATE: float = 0.001  # 0.1%
    
    # RL Parameters
    RL_ALGORITHM: RLAlgorithm = RLAlgorithm.DQN
    GAMMA: float = 0.99
    LEARNING_RATE: float = 0.0001
    BATCH_SIZE: int = 64
    MEMORY_SIZE: int = 10000
    TARGET_UPDATE_FREQ: int = 1000
    
    # Neural Network
    HIDDEN_LAYERS: List[int] = [256, 128, 64]
    DROPOUT_RATE: float = 0.2
    USE_BATCHNORM: bool = True
    
    # Risk Management
    MAX_DRAWDOWN: float = 0.20  # Stop trading at 20% drawdown
    VOLATILITY_CAP: float = 0.30  # Annualized volatility limit
    CORRELATION_THRESHOLD: float = 0.7
    
    # Firebase Configuration (CRITICAL - Ecosystem Standard)
    FIREBASE_CREDENTIALS_PATH: str = "firebase_credentials.json"
    FIRESTORE_COLLECTION: str = "trading_state"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_TO_FIRESTORE: bool = True
    
    # Feature Engineering
    TECHNICAL_INDICATORS: List[str] = Field(
        default=["sma_20", "sma_50", "rsi_14", "macd", "bb_upper", "bb_lower"]
    )
    SENTIMENT_ANALYSIS: bool = False
    NEWS_API_KEY: Optional[str] = None
    
    @validator("SYMBOLS")
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol must be specified")
        return v
    
    @validator("MAX_POSITION_SIZE")
    def validate_position_size(cls, v):
        if not 0 < v <= 1:
            raise ValueError("MAX_POSITION_SIZE must be between 0 and 1")
        return v
    
    class Config:
        env_file = ".env"
        env_prefix = "TRADING_"
        case_sensitive = False

# Global config instance
config = TradingConfig()

def validate_environment() -> bool:
    """Validate critical environment dependencies"""
    try:
        import firebase_admin
        import torch
        import gymnasium
        return True
    except ImportError as e:
        raise RuntimeError(f