"""
Strateji Modülü Testleri

Bu modül, trading stratejileri testlerini içerir.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.ml_strategy import MLStrategy


def generate_test_data(periods: int = 100, trend: str = 'up', seed: int = 42) -> pd.DataFrame:
    """
    Test için OHLCV verisi oluştur.
    
    Args:
        periods: Mum sayısı
        trend: 'up', 'down' veya 'sideways'
        seed: Rastgele sayı üreteci seed'i
        
    Returns:
        OHLCV DataFrame'i
    """
    rng = np.random.default_rng(seed)
    
    if trend == 'up':
        base_prices = np.linspace(50000, 55000, periods)
    elif trend == 'down':
        base_prices = np.linspace(55000, 50000, periods)
    else:
        base_prices = np.ones(periods) * 52500
    
    # Rastgele dalgalanma ekle
    noise = rng.standard_normal(periods) * 100
    close = base_prices + noise
    
    high = close + np.abs(rng.standard_normal(periods) * 50)
    low = close - np.abs(rng.standard_normal(periods) * 50)
    open_price = close + rng.standard_normal(periods) * 30
    volume = rng.integers(100, 1000, size=periods).astype(float)
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(hours=periods),
        periods=periods,
        freq='1h'
    )
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


class TestTradeSignal:
    """TradeSignal veri sınıfı testleri."""
    
    def test_create_buy_signal(self):
        """Buy sinyali oluşturma testi."""
        signal = TradeSignal(
            signal=Signal.BUY,
            symbol='BTCUSDT',
            price=50000.0,
            confidence=0.8,
            stop_loss=49000.0,
            take_profit=52000.0
        )
        
        assert signal.signal == Signal.BUY
        assert signal.symbol == 'BTCUSDT'
        assert signal.confidence == 0.8
    
    def test_to_dict(self):
        """Dictionary dönüşümü testi."""
        signal = TradeSignal(
            signal=Signal.SELL,
            symbol='ETHUSDT',
            price=3000.0,
            confidence=0.75
        )
        
        result = signal.to_dict()
        
        assert result['signal'] == 'SELL'
        assert result['symbol'] == 'ETHUSDT'
        assert result['price'] == 3000.0


class TestMomentumStrategy:
    """MomentumStrategy sınıfı testleri."""
    
    @pytest.fixture
    def strategy(self):
        """Momentum strateji oluştur."""
        config = {
            'symbol': 'BTCUSDT',
            'timeframe': '15m',
            'stop_loss_pct': 2,
            'take_profit_pct': 4,
            'momentum': {
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'volume_multiplier': 1.5
            }
        }
        return MomentumStrategy(config)
    
    def test_strategy_name(self, strategy):
        """Strateji ismi testi."""
        assert strategy.name == 'MomentumStrategy'
    
    def test_analyze_with_uptrend(self, strategy):
        """Yükselen trendde analiz testi."""
        df = generate_test_data(100, 'up')
        signal = strategy.analyze(df)
        
        assert isinstance(signal, TradeSignal)
        assert signal.symbol == 'BTCUSDT'
        assert signal.signal in [Signal.BUY, Signal.SELL, Signal.HOLD]
    
    def test_analyze_with_downtrend(self, strategy):
        """Düşen trendde analiz testi."""
        df = generate_test_data(100, 'down')
        signal = strategy.analyze(df)
        
        assert isinstance(signal, TradeSignal)
    
    def test_analyze_insufficient_data(self, strategy):
        """Yetersiz veri durumu testi."""
        df = generate_test_data(20, 'up')
        signal = strategy.analyze(df)
        
        assert signal.signal == Signal.HOLD
        assert signal.confidence == 0.0
    
    def test_calculate_stop_loss_long(self, strategy):
        """Long pozisyon için stop-loss hesaplama testi."""
        entry_price = 50000.0
        stop_loss = strategy.calculate_stop_loss(entry_price, 'LONG')
        
        expected = entry_price * (1 - 2 / 100)  # %2 stop-loss
        assert stop_loss == expected
    
    def test_calculate_stop_loss_short(self, strategy):
        """Short pozisyon için stop-loss hesaplama testi."""
        entry_price = 50000.0
        stop_loss = strategy.calculate_stop_loss(entry_price, 'SHORT')
        
        expected = entry_price * (1 + 2 / 100)  # %2 stop-loss
        assert stop_loss == expected
    
    def test_calculate_take_profit_long(self, strategy):
        """Long pozisyon için take-profit hesaplama testi."""
        entry_price = 50000.0
        take_profit = strategy.calculate_take_profit(entry_price, 'LONG')
        
        expected = entry_price * (1 + 4 / 100)  # %4 take-profit
        assert take_profit == expected
    
    def test_should_enter(self, strategy):
        """Giriş kontrolü testi."""
        df = generate_test_data(100, 'up')
        result = strategy.should_enter(df)
        
        assert isinstance(result, bool)
    
    def test_should_exit_long(self, strategy):
        """Long pozisyon çıkış kontrolü testi."""
        df = generate_test_data(100, 'down')
        result = strategy.should_exit(df, 'LONG')
        
        assert isinstance(result, bool)
    
    def test_should_exit_short(self, strategy):
        """Short pozisyon çıkış kontrolü testi."""
        df = generate_test_data(100, 'up')
        result = strategy.should_exit(df, 'SHORT')
        
        assert isinstance(result, bool)
    
    def test_validate_signal_low_confidence(self, strategy):
        """Düşük güvenli sinyal doğrulama testi."""
        signal = TradeSignal(
            signal=Signal.BUY,
            symbol='BTCUSDT',
            price=50000.0,
            confidence=0.3  # Düşük güven
        )
        
        result = strategy.validate_signal(signal)
        
        assert result is False
    
    def test_validate_signal_high_confidence(self, strategy):
        """Yüksek güvenli sinyal doğrulama testi."""
        signal = TradeSignal(
            signal=Signal.BUY,
            symbol='BTCUSDT',
            price=50000.0,
            confidence=0.8,
            stop_loss=49000.0,
            take_profit=52000.0
        )
        
        result = strategy.validate_signal(signal)
        
        assert result is True
    
    def test_signal_history(self, strategy):
        """Sinyal geçmişi testi."""
        df = generate_test_data(100, 'up')
        
        # Birkaç analiz yap
        for _ in range(5):
            strategy.analyze(df)
        
        history = strategy.get_signal_history()
        
        assert len(history) == 5
    
    def test_activate_deactivate(self, strategy):
        """Strateji aktiflik testi."""
        assert strategy.is_active is True
        
        strategy.deactivate()
        assert strategy.is_active is False
        
        strategy.activate()
        assert strategy.is_active is True


class TestMLStrategy:
    """MLStrategy sınıfı testleri."""
    
    @pytest.fixture
    def strategy(self):
        """ML strateji oluştur."""
        config = {
            'symbol': 'BTCUSDT',
            'timeframe': '15m',
            'stop_loss_pct': 2,
            'take_profit_pct': 4,
            'ai': {
                'lookback_period': 60,
                'prediction_horizon': 5,
                'confidence_threshold': 0.6
            },
            'ml': {
                'min_confidence': 0.65,
                'position_sizing_method': 'kelly'
            }
        }
        return MLStrategy(config)
    
    def test_strategy_name(self, strategy):
        """Strateji ismi testi."""
        assert strategy.name == 'MLStrategy'
    
    def test_analyze_without_predictor(self, strategy):
        """Predictor olmadan analiz testi."""
        df = generate_test_data(100, 'up')
        signal = strategy.analyze(df)
        
        assert isinstance(signal, TradeSignal)
        # Predictor olmadan basit momentum tahmini kullanılır
    
    def test_analyze_with_uptrend(self, strategy):
        """Yükselen trendde analiz testi."""
        df = generate_test_data(100, 'up')
        signal = strategy.analyze(df)
        
        assert isinstance(signal, TradeSignal)
        assert signal.symbol == 'BTCUSDT'
    
    def test_calculate_position_size_kelly(self, strategy):
        """Kelly pozisyon boyutu hesaplama testi."""
        balance = 10000.0
        confidence = 0.8
        
        risk_pct = strategy.calculate_position_size(balance, confidence)
        
        # Kelly method: güven arttıkça risk artar
        assert risk_pct > 2.0  # Base risk
        assert risk_pct <= 5.0  # Max limit
    
    def test_calculate_position_size_low_confidence(self, strategy):
        """Düşük güvenle pozisyon boyutu testi."""
        balance = 10000.0
        confidence = 0.5
        
        risk_pct = strategy.calculate_position_size(balance, confidence)
        
        # Düşük güvenle risk azalır
        assert risk_pct >= 0.5  # Min limit


class TestBacktest:
    """Backtest fonksiyonu testleri."""
    
    @pytest.fixture
    def strategy(self):
        """Test strateji oluştur."""
        config = {
            'symbol': 'BTCUSDT',
            'timeframe': '15m',
            'stop_loss_pct': 2,
            'take_profit_pct': 4,
            'momentum': {
                'rsi_overbought': 70,
                'rsi_oversold': 30
            }
        }
        return MomentumStrategy(config)
    
    def test_backtest_basic(self, strategy):
        """Temel backtest testi."""
        df = generate_test_data(200, 'up')
        
        results = strategy.backtest(
            df,
            initial_balance=10000.0,
            position_size_pct=10.0
        )
        
        assert 'initial_balance' in results
        assert 'final_balance' in results
        assert 'total_trades' in results
        assert 'win_rate' in results
        assert 'trades' in results
    
    def test_backtest_returns_valid_metrics(self, strategy):
        """Backtest metrikleri doğrulama testi."""
        df = generate_test_data(200, 'up')
        
        results = strategy.backtest(df)
        
        assert results['initial_balance'] == 10000.0
        assert isinstance(results['total_return'], float)
        assert 0 <= results['win_rate'] <= 100
    
    def test_backtest_insufficient_data(self, strategy):
        """Yetersiz veri ile backtest testi."""
        df = generate_test_data(30, 'up')
        
        results = strategy.backtest(df)
        
        # 50'den az veri olduğunda trade yapılmaz
        assert results['total_trades'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
