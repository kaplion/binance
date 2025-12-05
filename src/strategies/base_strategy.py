"""
Base Strategy Modülü

Bu modül, tüm trading stratejileri için temel abstract sınıfı içerir.
Yeni stratejiler bu sınıftan türetilmelidir.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import pandas as pd


class Signal(Enum):
    """Trading sinyalleri."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """Trade sinyali veri sınıfı."""
    signal: Signal
    symbol: str
    price: float
    confidence: float  # 0-1 arası güven skoru
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None
    reason: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Sinyali dictionary'ye çevir."""
        return {
            'signal': self.signal.value,
            'symbol': self.symbol,
            'price': self.price,
            'confidence': self.confidence,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp.isoformat(),
            'reason': self.reason,
            'metadata': self.metadata
        }


class BaseStrategy(ABC):
    """
    Temel strateji abstract sınıfı.
    
    Tüm trading stratejileri bu sınıftan türetilmeli ve
    gerekli metodları implement etmelidir.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        BaseStrategy'yi başlat.
        
        Args:
            config: Strateji konfigürasyonu
        """
        self.config = config
        self.symbol = config.get('symbol', 'BTCUSDT')
        self.timeframe = config.get('timeframe', '15m')
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2)
        self.stop_loss_pct = config.get('stop_loss_pct', 2)
        self.take_profit_pct = config.get('take_profit_pct', 4)
        self._is_active = True
        self._last_signal: Optional[TradeSignal] = None
        self._signal_history: List[TradeSignal] = []
    
    @property
    def name(self) -> str:
        """Strateji adını döndür."""
        return self.__class__.__name__
    
    @property
    def is_active(self) -> bool:
        """Strateji aktif mi kontrol et."""
        return self._is_active
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> TradeSignal:
        """
        Piyasa verisini analiz et ve sinyal üret.
        
        Args:
            data: OHLCV DataFrame'i (indikatörler dahil)
            
        Returns:
            TradeSignal nesnesi
        """
        pass
    
    @abstractmethod
    def should_enter(self, data: pd.DataFrame) -> bool:
        """
        Pozisyona girmeli mi kontrol et.
        
        Args:
            data: OHLCV DataFrame'i
            
        Returns:
            Girmeli ise True
        """
        pass
    
    @abstractmethod
    def should_exit(self, data: pd.DataFrame, position_side: str) -> bool:
        """
        Pozisyondan çıkmalı mı kontrol et.
        
        Args:
            data: OHLCV DataFrame'i
            position_side: Mevcut pozisyon yönü ('LONG' veya 'SHORT')
            
        Returns:
            Çıkmalı ise True
        """
        pass
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """
        Stop-loss fiyatını hesapla.
        
        Args:
            entry_price: Giriş fiyatı
            side: Pozisyon yönü ('LONG' veya 'SHORT')
            
        Returns:
            Stop-loss fiyatı
        """
        if side.upper() == 'LONG':
            return entry_price * (1 - self.stop_loss_pct / 100)
        else:
            return entry_price * (1 + self.stop_loss_pct / 100)
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """
        Take-profit fiyatını hesapla.
        
        Args:
            entry_price: Giriş fiyatı
            side: Pozisyon yönü ('LONG' veya 'SHORT')
            
        Returns:
            Take-profit fiyatı
        """
        if side.upper() == 'LONG':
            return entry_price * (1 + self.take_profit_pct / 100)
        else:
            return entry_price * (1 - self.take_profit_pct / 100)
    
    def validate_signal(self, signal: TradeSignal) -> bool:
        """
        Sinyalin geçerliliğini kontrol et.
        
        Args:
            signal: Kontrol edilecek sinyal
            
        Returns:
            Geçerli ise True
        """
        # Minimum güven kontrolü
        if signal.confidence < 0.5:
            return False
        
        # Risk/Reward kontrolü
        if signal.stop_loss and signal.take_profit:
            risk = abs(signal.price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.price)
            
            if risk > 0 and reward / risk < self.risk_reward_ratio:
                return False
        
        return True
    
    def update_signal_history(self, signal: TradeSignal) -> None:
        """
        Sinyal geçmişini güncelle.
        
        Args:
            signal: Eklenecek sinyal
        """
        self._signal_history.append(signal)
        self._last_signal = signal
        
        # Geçmiş boyutunu sınırla
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-500:]
    
    def get_last_signal(self) -> Optional[TradeSignal]:
        """Son sinyali döndür."""
        return self._last_signal
    
    def get_signal_history(self, limit: int = 100) -> List[TradeSignal]:
        """
        Sinyal geçmişini döndür.
        
        Args:
            limit: Maksimum sinyal sayısı
            
        Returns:
            Sinyal listesi
        """
        return self._signal_history[-limit:]
    
    def activate(self) -> None:
        """Stratejiyi aktifleştir."""
        self._is_active = True
    
    def deactivate(self) -> None:
        """Stratejiyi devre dışı bırak."""
        self._is_active = False
    
    def reset(self) -> None:
        """Strateji durumunu sıfırla."""
        self._signal_history = []
        self._last_signal = None
    
    def backtest(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        position_size_pct: float = 10.0
    ) -> Dict[str, Any]:
        """
        Stratejiyi geçmiş veri üzerinde test et.
        
        Args:
            data: OHLCV DataFrame'i
            initial_balance: Başlangıç bakiyesi
            position_size_pct: Pozisyon boyutu yüzdesi
            
        Returns:
            Backtest sonuçları
        """
        balance = initial_balance
        position = None
        trades = []
        
        for i in range(50, len(data)):
            current_data = data.iloc[:i+1].copy()
            current_price = current_data['close'].iloc[-1]
            
            signal = self.analyze(current_data)
            
            if position is None:
                # Pozisyon yok, giriş sinyali kontrol et
                if signal.signal == Signal.BUY:
                    # Long pozisyon aç
                    position_value = balance * (position_size_pct / 100)
                    quantity = position_value / current_price
                    position = {
                        'side': 'LONG',
                        'entry_price': current_price,
                        'quantity': quantity,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'entry_time': current_data.index[-1]
                    }
                elif signal.signal == Signal.SELL:
                    # Short pozisyon aç
                    position_value = balance * (position_size_pct / 100)
                    quantity = position_value / current_price
                    position = {
                        'side': 'SHORT',
                        'entry_price': current_price,
                        'quantity': quantity,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'entry_time': current_data.index[-1]
                    }
            else:
                # Pozisyon var, çıkış kontrol et
                should_close = False
                exit_reason = ''
                
                if position['side'] == 'LONG':
                    if position['stop_loss'] and current_price <= position['stop_loss']:
                        should_close = True
                        exit_reason = 'Stop-loss'
                    elif position['take_profit'] and current_price >= position['take_profit']:
                        should_close = True
                        exit_reason = 'Take-profit'
                    elif signal.signal == Signal.SELL:
                        should_close = True
                        exit_reason = 'Signal'
                else:  # SHORT
                    if position['stop_loss'] and current_price >= position['stop_loss']:
                        should_close = True
                        exit_reason = 'Stop-loss'
                    elif position['take_profit'] and current_price <= position['take_profit']:
                        should_close = True
                        exit_reason = 'Take-profit'
                    elif signal.signal == Signal.BUY:
                        should_close = True
                        exit_reason = 'Signal'
                
                if should_close:
                    # Pozisyonu kapat
                    if position['side'] == 'LONG':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    balance += pnl
                    
                    trades.append({
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                        'exit_reason': exit_reason,
                        'entry_time': position['entry_time'],
                        'exit_time': current_data.index[-1]
                    })
                    
                    position = None
        
        # Sonuçları hesapla
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            results = {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': ((balance - initial_balance) / initial_balance) * 100,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(trades)) * 100 if trades else 0,
                'avg_win': sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                'avg_loss': sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
                'max_drawdown': self._calculate_max_drawdown(trades, initial_balance),
                'trades': trades
            }
        else:
            results = {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'trades': []
            }
        
        return results
    
    def _calculate_max_drawdown(
        self,
        trades: List[Dict],
        initial_balance: float
    ) -> float:
        """Maksimum drawdown hesapla."""
        if not trades:
            return 0.0
        
        balance = initial_balance
        peak = balance
        max_dd = 0.0
        
        for trade in trades:
            balance += trade['pnl']
            peak = max(peak, balance)
            drawdown = ((peak - balance) / peak) * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
