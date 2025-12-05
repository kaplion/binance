"""
Risk Manager Modülü

Bu modül, trading işlemlerinde risk yönetimini sağlar.
Stop-loss, take-profit, pozisyon boyutlandırma ve kayıp limitleri yönetir.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.api.binance_client import BinanceClient
from src.trading.position_manager import PositionManager, Position, PositionSide
from src.utils.logger import get_logger
from src.utils.helpers import calculate_position_size

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk seviyeleri."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskMetrics:
    """Risk metrikleri veri sınıfı."""
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Metrikleri dictionary'ye çevir."""
        return {
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'risk_level': self.risk_level.value,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class TradeRecord:
    """Trade kaydı veri sınıfı."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percentage: float
    entry_time: datetime
    exit_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Trade kaydını dictionary'ye çevir."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat()
        }


class RiskManager:
    """
    Risk yönetim sınıfı.
    
    Bu sınıf, risk yönetimi işlemlerini yönetir:
    - Pozisyon boyutlandırma
    - Kayıp limitleri kontrolü
    - Drawdown hesaplama
    - Risk/Reward analizi
    - Margin call koruması
    """
    
    def __init__(
        self,
        client: BinanceClient,
        position_manager: PositionManager,
        config: Dict[str, Any]
    ):
        """
        RiskManager'ı başlat.
        
        Args:
            client: Binance API client
            position_manager: Pozisyon yöneticisi
            config: Risk konfigürasyonu
        """
        self.client = client
        self.position_manager = position_manager
        
        # Risk parametreleri
        self.stop_loss_pct = config.get('stop_loss_pct', 2)
        self.take_profit_pct = config.get('take_profit_pct', 4)
        self.max_daily_loss_pct = config.get('max_daily_loss_pct', 5)
        self.max_weekly_loss_pct = config.get('max_weekly_loss_pct', 15)
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2)
        self.max_position_pct = config.get('position_size_pct', 5)
        self.max_positions = config.get('max_positions', 3)
        
        # Trailing stop parametreleri
        self.trailing_stop_enabled = config.get('trailing_stop_enabled', True)
        self.trailing_stop_activation_pct = config.get('trailing_stop_activation_pct', 1.5)
        self.trailing_stop_callback_pct = config.get('trailing_stop_callback_pct', 0.5)
        
        # Risk metrikleri
        self.metrics = RiskMetrics()
        self._trade_history: List[TradeRecord] = []
        self._peak_balance: float = 0.0
        self._daily_start_balance: float = 0.0
        self._weekly_start_balance: float = 0.0
        self._last_daily_reset: datetime = datetime.now()
        self._last_weekly_reset: datetime = datetime.now()
        
        # İlk bakiyeyi al
        self._initialize_balances()
        
        logger.info("RiskManager başlatıldı")
    
    def _initialize_balances(self) -> None:
        """İlk bakiyeleri al."""
        try:
            balance = self.client.get_usdt_balance()
            self._peak_balance = balance
            self._daily_start_balance = balance
            self._weekly_start_balance = balance
            logger.debug(f"Başlangıç bakiyesi: {balance} USDT")
        except Exception as e:
            logger.error(f"Bakiye alınamadı: {e}")
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_percentage: Optional[float] = None
    ) -> float:
        """
        Risk bazlı pozisyon boyutu hesapla.
        
        Args:
            entry_price: Giriş fiyatı
            stop_loss_price: Stop-loss fiyatı
            risk_percentage: Risk yüzdesi (None ise varsayılan)
            
        Returns:
            Hesaplanan pozisyon boyutu
        """
        try:
            balance = self.client.get_usdt_balance()
            risk_pct = risk_percentage or self.max_position_pct
            leverage = self.position_manager.leverage
            
            quantity = calculate_position_size(
                balance=balance,
                risk_percentage=risk_pct,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                leverage=leverage
            )
            
            # Maksimum pozisyon kontrolü
            max_quantity = (balance * self.max_position_pct / 100) / entry_price * leverage
            quantity = min(quantity, max_quantity)
            
            logger.debug(f"Hesaplanan pozisyon boyutu: {quantity}")
            return quantity
            
        except Exception as e:
            logger.error(f"Pozisyon boyutu hesaplama hatası: {e}")
            return 0.0
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """
        Risk limitlerini kontrol et.
        
        Returns:
            Risk kontrol sonuçları
        """
        try:
            self._update_metrics()
            
            result = {
                'can_trade': True,
                'risk_level': self.metrics.risk_level.value,
                'warnings': [],
                'daily_loss_pct': abs(self.metrics.daily_pnl),
                'weekly_loss_pct': abs(self.metrics.weekly_pnl),
                'current_drawdown': self.metrics.current_drawdown
            }
            
            # Günlük kayıp kontrolü
            if self.metrics.daily_pnl < 0:
                daily_loss = abs(self.metrics.daily_pnl)
                if daily_loss >= self.max_daily_loss_pct:
                    result['can_trade'] = False
                    result['warnings'].append(
                        f"Günlük kayıp limiti aşıldı: {daily_loss:.2f}%"
                    )
                elif daily_loss >= self.max_daily_loss_pct * 0.8:
                    result['warnings'].append(
                        f"Günlük kayıp limite yaklaşıyor: {daily_loss:.2f}%"
                    )
            
            # Haftalık kayıp kontrolü
            if self.metrics.weekly_pnl < 0:
                weekly_loss = abs(self.metrics.weekly_pnl)
                if weekly_loss >= self.max_weekly_loss_pct:
                    result['can_trade'] = False
                    result['warnings'].append(
                        f"Haftalık kayıp limiti aşıldı: {weekly_loss:.2f}%"
                    )
                elif weekly_loss >= self.max_weekly_loss_pct * 0.8:
                    result['warnings'].append(
                        f"Haftalık kayıp limite yaklaşıyor: {weekly_loss:.2f}%"
                    )
            
            # Maksimum pozisyon kontrolü
            open_positions = len(self.position_manager.get_all_positions())
            if open_positions >= self.max_positions:
                result['can_trade'] = False
                result['warnings'].append(
                    f"Maksimum pozisyon sayısına ulaşıldı: {open_positions}"
                )
            
            # Drawdown kontrolü
            if self.metrics.current_drawdown >= 20:
                result['risk_level'] = RiskLevel.CRITICAL.value
                result['warnings'].append(
                    f"Kritik drawdown seviyesi: {self.metrics.current_drawdown:.2f}%"
                )
            elif self.metrics.current_drawdown >= 15:
                result['risk_level'] = RiskLevel.HIGH.value
            elif self.metrics.current_drawdown >= 10:
                result['risk_level'] = RiskLevel.MEDIUM.value
            
            return result
            
        except Exception as e:
            logger.error(f"Risk limitleri kontrol hatası: {e}")
            return {'can_trade': False, 'warnings': [str(e)]}
    
    def _update_metrics(self) -> None:
        """Risk metriklerini güncelle."""
        try:
            current_balance = self.client.get_usdt_balance()
            now = datetime.now()
            
            # Günlük/haftalık reset kontrolü
            if (now - self._last_daily_reset).days >= 1:
                self._daily_start_balance = current_balance
                self._last_daily_reset = now
            
            if (now - self._last_weekly_reset).days >= 7:
                self._weekly_start_balance = current_balance
                self._last_weekly_reset = now
            
            # PnL hesapla
            daily_pnl = ((current_balance - self._daily_start_balance) / 
                        self._daily_start_balance * 100) if self._daily_start_balance > 0 else 0
            
            weekly_pnl = ((current_balance - self._weekly_start_balance) / 
                         self._weekly_start_balance * 100) if self._weekly_start_balance > 0 else 0
            
            # Drawdown hesapla
            if current_balance > self._peak_balance:
                self._peak_balance = current_balance
            
            current_drawdown = ((self._peak_balance - current_balance) / 
                               self._peak_balance * 100) if self._peak_balance > 0 else 0
            
            # Metrikleri güncelle
            self.metrics.daily_pnl = daily_pnl
            self.metrics.weekly_pnl = weekly_pnl
            self.metrics.current_drawdown = current_drawdown
            
            if current_drawdown > self.metrics.max_drawdown:
                self.metrics.max_drawdown = current_drawdown
            
            # Win rate hesapla
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = (self.metrics.winning_trades / 
                                        self.metrics.total_trades * 100)
            
            # Risk seviyesini belirle
            if current_drawdown >= 20 or abs(daily_pnl) >= self.max_daily_loss_pct:
                self.metrics.risk_level = RiskLevel.CRITICAL
            elif current_drawdown >= 15 or abs(daily_pnl) >= self.max_daily_loss_pct * 0.8:
                self.metrics.risk_level = RiskLevel.HIGH
            elif current_drawdown >= 10 or abs(daily_pnl) >= self.max_daily_loss_pct * 0.5:
                self.metrics.risk_level = RiskLevel.MEDIUM
            else:
                self.metrics.risk_level = RiskLevel.LOW
            
            self.metrics.last_updated = now
            
        except Exception as e:
            logger.error(f"Metrik güncelleme hatası: {e}")
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime
    ) -> None:
        """
        Trade kaydı ekle.
        
        Args:
            symbol: Sembol
            side: LONG veya SHORT
            quantity: Miktar
            entry_price: Giriş fiyatı
            exit_price: Çıkış fiyatı
            entry_time: Giriş zamanı
            exit_time: Çıkış zamanı
        """
        # PnL hesapla
        if side.upper() == 'LONG':
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        pnl_percentage = (pnl / (entry_price * quantity)) * 100
        
        # Kaydı oluştur
        record = TradeRecord(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percentage=pnl_percentage,
            entry_time=entry_time,
            exit_time=exit_time
        )
        
        self._trade_history.append(record)
        
        # Metrikleri güncelle
        self.metrics.total_trades += 1
        self.metrics.total_pnl += pnl
        
        if pnl > 0:
            self.metrics.winning_trades += 1
        else:
            self.metrics.losing_trades += 1
        
        logger.debug(f"Trade kaydedildi: PnL={pnl:.2f} ({pnl_percentage:.2f}%)")
    
    def check_risk_reward(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float
    ) -> bool:
        """
        Risk/Reward oranını kontrol et.
        
        Args:
            entry_price: Giriş fiyatı
            stop_loss_price: Stop-loss fiyatı
            take_profit_price: Take-profit fiyatı
            
        Returns:
            Oran uygunsa True
        """
        risk = abs(entry_price - stop_loss_price)
        reward = abs(take_profit_price - entry_price)
        
        if risk == 0:
            return False
        
        ratio = reward / risk
        is_valid = ratio >= self.risk_reward_ratio
        
        if not is_valid:
            logger.warning(
                f"Risk/Reward oranı yetersiz: {ratio:.2f} "
                f"(minimum: {self.risk_reward_ratio})"
            )
        
        return is_valid
    
    def should_use_trailing_stop(self, position: Position, current_price: float) -> bool:
        """
        Trailing stop kullanılmalı mı kontrol et.
        
        Args:
            position: Pozisyon
            current_price: Mevcut fiyat
            
        Returns:
            Trailing stop aktive edilmeli ise True
        """
        if not self.trailing_stop_enabled:
            return False
        
        # Kar yüzdesini hesapla
        if position.side == PositionSide.LONG:
            profit_pct = ((current_price - position.entry_price) / 
                         position.entry_price * 100)
        else:
            profit_pct = ((position.entry_price - current_price) / 
                         position.entry_price * 100)
        
        return profit_pct >= self.trailing_stop_activation_pct
    
    def calculate_trailing_stop(
        self,
        position: Position,
        current_price: float,
        highest_price: float
    ) -> Optional[float]:
        """
        Trailing stop fiyatını hesapla.
        
        Args:
            position: Pozisyon
            current_price: Mevcut fiyat
            highest_price: En yüksek fiyat (LONG için), en düşük (SHORT için)
            
        Returns:
            Yeni stop-loss fiyatı veya None
        """
        if not self.should_use_trailing_stop(position, current_price):
            return None
        
        callback_distance = highest_price * (self.trailing_stop_callback_pct / 100)
        
        if position.side == PositionSide.LONG:
            new_stop = highest_price - callback_distance
            # Stop sadece yukarı hareket edebilir
            if position.stop_loss_price and new_stop <= position.stop_loss_price:
                return None
        else:
            new_stop = highest_price + callback_distance
            # Stop sadece aşağı hareket edebilir
            if position.stop_loss_price and new_stop >= position.stop_loss_price:
                return None
        
        return new_stop
    
    def get_metrics(self) -> Dict[str, Any]:
        """Risk metriklerini getir."""
        self._update_metrics()
        return self.metrics.to_dict()
    
    def get_trade_history(
        self,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Trade geçmişini getir.
        
        Args:
            limit: Maksimum kayıt sayısı
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            
        Returns:
            Trade kayıtları listesi
        """
        trades = self._trade_history
        
        if start_date:
            trades = [t for t in trades if t.entry_time >= start_date]
        
        if end_date:
            trades = [t for t in trades if t.exit_time <= end_date]
        
        return [t.to_dict() for t in trades[-limit:]]
    
    def emergency_close_all(self) -> bool:
        """
        Acil durum: Tüm pozisyonları kapat.
        
        Returns:
            Başarılı ise True
        """
        logger.warning("ACİL DURUM: Tüm pozisyonlar kapatılıyor!")
        
        try:
            positions = self.position_manager.get_all_positions()
            
            for position in positions:
                self.position_manager.close_position(position.side.value)
            
            logger.info("Tüm pozisyonlar kapatıldı")
            return True
            
        except Exception as e:
            logger.error(f"Acil pozisyon kapatma hatası: {e}")
            return False
