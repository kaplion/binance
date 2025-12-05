"""
Momentum Strategy Modülü

Bu modül, momentum ve trend takip stratejisini içerir.
RSI, MACD ve hacim analizini kullanarak trading sinyalleri üretir.
"""

from typing import Any, Dict, Optional
import pandas as pd

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal
from src.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum trading stratejisi.
    
    Bu strateji, RSI, MACD ve hacim analizini kullanarak
    trend yönünde işlem sinyalleri üretir.
    
    Giriş Koşulları (LONG):
    - RSI oversold bölgesinden çıkıyor (< 30'dan yukarı)
    - MACD çizgisi sinyal çizgisini yukarı kesiyor
    - Hacim ortalamanın üzerinde
    - EMA 9 > EMA 21 (trend doğrulaması)
    
    Giriş Koşulları (SHORT):
    - RSI overbought bölgesinden çıkıyor (> 70'den aşağı)
    - MACD çizgisi sinyal çizgisini aşağı kesiyor
    - Hacim ortalamanın üzerinde
    - EMA 9 < EMA 21 (trend doğrulaması)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        MomentumStrategy'yi başlat.
        
        Args:
            config: Strateji konfigürasyonu
        """
        super().__init__(config)
        
        # Momentum parametreleri
        momentum_config = config.get('momentum', {})
        self.rsi_overbought = momentum_config.get('rsi_overbought', 70)
        self.rsi_oversold = momentum_config.get('rsi_oversold', 30)
        self.macd_signal_period = momentum_config.get('macd_signal_period', 9)
        self.volume_multiplier = momentum_config.get('volume_multiplier', 1.5)
        
        # RSI parametreleri
        self.rsi_period = 14
        
        # EMA parametreleri
        self.ema_fast = 9
        self.ema_slow = 21
        
        logger.info(
            f"MomentumStrategy başlatıldı - "
            f"RSI: {self.rsi_oversold}/{self.rsi_overbought}, "
            f"Volume Multiplier: {self.volume_multiplier}"
        )
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """İndikatörleri hesapla ve veriye ekle."""
        df = data.copy()
        
        # Temel indikatörler yoksa hesapla
        if 'rsi' not in df.columns:
            df['rsi'] = TechnicalIndicators.rsi(df['close'], self.rsi_period)
        
        if 'macd' not in df.columns:
            macd, signal, hist = TechnicalIndicators.macd(df['close'])
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_histogram'] = hist
        
        if 'ema_9' not in df.columns:
            df['ema_9'] = TechnicalIndicators.ema(df['close'], self.ema_fast)
        
        if 'ema_21' not in df.columns:
            df['ema_21'] = TechnicalIndicators.ema(df['close'], self.ema_slow)
        
        if 'volume_sma' not in df.columns:
            df['volume_sma'] = TechnicalIndicators.sma(df['volume'], 20)
        
        if 'atr' not in df.columns:
            df['atr'] = TechnicalIndicators.atr(
                df['high'], df['low'], df['close']
            )
        
        return df
    
    def analyze(self, data: pd.DataFrame) -> TradeSignal:
        """
        Piyasa verisini analiz et ve sinyal üret.
        
        Args:
            data: OHLCV DataFrame'i
            
        Returns:
            TradeSignal nesnesi
        """
        df = self._prepare_data(data)
        
        if len(df) < 50:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=self.symbol,
                price=df['close'].iloc[-1],
                confidence=0.0,
                reason="Yetersiz veri"
            )
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        price = current['close']
        
        # Sinyalleri hesapla
        long_signal, long_confidence, long_reasons = self._check_long_signal(current, prev)
        short_signal, short_confidence, short_reasons = self._check_short_signal(current, prev)
        
        # En güçlü sinyali seç
        if long_signal and (not short_signal or long_confidence > short_confidence):
            stop_loss = self.calculate_stop_loss(price, 'LONG')
            take_profit = self.calculate_take_profit(price, 'LONG')
            
            # ATR bazlı stop-loss (daha dinamik)
            atr_stop = price - (current['atr'] * 2)
            stop_loss = max(stop_loss, atr_stop)
            
            signal = TradeSignal(
                signal=Signal.BUY,
                symbol=self.symbol,
                price=price,
                confidence=long_confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=", ".join(long_reasons),
                metadata={
                    'rsi': current['rsi'],
                    'macd': current['macd'],
                    'macd_signal': current['macd_signal'],
                    'ema_9': current['ema_9'],
                    'ema_21': current['ema_21'],
                    'atr': current['atr']
                }
            )
        elif short_signal:
            stop_loss = self.calculate_stop_loss(price, 'SHORT')
            take_profit = self.calculate_take_profit(price, 'SHORT')
            
            # ATR bazlı stop-loss (daha dinamik)
            atr_stop = price + (current['atr'] * 2)
            stop_loss = min(stop_loss, atr_stop)
            
            signal = TradeSignal(
                signal=Signal.SELL,
                symbol=self.symbol,
                price=price,
                confidence=short_confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=", ".join(short_reasons),
                metadata={
                    'rsi': current['rsi'],
                    'macd': current['macd'],
                    'macd_signal': current['macd_signal'],
                    'ema_9': current['ema_9'],
                    'ema_21': current['ema_21'],
                    'atr': current['atr']
                }
            )
        else:
            signal = TradeSignal(
                signal=Signal.HOLD,
                symbol=self.symbol,
                price=price,
                confidence=0.0,
                reason="Giriş koşulları sağlanmadı",
                metadata={
                    'rsi': current['rsi'],
                    'macd': current['macd'],
                    'ema_9': current['ema_9'],
                    'ema_21': current['ema_21']
                }
            )
        
        self.update_signal_history(signal)
        return signal
    
    def _check_long_signal(self, current, prev) -> tuple:
        """Long sinyali kontrol et."""
        signals = []
        reasons = []
        
        # RSI oversold'dan çıkıyor
        rsi_signal = (
            prev['rsi'] < self.rsi_oversold and 
            current['rsi'] > self.rsi_oversold
        )
        if rsi_signal:
            signals.append(0.3)
            reasons.append(f"RSI oversold çıkışı ({current['rsi']:.1f})")
        
        # RSI yükseliyor ve düşük bölgede
        rsi_rising = (
            current['rsi'] > prev['rsi'] and 
            current['rsi'] < 50
        )
        if rsi_rising:
            signals.append(0.1)
        
        # MACD crossover (yukarı)
        macd_crossover = (
            prev['macd'] < prev['macd_signal'] and 
            current['macd'] > current['macd_signal']
        )
        if macd_crossover:
            signals.append(0.3)
            reasons.append("MACD bullish crossover")
        
        # MACD histogram pozitife dönüyor
        macd_hist_positive = (
            prev['macd_histogram'] < 0 and 
            current['macd_histogram'] > 0
        )
        if macd_hist_positive:
            signals.append(0.1)
        
        # EMA trend
        ema_bullish = current['ema_9'] > current['ema_21']
        if ema_bullish:
            signals.append(0.15)
            reasons.append("EMA trend bullish")
        
        # Hacim doğrulaması
        volume_confirm = current['volume'] > (current['volume_sma'] * self.volume_multiplier)
        if volume_confirm:
            signals.append(0.15)
            reasons.append("Yüksek hacim")
        
        # Fiyat EMA'ların üzerinde
        price_above_ema = current['close'] > current['ema_21']
        if price_above_ema:
            signals.append(0.1)
        
        # Minimum koşul: En az 2 sinyal aktif olmalı
        if len(signals) >= 2:
            confidence = min(sum(signals), 0.95)
            return True, confidence, reasons
        
        return False, 0.0, []
    
    def _check_short_signal(self, current, prev) -> tuple:
        """Short sinyali kontrol et."""
        signals = []
        reasons = []
        
        # RSI overbought'tan çıkıyor
        rsi_signal = (
            prev['rsi'] > self.rsi_overbought and 
            current['rsi'] < self.rsi_overbought
        )
        if rsi_signal:
            signals.append(0.3)
            reasons.append(f"RSI overbought çıkışı ({current['rsi']:.1f})")
        
        # RSI düşüyor ve yüksek bölgede
        rsi_falling = (
            current['rsi'] < prev['rsi'] and 
            current['rsi'] > 50
        )
        if rsi_falling:
            signals.append(0.1)
        
        # MACD crossover (aşağı)
        macd_crossover = (
            prev['macd'] > prev['macd_signal'] and 
            current['macd'] < current['macd_signal']
        )
        if macd_crossover:
            signals.append(0.3)
            reasons.append("MACD bearish crossover")
        
        # MACD histogram negatife dönüyor
        macd_hist_negative = (
            prev['macd_histogram'] > 0 and 
            current['macd_histogram'] < 0
        )
        if macd_hist_negative:
            signals.append(0.1)
        
        # EMA trend
        ema_bearish = current['ema_9'] < current['ema_21']
        if ema_bearish:
            signals.append(0.15)
            reasons.append("EMA trend bearish")
        
        # Hacim doğrulaması
        volume_confirm = current['volume'] > (current['volume_sma'] * self.volume_multiplier)
        if volume_confirm:
            signals.append(0.15)
            reasons.append("Yüksek hacim")
        
        # Fiyat EMA'ların altında
        price_below_ema = current['close'] < current['ema_21']
        if price_below_ema:
            signals.append(0.1)
        
        # Minimum koşul: En az 2 sinyal aktif olmalı
        if len(signals) >= 2:
            confidence = min(sum(signals), 0.95)
            return True, confidence, reasons
        
        return False, 0.0, []
    
    def should_enter(self, data: pd.DataFrame) -> bool:
        """
        Pozisyona girmeli mi kontrol et.
        
        Args:
            data: OHLCV DataFrame'i
            
        Returns:
            Girmeli ise True
        """
        signal = self.analyze(data)
        return signal.signal in [Signal.BUY, Signal.SELL] and signal.confidence >= 0.5
    
    def should_exit(self, data: pd.DataFrame, position_side: str) -> bool:
        """
        Pozisyondan çıkmalı mı kontrol et.
        
        Args:
            data: OHLCV DataFrame'i
            position_side: Mevcut pozisyon yönü
            
        Returns:
            Çıkmalı ise True
        """
        df = self._prepare_data(data)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if position_side.upper() == 'LONG':
            # RSI overbought
            if current['rsi'] > self.rsi_overbought:
                return True
            
            # MACD bearish crossover
            if (prev['macd'] > prev['macd_signal'] and 
                current['macd'] < current['macd_signal']):
                return True
            
            # EMA trend değişimi
            if (prev['ema_9'] > prev['ema_21'] and 
                current['ema_9'] < current['ema_21']):
                return True
                
        else:  # SHORT
            # RSI oversold
            if current['rsi'] < self.rsi_oversold:
                return True
            
            # MACD bullish crossover
            if (prev['macd'] < prev['macd_signal'] and 
                current['macd'] > current['macd_signal']):
                return True
            
            # EMA trend değişimi
            if (prev['ema_9'] < prev['ema_21'] and 
                current['ema_9'] > current['ema_21']):
                return True
        
        return False
