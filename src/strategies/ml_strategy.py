"""
ML Strategy Modülü

Bu modül, machine learning tabanlı trading stratejisini içerir.
AI model tahminlerine dayalı trading sinyalleri üretir.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal
from src.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLStrategy(BaseStrategy):
    """
    Machine Learning tabanlı trading stratejisi.
    
    Bu strateji, LSTM veya Transformer modeli tahminlerini
    kullanarak trading sinyalleri üretir.
    
    Özellikler:
    - Model tahminlerine dayalı sinyal üretimi
    - Güven skoruna göre pozisyon boyutlandırma
    - Teknik indikatörlerle doğrulama
    - Dinamik stop-loss ve take-profit
    """
    
    def __init__(self, config: Dict[str, Any], predictor=None):
        """
        MLStrategy'yi başlat.
        
        Args:
            config: Strateji konfigürasyonu
            predictor: AI tahmin modülü (opsiyonel)
        """
        super().__init__(config)
        
        self.predictor = predictor
        
        # ML parametreleri
        ml_config = config.get('ml', {})
        self.min_confidence = ml_config.get('min_confidence', 0.65)
        self.position_sizing_method = ml_config.get('position_sizing_method', 'kelly')
        
        # AI parametreleri
        ai_config = config.get('ai', {})
        self.lookback_period = ai_config.get('lookback_period', 60)
        self.prediction_horizon = ai_config.get('prediction_horizon', 5)
        self.confidence_threshold = ai_config.get('confidence_threshold', 0.6)
        
        # Doğrulama için teknik indikatör parametreleri
        self.use_technical_confirmation = True
        self.rsi_filter_enabled = True
        self.trend_filter_enabled = True
        
        logger.info(
            f"MLStrategy başlatıldı - "
            f"Min Confidence: {self.min_confidence}, "
            f"Position Sizing: {self.position_sizing_method}"
        )
    
    def set_predictor(self, predictor) -> None:
        """
        Predictor'ı ayarla.
        
        Args:
            predictor: AI tahmin modülü
        """
        self.predictor = predictor
        logger.info("Predictor ayarlandı")
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """İndikatörleri hesapla ve veriye ekle."""
        df = data.copy()
        
        # Temel indikatörler
        if 'rsi' not in df.columns:
            df['rsi'] = TechnicalIndicators.rsi(df['close'], 14)
        
        if 'macd' not in df.columns:
            macd, signal, hist = TechnicalIndicators.macd(df['close'])
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_histogram'] = hist
        
        if 'ema_9' not in df.columns:
            df['ema_9'] = TechnicalIndicators.ema(df['close'], 9)
        
        if 'ema_21' not in df.columns:
            df['ema_21'] = TechnicalIndicators.ema(df['close'], 21)
        
        if 'ema_50' not in df.columns:
            df['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
        
        if 'bb_upper' not in df.columns:
            upper, middle, lower = TechnicalIndicators.bollinger_bands(df['close'])
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
        
        if 'atr' not in df.columns:
            df['atr'] = TechnicalIndicators.atr(
                df['high'], df['low'], df['close']
            )
        
        if 'volume_sma' not in df.columns:
            df['volume_sma'] = TechnicalIndicators.sma(df['volume'], 20)
        
        return df
    
    def _get_ml_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        ML modelinden tahmin al.
        
        Args:
            data: OHLCV DataFrame'i
            
        Returns:
            Tahmin sonuçları
        """
        if self.predictor is None:
            # Predictor yoksa basit momentum tahmini kullan
            return self._simple_momentum_prediction(data)
        
        try:
            # Predictor'dan tahmin al
            prediction = self.predictor.predict(data)
            return prediction
        except Exception as e:
            logger.warning(f"ML tahmin hatası: {e}")
            return self._simple_momentum_prediction(data)
    
    def _simple_momentum_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Basit momentum bazlı tahmin (fallback).
        Daha agresif sinyal üretimi için optimize edildi.
        
        Args:
            data: OHLCV DataFrame'i
            
        Returns:
            Tahmin sonuçları
        """
        df = self._prepare_data(data)
        current = df.iloc[-1]
        
        # Son 5 mum analizi
        recent_closes = df['close'].tail(5)
        momentum = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
        
        # Daha agresif trend skoru
        if current['ema_9'] > current['ema_21'] > current['ema_50']:
            trend_score = 0.75  # 0.7'den 0.75'e
            trend_direction = 'up'
        elif current['ema_9'] < current['ema_21'] < current['ema_50']:
            trend_score = 0.75
            trend_direction = 'down'
        elif current['ema_9'] > current['ema_21']:
            trend_score = 0.55  # Yeni - kısa vadeli trend
            trend_direction = 'up'
        elif current['ema_9'] < current['ema_21']:
            trend_score = 0.55
            trend_direction = 'down'
        else:
            trend_score = 0.35
            trend_direction = 'neutral'
        
        # RSI bazlı skor - daha geniş aralık
        rsi = current['rsi']
        if rsi < 35:  # 30'dan 35'e
            rsi_score = 0.75
            rsi_direction = 'up'
        elif rsi > 65:  # 70'den 65'e
            rsi_score = 0.75
            rsi_direction = 'down'
        elif rsi < 45:
            rsi_score = 0.55
            rsi_direction = 'up'
        elif rsi > 55:
            rsi_score = 0.55
            rsi_direction = 'down'
        else:
            rsi_score = 0.40
            rsi_direction = 'neutral'
        
        # MACD bazlı skor
        if current['macd'] > current['macd_signal'] and current['macd_histogram'] > 0:
            macd_score = 0.65
            macd_direction = 'up'
        elif current['macd'] < current['macd_signal'] and current['macd_histogram'] < 0:
            macd_score = 0.65
            macd_direction = 'down'
        elif current['macd'] > current['macd_signal']:
            macd_score = 0.50
            macd_direction = 'up'
        elif current['macd'] < current['macd_signal']:
            macd_score = 0.50
            macd_direction = 'down'
        else:
            macd_score = 0.35
            macd_direction = 'neutral'
        
        # Momentum bazlı bonus
        momentum_bonus = 0
        if abs(momentum) > 0.005:  # %0.5'ten fazla hareket
            momentum_bonus = 0.1
        
        # Yön belirleme - ağırlıklı oylama
        directions = {
            'up': 0,
            'down': 0,
            'neutral': 0
        }
        
        directions[trend_direction] += trend_score
        directions[macd_direction] += macd_score
        directions[rsi_direction] += rsi_score * 0.8  # RSI biraz daha az ağırlık
        
        # En yüksek skoru bul
        max_direction = max(directions, key=directions.get)
        max_score = directions[max_direction]
        
        if max_direction != 'neutral' and max_score > 0.8:
            direction = max_direction
            confidence = min((max_score / 2.5) + momentum_bonus, 0.85)  # Normalize et
        else:
            direction = 'neutral'
            confidence = 0.35
        
        # Fiyat tahmini
        if momentum > 0.005:
            predicted_change = momentum * 0.5
        elif momentum < -0.005:
            predicted_change = momentum * 0.5
        else:
            predicted_change = 0
        
        predicted_price = current['close'] * (1 + predicted_change)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'predicted_change_pct': predicted_change * 100,
            'trend_score': trend_score,
            'rsi_score': rsi_score,
            'macd_score': macd_score,
            'momentum': momentum,
            'is_fallback': True
        }
    
    def _apply_technical_filters(
        self,
        prediction: Dict[str, Any],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Teknik filtreler uygula - yumuşatılmış versiyon.
        
        Args:
            prediction: ML tahmini
            data: OHLCV DataFrame'i
            
        Returns:
            Filtrelenmiş tahmin
        """
        df = self._prepare_data(data)
        current = df.iloc[-1]
        
        direction = prediction['direction']
        confidence = prediction['confidence']
        
        # RSI filtresi - daha yumuşak
        if self.rsi_filter_enabled:
            rsi = current['rsi']
            
            if direction == 'up' and rsi > 75:  # 70'den 75'e
                confidence *= 0.75  # 0.6'dan 0.75'e
                prediction['filter_reason'] = 'RSI overbought'
            elif direction == 'down' and rsi < 25:  # 30'dan 25'e
                confidence *= 0.75
                prediction['filter_reason'] = 'RSI oversold'
        
        # Trend filtresi - daha yumuşak
        if self.trend_filter_enabled:
            ema_bullish = current['ema_9'] > current['ema_21']
            
            if direction == 'up' and not ema_bullish:
                confidence *= 0.85  # 0.7'den 0.85'e
            elif direction == 'down' and ema_bullish:
                confidence *= 0.85
        
        # Bollinger Bands filtresi - daha yumuşak
        bb_position = (current['close'] - current['bb_lower']) / (
            current['bb_upper'] - current['bb_lower']
        )
        
        if direction == 'up' and bb_position > 0.95:  # 0.9'dan 0.95'e
            confidence *= 0.9  # 0.8'den 0.9'a
        elif direction == 'down' and bb_position < 0.05:  # 0.1'den 0.05'e
            confidence *= 0.9
        
        prediction['confidence'] = confidence
        prediction['bb_position'] = bb_position
        
        return prediction
    
    def analyze(self, data: pd.DataFrame) -> TradeSignal:
        """
        Piyasa verisini analiz et ve sinyal üret.
        
        Args:
            data: OHLCV DataFrame'i
            
        Returns:
            TradeSignal nesnesi
        """
        df = self._prepare_data(data)
        
        if len(df) < self.lookback_period:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=self.symbol,
                price=df['close'].iloc[-1],
                confidence=0.0,
                reason="Yetersiz veri"
            )
        
        current = df.iloc[-1]
        price = current['close']
        
        # ML tahminini al
        prediction = self._get_ml_prediction(df)
        
        # Teknik filtreleri uygula
        if self.use_technical_confirmation:
            prediction = self._apply_technical_filters(prediction, df)
        
        direction = prediction['direction']
        confidence = prediction['confidence']
        
        # Minimum güven kontrolü
        if confidence < self.min_confidence:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=self.symbol,
                price=price,
                confidence=confidence,
                reason=f"Düşük güven skoru ({confidence:.2f} < {self.min_confidence})",
                metadata=prediction
            )
        
        # Sinyal oluştur
        if direction == 'up':
            stop_loss = self.calculate_stop_loss(price, 'LONG')
            take_profit = self.calculate_take_profit(price, 'LONG')
            
            # ATR bazlı dinamik stop-loss
            atr_stop = price - (current['atr'] * 2)
            stop_loss = max(stop_loss, atr_stop)
            
            signal = TradeSignal(
                signal=Signal.BUY,
                symbol=self.symbol,
                price=price,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"ML Long sinyali (güven: {confidence:.2f})",
                metadata=prediction
            )
            
        elif direction == 'down':
            stop_loss = self.calculate_stop_loss(price, 'SHORT')
            take_profit = self.calculate_take_profit(price, 'SHORT')
            
            # ATR bazlı dinamik stop-loss
            atr_stop = price + (current['atr'] * 2)
            stop_loss = min(stop_loss, atr_stop)
            
            signal = TradeSignal(
                signal=Signal.SELL,
                symbol=self.symbol,
                price=price,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"ML Short sinyali (güven: {confidence:.2f})",
                metadata=prediction
            )
            
        else:
            signal = TradeSignal(
                signal=Signal.HOLD,
                symbol=self.symbol,
                price=price,
                confidence=confidence,
                reason="Nötr tahmin",
                metadata=prediction
            )
        
        self.update_signal_history(signal)
        return signal
    
    def calculate_position_size(
        self,
        balance: float,
        confidence: float,
        base_risk_pct: float = 2.0
    ) -> float:
        """
        Güven skoruna göre pozisyon boyutu hesapla.
        
        Args:
            balance: Hesap bakiyesi
            confidence: Güven skoru (0-1)
            base_risk_pct: Baz risk yüzdesi
            
        Returns:
            Risk yüzdesi
        """
        if self.position_sizing_method == 'kelly':
            # Kelly Criterion benzeri ölçeklendirme
            # Güven skoru arttıkça risk artır
            kelly_factor = (confidence - 0.5) * 2  # -1 ile 1 arası
            risk_pct = base_risk_pct * (1 + kelly_factor * 0.5)
        elif self.position_sizing_method == 'volatility':
            # Volatilite bazlı (ATR kullanılabilir)
            risk_pct = base_risk_pct * confidence
        else:
            # Sabit
            risk_pct = base_risk_pct
        
        # Sınırlar
        risk_pct = max(0.5, min(risk_pct, 5.0))
        
        return risk_pct
    
    def should_enter(self, data: pd.DataFrame) -> bool:
        """
        Pozisyona girmeli mi kontrol et.
        
        Args:
            data: OHLCV DataFrame'i
            
        Returns:
            Girmeli ise True
        """
        signal = self.analyze(data)
        return (
            signal.signal in [Signal.BUY, Signal.SELL] and 
            signal.confidence >= self.min_confidence
        )
    
    def should_exit(self, data: pd.DataFrame, position_side: str) -> bool:
        """
        Pozisyondan çıkmalı mı kontrol et.
        
        Args:
            data: OHLCV DataFrame'i
            position_side: Mevcut pozisyon yönü
            
        Returns:
            Çıkmalı ise True
        """
        signal = self.analyze(data)
        
        # Ters sinyal geldi mi
        if position_side.upper() == 'LONG' and signal.signal == Signal.SELL:
            if signal.confidence >= self.min_confidence * 0.8:  # Biraz daha düşük eşik
                return True
        elif position_side.upper() == 'SHORT' and signal.signal == Signal.BUY:
            if signal.confidence >= self.min_confidence * 0.8:
                return True
        
        # Düşük güvenli HOLD sinyali - pozisyonu koru
        return False
