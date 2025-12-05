"""
Hibrit Strateji Modülü

Bu modül, Momentum ve ML stratejilerini birleştiren
hibrit bir trading stratejisi içerir.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.ml_strategy import MLStrategy
from src.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridStrategy(BaseStrategy):
    """
    Hibrit trading stratejisi.
    
    Momentum ve ML stratejilerini birleştirerek
    daha güvenilir sinyaller üretir.
    
    Sinyal Kombinasyonları:
    - Tam Uyum: Momentum UP + ML UP = STRONG LONG (+20% confidence bonus)
    - Tam Uyum: Momentum DOWN + ML DOWN = STRONG SHORT (+20% confidence bonus)
    - Kısmi Uyum: Momentum UP + ML Nötr = LONG (normal confidence)
    - Kısmi Uyum: ML DOWN + Momentum Nötr = SHORT (normal confidence)
    - Çelişki: Momentum UP + ML DOWN = HOLD (trade yok)
    """
    
    def __init__(self, config: Dict[str, Any], predictor=None):
        """
        HybridStrategy'yi başlat.
        
        Args:
            config: Strateji konfigürasyonu
            predictor: AI tahmin modülü (opsiyonel)
        """
        super().__init__(config)
        
        self.predictor = predictor
        
        # Hibrit parametreleri
        hybrid_config = config.get('hybrid', {})
        self.momentum_weight = hybrid_config.get('momentum_weight', 0.5)
        self.ml_weight = hybrid_config.get('ml_weight', 0.5)
        self.min_combined_confidence = hybrid_config.get('min_combined_confidence', 0.45)
        self.require_agreement = hybrid_config.get('require_agreement', False)
        self.agreement_bonus = hybrid_config.get('agreement_bonus', 0.2)
        self.conflict_penalty = hybrid_config.get('conflict_penalty', 0.5)
        
        # Alt stratejiler
        self.momentum_strategy = MomentumStrategy(config)
        self.ml_strategy = MLStrategy(config, predictor)
        
        # Maksimum güven skoru sınırı
        self.max_confidence = 0.95
        
        logger.info(
            f"HybridStrategy başlatıldı - "
            f"Momentum Ağırlık: {self.momentum_weight}, "
            f"ML Ağırlık: {self.ml_weight}, "
            f"Min Confidence: {self.min_combined_confidence}"
        )
    
    def set_predictor(self, predictor) -> None:
        """
        Predictor'ı ayarla.
        
        Args:
            predictor: AI tahmin modülü
        """
        self.predictor = predictor
        self.ml_strategy.set_predictor(predictor)
        logger.info("Predictor ayarlandı")
    
    def _get_momentum_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Momentum stratejisinden sinyal al.
        
        Args:
            data: OHLCV DataFrame'i
            
        Returns:
            Momentum sinyal sonuçları
        """
        signal = self.momentum_strategy.analyze(data)
        
        if signal.signal == Signal.BUY:
            direction = 'up'
        elif signal.signal == Signal.SELL:
            direction = 'down'
        else:
            direction = 'neutral'
        
        return {
            'direction': direction,
            'confidence': signal.confidence,
            'signal': signal,
            'reason': signal.reason
        }
    
    def _get_ml_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        ML stratejisinden sinyal al.
        
        Args:
            data: OHLCV DataFrame'i
            
        Returns:
            ML sinyal sonuçları
        """
        signal = self.ml_strategy.analyze(data)
        
        if signal.signal == Signal.BUY:
            direction = 'up'
        elif signal.signal == Signal.SELL:
            direction = 'down'
        else:
            direction = 'neutral'
        
        return {
            'direction': direction,
            'confidence': signal.confidence,
            'signal': signal,
            'reason': signal.reason,
            'metadata': signal.metadata
        }
    
    def _combine_signals(
        self,
        momentum: Dict[str, Any],
        ml: Dict[str, Any],
        current_price: float
    ) -> TradeSignal:
        """
        Momentum ve ML sinyallerini birleştir.
        
        Args:
            momentum: Momentum sinyal sonuçları
            ml: ML sinyal sonuçları
            current_price: Güncel fiyat
            
        Returns:
            Birleştirilmiş TradeSignal
        """
        mom_dir = momentum['direction']
        ml_dir = ml['direction']
        mom_conf = momentum['confidence']
        ml_conf = ml['confidence']
        
        # Ağırlıklı güven skorları
        weighted_mom = mom_conf * self.momentum_weight
        weighted_ml = ml_conf * self.ml_weight
        
        # Uyum durumunu belirle
        final_direction = 'neutral'
        confidence_modifier = 1.0
        
        if mom_dir == ml_dir and mom_dir != 'neutral':
            # Tam uyum
            agreement = 'full'
            final_direction = mom_dir
            confidence_modifier = 1 + self.agreement_bonus
            
        elif mom_dir != 'neutral' and ml_dir != 'neutral' and mom_dir != ml_dir:
            # Çelişki
            agreement = 'conflict'
            final_direction = 'neutral'
            confidence_modifier = self.conflict_penalty
            
        elif mom_dir != 'neutral' and ml_dir == 'neutral':
            # Kısmi uyum - Momentum tarafı
            agreement = 'partial_momentum'
            final_direction = mom_dir
            confidence_modifier = 0.9
            
        elif ml_dir != 'neutral' and mom_dir == 'neutral':
            # Kısmi uyum - ML tarafı
            agreement = 'partial_ml'
            final_direction = ml_dir
            confidence_modifier = 0.9
            
        else:
            # Her ikisi de nötr
            agreement = 'neutral'
            final_direction = 'neutral'
        
        # Final güven skorunu hesapla
        base_confidence = weighted_mom + weighted_ml
        final_confidence = min(base_confidence * confidence_modifier, self.max_confidence)
        
        # require_agreement kontrolü
        if self.require_agreement and agreement not in ['full']:
            final_direction = 'neutral'
            final_confidence = min(final_confidence, 0.3)
        
        # Sinyal oluştur
        if final_direction == 'up' and final_confidence >= self.min_combined_confidence:
            stop_loss = self.calculate_stop_loss(current_price, 'LONG')
            take_profit = self.calculate_take_profit(current_price, 'LONG')
            
            signal = TradeSignal(
                signal=Signal.BUY,
                symbol=self.symbol,
                price=current_price,
                confidence=final_confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Hibrit LONG - Uyum: {agreement}, Mom: {mom_dir}({mom_conf:.2f}), ML: {ml_dir}({ml_conf:.2f})",
                metadata={
                    'agreement': agreement,
                    'momentum_direction': mom_dir,
                    'ml_direction': ml_dir,
                    'momentum_confidence': mom_conf,
                    'ml_confidence': ml_conf
                }
            )
            
        elif final_direction == 'down' and final_confidence >= self.min_combined_confidence:
            stop_loss = self.calculate_stop_loss(current_price, 'SHORT')
            take_profit = self.calculate_take_profit(current_price, 'SHORT')
            
            signal = TradeSignal(
                signal=Signal.SELL,
                symbol=self.symbol,
                price=current_price,
                confidence=final_confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Hibrit SHORT - Uyum: {agreement}, Mom: {mom_dir}({mom_conf:.2f}), ML: {ml_dir}({ml_conf:.2f})",
                metadata={
                    'agreement': agreement,
                    'momentum_direction': mom_dir,
                    'ml_direction': ml_dir,
                    'momentum_confidence': mom_conf,
                    'ml_confidence': ml_conf
                }
            )
            
        else:
            signal = TradeSignal(
                signal=Signal.HOLD,
                symbol=self.symbol,
                price=current_price,
                confidence=final_confidence,
                reason=f"Hibrit HOLD - Uyum: {agreement}, Mom: {mom_dir}({mom_conf:.2f}), ML: {ml_dir}({ml_conf:.2f})",
                metadata={
                    'agreement': agreement,
                    'momentum_direction': mom_dir,
                    'ml_direction': ml_dir,
                    'momentum_confidence': mom_conf,
                    'ml_confidence': ml_conf
                }
            )
        
        return signal
    
    def analyze(self, data: pd.DataFrame) -> TradeSignal:
        """
        Piyasa verisini analiz et ve sinyal üret.
        
        Args:
            data: OHLCV DataFrame'i
            
        Returns:
            TradeSignal nesnesi
        """
        if len(data) < 60:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=self.symbol,
                price=data['close'].iloc[-1],
                confidence=0.0,
                reason="Yetersiz veri"
            )
        
        current_price = data['close'].iloc[-1]
        
        # Alt stratejilerden sinyalleri al
        momentum_result = self._get_momentum_signal(data)
        ml_result = self._get_ml_signal(data)
        
        logger.debug(
            f"Momentum: {momentum_result['direction']} ({momentum_result['confidence']:.2f}), "
            f"ML: {ml_result['direction']} ({ml_result['confidence']:.2f})"
        )
        
        # Sinyalleri birleştir
        signal = self._combine_signals(momentum_result, ml_result, current_price)
        
        self.update_signal_history(signal)
        return signal
    
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
            signal.confidence >= self.min_combined_confidence
        )
    
    def should_exit(self, data: pd.DataFrame, position_side: str) -> bool:
        """
        Pozisyondan çıkmalı mı kontrol et.
        
        Args:
            data: OHLCV DataFrame'i
            position_side: Mevcut pozisyon yönü ('LONG' veya 'SHORT')
            
        Returns:
            Çıkmalı ise True
        """
        signal = self.analyze(data)
        
        # Ters sinyal geldi ve tam uyum var ise çık
        if position_side.upper() == 'LONG' and signal.signal == Signal.SELL:
            if signal.metadata and signal.metadata.get('agreement') == 'full':
                return True
        elif position_side.upper() == 'SHORT' and signal.signal == Signal.BUY:
            if signal.metadata and signal.metadata.get('agreement') == 'full':
                return True
        
        return False
    
    @property
    def name(self) -> str:
        """Strateji adını döndür."""
        return "HybridStrategy"
