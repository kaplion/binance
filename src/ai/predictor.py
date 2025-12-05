"""
Predictor Modülü

Bu modül, eğitilmiş AI modellerini kullanarak
fiyat/trend tahminleri yapar.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from src.ai.data_processor import DataProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Tahmin sınıfı.
    
    Bu sınıf, eğitilmiş modelleri kullanarak
    piyasa tahminleri yapar.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_type: str = 'lstm',
        config: Dict[str, Any] = None
    ):
        """
        Predictor'ı başlat.
        
        Args:
            model_path: Eğitilmiş model dosyası yolu
            scaler_path: Scaler dosyası yolu
            model_type: Model tipi ('lstm' veya 'transformer')
            config: Konfigürasyon
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model_type = model_type
        self.config = config or {}
        
        self.model = None
        self.data_processor = None
        self._is_ready = False
        
        # Varsayılan parametreler
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.lookback_period = self.config.get('lookback_period', 60)
        self.prediction_horizon = self.config.get('prediction_horizon', 5)
        
        # Model ve scaler'ı yükle
        if model_path and scaler_path:
            self.load(model_path, scaler_path)
        
        logger.info(f"Predictor başlatıldı - Model tipi: {model_type}")
    
    def load(self, model_path: str, scaler_path: str) -> None:
        """
        Model ve scaler'ı yükle.
        
        Args:
            model_path: Model dosyası yolu
            scaler_path: Scaler dosyası yolu
        """
        try:
            # Scaler'ı yükle
            self.data_processor = DataProcessor(
                lookback_period=self.lookback_period,
                prediction_horizon=self.prediction_horizon
            )
            self.data_processor.load_scalers(scaler_path)
            
            # Modeli yükle
            if self.model_type == 'lstm':
                from src.ai.models.lstm_model import LSTMModel
                self.model = LSTMModel.load(model_path)
            elif self.model_type == 'transformer':
                from src.ai.models.transformer_model import TransformerModel
                self.model = TransformerModel.load(model_path)
            else:
                raise ValueError(f"Bilinmeyen model tipi: {self.model_type}")
            
            self._is_ready = True
            self.model_path = model_path
            self.scaler_path = scaler_path
            
            logger.info("Model ve scaler yüklendi")
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            self._is_ready = False
    
    def set_model(self, model: Any, data_processor: DataProcessor) -> None:
        """
        Model ve data processor'ı ayarla.
        
        Args:
            model: Eğitilmiş model
            data_processor: Veri işlemci
        """
        self.model = model
        self.data_processor = data_processor
        self._is_ready = True
        logger.info("Model ve data processor ayarlandı")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tahmin yap.
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            Tahmin sonuçları
        """
        if not self._is_ready:
            logger.warning("Model hazır değil, basit tahmin kullanılıyor")
            return self._simple_prediction(df)
        
        try:
            # Veriyi hazırla
            X = self.data_processor.prepare_inference_data(df)
            
            # Tahmin yap
            raw_prediction = self.model.predict(X)
            
            # Tahmin tipine göre yorumla
            if hasattr(self.model, 'output_type') and self.model.output_type == 'regression':
                return self._interpret_regression(raw_prediction, df)
            else:
                return self._interpret_classification(raw_prediction, df)
                
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return self._simple_prediction(df)
    
    def _interpret_classification(
        self,
        prediction: np.ndarray,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Sınıflandırma tahminini yorumla.
        
        Args:
            prediction: Ham tahmin
            df: Orijinal veri
            
        Returns:
            Yorumlanmış tahmin
        """
        current_price = df['close'].iloc[-1]
        
        # Sigmoid/softmax çıktısı varsayıyoruz
        prob = float(prediction[0][0]) if len(prediction.shape) > 1 else float(prediction[0])
        
        # Yön belirleme
        if prob > 0.5 + (self.confidence_threshold - 0.5):
            direction = 'up'
            confidence = prob
        elif prob < 0.5 - (self.confidence_threshold - 0.5):
            direction = 'down'
            confidence = 1 - prob
        else:
            direction = 'neutral'
            confidence = 0.5
        
        # Tahmini fiyat değişimi (basit tahmin)
        if direction == 'up':
            predicted_change_pct = confidence * 2  # Max %2
        elif direction == 'down':
            predicted_change_pct = -confidence * 2
        else:
            predicted_change_pct = 0
        
        predicted_price = current_price * (1 + predicted_change_pct / 100)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'probability': prob,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change_pct': predicted_change_pct,
            'prediction_horizon': self.prediction_horizon
        }
    
    def _interpret_regression(
        self,
        prediction: np.ndarray,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Regresyon tahminini yorumla.
        
        Args:
            prediction: Ham tahmin
            df: Orijinal veri
            
        Returns:
            Yorumlanmış tahmin
        """
        current_price = df['close'].iloc[-1]
        
        # Tahmin edilen değer
        predicted_value = float(prediction[0][0]) if len(prediction.shape) > 1 else float(prediction[0])
        
        # Orijinal ölçeğe geri dönüştür
        if self.data_processor._target_scaler is not None:
            predicted_value = self.data_processor.inverse_transform_target(
                np.array([predicted_value])
            )[0]
        
        # Yön ve güven belirleme
        change_pct = (predicted_value - current_price) / current_price * 100
        
        if change_pct > 0.5:
            direction = 'up'
            confidence = min(0.5 + abs(change_pct) / 10, 0.95)
        elif change_pct < -0.5:
            direction = 'down'
            confidence = min(0.5 + abs(change_pct) / 10, 0.95)
        else:
            direction = 'neutral'
            confidence = 0.5
        
        return {
            'direction': direction,
            'confidence': confidence,
            'current_price': current_price,
            'predicted_price': predicted_value,
            'predicted_change_pct': change_pct,
            'prediction_horizon': self.prediction_horizon
        }
    
    def _simple_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Basit momentum bazlı tahmin (fallback).
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            Basit tahmin sonuçları
        """
        current_price = df['close'].iloc[-1]
        
        # Son 10 mumun momentum'u
        if len(df) < 10:
            return {
                'direction': 'neutral',
                'confidence': 0.3,
                'current_price': current_price,
                'predicted_price': current_price,
                'predicted_change_pct': 0,
                'prediction_horizon': self.prediction_horizon
            }
        
        recent = df.tail(10)
        momentum = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        if momentum > 0.01:
            direction = 'up'
            confidence = min(0.5 + abs(momentum) * 10, 0.8)
        elif momentum < -0.01:
            direction = 'down'
            confidence = min(0.5 + abs(momentum) * 10, 0.8)
        else:
            direction = 'neutral'
            confidence = 0.4
        
        predicted_change = momentum * 0.5  # Yarı momentum
        predicted_price = current_price * (1 + predicted_change)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change_pct': predicted_change * 100,
            'prediction_horizon': self.prediction_horizon,
            'is_fallback': True
        }
    
    def predict_batch(self, data_list: list) -> list:
        """
        Toplu tahmin yap.
        
        Args:
            data_list: DataFrame listesi
            
        Returns:
            Tahmin listesi
        """
        return [self.predict(df) for df in data_list]
    
    def get_signal(self, df: pd.DataFrame) -> str:
        """
        Trading sinyali al.
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            'BUY', 'SELL' veya 'HOLD'
        """
        prediction = self.predict(df)
        
        if prediction['confidence'] < self.confidence_threshold:
            return 'HOLD'
        
        if prediction['direction'] == 'up':
            return 'BUY'
        elif prediction['direction'] == 'down':
            return 'SELL'
        else:
            return 'HOLD'
    
    def is_ready(self) -> bool:
        """Model hazır mı kontrol et."""
        return self._is_ready
    
    def get_model_info(self) -> Dict[str, Any]:
        """Model bilgilerini döndür."""
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'is_ready': self._is_ready,
            'lookback_period': self.lookback_period,
            'prediction_horizon': self.prediction_horizon,
            'confidence_threshold': self.confidence_threshold
        }
