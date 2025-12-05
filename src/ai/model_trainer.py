"""
Model Trainer Modülü

Bu modül, AI modellerini eğitmek için gerekli
işlevleri sağlar.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

from src.ai.data_processor import DataProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Model eğitim sınıfı.
    
    Bu sınıf, LSTM ve Transformer modellerini eğitir,
    değerlendirir ve kaydeder.
    """
    
    def __init__(
        self,
        model_type: str = 'lstm',
        config: Dict[str, Any] = None
    ):
        """
        ModelTrainer'ı başlat.
        
        Args:
            model_type: Model tipi ('lstm' veya 'transformer')
            config: Model konfigürasyonu
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.history = None
        self.data_processor = None
        
        # Eğitim parametreleri
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.validation_split = self.config.get('validation_split', 0.2)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        logger.info(f"ModelTrainer başlatıldı - Model tipi: {model_type}")
    
    def _create_model(
        self,
        input_shape: Tuple[int, int],
        output_type: str = 'classification'
    ) -> Any:
        """
        Model oluştur.
        
        Args:
            input_shape: Giriş şekli (lookback, features)
            output_type: 'classification' veya 'regression'
            
        Returns:
            Oluşturulan model
        """
        if self.model_type == 'lstm':
            from src.ai.models.lstm_model import LSTMModel
            model = LSTMModel(
                input_shape=input_shape,
                output_type=output_type,
                config=self.config
            )
        elif self.model_type == 'transformer':
            from src.ai.models.transformer_model import TransformerModel
            model = TransformerModel(
                input_shape=input_shape,
                output_type=output_type,
                config=self.config
            )
        else:
            raise ValueError(f"Bilinmeyen model tipi: {self.model_type}")
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        output_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Modeli eğit.
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedefleri
            X_val: Validasyon özellikleri (opsiyonel)
            y_val: Validasyon hedefleri (opsiyonel)
            output_type: 'classification' veya 'regression'
            
        Returns:
            Eğitim sonuçları
        """
        logger.info(f"Model eğitimi başlıyor - Shape: {X_train.shape}")
        
        # Model oluştur
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self._create_model(input_shape, output_type)
        
        # Modeli derle
        self.model.compile(learning_rate=self.learning_rate)
        
        # Validasyon verisi
        if X_val is None:
            validation_data = None
        else:
            validation_data = (X_val, y_val)
        
        # Eğit
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            validation_split=self.validation_split if validation_data is None else 0,
            early_stopping_patience=self.early_stopping_patience
        )
        
        # Sonuçları topla
        results = {
            'epochs_trained': len(self.history.get('loss', [])),
            'final_loss': self.history.get('loss', [0])[-1],
            'final_val_loss': self.history.get('val_loss', [0])[-1] if 'val_loss' in self.history else None,
            'history': self.history
        }
        
        if output_type == 'classification':
            results['final_accuracy'] = self.history.get('accuracy', [0])[-1]
            results['final_val_accuracy'] = self.history.get('val_accuracy', [0])[-1] if 'val_accuracy' in self.history else None
        
        logger.info(
            f"Eğitim tamamlandı - "
            f"Loss: {results['final_loss']:.4f}, "
            f"Val Loss: {results.get('final_val_loss', 'N/A')}"
        )
        
        return results
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Modeli değerlendir.
        
        Args:
            X_test: Test özellikleri
            y_test: Test hedefleri
            
        Returns:
            Değerlendirme metrikleri
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi")
        
        metrics = self.model.evaluate(X_test, y_test)
        
        logger.info(f"Değerlendirme sonuçları: {metrics}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin yap.
        
        Args:
            X: Giriş verisi
            
        Returns:
            Tahminler
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi")
        
        return self.model.predict(X)
    
    def save_model(self, path: str) -> None:
        """
        Modeli kaydet.
        
        Args:
            path: Kayıt yolu
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi")
        
        self.model.save(path)
        logger.info(f"Model kaydedildi: {path}")
    
    def load_model(self, path: str) -> None:
        """
        Modeli yükle.
        
        Args:
            path: Model dosyası yolu
        """
        if self.model_type == 'lstm':
            from src.ai.models.lstm_model import LSTMModel
            self.model = LSTMModel.load(path)
        elif self.model_type == 'transformer':
            from src.ai.models.transformer_model import TransformerModel
            self.model = TransformerModel.load(path)
        
        logger.info(f"Model yüklendi: {path}")
    
    def train_from_dataframe(
        self,
        df,
        features: List[str] = None,
        target_type: str = 'direction',
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        DataFrame'den model eğit.
        
        Args:
            df: OHLCV DataFrame'i
            features: Kullanılacak özellikler
            target_type: Hedef tipi
            test_size: Test seti oranı
            
        Returns:
            Eğitim ve değerlendirme sonuçları
        """
        # Data processor oluştur
        self.data_processor = DataProcessor(
            features=features,
            lookback_period=self.config.get('lookback_period', 60),
            prediction_horizon=self.config.get('prediction_horizon', 5)
        )
        
        # Veriyi hazırla
        data = self.data_processor.prepare_training_data(
            df, target_type, test_size
        )
        
        # Modeli eğit
        output_type = 'classification' if target_type == 'direction' else 'regression'
        
        train_results = self.train(
            data['X_train'],
            data['y_train'],
            data['X_test'],
            data['y_test'],
            output_type
        )
        
        # Değerlendir
        eval_results = self.evaluate(data['X_test'], data['y_test'])
        
        return {
            'training': train_results,
            'evaluation': eval_results
        }
    
    def get_model_summary(self) -> str:
        """Model özetini döndür."""
        if self.model is None:
            return "Model henüz oluşturulmadı"
        
        return self.model.summary()
