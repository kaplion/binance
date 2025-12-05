"""
Transformer Model Modülü

Bu modül, attention mekanizmalı Transformer mimarisini
fiyat tahmini için kullanır.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransformerModel:
    """
    Transformer tabanlı tahmin modeli.
    
    Bu sınıf, self-attention mekanizması kullanan
    Transformer mimarisini uygular.
    
    Mimari:
    - Multi-Head Self Attention
    - Position-wise Feed Forward Networks
    - Layer Normalization
    - Residual Connections
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_type: str = 'classification',
        config: Dict[str, Any] = None
    ):
        """
        TransformerModel'i başlat.
        
        Args:
            input_shape: Giriş şekli (sequence_length, features)
            output_type: 'classification' veya 'regression'
            config: Model konfigürasyonu
        """
        self.input_shape = input_shape
        self.output_type = output_type
        self.config = config or {}
        
        # Model parametreleri
        self.d_model = self.config.get('d_model', 64)
        self.num_heads = self.config.get('num_heads', 4)
        self.num_layers = self.config.get('num_layers', 2)
        self.dff = self.config.get('dff', 128)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        
        self.model = None
        self._build_model()
        
        logger.info(
            f"TransformerModel oluşturuldu - "
            f"Input: {input_shape}, d_model: {self.d_model}"
        )
    
    def _build_model(self) -> None:
        """Keras Transformer modeli oluştur."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            inputs = keras.Input(shape=self.input_shape)
            
            # Input projection
            x = layers.Dense(self.d_model)(inputs)
            
            # Positional encoding (learnable)
            seq_length = self.input_shape[0]
            position_embedding = layers.Embedding(
                input_dim=seq_length,
                output_dim=self.d_model
            )
            
            positions = keras.backend.arange(start=0, stop=seq_length, step=1)
            position_encodings = position_embedding(positions)
            x = x + position_encodings
            
            # Transformer encoder layers
            for _ in range(self.num_layers):
                x = self._transformer_encoder_layer(x)
            
            # Global average pooling
            x = layers.GlobalAveragePooling1D()(x)
            
            # Dense layers
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            # Output layer
            if self.output_type == 'classification':
                outputs = layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = layers.Dense(1, activation='linear')(x)
            
            self.model = keras.Model(inputs, outputs)
            
        except ImportError:
            logger.warning("TensorFlow bulunamadı, mock model kullanılıyor")
            self.model = MockTransformerModel(self.input_shape, self.output_type)
    
    def _transformer_encoder_layer(self, x):
        """
        Tek bir Transformer encoder katmanı.
        
        Args:
            x: Giriş tensor
            
        Returns:
            Çıkış tensor
        """
        from tensorflow.keras import layers
        
        # Multi-head self attention
        attn_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate
        )(x, x)
        
        # Residual connection and layer normalization
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed forward network
        ffn_output = layers.Dense(self.dff, activation='relu')(x)
        ffn_output = layers.Dense(self.d_model)(ffn_output)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        
        # Residual connection and layer normalization
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        return x
    
    def compile(self, learning_rate: float = 0.001) -> None:
        """
        Modeli derle.
        
        Args:
            learning_rate: Öğrenme oranı
        """
        if isinstance(self.model, MockTransformerModel):
            return
        
        from tensorflow.keras import optimizers
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        if self.output_type == 'classification':
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: Optional[Tuple] = None,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10
    ) -> Dict[str, list]:
        """
        Modeli eğit.
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedefleri
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            validation_data: (X_val, y_val) tuple'ı
            validation_split: Validasyon oranı
            early_stopping_patience: Early stopping sabır
            
        Returns:
            Eğitim geçmişi
        """
        if isinstance(self.model, MockTransformerModel):
            return self.model.fit(X_train, y_train, epochs)
        
        from tensorflow.keras import callbacks
        
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else 0,
            callbacks=callback_list,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin yap.
        
        Args:
            X: Giriş verisi
            
        Returns:
            Tahminler
        """
        if isinstance(self.model, MockTransformerModel):
            return self.model.predict(X)
        
        return self.model.predict(X, verbose=0)
    
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
        if isinstance(self.model, MockTransformerModel):
            return self.model.evaluate(X_test, y_test)
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        if self.output_type == 'classification':
            return {'loss': results[0], 'accuracy': results[1]}
        else:
            return {'loss': results[0], 'mae': results[1]}
    
    def save(self, path: str) -> None:
        """
        Modeli kaydet.
        
        Args:
            path: Kayıt yolu
        """
        if isinstance(self.model, MockTransformerModel):
            import pickle
            with open(path, 'wb') as f:
                pickle.dump({
                    'type': 'mock',
                    'input_shape': self.input_shape,
                    'output_type': self.output_type,
                    'config': self.config
                }, f)
        else:
            self.model.save(path)
        
        logger.info(f"Model kaydedildi: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TransformerModel':
        """
        Modeli yükle.
        
        Args:
            path: Model dosyası yolu
            
        Returns:
            Yüklenmiş model
        """
        try:
            from tensorflow import keras
            
            # Keras modeli yüklemeyi dene
            keras_model = keras.models.load_model(path)
            
            # Wrapper oluştur
            input_shape = keras_model.input_shape[1:]
            instance = cls.__new__(cls)
            instance.input_shape = input_shape
            instance.output_type = 'classification'  # Varsayılan
            instance.config = {}
            instance.model = keras_model
            
            return instance
            
        except Exception:
            # Mock model yükle
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            instance = cls(
                input_shape=data['input_shape'],
                output_type=data['output_type'],
                config=data['config']
            )
            return instance
    
    def summary(self) -> str:
        """Model özetini döndür."""
        if isinstance(self.model, MockTransformerModel):
            return f"MockTransformerModel - Input: {self.input_shape}"
        
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Attention ağırlıklarını döndür.
        
        Args:
            X: Giriş verisi
            
        Returns:
            Attention ağırlıkları
        """
        if isinstance(self.model, MockTransformerModel):
            # Mock attention weights
            seq_len = self.input_shape[0]
            return np.random.random((len(X), seq_len, seq_len))
        
        # Gerçek attention ağırlıkları için model modifikasyonu gerekir
        # Bu basitleştirilmiş versiyon
        return None


class MockTransformerModel:
    """
    TensorFlow olmadığında kullanılan mock model.
    Basit rastgele tahminler üretir.
    """
    
    def __init__(self, input_shape: Tuple[int, int], output_type: str):
        self.input_shape = input_shape
        self.output_type = output_type
    
    def fit(self, X, y, epochs) -> Dict[str, list]:
        """Mock eğitim."""
        history = {
            'loss': [1.0 - (i / epochs * 0.5) for i in range(epochs)],
            'val_loss': [1.0 - (i / epochs * 0.4) for i in range(epochs)]
        }
        if self.output_type == 'classification':
            history['accuracy'] = [0.5 + (i / epochs * 0.3) for i in range(epochs)]
            history['val_accuracy'] = [0.5 + (i / epochs * 0.25) for i in range(epochs)]
        return history
    
    def predict(self, X) -> np.ndarray:
        """Mock tahmin."""
        if self.output_type == 'classification':
            return np.random.random((len(X), 1))
        else:
            return np.random.randn(len(X), 1)
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """Mock değerlendirme."""
        if self.output_type == 'classification':
            return {'loss': 0.5, 'accuracy': 0.6}
        else:
            return {'loss': 0.5, 'mae': 0.3}
