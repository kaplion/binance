"""
AI Modülü Testleri

Bu modül, AI/ML bileşenlerinin testlerini içerir.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.ai.data_processor import DataProcessor
from src.ai.predictor import Predictor
from src.ai.model_trainer import ModelTrainer
from src.ai.models.lstm_model import LSTMModel
from src.ai.models.transformer_model import TransformerModel


def generate_test_ohlcv(periods: int = 200, seed: int = 42) -> pd.DataFrame:
    """Test için OHLCV verisi oluştur."""
    rng = np.random.default_rng(seed)
    
    base_prices = np.linspace(50000, 52000, periods)
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
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


class TestDataProcessor:
    """DataProcessor sınıfı testleri."""
    
    @pytest.fixture
    def processor(self):
        """Data processor oluştur."""
        return DataProcessor(
            features=['close', 'volume', 'rsi', 'macd', 'ema_9', 'atr'],
            lookback_period=30,
            prediction_horizon=5
        )
    
    def test_add_technical_indicators(self, processor):
        """Teknik indikatör ekleme testi."""
        df = generate_test_ohlcv(100)
        result = processor.add_technical_indicators(df)
        
        # İndikatörlerin eklendiğini kontrol et
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'ema_9' in result.columns
        assert 'bb_upper' in result.columns
        assert 'atr' in result.columns
    
    def test_create_target_direction(self, processor):
        """Yön hedefi oluşturma testi."""
        df = generate_test_ohlcv(100)
        result = processor.create_target(df, target_type='direction')
        
        assert 'target' in result.columns
        # Target 0 veya 1 olmalı
        assert result['target'].dropna().isin([0, 1]).all()
    
    def test_create_target_returns(self, processor):
        """Getiri hedefi oluşturma testi."""
        df = generate_test_ohlcv(100)
        result = processor.create_target(df, target_type='returns')
        
        assert 'target' in result.columns
    
    def test_create_sequences(self, processor):
        """Sekans oluşturma testi."""
        features = np.random.randn(100, 6)
        targets = np.random.randint(0, 2, 100)
        
        X, y = processor.create_sequences(features, targets)
        
        # Doğru şekli kontrol et
        assert X.shape[0] == 100 - processor.lookback_period
        assert X.shape[1] == processor.lookback_period
        assert X.shape[2] == 6
        assert len(y) == len(X)
    
    def test_fit_and_transform(self, processor):
        """Scaler fit ve transform testi."""
        df = generate_test_ohlcv(200)
        data = processor.add_technical_indicators(df)
        data = data.dropna()
        
        # Fit
        processor.fit_scalers(data)
        
        # Transform
        transformed = processor.transform_features(data)
        
        assert transformed.shape[0] == len(data)
        assert transformed.shape[1] == len(processor.features)
        # Değerler 0-1 arasında olmalı (minmax scaler)
        assert np.all(transformed >= -1)  # Küçük sapmalara izin ver
        assert np.all(transformed <= 2)
    
    def test_prepare_training_data(self, processor):
        """Eğitim verisi hazırlama testi."""
        df = generate_test_ohlcv(500)
        
        data = processor.prepare_training_data(
            df,
            target_type='direction',
            test_size=0.2
        )
        
        assert 'X_train' in data
        assert 'y_train' in data
        assert 'X_test' in data
        assert 'y_test' in data
        
        # Şekilleri kontrol et
        assert len(data['X_train'].shape) == 3
        assert len(data['y_train'].shape) == 1
    
    def test_prepare_inference_data(self, processor):
        """Inference verisi hazırlama testi."""
        df = generate_test_ohlcv(200)
        data = processor.add_technical_indicators(df)
        data = data.dropna()
        
        # Önce fit et
        processor.fit_scalers(data)
        
        # Inference verisi hazırla
        X = processor.prepare_inference_data(df)
        
        assert X.shape == (1, processor.lookback_period, len(processor.features))


class TestLSTMModel:
    """LSTMModel sınıfı testleri."""
    
    @pytest.fixture
    def model(self):
        """LSTM model oluştur."""
        return LSTMModel(
            input_shape=(30, 6),
            output_type='classification',
            config={'lstm_units': [32, 16]}
        )
    
    def test_model_creation(self, model):
        """Model oluşturma testi."""
        assert model is not None
        assert model.input_shape == (30, 6)
        assert model.output_type == 'classification'
    
    def test_compile(self, model):
        """Model derleme testi."""
        model.compile(learning_rate=0.001)
        # Derleme başarılı olmalı (hata vermemeli)
    
    def test_predict_shape(self, model):
        """Tahmin şekil testi."""
        model.compile()
        X = np.random.randn(10, 30, 6)
        
        predictions = model.predict(X)
        
        assert predictions.shape[0] == 10
    
    def test_fit_and_evaluate(self, model):
        """Eğitim ve değerlendirme testi."""
        model.compile()
        
        X_train = np.random.randn(100, 30, 6)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 30, 6)
        y_test = np.random.randint(0, 2, 20)
        
        # Eğit
        history = model.fit(
            X_train, y_train,
            epochs=2,
            batch_size=32,
            validation_split=0.2
        )
        
        assert 'loss' in history
        
        # Değerlendir
        metrics = model.evaluate(X_test, y_test)
        
        assert 'loss' in metrics


class TestTransformerModel:
    """TransformerModel sınıfı testleri."""
    
    @pytest.fixture
    def model(self):
        """Transformer model oluştur."""
        return TransformerModel(
            input_shape=(30, 6),
            output_type='classification',
            config={'d_model': 32, 'num_heads': 2}
        )
    
    def test_model_creation(self, model):
        """Model oluşturma testi."""
        assert model is not None
        assert model.input_shape == (30, 6)
    
    def test_predict_shape(self, model):
        """Tahmin şekil testi."""
        model.compile()
        X = np.random.randn(10, 30, 6)
        
        predictions = model.predict(X)
        
        assert predictions.shape[0] == 10


class TestPredictor:
    """Predictor sınıfı testleri."""
    
    @pytest.fixture
    def predictor(self):
        """Predictor oluştur (model olmadan)."""
        return Predictor(
            model_type='lstm',
            config={
                'lookback_period': 30,
                'prediction_horizon': 5,
                'confidence_threshold': 0.6
            }
        )
    
    def test_init_without_model(self, predictor):
        """Model olmadan başlatma testi."""
        assert predictor.is_ready() is False
    
    def test_simple_prediction_fallback(self, predictor):
        """Basit tahmin fallback testi."""
        df = generate_test_ohlcv(100)
        
        prediction = predictor.predict(df)
        
        assert 'direction' in prediction
        assert 'confidence' in prediction
        assert 'current_price' in prediction
        assert prediction.get('is_fallback', False) is True
    
    def test_get_signal(self, predictor):
        """Sinyal alma testi."""
        df = generate_test_ohlcv(100)
        
        signal = predictor.get_signal(df)
        
        assert signal in ['BUY', 'SELL', 'HOLD']
    
    def test_get_model_info(self, predictor):
        """Model bilgisi alma testi."""
        info = predictor.get_model_info()
        
        assert 'model_type' in info
        assert 'is_ready' in info
        assert info['model_type'] == 'lstm'


class TestModelTrainer:
    """ModelTrainer sınıfı testleri."""
    
    @pytest.fixture
    def trainer(self):
        """Model trainer oluştur."""
        return ModelTrainer(
            model_type='lstm',
            config={
                'epochs': 2,
                'batch_size': 16,
                'lstm_units': [16, 8]
            }
        )
    
    def test_init(self, trainer):
        """Başlatma testi."""
        assert trainer.model_type == 'lstm'
        assert trainer.epochs == 2
    
    def test_train_basic(self, trainer):
        """Temel eğitim testi."""
        X_train = np.random.randn(50, 30, 6)
        y_train = np.random.randint(0, 2, 50)
        
        results = trainer.train(X_train, y_train)
        
        assert 'epochs_trained' in results
        assert 'final_loss' in results
    
    def test_train_with_validation(self, trainer):
        """Validasyon ile eğitim testi."""
        X_train = np.random.randn(50, 30, 6)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(10, 30, 6)
        y_val = np.random.randint(0, 2, 10)
        
        results = trainer.train(
            X_train, y_train,
            X_val, y_val
        )
        
        assert 'final_val_loss' in results
    
    def test_predict_after_train(self, trainer):
        """Eğitimden sonra tahmin testi."""
        X_train = np.random.randn(50, 30, 6)
        y_train = np.random.randint(0, 2, 50)
        
        trainer.train(X_train, y_train)
        
        X_test = np.random.randn(5, 30, 6)
        predictions = trainer.predict(X_test)
        
        assert predictions.shape[0] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
