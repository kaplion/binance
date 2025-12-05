"""
Data Processor Modülü

Bu modül, OHLCV verilerini işler, normalize eder ve
ML modelleri için feature engineering yapar.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Veri işleme sınıfı.
    
    Bu sınıf, ham piyasa verilerini ML modelleri için
    uygun formata dönüştürür.
    
    Özellikler:
    - OHLCV veri işleme
    - Teknik indikatör hesaplama
    - Feature engineering
    - Veri normalizasyonu
    - Sekans oluşturma
    """
    
    def __init__(
        self,
        features: List[str] = None,
        lookback_period: int = 60,
        prediction_horizon: int = 5,
        scaler_type: str = 'minmax'
    ):
        """
        DataProcessor'ı başlat.
        
        Args:
            features: Kullanılacak özellik listesi
            lookback_period: Geriye bakış periyodu
            prediction_horizon: Tahmin ufku
            scaler_type: Normalizasyon tipi ('minmax' veya 'standard')
        """
        self.features = features or [
            'close', 'volume', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'ema_9', 'ema_21', 'atr'
        ]
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.scaler_type = scaler_type
        
        # Scaler'lar
        self._feature_scaler = None
        self._target_scaler = None
        self._is_fitted = False
        
        logger.info(
            f"DataProcessor başlatıldı - "
            f"Features: {len(self.features)}, Lookback: {lookback_period}"
        )
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Teknik indikatörleri hesapla ve ekle.
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            İndikatörler eklenmiş DataFrame
        """
        result = df.copy()
        
        # EMA'lar
        result['ema_9'] = TechnicalIndicators.ema(df['close'], 9)
        result['ema_21'] = TechnicalIndicators.ema(df['close'], 21)
        result['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
        result['ema_200'] = TechnicalIndicators.ema(df['close'], 200)
        
        # SMA'lar
        result['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        result['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        
        # RSI
        result['rsi'] = TechnicalIndicators.rsi(df['close'], 14)
        
        # MACD
        macd, signal, hist = TechnicalIndicators.macd(df['close'])
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_histogram'] = hist
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(df['close'])
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower
        
        # ATR
        result['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Stochastic
        k, d = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        result['stoch_k'] = k
        result['stoch_d'] = d
        
        # OBV
        result['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        
        # Momentum
        result['momentum'] = TechnicalIndicators.momentum(df['close'], 10)
        result['roc'] = TechnicalIndicators.roc(df['close'], 10)
        
        # ADX
        adx, plus_di, minus_di = TechnicalIndicators.adx(df['high'], df['low'], df['close'])
        result['adx'] = adx
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di
        
        # Fiyat değişim özellikleri
        result['price_change'] = df['close'].pct_change()
        result['price_change_5'] = df['close'].pct_change(5)
        result['price_change_10'] = df['close'].pct_change(10)
        
        # Hacim değişim özellikleri
        result['volume_change'] = df['volume'].pct_change()
        result['volume_sma'] = TechnicalIndicators.sma(df['volume'], 20)
        result['volume_ratio'] = df['volume'] / result['volume_sma']
        
        # Bollinger Band pozisyonu (0-1 arası)
        bb_range = result['bb_upper'] - result['bb_lower']
        result['bb_position'] = (df['close'] - result['bb_lower']) / bb_range
        
        # High-Low range
        result['hl_range'] = (df['high'] - df['low']) / df['close']
        
        return result
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Gecikme (lag) özelliklerini ekle.
        
        Args:
            df: DataFrame
            columns: Lag eklenecek kolonlar
            lags: Lag periyotları
            
        Returns:
            Lag özellikleri eklenmiş DataFrame
        """
        result = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    result[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return result
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        column: str,
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Kayan pencere özelliklerini ekle.
        
        Args:
            df: DataFrame
            column: İşlenecek kolon
            windows: Pencere boyutları
            
        Returns:
            Rolling özellikleri eklenmiş DataFrame
        """
        result = df.copy()
        
        if column in df.columns:
            for window in windows:
                result[f'{column}_mean_{window}'] = df[column].rolling(window).mean()
                result[f'{column}_std_{window}'] = df[column].rolling(window).std()
                result[f'{column}_min_{window}'] = df[column].rolling(window).min()
                result[f'{column}_max_{window}'] = df[column].rolling(window).max()
        
        return result
    
    def create_target(
        self,
        df: pd.DataFrame,
        target_type: str = 'direction'
    ) -> pd.DataFrame:
        """
        Hedef değişkeni oluştur.
        
        Args:
            df: DataFrame
            target_type: 'direction' (0/1), 'returns' (yüzde), 'price' (fiyat)
            
        Returns:
            Hedef eklenmiş DataFrame
        """
        result = df.copy()
        
        if target_type == 'direction':
            # Gelecek fiyat yönü (1: yukarı, 0: aşağı/yatay)
            future_price = df['close'].shift(-self.prediction_horizon)
            result['target'] = (future_price > df['close']).astype(int)
            
        elif target_type == 'returns':
            # Gelecek getiri yüzdesi
            future_price = df['close'].shift(-self.prediction_horizon)
            result['target'] = (future_price - df['close']) / df['close'] * 100
            
        elif target_type == 'price':
            # Gelecek fiyat
            result['target'] = df['close'].shift(-self.prediction_horizon)
        
        return result
    
    def fit_scalers(self, df: pd.DataFrame) -> None:
        """
        Scaler'ları veriye göre eğit.
        
        Args:
            df: Eğitim DataFrame'i
        """
        # Feature scaler
        feature_data = df[self.features].dropna()
        
        if self.scaler_type == 'minmax':
            self._feature_scaler = MinMaxScaler()
        else:
            self._feature_scaler = StandardScaler()
        
        self._feature_scaler.fit(feature_data)
        
        # Target scaler (eğer varsa)
        if 'target' in df.columns:
            target_data = df[['target']].dropna()
            
            if self.scaler_type == 'minmax':
                self._target_scaler = MinMaxScaler()
            else:
                self._target_scaler = StandardScaler()
            
            self._target_scaler.fit(target_data)
        
        self._is_fitted = True
        logger.debug("Scaler'lar eğitildi")
    
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Özellikleri normalize et.
        
        Args:
            df: DataFrame
            
        Returns:
            Normalize edilmiş özellik matrisi
        """
        if not self._is_fitted:
            raise ValueError("Scaler'lar henüz eğitilmedi. fit_scalers() çağırın.")
        
        feature_data = df[self.features].copy()
        
        # NaN değerleri doldur
        feature_data = feature_data.ffill().bfill()
        
        return self._feature_scaler.transform(feature_data)
    
    def inverse_transform_target(self, scaled_target: np.ndarray) -> np.ndarray:
        """
        Normalize edilmiş hedefi orijinal ölçeğe çevir.
        
        Args:
            scaled_target: Normalize edilmiş hedef
            
        Returns:
            Orijinal ölçekte hedef
        """
        if self._target_scaler is None:
            return scaled_target
        
        return self._target_scaler.inverse_transform(scaled_target.reshape(-1, 1)).flatten()
    
    def create_sequences(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        LSTM/Transformer için sekanslar oluştur.
        
        Args:
            features: Özellik matrisi
            targets: Hedef dizisi (opsiyonel)
            
        Returns:
            (X, y) tuple'ı - X: (samples, lookback, features), y: (samples,)
        """
        X = []
        y = [] if targets is not None else None
        
        for i in range(len(features) - self.lookback_period):
            X.append(features[i:i + self.lookback_period])
            
            if targets is not None:
                y.append(targets[i + self.lookback_period])
        
        X = np.array(X)
        
        if y is not None:
            y = np.array(y)
        
        return X, y
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_type: str = 'direction',
        test_size: float = 0.2
    ) -> Dict[str, np.ndarray]:
        """
        Eğitim için veri hazırla.
        
        Args:
            df: OHLCV DataFrame'i
            target_type: Hedef tipi
            test_size: Test seti oranı
            
        Returns:
            Eğitim ve test verilerini içeren dictionary
        """
        # İndikatörleri ekle
        data = self.add_technical_indicators(df)
        
        # Hedef oluştur
        data = self.create_target(data, target_type)
        
        # NaN satırları temizle
        data = data.dropna()
        
        if len(data) < self.lookback_period + 100:
            raise ValueError(f"Yetersiz veri: {len(data)} satır")
        
        # Train-test split
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Scaler'ları eğitim verisine göre eğit
        self.fit_scalers(train_data)
        
        # Özellikleri dönüştür
        train_features = self.transform_features(train_data)
        test_features = self.transform_features(test_data)
        
        # Hedefleri al
        train_targets = train_data['target'].values
        test_targets = test_data['target'].values
        
        # Sekansları oluştur
        X_train, y_train = self.create_sequences(train_features, train_targets)
        X_test, y_test = self.create_sequences(test_features, test_targets)
        
        logger.info(
            f"Eğitim verisi hazırlandı - "
            f"Train: {X_train.shape}, Test: {X_test.shape}"
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def prepare_inference_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Tahmin için veri hazırla.
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            Model girişi için hazır dizi
        """
        if not self._is_fitted:
            raise ValueError("Scaler'lar henüz eğitilmedi. fit_scalers() çağırın.")
        
        # İndikatörleri ekle
        data = self.add_technical_indicators(df)
        
        # NaN değerleri doldur
        data = data.ffill().bfill()
        
        # Son lookback_period kadar veri al
        data = data.tail(self.lookback_period)
        
        if len(data) < self.lookback_period:
            raise ValueError(
                f"Yetersiz veri: {len(data)} < {self.lookback_period}"
            )
        
        # Özellikleri dönüştür
        features = self.transform_features(data)
        
        # Sekans formatına çevir
        X = features.reshape(1, self.lookback_period, len(self.features))
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Özellik isimlerini döndür."""
        return self.features.copy()
    
    def save_scalers(self, path: str) -> None:
        """
        Scaler'ları dosyaya kaydet.
        
        Args:
            path: Kayıt yolu
        """
        import pickle
        
        with open(path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self._feature_scaler,
                'target_scaler': self._target_scaler,
                'features': self.features,
                'lookback_period': self.lookback_period,
                'prediction_horizon': self.prediction_horizon,
                'scaler_type': self.scaler_type
            }, f)
        
        logger.info(f"Scaler'lar kaydedildi: {path}")
    
    def load_scalers(self, path: str) -> None:
        """
        Scaler'ları dosyadan yükle.
        
        Args:
            path: Dosya yolu
        """
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self._feature_scaler = data['feature_scaler']
        self._target_scaler = data['target_scaler']
        self.features = data['features']
        self.lookback_period = data['lookback_period']
        self.prediction_horizon = data['prediction_horizon']
        self.scaler_type = data['scaler_type']
        self._is_fitted = True
        
        logger.info(f"Scaler'lar yüklendi: {path}")
