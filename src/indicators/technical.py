"""
Teknik İndikatörler Modülü

Bu modül, teknik analiz indikatörlerini hesaplar.
RSI, MACD, Bollinger Bands, EMA, SMA, ATR ve hacim analizi desteklenir.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    Teknik analiz indikatörleri sınıfı.
    
    Bu sınıf, yaygın teknik analiz indikatörlerini hesaplar.
    Tüm metodlar pandas DataFrame veya Series üzerinde çalışır.
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Simple Moving Average (Basit Hareketli Ortalama) hesapla.
        
        Args:
            data: Fiyat verisi
            period: Periyot uzunluğu
            
        Returns:
            SMA serisi
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Exponential Moving Average (Üssel Hareketli Ortalama) hesapla.
        
        Args:
            data: Fiyat verisi
            period: Periyot uzunluğu
            
        Returns:
            EMA serisi
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (Göreceli Güç Endeksi) hesapla.
        
        Args:
            data: Fiyat verisi (genellikle close)
            period: RSI periyodu
            
        Returns:
            RSI serisi (0-100 arası)
        """
        delta = data.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence) hesapla.
        
        Args:
            data: Fiyat verisi (genellikle close)
            fast_period: Hızlı EMA periyodu
            slow_period: Yavaş EMA periyodu
            signal_period: Sinyal çizgisi periyodu
            
        Returns:
            (macd_line, signal_line, histogram) tuple'ı
        """
        fast_ema = data.ewm(span=fast_period, adjust=False).mean()
        slow_ema = data.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands hesapla.
        
        Args:
            data: Fiyat verisi (genellikle close)
            period: SMA periyodu
            std_dev: Standart sapma çarpanı
            
        Returns:
            (upper_band, middle_band, lower_band) tuple'ı
        """
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range hesapla.
        
        Args:
            high: Yüksek fiyat serisi
            low: Düşük fiyat serisi
            close: Kapanış fiyatı serisi
            period: ATR periyodu
            
        Returns:
            ATR serisi
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator hesapla.
        
        Args:
            high: Yüksek fiyat serisi
            low: Düşük fiyat serisi
            close: Kapanış fiyatı serisi
            k_period: %K periyodu
            d_period: %D periyodu
            
        Returns:
            (%K, %D) tuple'ı
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Williams %R hesapla.
        
        Args:
            high: Yüksek fiyat serisi
            low: Düşük fiyat serisi
            close: Kapanış fiyatı serisi
            period: Periyot uzunluğu
            
        Returns:
            Williams %R serisi (-100 ile 0 arası)
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return wr
    
    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Commodity Channel Index hesapla.
        
        Args:
            high: Yüksek fiyat serisi
            low: Düşük fiyat serisi
            close: Kapanış fiyatı serisi
            period: Periyot uzunluğu
            
        Returns:
            CCI serisi
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume hesapla.
        
        Args:
            close: Kapanış fiyatı serisi
            volume: Hacim serisi
            
        Returns:
            OBV serisi
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Volume Weighted Average Price hesapla.
        
        Args:
            high: Yüksek fiyat serisi
            low: Düşük fiyat serisi
            close: Kapanış fiyatı serisi
            volume: Hacim serisi
            
        Returns:
            VWAP serisi
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum indikatörü hesapla.
        
        Args:
            data: Fiyat verisi
            period: Momentum periyodu
            
        Returns:
            Momentum serisi
        """
        return data.diff(period)
    
    @staticmethod
    def roc(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of Change hesapla.
        
        Args:
            data: Fiyat verisi
            period: ROC periyodu
            
        Returns:
            ROC serisi (yüzde olarak)
        """
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index hesapla.
        
        Args:
            high: Yüksek fiyat serisi
            low: Düşük fiyat serisi
            close: Kapanış fiyatı serisi
            period: ADX periyodu
            
        Returns:
            (ADX, +DI, -DI) tuple'ı
        """
        # True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Smoothed averages
        atr = true_range.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52
    ) -> Dict[str, pd.Series]:
        """
        Ichimoku Cloud hesapla.
        
        Args:
            high: Yüksek fiyat serisi
            low: Düşük fiyat serisi
            close: Kapanış fiyatı serisi
            tenkan_period: Tenkan-sen periyodu
            kijun_period: Kijun-sen periyodu
            senkou_b_period: Senkou Span B periyodu
            
        Returns:
            Ichimoku bileşenlerini içeren dictionary
        """
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=senkou_b_period).max()
        senkou_low = low.rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun_period)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def pivot_points(
        high: float,
        low: float,
        close: float
    ) -> Dict[str, float]:
        """
        Pivot Points hesapla.
        
        Args:
            high: Önceki dönemin yüksek fiyatı
            low: Önceki dönemin düşük fiyatı
            close: Önceki dönemin kapanış fiyatı
            
        Returns:
            Pivot noktalarını içeren dictionary
        """
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }
    
    @staticmethod
    def calculate_all(
        df: pd.DataFrame,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14
    ) -> pd.DataFrame:
        """
        Tüm temel indikatörleri hesapla ve DataFrame'e ekle.
        
        Args:
            df: OHLCV DataFrame'i (open, high, low, close, volume kolonları)
            rsi_period: RSI periyodu
            macd_fast: MACD hızlı periyot
            macd_slow: MACD yavaş periyot
            macd_signal: MACD sinyal periyodu
            bb_period: Bollinger Bands periyodu
            bb_std: Bollinger Bands standart sapma
            atr_period: ATR periyodu
            
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
        result['rsi'] = TechnicalIndicators.rsi(df['close'], rsi_period)
        
        # MACD
        macd, signal, hist = TechnicalIndicators.macd(
            df['close'], macd_fast, macd_slow, macd_signal
        )
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_histogram'] = hist
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            df['close'], bb_period, bb_std
        )
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower
        
        # ATR
        result['atr'] = TechnicalIndicators.atr(
            df['high'], df['low'], df['close'], atr_period
        )
        
        # OBV
        if 'volume' in df.columns:
            result['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        
        # Stochastic
        k, d = TechnicalIndicators.stochastic(
            df['high'], df['low'], df['close']
        )
        result['stoch_k'] = k
        result['stoch_d'] = d
        
        return result
