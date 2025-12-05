"""
Yardımcı Fonksiyonlar Modülü

Bu modül, uygulama genelinde kullanılacak yardımcı fonksiyonları içerir.
"""

from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Union

import numpy as np


def format_price(price: Union[float, Decimal], precision: int = 2) -> str:
    """
    Fiyatı belirtilen hassasiyette formatla.
    
    Args:
        price: Formatlanacak fiyat
        precision: Ondalık basamak sayısı
        
    Returns:
        Formatlanmış fiyat string'i
    """
    if isinstance(price, float):
        price = Decimal(str(price))
    
    quantize_str = '0.' + '0' * precision
    formatted = price.quantize(Decimal(quantize_str), rounding=ROUND_DOWN)
    return str(formatted)


def format_quantity(quantity: Union[float, Decimal], precision: int = 3) -> str:
    """
    Miktarı belirtilen hassasiyette formatla.
    
    Args:
        quantity: Formatlanacak miktar
        precision: Ondalık basamak sayısı
        
    Returns:
        Formatlanmış miktar string'i
    """
    if isinstance(quantity, float):
        quantity = Decimal(str(quantity))
    
    quantize_str = '0.' + '0' * precision
    formatted = quantity.quantize(Decimal(quantize_str), rounding=ROUND_DOWN)
    return str(formatted)


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str,
    leverage: int = 1
) -> dict:
    """
    Kar/zarar hesapla.
    
    Args:
        entry_price: Giriş fiyatı
        exit_price: Çıkış fiyatı
        quantity: Pozisyon miktarı
        side: Pozisyon yönü ('LONG' veya 'SHORT')
        leverage: Kaldıraç oranı
        
    Returns:
        PnL detaylarını içeren dictionary
    """
    if side.upper() == 'LONG':
        price_change = exit_price - entry_price
    else:
        price_change = entry_price - exit_price
    
    pnl = price_change * quantity
    pnl_percentage = (price_change / entry_price) * 100 * leverage
    
    return {
        'pnl': round(pnl, 4),
        'pnl_percentage': round(pnl_percentage, 2),
        'entry_price': entry_price,
        'exit_price': exit_price,
        'quantity': quantity,
        'side': side,
        'leverage': leverage
    }


def timestamp_to_datetime(timestamp: int) -> datetime:
    """
    Milisaniye cinsinden timestamp'i datetime'a çevir.
    
    Args:
        timestamp: Milisaniye cinsinden Unix timestamp
        
    Returns:
        datetime nesnesi
    """
    return datetime.fromtimestamp(timestamp / 1000)


def datetime_to_timestamp(dt: datetime) -> int:
    """
    datetime'ı milisaniye cinsinden timestamp'e çevir.
    
    Args:
        dt: datetime nesnesi
        
    Returns:
        Milisaniye cinsinden Unix timestamp
    """
    return int(dt.timestamp() * 1000)


def calculate_position_size(
    balance: float,
    risk_percentage: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: int = 1
) -> float:
    """
    Risk bazlı pozisyon boyutu hesapla.
    
    Args:
        balance: Hesap bakiyesi
        risk_percentage: Risk yüzdesi (0-100)
        entry_price: Planlanan giriş fiyatı
        stop_loss_price: Stop-loss fiyatı
        leverage: Kaldıraç oranı
        
    Returns:
        Hesaplanan pozisyon boyutu
    """
    # Risk miktarı
    risk_amount = balance * (risk_percentage / 100)
    
    # Fiyat farkı yüzdesi
    price_diff_pct = abs(entry_price - stop_loss_price) / entry_price
    
    # Pozisyon değeri
    if price_diff_pct > 0:
        position_value = risk_amount / price_diff_pct
    else:
        position_value = balance * (risk_percentage / 100)
    
    # Miktar hesapla
    quantity = position_value / entry_price
    
    return quantity


def calculate_stop_loss(
    entry_price: float,
    side: str,
    stop_loss_pct: float
) -> float:
    """
    Stop-loss fiyatını hesapla.
    
    Args:
        entry_price: Giriş fiyatı
        side: Pozisyon yönü ('LONG' veya 'SHORT')
        stop_loss_pct: Stop-loss yüzdesi
        
    Returns:
        Stop-loss fiyatı
    """
    if side.upper() == 'LONG':
        return entry_price * (1 - stop_loss_pct / 100)
    else:
        return entry_price * (1 + stop_loss_pct / 100)


def calculate_take_profit(
    entry_price: float,
    side: str,
    take_profit_pct: float
) -> float:
    """
    Take-profit fiyatını hesapla.
    
    Args:
        entry_price: Giriş fiyatı
        side: Pozisyon yönü ('LONG' veya 'SHORT')
        take_profit_pct: Take-profit yüzdesi
        
    Returns:
        Take-profit fiyatı
    """
    if side.upper() == 'LONG':
        return entry_price * (1 + take_profit_pct / 100)
    else:
        return entry_price * (1 - take_profit_pct / 100)


def normalize_data(data: np.ndarray, method: str = 'minmax') -> tuple:
    """
    Veriyi normalize et.
    
    Args:
        data: Normalize edilecek veri
        method: Normalizasyon yöntemi ('minmax' veya 'zscore')
        
    Returns:
        (normalize edilmiş veri, parametreler) tuple'ı
    """
    if method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Sıfıra bölmeyi önle
        normalized = (data - min_val) / range_val
        params = {'min': min_val, 'max': max_val}
    elif method == 'zscore':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Sıfıra bölmeyi önle
        normalized = (data - mean) / std
        params = {'mean': mean, 'std': std}
    else:
        raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")
    
    return normalized, params


def denormalize_data(
    data: np.ndarray,
    params: dict,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize edilmiş veriyi orijinal ölçeğe geri çevir.
    
    Args:
        data: Denormalize edilecek veri
        params: Normalizasyon parametreleri
        method: Normalizasyon yöntemi ('minmax' veya 'zscore')
        
    Returns:
        Denormalize edilmiş veri
    """
    if method == 'minmax':
        return data * (params['max'] - params['min']) + params['min']
    elif method == 'zscore':
        return data * params['std'] + params['mean']
    else:
        raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")


def round_step_size(quantity: float, step_size: float) -> float:
    """
    Miktarı Binance step size'a göre yuvarla.
    
    Args:
        quantity: Yuvarlanacak miktar
        step_size: Binance step size
        
    Returns:
        Yuvarlanmış miktar
    """
    precision = int(round(-np.log10(step_size), 0))
    return round(quantity - (quantity % step_size), precision)


def round_tick_size(price: float, tick_size: float) -> float:
    """
    Fiyatı Binance tick size'a göre yuvarla.
    
    Args:
        price: Yuvarlanacak fiyat
        tick_size: Binance tick size
        
    Returns:
        Yuvarlanmış fiyat
    """
    precision = int(round(-np.log10(tick_size), 0))
    return round(price - (price % tick_size), precision)


def validate_symbol(symbol: str) -> bool:
    """
    Sembol formatını doğrula.
    
    Args:
        symbol: Doğrulanacak sembol
        
    Returns:
        Geçerli ise True
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Temel format kontrolü
    symbol = symbol.upper()
    valid_quotes = ['USDT', 'BUSD', 'BTC', 'ETH', 'BNB']
    
    for quote in valid_quotes:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return True
    
    return False


def get_timeframe_minutes(timeframe: str) -> int:
    """
    Zaman dilimini dakika cinsinden döndür.
    
    Args:
        timeframe: Zaman dilimi string'i (örn: '1m', '5m', '1h')
        
    Returns:
        Dakika cinsinden zaman dilimi
    """
    timeframe = timeframe.lower()
    
    timeframe_map = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
        '1d': 1440,
        '3d': 4320,
        '1w': 10080,
        '1M': 43200
    }
    
    return timeframe_map.get(timeframe, 15)
