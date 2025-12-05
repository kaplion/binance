"""
Binance Futures API Client Modülü

Bu modül, Binance Futures API ile iletişim kurmak için gerekli
istemci sınıfını içerir. Hem USDT-M hem de COIN-M futures desteği sağlar.
"""

import asyncio
from typing import Any, Dict, List, Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BinanceClient:
    """
    Binance Futures API istemcisi.
    
    Bu sınıf, Binance Futures API'sine bağlanmak, order vermek,
    pozisyon bilgilerini sorgulamak ve hesap bilgilerine erişmek
    için gerekli metodları sağlar.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        futures_type: str = 'usdt_m'
    ):
        """
        BinanceClient'ı başlat.
        
        Args:
            api_key: Binance API anahtarı
            api_secret: Binance API secret
            testnet: Testnet modunu kullan
            futures_type: Futures tipi ('usdt_m' veya 'coin_m')
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.futures_type = futures_type
        
        # Client'ı oluştur
        self.client = Client(api_key, api_secret, testnet=testnet)
        
        # Testnet için endpoint'leri ayarla
        if testnet:
            if futures_type == 'usdt_m':
                self.client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
            else:
                self.client.FUTURES_COIN_URL = 'https://testnet.binancefuture.com/dapi'
        
        logger.info(
            f"BinanceClient başlatıldı - Testnet: {testnet}, "
            f"Futures Tipi: {futures_type}"
        )
    
    def get_account_balance(self) -> List[Dict[str, Any]]:
        """
        Hesap bakiyelerini getir.
        
        Returns:
            Bakiye listesi
        """
        try:
            if self.futures_type == 'usdt_m':
                return self.client.futures_account_balance()
            else:
                return self.client.futures_coin_account_balance()
        except BinanceAPIException as e:
            logger.error(f"Bakiye bilgisi alınamadı: {e}")
            raise
    
    def get_usdt_balance(self) -> float:
        """
        USDT bakiyesini getir.
        
        Returns:
            USDT bakiyesi
        """
        balances = self.get_account_balance()
        for balance in balances:
            if balance.get('asset') == 'USDT':
                return float(balance.get('balance', 0))
        return 0.0
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Hesap bilgilerini getir.
        
        Returns:
            Hesap bilgileri dictionary'si
        """
        try:
            if self.futures_type == 'usdt_m':
                return self.client.futures_account()
            else:
                return self.client.futures_coin_account()
        except BinanceAPIException as e:
            logger.error(f"Hesap bilgisi alınamadı: {e}")
            raise
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Açık pozisyonları getir.
        
        Args:
            symbol: Belirli bir sembol için filtrele (opsiyonel)
            
        Returns:
            Pozisyon listesi
        """
        try:
            if self.futures_type == 'usdt_m':
                positions = self.client.futures_position_information()
            else:
                positions = self.client.futures_coin_position_information()
            
            # Sıfır olmayan pozisyonları filtrele
            active_positions = [
                p for p in positions
                if float(p.get('positionAmt', 0)) != 0
            ]
            
            if symbol:
                active_positions = [
                    p for p in active_positions
                    if p.get('symbol') == symbol
                ]
            
            return active_positions
        except BinanceAPIException as e:
            logger.error(f"Pozisyon bilgisi alınamadı: {e}")
            raise
    
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Kaldıraç oranını ayarla.
        
        Args:
            symbol: Sembol
            leverage: Kaldıraç oranı (1-125)
            
        Returns:
            API yanıtı
        """
        try:
            if self.futures_type == 'usdt_m':
                return self.client.futures_change_leverage(
                    symbol=symbol,
                    leverage=leverage
                )
            else:
                return self.client.futures_coin_change_leverage(
                    symbol=symbol,
                    leverage=leverage
                )
        except BinanceAPIException as e:
            logger.error(f"Kaldıraç ayarlanamadı: {e}")
            raise
    
    def set_margin_mode(self, symbol: str, mode: str = 'ISOLATED') -> Dict[str, Any]:
        """
        Margin modunu ayarla.
        
        Args:
            symbol: Sembol
            mode: Margin modu ('ISOLATED' veya 'CROSSED')
            
        Returns:
            API yanıtı
        """
        try:
            if self.futures_type == 'usdt_m':
                return self.client.futures_change_margin_type(
                    symbol=symbol,
                    marginType=mode
                )
            else:
                return self.client.futures_coin_change_margin_type(
                    symbol=symbol,
                    marginType=mode
                )
        except BinanceAPIException as e:
            # Zaten aynı moddaysa hata verme
            if 'No need to change margin type' in str(e):
                logger.debug(f"Margin modu zaten {mode}")
                return {'code': 0, 'msg': 'No change needed'}
            logger.error(f"Margin modu ayarlanamadı: {e}")
            raise
    
    def market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Market order ver.
        
        Args:
            symbol: Sembol
            side: 'BUY' veya 'SELL'
            quantity: Miktar
            reduce_only: Sadece pozisyon azaltma
            
        Returns:
            Order bilgisi
        """
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': quantity
            }
            
            if reduce_only:
                params['reduceOnly'] = 'true'
            
            if self.futures_type == 'usdt_m':
                return self.client.futures_create_order(**params)
            else:
                return self.client.futures_coin_create_order(**params)
        except BinanceAPIException as e:
            logger.error(f"Market order verilemedi: {e}")
            raise
    
    def limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str = 'GTC',
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Limit order ver.
        
        Args:
            symbol: Sembol
            side: 'BUY' veya 'SELL'
            quantity: Miktar
            price: Fiyat
            time_in_force: GTC, IOC veya FOK
            reduce_only: Sadece pozisyon azaltma
            
        Returns:
            Order bilgisi
        """
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'quantity': quantity,
                'price': price,
                'timeInForce': time_in_force
            }
            
            if reduce_only:
                params['reduceOnly'] = 'true'
            
            if self.futures_type == 'usdt_m':
                return self.client.futures_create_order(**params)
            else:
                return self.client.futures_coin_create_order(**params)
        except BinanceAPIException as e:
            logger.error(f"Limit order verilemedi: {e}")
            raise
    
    def stop_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True
    ) -> Dict[str, Any]:
        """
        Stop market order ver.
        
        Args:
            symbol: Sembol
            side: 'BUY' veya 'SELL'
            quantity: Miktar
            stop_price: Stop fiyatı
            reduce_only: Sadece pozisyon azaltma
            
        Returns:
            Order bilgisi
        """
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': 'STOP_MARKET',
                'quantity': quantity,
                'stopPrice': stop_price
            }
            
            if reduce_only:
                params['reduceOnly'] = 'true'
            
            if self.futures_type == 'usdt_m':
                return self.client.futures_create_order(**params)
            else:
                return self.client.futures_coin_create_order(**params)
        except BinanceAPIException as e:
            logger.error(f"Stop market order verilemedi: {e}")
            raise
    
    def take_profit_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True
    ) -> Dict[str, Any]:
        """
        Take profit market order ver.
        
        Args:
            symbol: Sembol
            side: 'BUY' veya 'SELL'
            quantity: Miktar
            stop_price: Take profit fiyatı
            reduce_only: Sadece pozisyon azaltma
            
        Returns:
            Order bilgisi
        """
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': 'TAKE_PROFIT_MARKET',
                'quantity': quantity,
                'stopPrice': stop_price
            }
            
            if reduce_only:
                params['reduceOnly'] = 'true'
            
            if self.futures_type == 'usdt_m':
                return self.client.futures_create_order(**params)
            else:
                return self.client.futures_coin_create_order(**params)
        except BinanceAPIException as e:
            logger.error(f"Take profit order verilemedi: {e}")
            raise
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Order iptal et.
        
        Args:
            symbol: Sembol
            order_id: Order ID
            
        Returns:
            İptal edilen order bilgisi
        """
        try:
            if self.futures_type == 'usdt_m':
                return self.client.futures_cancel_order(
                    symbol=symbol,
                    orderId=order_id
                )
            else:
                return self.client.futures_coin_cancel_order(
                    symbol=symbol,
                    orderId=order_id
                )
        except BinanceAPIException as e:
            logger.error(f"Order iptal edilemedi: {e}")
            raise
    
    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """
        Semboldeki tüm açık order'ları iptal et.
        
        Args:
            symbol: Sembol
            
        Returns:
            API yanıtı
        """
        try:
            if self.futures_type == 'usdt_m':
                return self.client.futures_cancel_all_open_orders(symbol=symbol)
            else:
                return self.client.futures_coin_cancel_all_open_orders(symbol=symbol)
        except BinanceAPIException as e:
            logger.error(f"Order'lar iptal edilemedi: {e}")
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Açık order'ları getir.
        
        Args:
            symbol: Sembol (opsiyonel)
            
        Returns:
            Açık order listesi
        """
        try:
            if self.futures_type == 'usdt_m':
                if symbol:
                    return self.client.futures_get_open_orders(symbol=symbol)
                return self.client.futures_get_open_orders()
            else:
                if symbol:
                    return self.client.futures_coin_get_open_orders(symbol=symbol)
                return self.client.futures_coin_get_open_orders()
        except BinanceAPIException as e:
            logger.error(f"Açık order'lar alınamadı: {e}")
            raise
    
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """
        Kline (mum) verilerini getir.
        
        Args:
            symbol: Sembol
            interval: Zaman aralığı (1m, 5m, 15m, 1h, vb.)
            limit: Maksimum mum sayısı
            start_time: Başlangıç zamanı (ms)
            end_time: Bitiş zamanı (ms)
            
        Returns:
            Kline listesi
        """
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            if self.futures_type == 'usdt_m':
                return self.client.futures_klines(**params)
            else:
                return self.client.futures_coin_klines(**params)
        except BinanceAPIException as e:
            logger.error(f"Kline verileri alınamadı: {e}")
            raise
    
    def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """
        Güncel fiyat bilgisini getir.
        
        Args:
            symbol: Sembol
            
        Returns:
            Fiyat bilgisi
        """
        try:
            if self.futures_type == 'usdt_m':
                return self.client.futures_symbol_ticker(symbol=symbol)
            else:
                return self.client.futures_coin_symbol_ticker(symbol=symbol)
        except BinanceAPIException as e:
            logger.error(f"Fiyat bilgisi alınamadı: {e}")
            raise
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Exchange bilgilerini getir.
        
        Args:
            symbol: Sembol (opsiyonel)
            
        Returns:
            Exchange bilgileri
        """
        try:
            if self.futures_type == 'usdt_m':
                return self.client.futures_exchange_info()
            else:
                return self.client.futures_coin_exchange_info()
        except BinanceAPIException as e:
            logger.error(f"Exchange bilgisi alınamadı: {e}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Sembol bilgilerini getir.
        
        Args:
            symbol: Sembol
            
        Returns:
            Sembol bilgileri veya None
        """
        exchange_info = self.get_exchange_info()
        symbols = exchange_info.get('symbols', [])
        
        for s in symbols:
            if s.get('symbol') == symbol:
                return s
        
        return None
    
    def close(self) -> None:
        """
        Client bağlantısını kapat.
        """
        if self.client:
            self.client.close_connection()
            logger.info("Binance client bağlantısı kapatıldı")
