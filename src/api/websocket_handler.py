"""
WebSocket Handler Modülü

Bu modül, Binance Futures WebSocket bağlantılarını yönetir.
Gerçek zamanlı fiyat akışı, order book güncellemeleri ve
user data stream desteği sağlar.
"""

import asyncio
import json
import threading
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

import websocket

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WebSocketHandler:
    """
    Binance Futures WebSocket bağlantı yöneticisi.
    
    Bu sınıf, gerçek zamanlı veri akışı için WebSocket bağlantılarını
    yönetir ve otomatik yeniden bağlanma mekanizması sağlar.
    """
    
    # WebSocket endpoint'leri
    TESTNET_WS_URL = "wss://stream.binancefuture.com"
    MAINNET_WS_URL = "wss://fstream.binance.com"
    
    def __init__(
        self,
        testnet: bool = True,
        auto_reconnect: bool = True,
        reconnect_attempts: int = 5,
        reconnect_delay: int = 5,
        ping_interval: int = 30
    ):
        """
        WebSocketHandler'ı başlat.
        
        Args:
            testnet: Testnet modunu kullan
            auto_reconnect: Otomatik yeniden bağlanma
            reconnect_attempts: Yeniden bağlanma deneme sayısı
            reconnect_delay: Yeniden bağlanma bekleme süresi (saniye)
            ping_interval: Ping aralığı (saniye)
        """
        self.testnet = testnet
        self.auto_reconnect = auto_reconnect
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.ping_interval = ping_interval
        
        self.base_url = self.TESTNET_WS_URL if testnet else self.MAINNET_WS_URL
        
        # Aktif WebSocket bağlantıları
        self._connections: Dict[str, websocket.WebSocketApp] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        
        # Veri depolama
        self._klines: Dict[str, Dict] = {}
        self._orderbooks: Dict[str, Dict] = {}
        self._tickers: Dict[str, Dict] = {}
        
        logger.info(f"WebSocketHandler başlatıldı - Testnet: {testnet}")
    
    def _get_stream_url(self, stream: str) -> str:
        """WebSocket stream URL'ini oluştur."""
        return f"{self.base_url}/ws/{stream}"
    
    def _get_combined_stream_url(self, streams: List[str]) -> str:
        """Birleşik stream URL'ini oluştur."""
        streams_str = "/".join(streams)
        return f"{self.base_url}/stream?streams={streams_str}"
    
    def _on_message(self, ws, message: str, stream_key: str) -> None:
        """WebSocket mesaj callback'i."""
        try:
            data = json.loads(message)
            
            # Birleşik stream ise veriyi çıkar
            if 'stream' in data:
                stream_name = data['stream']
                data = data['data']
            else:
                stream_name = stream_key
            
            # Veri tipine göre işle
            event_type = data.get('e', '')
            
            if event_type == 'kline':
                self._handle_kline(data)
            elif event_type == 'depthUpdate':
                self._handle_depth_update(data)
            elif event_type == '24hrTicker':
                self._handle_ticker(data)
            elif event_type == 'aggTrade':
                self._handle_agg_trade(data)
            
            # Kayıtlı callback'leri çağır
            if stream_key in self._callbacks:
                for callback in self._callbacks[stream_key]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Callback hatası: {e}")
                        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse hatası: {e}")
        except Exception as e:
            logger.error(f"Mesaj işleme hatası: {e}")
    
    def _on_error(self, ws, error) -> None:
        """WebSocket hata callback'i."""
        logger.error(f"WebSocket hatası: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """WebSocket kapanış callback'i."""
        logger.warning(
            f"WebSocket bağlantısı kapandı - "
            f"Kod: {close_status_code}, Mesaj: {close_msg}"
        )
        
        if self.auto_reconnect and self._running:
            self._schedule_reconnect(ws)
    
    def _on_open(self, ws) -> None:
        """WebSocket açılış callback'i."""
        logger.info("WebSocket bağlantısı açıldı")
    
    def _handle_kline(self, data: Dict) -> None:
        """Kline verisini işle."""
        kline = data.get('k', {})
        symbol = kline.get('s', '')
        interval = kline.get('i', '')
        
        key = f"{symbol}_{interval}"
        self._klines[key] = {
            'symbol': symbol,
            'interval': interval,
            'open_time': kline.get('t'),
            'close_time': kline.get('T'),
            'open': float(kline.get('o', 0)),
            'high': float(kline.get('h', 0)),
            'low': float(kline.get('l', 0)),
            'close': float(kline.get('c', 0)),
            'volume': float(kline.get('v', 0)),
            'quote_volume': float(kline.get('q', 0)),
            'trades': kline.get('n'),
            'is_closed': kline.get('x', False),
            'timestamp': datetime.now()
        }
    
    def _handle_depth_update(self, data: Dict) -> None:
        """Depth (order book) verisini işle."""
        symbol = data.get('s', '')
        
        self._orderbooks[symbol] = {
            'symbol': symbol,
            'event_time': data.get('E'),
            'transaction_time': data.get('T'),
            'first_update_id': data.get('U'),
            'final_update_id': data.get('u'),
            'bids': data.get('b', []),
            'asks': data.get('a', []),
            'timestamp': datetime.now()
        }
    
    def _handle_ticker(self, data: Dict) -> None:
        """24hr ticker verisini işle."""
        symbol = data.get('s', '')
        
        self._tickers[symbol] = {
            'symbol': symbol,
            'price_change': float(data.get('p', 0)),
            'price_change_percent': float(data.get('P', 0)),
            'weighted_avg_price': float(data.get('w', 0)),
            'last_price': float(data.get('c', 0)),
            'last_qty': float(data.get('Q', 0)),
            'open_price': float(data.get('o', 0)),
            'high_price': float(data.get('h', 0)),
            'low_price': float(data.get('l', 0)),
            'volume': float(data.get('v', 0)),
            'quote_volume': float(data.get('q', 0)),
            'open_time': data.get('O'),
            'close_time': data.get('C'),
            'first_trade_id': data.get('F'),
            'last_trade_id': data.get('L'),
            'trade_count': data.get('n'),
            'timestamp': datetime.now()
        }
    
    def _handle_agg_trade(self, data: Dict) -> None:
        """Aggregated trade verisini işle."""
        pass  # Gerekirse implement edilebilir
    
    def _schedule_reconnect(self, ws) -> None:
        """Yeniden bağlanmayı planla."""
        for attempt in range(self.reconnect_attempts):
            logger.info(
                f"Yeniden bağlanma denemesi {attempt + 1}/{self.reconnect_attempts}"
            )
            try:
                asyncio.sleep(self.reconnect_delay)
                ws.run_forever()
                return
            except Exception as e:
                logger.error(f"Yeniden bağlanma hatası: {e}")
        
        logger.error("Yeniden bağlanma başarısız")
    
    def subscribe_kline(
        self,
        symbol: str,
        interval: str,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Kline (mum) stream'ine abone ol.
        
        Args:
            symbol: Sembol (örn: 'btcusdt')
            interval: Zaman aralığı (1m, 5m, 15m, vb.)
            callback: Veri geldiğinde çağrılacak fonksiyon
            
        Returns:
            Stream anahtarı
        """
        stream = f"{symbol.lower()}@kline_{interval}"
        stream_key = f"kline_{symbol}_{interval}"
        
        self._subscribe(stream, stream_key, callback)
        return stream_key
    
    def subscribe_depth(
        self,
        symbol: str,
        levels: int = 10,
        update_speed: str = '100ms',
        callback: Optional[Callable] = None
    ) -> str:
        """
        Order book depth stream'ine abone ol.
        
        Args:
            symbol: Sembol
            levels: Seviye sayısı (5, 10, veya 20)
            update_speed: Güncelleme hızı ('100ms' veya '250ms')
            callback: Veri geldiğinde çağrılacak fonksiyon
            
        Returns:
            Stream anahtarı
        """
        stream = f"{symbol.lower()}@depth{levels}@{update_speed}"
        stream_key = f"depth_{symbol}"
        
        self._subscribe(stream, stream_key, callback)
        return stream_key
    
    def subscribe_ticker(
        self,
        symbol: str,
        callback: Optional[Callable] = None
    ) -> str:
        """
        24hr ticker stream'ine abone ol.
        
        Args:
            symbol: Sembol
            callback: Veri geldiğinde çağrılacak fonksiyon
            
        Returns:
            Stream anahtarı
        """
        stream = f"{symbol.lower()}@ticker"
        stream_key = f"ticker_{symbol}"
        
        self._subscribe(stream, stream_key, callback)
        return stream_key
    
    def subscribe_agg_trade(
        self,
        symbol: str,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Aggregated trade stream'ine abone ol.
        
        Args:
            symbol: Sembol
            callback: Veri geldiğinde çağrılacak fonksiyon
            
        Returns:
            Stream anahtarı
        """
        stream = f"{symbol.lower()}@aggTrade"
        stream_key = f"aggtrade_{symbol}"
        
        self._subscribe(stream, stream_key, callback)
        return stream_key
    
    def _subscribe(
        self,
        stream: str,
        stream_key: str,
        callback: Optional[Callable] = None
    ) -> None:
        """Stream'e abone ol."""
        url = self._get_stream_url(stream)
        
        # Callback'i kaydet
        if callback:
            if stream_key not in self._callbacks:
                self._callbacks[stream_key] = []
            self._callbacks[stream_key].append(callback)
        
        # WebSocket oluştur
        ws = websocket.WebSocketApp(
            url,
            on_message=lambda ws, msg: self._on_message(ws, msg, stream_key),
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        self._connections[stream_key] = ws
        
        # Thread'de çalıştır
        thread = threading.Thread(
            target=ws.run_forever,
            kwargs={'ping_interval': self.ping_interval}
        )
        thread.daemon = True
        thread.start()
        
        self._threads[stream_key] = thread
        self._running = True
        
        logger.info(f"Stream'e abone olundu: {stream}")
    
    def unsubscribe(self, stream_key: str) -> None:
        """
        Stream aboneliğini iptal et.
        
        Args:
            stream_key: Stream anahtarı
        """
        if stream_key in self._connections:
            ws = self._connections[stream_key]
            ws.close()
            del self._connections[stream_key]
            
            if stream_key in self._callbacks:
                del self._callbacks[stream_key]
            
            logger.info(f"Stream aboneliği iptal edildi: {stream_key}")
    
    def get_kline(self, symbol: str, interval: str) -> Optional[Dict]:
        """
        Son kline verisini getir.
        
        Args:
            symbol: Sembol
            interval: Zaman aralığı
            
        Returns:
            Kline verisi veya None
        """
        key = f"{symbol}_{interval}"
        return self._klines.get(key)
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """
        Son order book verisini getir.
        
        Args:
            symbol: Sembol
            
        Returns:
            Order book verisi veya None
        """
        return self._orderbooks.get(symbol)
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Son ticker verisini getir.
        
        Args:
            symbol: Sembol
            
        Returns:
            Ticker verisi veya None
        """
        return self._tickers.get(symbol)
    
    def close_all(self) -> None:
        """Tüm WebSocket bağlantılarını kapat."""
        self._running = False
        
        for stream_key in list(self._connections.keys()):
            self.unsubscribe(stream_key)
        
        logger.info("Tüm WebSocket bağlantıları kapatıldı")
    
    def is_connected(self, stream_key: str) -> bool:
        """
        Belirtilen stream'in bağlı olup olmadığını kontrol et.
        
        Args:
            stream_key: Stream anahtarı
            
        Returns:
            Bağlı ise True
        """
        if stream_key in self._connections:
            ws = self._connections[stream_key]
            return ws.sock and ws.sock.connected
        return False
