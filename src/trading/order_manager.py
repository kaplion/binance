"""
Order Manager Modülü

Bu modül, trading order'larını yönetir.
Market, limit, stop-market, stop-limit ve take-profit order'larını destekler.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.api.binance_client import BinanceClient
from src.utils.logger import get_logger
from src.utils.helpers import format_price, format_quantity

logger = get_logger(__name__)


class OrderType(Enum):
    """Order tipleri."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class OrderSide(Enum):
    """Order yönleri."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order durumları."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Order geçerlilik süreleri."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTX = "GTX"  # Good Till Crossing (Post Only)


@dataclass
class Order:
    """Order veri sınıfı."""
    order_id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    reduce_only: bool = False
    time_in_force: TimeInForce = TimeInForce.GTC
    
    def to_dict(self) -> Dict[str, Any]:
        """Order'ı dictionary'ye çevir."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_price': self.avg_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'reduce_only': self.reduce_only,
            'time_in_force': self.time_in_force.value
        }


class OrderManager:
    """
    Order yönetim sınıfı.
    
    Bu sınıf, tüm order işlemlerini yönetir:
    - Market/Limit order verme
    - Stop-loss ve take-profit order'ları
    - Order iptali ve güncelleme
    - Order durumu takibi
    """
    
    def __init__(self, client: BinanceClient, symbol: str):
        """
        OrderManager'ı başlat.
        
        Args:
            client: Binance API client
            symbol: İşlem yapılacak sembol
        """
        self.client = client
        self.symbol = symbol
        self._orders: Dict[int, Order] = {}
        self._symbol_info: Optional[Dict] = None
        
        # Sembol bilgilerini al
        self._load_symbol_info()
        
        logger.info(f"OrderManager başlatıldı - Sembol: {symbol}")
    
    def _load_symbol_info(self) -> None:
        """Sembol bilgilerini yükle."""
        try:
            self._symbol_info = self.client.get_symbol_info(self.symbol)
            if self._symbol_info:
                logger.debug(f"Sembol bilgileri yüklendi: {self.symbol}")
        except Exception as e:
            logger.error(f"Sembol bilgileri yüklenemedi: {e}")
    
    def _get_precision(self) -> tuple:
        """
        Sembol için fiyat ve miktar hassasiyetini getir.
        
        Returns:
            (price_precision, quantity_precision) tuple'ı
        """
        if not self._symbol_info:
            return (2, 3)  # Varsayılan değerler
        
        price_precision = self._symbol_info.get('pricePrecision', 2)
        quantity_precision = self._symbol_info.get('quantityPrecision', 3)
        
        return (price_precision, quantity_precision)
    
    def _parse_order_response(self, response: Dict) -> Order:
        """API yanıtından Order nesnesi oluştur."""
        return Order(
            order_id=response.get('orderId'),
            symbol=response.get('symbol'),
            side=OrderSide(response.get('side', 'BUY')),
            order_type=OrderType(response.get('type', 'MARKET')),
            quantity=float(response.get('origQty', 0)),
            price=float(response.get('price', 0)) if response.get('price') else None,
            stop_price=float(response.get('stopPrice', 0)) if response.get('stopPrice') else None,
            status=OrderStatus(response.get('status', 'NEW')),
            filled_quantity=float(response.get('executedQty', 0)),
            avg_price=float(response.get('avgPrice', 0)),
            reduce_only=response.get('reduceOnly', False),
            time_in_force=TimeInForce(response.get('timeInForce', 'GTC'))
        )
    
    def market_order(
        self,
        side: str,
        quantity: float,
        reduce_only: bool = False
    ) -> Optional[Order]:
        """
        Market order ver.
        
        Args:
            side: 'BUY' veya 'SELL'
            quantity: Miktar
            reduce_only: Sadece pozisyon azaltma
            
        Returns:
            Order nesnesi veya None
        """
        try:
            _, qty_precision = self._get_precision()
            formatted_qty = float(format_quantity(quantity, qty_precision))
            
            response = self.client.market_order(
                symbol=self.symbol,
                side=side,
                quantity=formatted_qty,
                reduce_only=reduce_only
            )
            
            order = self._parse_order_response(response)
            self._orders[order.order_id] = order
            
            logger.info(
                f"Market order verildi - ID: {order.order_id}, "
                f"Side: {side}, Quantity: {formatted_qty}"
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Market order hatası: {e}")
            return None
    
    def limit_order(
        self,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str = 'GTC',
        reduce_only: bool = False
    ) -> Optional[Order]:
        """
        Limit order ver.
        
        Args:
            side: 'BUY' veya 'SELL'
            quantity: Miktar
            price: Fiyat
            time_in_force: GTC, IOC veya FOK
            reduce_only: Sadece pozisyon azaltma
            
        Returns:
            Order nesnesi veya None
        """
        try:
            price_precision, qty_precision = self._get_precision()
            formatted_qty = float(format_quantity(quantity, qty_precision))
            formatted_price = float(format_price(price, price_precision))
            
            response = self.client.limit_order(
                symbol=self.symbol,
                side=side,
                quantity=formatted_qty,
                price=formatted_price,
                time_in_force=time_in_force,
                reduce_only=reduce_only
            )
            
            order = self._parse_order_response(response)
            self._orders[order.order_id] = order
            
            logger.info(
                f"Limit order verildi - ID: {order.order_id}, "
                f"Side: {side}, Price: {formatted_price}, Quantity: {formatted_qty}"
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Limit order hatası: {e}")
            return None
    
    def stop_loss_order(
        self,
        side: str,
        quantity: float,
        stop_price: float
    ) -> Optional[Order]:
        """
        Stop-loss order ver.
        
        Args:
            side: 'BUY' veya 'SELL'
            quantity: Miktar
            stop_price: Stop fiyatı
            
        Returns:
            Order nesnesi veya None
        """
        try:
            price_precision, qty_precision = self._get_precision()
            formatted_qty = float(format_quantity(quantity, qty_precision))
            formatted_stop_price = float(format_price(stop_price, price_precision))
            
            response = self.client.stop_market_order(
                symbol=self.symbol,
                side=side,
                quantity=formatted_qty,
                stop_price=formatted_stop_price,
                reduce_only=True
            )
            
            order = self._parse_order_response(response)
            self._orders[order.order_id] = order
            
            logger.info(
                f"Stop-loss order verildi - ID: {order.order_id}, "
                f"Side: {side}, Stop: {formatted_stop_price}"
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Stop-loss order hatası: {e}")
            return None
    
    def take_profit_order(
        self,
        side: str,
        quantity: float,
        take_profit_price: float
    ) -> Optional[Order]:
        """
        Take-profit order ver.
        
        Args:
            side: 'BUY' veya 'SELL'
            quantity: Miktar
            take_profit_price: Take-profit fiyatı
            
        Returns:
            Order nesnesi veya None
        """
        try:
            price_precision, qty_precision = self._get_precision()
            formatted_qty = float(format_quantity(quantity, qty_precision))
            formatted_tp_price = float(format_price(take_profit_price, price_precision))
            
            response = self.client.take_profit_market_order(
                symbol=self.symbol,
                side=side,
                quantity=formatted_qty,
                stop_price=formatted_tp_price,
                reduce_only=True
            )
            
            order = self._parse_order_response(response)
            self._orders[order.order_id] = order
            
            logger.info(
                f"Take-profit order verildi - ID: {order.order_id}, "
                f"Side: {side}, TP: {formatted_tp_price}"
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Take-profit order hatası: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Order iptal et.
        
        Args:
            order_id: Order ID
            
        Returns:
            Başarılı ise True
        """
        try:
            self.client.cancel_order(self.symbol, order_id)
            
            if order_id in self._orders:
                self._orders[order_id].status = OrderStatus.CANCELED
                self._orders[order_id].updated_at = datetime.now()
            
            logger.info(f"Order iptal edildi - ID: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Order iptal hatası: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """
        Tüm açık order'ları iptal et.
        
        Returns:
            Başarılı ise True
        """
        try:
            self.client.cancel_all_orders(self.symbol)
            
            for order in self._orders.values():
                if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                    order.status = OrderStatus.CANCELED
                    order.updated_at = datetime.now()
            
            logger.info("Tüm order'lar iptal edildi")
            return True
            
        except Exception as e:
            logger.error(f"Toplu order iptal hatası: {e}")
            return False
    
    def get_order(self, order_id: int) -> Optional[Order]:
        """
        Order bilgisini getir.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order nesnesi veya None
        """
        return self._orders.get(order_id)
    
    def get_open_orders(self) -> List[Order]:
        """
        Açık order'ları getir.
        
        Returns:
            Açık order listesi
        """
        try:
            open_orders = self.client.get_open_orders(self.symbol)
            
            orders = []
            for order_data in open_orders:
                order = self._parse_order_response(order_data)
                self._orders[order.order_id] = order
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Açık order'lar alınamadı: {e}")
            return []
    
    def sync_orders(self) -> None:
        """Order durumlarını Binance ile senkronize et."""
        try:
            open_orders = self.client.get_open_orders(self.symbol)
            open_order_ids = {o['orderId'] for o in open_orders}
            
            for order_id, order in self._orders.items():
                if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                    if order_id not in open_order_ids:
                        # Order kapatılmış, durumu güncelle
                        order.status = OrderStatus.FILLED
                        order.updated_at = datetime.now()
            
            logger.debug("Order'lar senkronize edildi")
            
        except Exception as e:
            logger.error(f"Order senkronizasyon hatası: {e}")
    
    def get_all_orders(self) -> Dict[int, Order]:
        """Tüm order'ları getir."""
        return self._orders.copy()
