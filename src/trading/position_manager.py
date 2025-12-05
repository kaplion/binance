"""
Position Manager Modülü

Bu modül, açık pozisyonları yönetir.
Pozisyon açma, kapatma, güncelleme ve takip işlemlerini sağlar.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.api.binance_client import BinanceClient
from src.trading.order_manager import OrderManager, OrderSide
from src.utils.logger import get_logger
from src.utils.helpers import calculate_pnl, calculate_stop_loss, calculate_take_profit

logger = get_logger(__name__)


class PositionSide(Enum):
    """Pozisyon yönleri."""
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"  # Hedge mode için


class PositionStatus(Enum):
    """Pozisyon durumları."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"


@dataclass
class Position:
    """Pozisyon veri sınıfı."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    leverage: int
    margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    liquidation_price: float = 0.0
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    stop_loss_order_id: Optional[int] = None
    take_profit_order_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Pozisyonu dictionary'ye çevir."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'leverage': self.leverage,
            'margin': self.margin,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'liquidation_price': self.liquidation_price,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def calculate_pnl(self, current_price: float) -> Dict[str, float]:
        """
        Mevcut fiyata göre PnL hesapla.
        
        Args:
            current_price: Mevcut fiyat
            
        Returns:
            PnL detayları
        """
        return calculate_pnl(
            entry_price=self.entry_price,
            exit_price=current_price,
            quantity=self.quantity,
            side=self.side.value,
            leverage=self.leverage
        )


class PositionManager:
    """
    Pozisyon yönetim sınıfı.
    
    Bu sınıf, pozisyon işlemlerini yönetir:
    - Pozisyon açma/kapatma
    - Stop-loss ve take-profit ayarlama
    - Pozisyon boyutlandırma
    - PnL hesaplama ve takip
    """
    
    def __init__(
        self,
        client: BinanceClient,
        order_manager: OrderManager,
        symbol: str,
        leverage: int = 10
    ):
        """
        PositionManager'ı başlat.
        
        Args:
            client: Binance API client
            order_manager: Order yöneticisi
            symbol: İşlem yapılacak sembol
            leverage: Kaldıraç oranı
        """
        self.client = client
        self.order_manager = order_manager
        self.symbol = symbol
        self.leverage = leverage
        self._positions: Dict[str, Position] = {}
        
        # Leverage'ı ayarla
        self._setup_leverage()
        
        logger.info(
            f"PositionManager başlatıldı - Sembol: {symbol}, Leverage: {leverage}"
        )
    
    def _setup_leverage(self) -> None:
        """Leverage'ı ayarla."""
        try:
            self.client.set_leverage(self.symbol, self.leverage)
            logger.debug(f"Leverage ayarlandı: {self.leverage}x")
        except Exception as e:
            logger.warning(f"Leverage ayarlanamadı: {e}")
    
    def _get_position_key(self, side: PositionSide) -> str:
        """Pozisyon için benzersiz anahtar oluştur."""
        return f"{self.symbol}_{side.value}"
    
    def open_position(
        self,
        side: str,
        quantity: float,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        limit_price: Optional[float] = None
    ) -> Optional[Position]:
        """
        Yeni pozisyon aç.
        
        Args:
            side: 'LONG' veya 'SHORT'
            quantity: Pozisyon miktarı
            stop_loss_pct: Stop-loss yüzdesi
            take_profit_pct: Take-profit yüzdesi
            limit_price: Limit order için fiyat (None ise market order)
            
        Returns:
            Position nesnesi veya None
        """
        try:
            # Order yönünü belirle
            order_side = 'BUY' if side.upper() == 'LONG' else 'SELL'
            
            # Order ver
            if limit_price:
                order = self.order_manager.limit_order(
                    side=order_side,
                    quantity=quantity,
                    price=limit_price
                )
            else:
                order = self.order_manager.market_order(
                    side=order_side,
                    quantity=quantity
                )
            
            if not order:
                return None
            
            # Giriş fiyatını al
            entry_price = order.avg_price if order.avg_price > 0 else self._get_current_price()
            
            # Pozisyon oluştur
            position = Position(
                symbol=self.symbol,
                side=PositionSide(side.upper()),
                quantity=quantity,
                entry_price=entry_price,
                leverage=self.leverage,
                margin=quantity * entry_price / self.leverage
            )
            
            # Stop-loss ve take-profit ayarla
            if stop_loss_pct:
                self._set_stop_loss(position, stop_loss_pct)
            
            if take_profit_pct:
                self._set_take_profit(position, take_profit_pct)
            
            # Pozisyonu kaydet
            key = self._get_position_key(position.side)
            self._positions[key] = position
            
            logger.info(
                f"Pozisyon açıldı - Side: {side}, Quantity: {quantity}, "
                f"Entry: {entry_price}, Leverage: {self.leverage}x"
            )
            
            return position
            
        except Exception as e:
            logger.error(f"Pozisyon açma hatası: {e}")
            return None
    
    def close_position(
        self,
        side: str,
        quantity: Optional[float] = None
    ) -> bool:
        """
        Pozisyonu kapat.
        
        Args:
            side: 'LONG' veya 'SHORT'
            quantity: Kapatılacak miktar (None ise tamamı)
            
        Returns:
            Başarılı ise True
        """
        try:
            key = self._get_position_key(PositionSide(side.upper()))
            position = self._positions.get(key)
            
            if not position:
                logger.warning(f"Kapatılacak pozisyon bulunamadı: {side}")
                return False
            
            # Kapatılacak miktarı belirle
            close_quantity = quantity if quantity else position.quantity
            
            # Ters yönde order ver
            order_side = 'SELL' if side.upper() == 'LONG' else 'BUY'
            
            order = self.order_manager.market_order(
                side=order_side,
                quantity=close_quantity,
                reduce_only=True
            )
            
            if not order:
                return False
            
            # PnL hesapla
            exit_price = order.avg_price if order.avg_price > 0 else self._get_current_price()
            pnl = position.calculate_pnl(exit_price)
            position.realized_pnl += pnl['pnl']
            
            # Pozisyon durumunu güncelle
            if close_quantity >= position.quantity:
                position.status = PositionStatus.CLOSED
                position.quantity = 0
                
                # İlgili order'ları iptal et
                if position.stop_loss_order_id:
                    self.order_manager.cancel_order(position.stop_loss_order_id)
                if position.take_profit_order_id:
                    self.order_manager.cancel_order(position.take_profit_order_id)
            else:
                position.status = PositionStatus.PARTIALLY_CLOSED
                position.quantity -= close_quantity
            
            position.updated_at = datetime.now()
            
            logger.info(
                f"Pozisyon kapatıldı - Side: {side}, Quantity: {close_quantity}, "
                f"PnL: {pnl['pnl']:.2f} ({pnl['pnl_percentage']:.2f}%)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Pozisyon kapatma hatası: {e}")
            return False
    
    def _set_stop_loss(self, position: Position, stop_loss_pct: float) -> None:
        """Stop-loss order'ı ayarla."""
        stop_loss_price = calculate_stop_loss(
            entry_price=position.entry_price,
            side=position.side.value,
            stop_loss_pct=stop_loss_pct
        )
        
        # Stop-loss order yönü
        order_side = 'SELL' if position.side == PositionSide.LONG else 'BUY'
        
        order = self.order_manager.stop_loss_order(
            side=order_side,
            quantity=position.quantity,
            stop_price=stop_loss_price
        )
        
        if order:
            position.stop_loss_price = stop_loss_price
            position.stop_loss_order_id = order.order_id
            logger.debug(f"Stop-loss ayarlandı: {stop_loss_price}")
    
    def _set_take_profit(self, position: Position, take_profit_pct: float) -> None:
        """Take-profit order'ı ayarla."""
        take_profit_price = calculate_take_profit(
            entry_price=position.entry_price,
            side=position.side.value,
            take_profit_pct=take_profit_pct
        )
        
        # Take-profit order yönü
        order_side = 'SELL' if position.side == PositionSide.LONG else 'BUY'
        
        order = self.order_manager.take_profit_order(
            side=order_side,
            quantity=position.quantity,
            take_profit_price=take_profit_price
        )
        
        if order:
            position.take_profit_price = take_profit_price
            position.take_profit_order_id = order.order_id
            logger.debug(f"Take-profit ayarlandı: {take_profit_price}")
    
    def update_stop_loss(self, side: str, new_stop_price: float) -> bool:
        """
        Stop-loss fiyatını güncelle.
        
        Args:
            side: 'LONG' veya 'SHORT'
            new_stop_price: Yeni stop-loss fiyatı
            
        Returns:
            Başarılı ise True
        """
        try:
            key = self._get_position_key(PositionSide(side.upper()))
            position = self._positions.get(key)
            
            if not position:
                return False
            
            # Eski order'ı iptal et
            if position.stop_loss_order_id:
                self.order_manager.cancel_order(position.stop_loss_order_id)
            
            # Yeni order ver
            order_side = 'SELL' if position.side == PositionSide.LONG else 'BUY'
            
            order = self.order_manager.stop_loss_order(
                side=order_side,
                quantity=position.quantity,
                stop_price=new_stop_price
            )
            
            if order:
                position.stop_loss_price = new_stop_price
                position.stop_loss_order_id = order.order_id
                position.updated_at = datetime.now()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Stop-loss güncelleme hatası: {e}")
            return False
    
    def _get_current_price(self) -> float:
        """Mevcut fiyatı al."""
        try:
            ticker = self.client.get_ticker_price(self.symbol)
            return float(ticker.get('price', 0))
        except Exception:
            return 0.0
    
    def get_position(self, side: str) -> Optional[Position]:
        """
        Pozisyon bilgisini getir.
        
        Args:
            side: 'LONG' veya 'SHORT'
            
        Returns:
            Position nesnesi veya None
        """
        key = self._get_position_key(PositionSide(side.upper()))
        return self._positions.get(key)
    
    def get_all_positions(self) -> List[Position]:
        """Tüm açık pozisyonları getir."""
        return [p for p in self._positions.values() if p.status == PositionStatus.OPEN]
    
    def sync_positions(self) -> None:
        """Pozisyonları Binance ile senkronize et."""
        try:
            binance_positions = self.client.get_positions(self.symbol)
            
            for bp in binance_positions:
                position_amt = float(bp.get('positionAmt', 0))
                
                if position_amt != 0:
                    side = PositionSide.LONG if position_amt > 0 else PositionSide.SHORT
                    key = self._get_position_key(side)
                    
                    if key not in self._positions:
                        # Yeni pozisyon bulduk
                        self._positions[key] = Position(
                            symbol=self.symbol,
                            side=side,
                            quantity=abs(position_amt),
                            entry_price=float(bp.get('entryPrice', 0)),
                            leverage=int(bp.get('leverage', self.leverage)),
                            unrealized_pnl=float(bp.get('unRealizedProfit', 0)),
                            liquidation_price=float(bp.get('liquidationPrice', 0))
                        )
                    else:
                        # Mevcut pozisyonu güncelle
                        position = self._positions[key]
                        position.quantity = abs(position_amt)
                        position.unrealized_pnl = float(bp.get('unRealizedProfit', 0))
                        position.updated_at = datetime.now()
            
            logger.debug("Pozisyonlar senkronize edildi")
            
        except Exception as e:
            logger.error(f"Pozisyon senkronizasyon hatası: {e}")
    
    def has_open_position(self, side: Optional[str] = None) -> bool:
        """
        Açık pozisyon var mı kontrol et.
        
        Args:
            side: 'LONG' veya 'SHORT' (None ise herhangi biri)
            
        Returns:
            Açık pozisyon varsa True
        """
        if side:
            key = self._get_position_key(PositionSide(side.upper()))
            position = self._positions.get(key)
            return position is not None and position.status == PositionStatus.OPEN
        
        return any(p.status == PositionStatus.OPEN for p in self._positions.values())
    
    def get_total_pnl(self) -> Dict[str, float]:
        """
        Toplam PnL hesapla.
        
        Returns:
            PnL detayları
        """
        total_realized = sum(p.realized_pnl for p in self._positions.values())
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        
        return {
            'realized_pnl': total_realized,
            'unrealized_pnl': total_unrealized,
            'total_pnl': total_realized + total_unrealized
        }
