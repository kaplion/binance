"""
Ana Uygulama Modülü

Binance Futures AI Trading Bot'un ana giriş noktası.
Bu modül, tüm bileşenleri koordine eder ve trading döngüsünü yönetir.
"""

import asyncio
import signal
import sys
import threading
from typing import Any, Dict, Optional
from pathlib import Path

import pandas as pd

from config.settings import get_settings, Settings
from src.api.binance_client import BinanceClient
from src.api.websocket_handler import WebSocketHandler
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.ml_strategy import MLStrategy
from src.ai.predictor import Predictor
from src.indicators.technical import TechnicalIndicators
from src.utils.logger import setup_logger, get_logger


class TradingBot:
    """
    Ana trading bot sınıfı.
    
    Bu sınıf, tüm trading bileşenlerini koordine eder:
    - API bağlantısı
    - WebSocket veri akışı
    - Strateji çalıştırma
    - Order ve pozisyon yönetimi
    - Risk kontrolü
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        TradingBot'u başlat.
        
        Args:
            config_path: Konfigürasyon dosyası yolu
        """
        # Ayarları yükle
        self.settings = get_settings(config_path)
        
        # Logger'ı ayarla
        logging_config = self.settings.logging_config
        setup_logger(
            level=logging_config.get('level', 'INFO'),
            log_file=logging_config.get('file_path'),
            max_size=logging_config.get('max_size', 10),
            backup_count=logging_config.get('backup_count', 5),
            console_output=logging_config.get('console_output', True)
        )
        
        self.logger = get_logger('TradingBot')
        self.logger.info("Trading bot başlatılıyor...")
        
        # Trading parametreleri
        self.symbol = self.settings.trading.get('symbol', 'BTCUSDT')
        self.timeframe = self.settings.strategy.get('timeframe', '15m')
        self.leverage = self.settings.trading.get('leverage', 10)
        
        # Bileşenler
        self.client: Optional[BinanceClient] = None
        self.ws_handler: Optional[WebSocketHandler] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.strategy = None
        self.predictor: Optional[Predictor] = None
        
        # Durum
        self._running = False
        self._kline_data: pd.DataFrame = pd.DataFrame()
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Sinyal yakalayıcı."""
        self.logger.info("Kapatma sinyali alındı...")
        self.stop()
    
    def _initialize_components(self) -> None:
        """Bileşenleri başlat."""
        binance_config = self.settings.binance
        
        # API client
        self.client = BinanceClient(
            api_key=binance_config.get('api_key', ''),
            api_secret=binance_config.get('api_secret', ''),
            testnet=binance_config.get('testnet', True),
            futures_type=binance_config.get('futures_type', 'usdt_m')
        )
        
        # WebSocket handler
        ws_config = self.settings.websocket
        self.ws_handler = WebSocketHandler(
            testnet=binance_config.get('testnet', True),
            auto_reconnect=ws_config.get('auto_reconnect', True),
            reconnect_attempts=ws_config.get('reconnect_attempts', 5),
            reconnect_delay=ws_config.get('reconnect_delay', 5),
            ping_interval=ws_config.get('ping_interval', 30)
        )
        
        # Order manager
        self.order_manager = OrderManager(self.client, self.symbol)
        
        # Position manager
        self.position_manager = PositionManager(
            self.client,
            self.order_manager,
            self.symbol,
            self.leverage
        )
        
        # Risk manager
        risk_config = self.settings.risk
        trading_config = self.settings.trading
        risk_config['position_size_pct'] = trading_config.get('position_size_pct', 5)
        risk_config['max_positions'] = trading_config.get('max_positions', 3)
        
        self.risk_manager = RiskManager(
            self.client,
            self.position_manager,
            risk_config
        )
        
        # Strateji
        self._initialize_strategy()
        
        self.logger.info("Tüm bileşenler başlatıldı")
    
    def _initialize_strategy(self) -> None:
        """Stratejiyi başlat."""
        strategy_config = self.settings.strategy
        strategy_type = strategy_config.get('type', 'ml')
        
        # Strateji konfigürasyonu
        full_config = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'risk_reward_ratio': self.settings.risk.get('risk_reward_ratio', 2),
            'stop_loss_pct': self.settings.risk.get('stop_loss_pct', 2),
            'take_profit_pct': self.settings.risk.get('take_profit_pct', 4),
            'momentum': strategy_config.get('momentum', {}),
            'ml': strategy_config.get('ml', {}),
            'ai': self.settings.ai
        }
        
        if strategy_type == 'momentum':
            self.strategy = MomentumStrategy(full_config)
        else:
            # ML stratejisi
            self.strategy = MLStrategy(full_config)
            
            # Predictor'ı ayarla
            ai_config = self.settings.ai
            self.predictor = Predictor(
                model_type=ai_config.get('model_type', 'lstm'),
                config=ai_config
            )
            self.strategy.set_predictor(self.predictor)
        
        self.logger.info(f"Strateji başlatıldı: {strategy_type}")
    
    def _fetch_historical_data(self, limit: int = 500) -> pd.DataFrame:
        """
        Geçmiş veri çek.
        
        Args:
            limit: Maksimum mum sayısı
            
        Returns:
            OHLCV DataFrame'i
        """
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Veri tiplerini düzelt
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Geçmiş veri çekme hatası: {e}")
            return pd.DataFrame()
    
    def _on_kline_update(self, data: Dict[str, Any]) -> None:
        """
        Yeni kline verisi geldiğinde çağrılır.
        
        Args:
            data: Kline verisi
        """
        try:
            kline = data.get('k', {})
            
            if kline.get('x', False):  # Mum kapandı mı
                self.logger.debug(f"Mum kapandı: {kline.get('c')}")
                
                # Trading döngüsünü ayrı thread'de çalıştır
                thread = threading.Thread(target=self._run_trading_loop, daemon=True)
                thread.start()
                
        except Exception as e:
            self.logger.error(f"Kline güncelleme hatası: {e}")
    
    def _run_trading_loop(self) -> None:
        """Trading döngüsünü yeni event loop'ta çalıştır."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._trading_loop())
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Trading loop hatası: {e}")
    
    async def _trading_loop(self) -> None:
        """Ana trading döngüsü."""
        try:
            # Güncel veriyi çek
            df = self._fetch_historical_data(200)
            
            if df.empty:
                self.logger.warning("Veri boş, trading atlanıyor")
                return
            
            self.logger.debug(f"Veri alındı: {len(df)} mum, son fiyat: {df['close'].iloc[-1]:.2f}")
            
            # Risk kontrolü
            risk_check = self.risk_manager.check_risk_limits()
            
            if not risk_check['can_trade']:
                for warning in risk_check.get('warnings', []):
                    self.logger.warning(warning)
                return
            
            self.logger.debug("Risk kontrolü geçti")
            
            # Mevcut pozisyon kontrol
            has_position = self.position_manager.has_open_position()
            self.logger.debug(f"Açık pozisyon var mı: {has_position}")
            
            if has_position:
                # Pozisyon varsa çıkış kontrol
                await self._check_exit(df)
            else:
                # Pozisyon yoksa giriş kontrol
                await self._check_entry(df)
                
        except Exception as e:
            self.logger.error(f"Trading döngüsü hatası: {e}")
    
    async def _check_entry(self, df: pd.DataFrame) -> None:
        """Giriş sinyali kontrol et."""
        signal = self.strategy.analyze(df)
        
        # Detaylı sinyal logu
        self.logger.info(
            f"Sinyal analizi - "
            f"Yön: {signal.signal.value}, "
            f"Confidence: {signal.confidence:.2f}, "
            f"Fiyat: {signal.price:.2f}, "
            f"Neden: {signal.reason}"
        )
        
        if signal.metadata:
            self.logger.debug(f"Sinyal metadata: {signal.metadata}")
        
        if not self.strategy.validate_signal(signal):
            self.logger.debug("Sinyal validasyonu başarısız")
            return
        
        if signal.signal.value == 'BUY':
            self.logger.info(f"🟢 LONG sinyali tespit edildi! Confidence: {signal.confidence:.2f}")
            await self._open_position('LONG', signal)
        elif signal.signal.value == 'SELL':
            self.logger.info(f"🔴 SHORT sinyali tespit edildi! Confidence: {signal.confidence:.2f}")
            await self._open_position('SHORT', signal)
    
    async def _open_position(self, side: str, signal) -> None:
        """
        Pozisyon aç.
        
        Args:
            side: 'LONG' veya 'SHORT'
            signal: Trading sinyali
        """
        try:
            # Pozisyon boyutunu hesapla
            entry_price = signal.price
            stop_loss_price = signal.stop_loss
            
            balance = self.client.get_usdt_balance()
            risk_pct = self.settings.trading.get('position_size_pct', 5)
            
            quantity = self.risk_manager.calculate_position_size(
                entry_price, stop_loss_price, risk_pct
            )
            
            if quantity <= 0:
                self.logger.warning("Hesaplanan pozisyon boyutu sıfır veya negatif")
                return
            
            # Pozisyonu aç
            position = self.position_manager.open_position(
                side=side,
                quantity=quantity,
                stop_loss_pct=self.settings.risk.get('stop_loss_pct', 2),
                take_profit_pct=self.settings.risk.get('take_profit_pct', 4)
            )
            
            if position:
                self.logger.info(
                    f"Pozisyon açıldı - Side: {side}, "
                    f"Quantity: {quantity:.4f}, "
                    f"Entry: {entry_price:.2f}, "
                    f"Confidence: {signal.confidence:.2f}"
                )
                
        except Exception as e:
            self.logger.error(f"Pozisyon açma hatası: {e}")
    
    async def _check_exit(self, df: pd.DataFrame) -> None:
        """Çıkış sinyali kontrol et."""
        positions = self.position_manager.get_all_positions()
        
        for position in positions:
            side = position.side.value
            
            if self.strategy.should_exit(df, side):
                self.position_manager.close_position(side)
                self.logger.info(f"Pozisyon kapatıldı: {side}")
    
    def start(self) -> None:
        """Bot'u başlat."""
        try:
            # Konfigürasyonu doğrula
            self.settings.validate()
            
            # Bileşenleri başlat
            self._initialize_components()
            
            # Geçmiş veriyi çek
            self._kline_data = self._fetch_historical_data()
            self.logger.info(f"Geçmiş veri yüklendi: {len(self._kline_data)} mum")
            
            # WebSocket'e abone ol
            self.ws_handler.subscribe_kline(
                symbol=self.symbol,
                interval=self.timeframe,
                callback=self._on_kline_update
            )
            
            self._running = True
            self.logger.info(
                f"Bot başlatıldı - "
                f"Sembol: {self.symbol}, "
                f"Timeframe: {self.timeframe}, "
                f"Strateji: {self.strategy.name}"
            )
            
            # Ana döngü
            import time
            while self._running:
                time.sleep(1)
                
        except ValueError as e:
            self.logger.error(f"Konfigürasyon hatası: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            self.logger.error(f"Başlatma hatası: {e}")
            self.stop()
    
    def stop(self) -> None:
        """Bot'u durdur."""
        self.logger.info("Bot durduruluyor...")
        self._running = False
        
        # WebSocket bağlantılarını kapat
        if self.ws_handler:
            self.ws_handler.close_all()
        
        # API client'ı kapat
        if self.client:
            self.client.close()
        
        self.logger.info("Bot durduruldu")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Bot durumunu döndür.
        
        Returns:
            Durum bilgileri
        """
        return {
            'running': self._running,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'strategy': self.strategy.name if self.strategy else None,
            'positions': [
                p.to_dict() 
                for p in self.position_manager.get_all_positions()
            ] if self.position_manager else [],
            'risk_metrics': self.risk_manager.get_metrics() if self.risk_manager else {},
            'testnet': self.settings.is_testnet()
        }


def main():
    """Ana giriş noktası."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Binance Futures AI Trading Bot')
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/config.yaml',
        help='Konfigürasyon dosyası yolu'
    )
    parser.add_argument(
        '--testnet',
        action='store_true',
        help='Testnet modunda çalıştır'
    )
    
    args = parser.parse_args()
    
    # Bot'u başlat
    bot = TradingBot(config_path=args.config)
    bot.start()


if __name__ == '__main__':
    main()
