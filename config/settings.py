"""
Uygulama Ayarları Modülü

Bu modül, YAML konfigürasyon dosyasını ve environment variable'ları
yükleyerek uygulama genelinde kullanılacak ayarları sağlar.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class Settings:
    """
    Uygulama ayarlarını yöneten sınıf.
    
    YAML konfigürasyon dosyası ve environment variable'ları birleştirerek
    tek bir ayar nesnesi oluşturur.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Settings sınıfını başlat.
        
        Args:
            config_path: Konfigürasyon dosyasının yolu. None ise varsayılan yol kullanılır.
        """
        # .env dosyasını yükle
        load_dotenv()
        
        # Varsayılan konfigürasyon yolu
        if config_path is None:
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        
        # Konfigürasyon dosyasını yükle
        self._load_config()
        
        # Environment variable'lardan API bilgilerini al
        self._load_env_overrides()
    
    def _load_config(self) -> None:
        """YAML konfigürasyon dosyasını yükle."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Örnek konfigürasyon dosyasını dene
            example_path = self.config_path.with_suffix('.example.yaml')
            if example_path.exists():
                with open(example_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
            else:
                self._config = self._get_default_config()
    
    def _load_env_overrides(self) -> None:
        """Environment variable'lardan API bilgilerini yükle ve override et."""
        # Binance API bilgileri
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        if 'binance' not in self._config:
            self._config['binance'] = {}
        
        if api_key:
            self._config['binance']['api_key'] = api_key
        if api_secret:
            self._config['binance']['api_secret'] = api_secret
        
        self._config['binance']['testnet'] = testnet
        
        # Log seviyesi
        log_level = os.getenv('LOG_LEVEL')
        if log_level:
            if 'logging' not in self._config:
                self._config['logging'] = {}
            self._config['logging']['level'] = log_level
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Varsayılan konfigürasyonu döndür."""
        return {
            'binance': {
                'api_key': '',
                'api_secret': '',
                'testnet': True,
                'futures_type': 'usdt_m'
            },
            'trading': {
                'symbol': 'BTCUSDT',
                'leverage': 10,
                'position_size_pct': 5,
                'max_positions': 3,
                'margin_mode': 'isolated'
            },
            'risk': {
                'stop_loss_pct': 2,
                'take_profit_pct': 4,
                'max_daily_loss_pct': 5,
                'max_weekly_loss_pct': 15,
                'risk_reward_ratio': 2,
                'trailing_stop_enabled': True,
                'trailing_stop_activation_pct': 1.5,
                'trailing_stop_callback_pct': 0.5
            },
            'ai': {
                'model_type': 'lstm',
                'lookback_period': 60,
                'prediction_horizon': 5,
                'retrain_interval': 24,
                'min_training_data': 1000,
                'confidence_threshold': 0.6,
                'features': ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_9', 'ema_21', 'atr']
            },
            'strategy': {
                'type': 'ml',
                'timeframe': '15m',
                'momentum': {
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'macd_signal_period': 9,
                    'volume_multiplier': 1.5
                },
                'ml': {
                    'min_confidence': 0.65,
                    'position_sizing_method': 'kelly'
                }
            },
            'websocket': {
                'auto_reconnect': True,
                'reconnect_attempts': 5,
                'reconnect_delay': 5,
                'ping_interval': 30
            },
            'logging': {
                'level': 'INFO',
                'file_path': 'logs/trading.log',
                'max_size': 10,
                'backup_count': 5,
                'console_output': True
            },
            'notifications': {
                'enabled': False
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Belirtilen anahtara göre ayar değerini döndür.
        
        Args:
            key: Nokta ile ayrılmış anahtar (örn: 'binance.api_key')
            default: Anahtar bulunamazsa döndürülecek varsayılan değer
            
        Returns:
            Ayar değeri veya varsayılan değer
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Belirtilen anahtara değer ata.
        
        Args:
            key: Nokta ile ayrılmış anahtar (örn: 'binance.api_key')
            value: Atanacak değer
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def binance(self) -> Dict[str, Any]:
        """Binance ayarlarını döndür."""
        return self._config.get('binance', {})
    
    @property
    def trading(self) -> Dict[str, Any]:
        """Trading ayarlarını döndür."""
        return self._config.get('trading', {})
    
    @property
    def risk(self) -> Dict[str, Any]:
        """Risk yönetimi ayarlarını döndür."""
        return self._config.get('risk', {})
    
    @property
    def ai(self) -> Dict[str, Any]:
        """AI/ML ayarlarını döndür."""
        return self._config.get('ai', {})
    
    @property
    def strategy(self) -> Dict[str, Any]:
        """Strateji ayarlarını döndür."""
        return self._config.get('strategy', {})
    
    @property
    def websocket(self) -> Dict[str, Any]:
        """WebSocket ayarlarını döndür."""
        return self._config.get('websocket', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Loglama ayarlarını döndür."""
        return self._config.get('logging', {})
    
    @property
    def notifications(self) -> Dict[str, Any]:
        """Bildirim ayarlarını döndür."""
        return self._config.get('notifications', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Tüm ayarları dictionary olarak döndür."""
        return self._config.copy()
    
    def is_testnet(self) -> bool:
        """Testnet modunda mı kontrol et."""
        return self.get('binance.testnet', True)
    
    def validate(self) -> bool:
        """
        Konfigürasyonun geçerliliğini kontrol et.
        
        Returns:
            bool: Konfigürasyon geçerli ise True
            
        Raises:
            ValueError: Konfigürasyon geçersiz ise
        """
        # API anahtarları kontrol
        api_key = self.get('binance.api_key', '')
        api_secret = self.get('binance.api_secret', '')
        
        if not api_key or api_key == 'YOUR_API_KEY':
            raise ValueError("Geçerli bir Binance API anahtarı gerekli")
        
        if not api_secret or api_secret == 'YOUR_API_SECRET':
            raise ValueError("Geçerli bir Binance API secret gerekli")
        
        # Leverage kontrol
        leverage = self.get('trading.leverage', 10)
        if not 1 <= leverage <= 125:
            raise ValueError("Leverage 1-125 arasında olmalı")
        
        # Risk parametreleri kontrol
        stop_loss = self.get('risk.stop_loss_pct', 2)
        take_profit = self.get('risk.take_profit_pct', 4)
        
        if stop_loss <= 0 or stop_loss > 100:
            raise ValueError("Stop-loss yüzdesi 0-100 arasında olmalı")
        
        if take_profit <= 0 or take_profit > 100:
            raise ValueError("Take-profit yüzdesi 0-100 arasında olmalı")
        
        return True


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Global settings instance'ını döndür.
    
    Args:
        config_path: Konfigürasyon dosyasının yolu
        
    Returns:
        Settings: Ayarlar nesnesi
    """
    global _settings
    if _settings is None or config_path is not None:
        _settings = Settings(config_path)
    return _settings
