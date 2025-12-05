"""
Main Modülü Testleri

Bu modül, TradingBot sınıfının _on_kline_update ve _run_trading_loop
metodları için testlerini içerir.
"""

import asyncio
import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock


class TestTradingBotKlineUpdate:
    """TradingBot kline güncelleme testleri."""
    
    def test_on_kline_update_closed_candle_starts_thread(self):
        """Mum kapandığında thread başlatılıyor mu testi."""
        with patch('src.main.get_settings') as mock_settings, \
             patch('src.main.setup_logger'), \
             patch('src.main.get_logger') as mock_logger:
            
            # Mock settings
            mock_settings_obj = MagicMock()
            mock_settings_obj.logging_config = {
                'level': 'INFO',
                'file_path': None,
                'max_size': 10,
                'backup_count': 5,
                'console_output': True
            }
            mock_settings_obj.trading = {'symbol': 'BTCUSDT', 'leverage': 10}
            mock_settings_obj.strategy = {'timeframe': '15m'}
            mock_settings.return_value = mock_settings_obj
            mock_logger.return_value = MagicMock()
            
            from src.main import TradingBot
            
            bot = TradingBot.__new__(TradingBot)
            bot.logger = MagicMock()
            bot._run_trading_loop = MagicMock()
            
            # Mum kapandı verisi
            kline_data = {
                'k': {
                    'x': True,  # Mum kapandı
                    'c': '50000'
                }
            }
            
            # Thread başlatılıyor mu kontrol et
            with patch('threading.Thread') as mock_thread:
                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance
                
                bot._on_kline_update(kline_data)
                
                # Thread oluşturuldu mu?
                mock_thread.assert_called_once_with(target=bot._run_trading_loop)
                # Thread başlatıldı mı?
                mock_thread_instance.start.assert_called_once()
    
    def test_on_kline_update_open_candle_no_thread(self):
        """Mum kapanmadığında thread başlatılmıyor testi."""
        with patch('src.main.get_settings') as mock_settings, \
             patch('src.main.setup_logger'), \
             patch('src.main.get_logger') as mock_logger:
            
            mock_settings_obj = MagicMock()
            mock_settings_obj.logging_config = {
                'level': 'INFO',
                'file_path': None,
                'max_size': 10,
                'backup_count': 5,
                'console_output': True
            }
            mock_settings_obj.trading = {'symbol': 'BTCUSDT', 'leverage': 10}
            mock_settings_obj.strategy = {'timeframe': '15m'}
            mock_settings.return_value = mock_settings_obj
            mock_logger.return_value = MagicMock()
            
            from src.main import TradingBot
            
            bot = TradingBot.__new__(TradingBot)
            bot.logger = MagicMock()
            bot._run_trading_loop = MagicMock()
            
            # Mum kapanmadı verisi
            kline_data = {
                'k': {
                    'x': False,  # Mum kapanmadı
                    'c': '50000'
                }
            }
            
            # Thread başlatılmıyor mu kontrol et
            with patch('threading.Thread') as mock_thread:
                bot._on_kline_update(kline_data)
                
                # Thread oluşturulmadı mı?
                mock_thread.assert_not_called()
    
    def test_on_kline_update_handles_exception(self):
        """Hata durumunda doğru loglama yapılıyor mu testi."""
        with patch('src.main.get_settings') as mock_settings, \
             patch('src.main.setup_logger'), \
             patch('src.main.get_logger') as mock_logger:
            
            mock_settings_obj = MagicMock()
            mock_settings_obj.logging_config = {
                'level': 'INFO',
                'file_path': None,
                'max_size': 10,
                'backup_count': 5,
                'console_output': True
            }
            mock_settings_obj.trading = {'symbol': 'BTCUSDT', 'leverage': 10}
            mock_settings_obj.strategy = {'timeframe': '15m'}
            mock_settings.return_value = mock_settings_obj
            mock_logger.return_value = MagicMock()
            
            from src.main import TradingBot
            
            bot = TradingBot.__new__(TradingBot)
            bot.logger = MagicMock()
            
            # Geçersiz veri (KeyError oluşturacak)
            kline_data = None
            
            # Exception yakalanıyor mu?
            bot._on_kline_update(kline_data)
            
            # Error log çağrıldı mı?
            bot.logger.error.assert_called_once()
            assert "Kline güncelleme hatası" in str(bot.logger.error.call_args)


class TestRunTradingLoop:
    """_run_trading_loop metodu testleri."""
    
    def test_run_trading_loop_creates_new_event_loop(self):
        """Yeni event loop oluşturuluyor mu testi."""
        with patch('src.main.get_settings') as mock_settings, \
             patch('src.main.setup_logger'), \
             patch('src.main.get_logger') as mock_logger:
            
            mock_settings_obj = MagicMock()
            mock_settings_obj.logging_config = {
                'level': 'INFO',
                'file_path': None,
                'max_size': 10,
                'backup_count': 5,
                'console_output': True
            }
            mock_settings_obj.trading = {'symbol': 'BTCUSDT', 'leverage': 10}
            mock_settings_obj.strategy = {'timeframe': '15m'}
            mock_settings.return_value = mock_settings_obj
            mock_logger.return_value = MagicMock()
            
            from src.main import TradingBot
            
            bot = TradingBot.__new__(TradingBot)
            bot.logger = MagicMock()
            
            # Mock async trading loop
            async def mock_trading_loop():
                pass
            
            bot._trading_loop = mock_trading_loop
            
            # _run_trading_loop çalıştır
            with patch('asyncio.new_event_loop') as mock_new_loop, \
                 patch('asyncio.set_event_loop') as mock_set_loop:
                
                mock_loop = MagicMock()
                mock_new_loop.return_value = mock_loop
                
                bot._run_trading_loop()
                
                # Yeni event loop oluşturuldu mu?
                mock_new_loop.assert_called_once()
                # Event loop ayarlandı mı?
                mock_set_loop.assert_called_once_with(mock_loop)
                # Loop kapatıldı mı?
                mock_loop.close.assert_called_once()
    
    def test_run_trading_loop_handles_exception(self):
        """Hata durumunda doğru loglama yapılıyor mu testi."""
        with patch('src.main.get_settings') as mock_settings, \
             patch('src.main.setup_logger'), \
             patch('src.main.get_logger') as mock_logger:
            
            mock_settings_obj = MagicMock()
            mock_settings_obj.logging_config = {
                'level': 'INFO',
                'file_path': None,
                'max_size': 10,
                'backup_count': 5,
                'console_output': True
            }
            mock_settings_obj.trading = {'symbol': 'BTCUSDT', 'leverage': 10}
            mock_settings_obj.strategy = {'timeframe': '15m'}
            mock_settings.return_value = mock_settings_obj
            mock_logger.return_value = MagicMock()
            
            from src.main import TradingBot
            
            bot = TradingBot.__new__(TradingBot)
            bot.logger = MagicMock()
            
            # Mock async trading loop that raises exception
            async def mock_trading_loop_error():
                raise Exception("Test error")
            
            bot._trading_loop = mock_trading_loop_error
            
            # _run_trading_loop çalıştır
            bot._run_trading_loop()
            
            # Error log çağrıldı mı?
            bot.logger.error.assert_called_once()
            assert "Trading loop hatası" in str(bot.logger.error.call_args)


class TestIntegrationKlineCallback:
    """Kline callback entegrasyon testi."""
    
    def test_kline_callback_actually_runs_trading_loop(self):
        """Callback'in gerçekten trading loop'u çalıştırdığı testi."""
        with patch('src.main.get_settings') as mock_settings, \
             patch('src.main.setup_logger'), \
             patch('src.main.get_logger') as mock_logger:
            
            mock_settings_obj = MagicMock()
            mock_settings_obj.logging_config = {
                'level': 'INFO',
                'file_path': None,
                'max_size': 10,
                'backup_count': 5,
                'console_output': True
            }
            mock_settings_obj.trading = {'symbol': 'BTCUSDT', 'leverage': 10}
            mock_settings_obj.strategy = {'timeframe': '15m'}
            mock_settings.return_value = mock_settings_obj
            mock_logger.return_value = MagicMock()
            
            from src.main import TradingBot
            
            bot = TradingBot.__new__(TradingBot)
            bot.logger = MagicMock()
            
            # Trading loop çalıştı mı takibi
            trading_loop_called = threading.Event()
            
            async def mock_trading_loop():
                trading_loop_called.set()
            
            bot._trading_loop = mock_trading_loop
            
            # Kline update çağır
            kline_data = {
                'k': {
                    'x': True,  # Mum kapandı
                    'c': '50000'
                }
            }
            
            bot._on_kline_update(kline_data)
            
            # Thread'in çalışmasını bekle
            result = trading_loop_called.wait(timeout=2.0)
            
            # Trading loop çağrıldı mı?
            assert result is True, "Trading loop çağrılmadı"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
