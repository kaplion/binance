"""
API Modülü Testleri

Bu modül, Binance API client ve WebSocket handler testlerini içerir.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.api.binance_client import BinanceClient
from src.api.websocket_handler import WebSocketHandler


class TestBinanceClient:
    """BinanceClient sınıfı testleri."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock Binance client oluştur."""
        with patch('src.api.binance_client.Client') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            
            client = BinanceClient(
                api_key='test_key',
                api_secret='test_secret',
                testnet=True
            )
            client.client = mock_instance
            yield client
    
    def test_init_testnet(self, mock_client):
        """Testnet modunda başlatma testi."""
        assert mock_client.testnet is True
        assert mock_client.futures_type == 'usdt_m'
    
    def test_get_usdt_balance(self, mock_client):
        """USDT bakiyesi alma testi."""
        mock_client.client.futures_account_balance.return_value = [
            {'asset': 'USDT', 'balance': '1000.0'},
            {'asset': 'BTC', 'balance': '0.1'}
        ]
        
        balance = mock_client.get_usdt_balance()
        
        assert balance == 1000.0
    
    def test_get_usdt_balance_not_found(self, mock_client):
        """USDT bakiyesi bulunamadığında testi."""
        mock_client.client.futures_account_balance.return_value = [
            {'asset': 'BTC', 'balance': '0.1'}
        ]
        
        balance = mock_client.get_usdt_balance()
        
        assert balance == 0.0
    
    def test_get_positions(self, mock_client):
        """Pozisyon bilgisi alma testi."""
        mock_client.client.futures_position_information.return_value = [
            {'symbol': 'BTCUSDT', 'positionAmt': '0.1', 'entryPrice': '50000'},
            {'symbol': 'ETHUSDT', 'positionAmt': '0', 'entryPrice': '0'}
        ]
        
        positions = mock_client.get_positions()
        
        assert len(positions) == 1
        assert positions[0]['symbol'] == 'BTCUSDT'
    
    def test_get_positions_with_symbol_filter(self, mock_client):
        """Sembol filtreli pozisyon testi."""
        mock_client.client.futures_position_information.return_value = [
            {'symbol': 'BTCUSDT', 'positionAmt': '0.1'},
            {'symbol': 'ETHUSDT', 'positionAmt': '0.5'}
        ]
        
        positions = mock_client.get_positions(symbol='BTCUSDT')
        
        assert len(positions) == 1
        assert positions[0]['symbol'] == 'BTCUSDT'
    
    def test_set_leverage(self, mock_client):
        """Leverage ayarlama testi."""
        mock_client.client.futures_change_leverage.return_value = {
            'leverage': 10,
            'symbol': 'BTCUSDT'
        }
        
        result = mock_client.set_leverage('BTCUSDT', 10)
        
        mock_client.client.futures_change_leverage.assert_called_once_with(
            symbol='BTCUSDT',
            leverage=10
        )
        assert result['leverage'] == 10
    
    def test_market_order(self, mock_client):
        """Market order testi."""
        mock_client.client.futures_create_order.return_value = {
            'orderId': 12345,
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'type': 'MARKET',
            'status': 'FILLED'
        }
        
        result = mock_client.market_order(
            symbol='BTCUSDT',
            side='BUY',
            quantity=0.001
        )
        
        assert result['orderId'] == 12345
        assert result['status'] == 'FILLED'
    
    def test_limit_order(self, mock_client):
        """Limit order testi."""
        mock_client.client.futures_create_order.return_value = {
            'orderId': 12346,
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'type': 'LIMIT',
            'status': 'NEW'
        }
        
        result = mock_client.limit_order(
            symbol='BTCUSDT',
            side='BUY',
            quantity=0.001,
            price=50000.0
        )
        
        assert result['orderId'] == 12346
        assert result['status'] == 'NEW'
    
    def test_cancel_order(self, mock_client):
        """Order iptal testi."""
        mock_client.client.futures_cancel_order.return_value = {
            'orderId': 12345,
            'status': 'CANCELED'
        }
        
        result = mock_client.cancel_order('BTCUSDT', 12345)
        
        assert result['status'] == 'CANCELED'
    
    def test_get_klines(self, mock_client):
        """Kline verisi alma testi."""
        mock_klines = [
            [1609459200000, '50000', '51000', '49000', '50500', '100'],
            [1609459260000, '50500', '51500', '50000', '51000', '150']
        ]
        mock_client.client.futures_klines.return_value = mock_klines
        
        result = mock_client.get_klines('BTCUSDT', '1m', limit=2)
        
        assert len(result) == 2


class TestWebSocketHandler:
    """WebSocketHandler sınıfı testleri."""
    
    @pytest.fixture
    def ws_handler(self):
        """WebSocket handler oluştur."""
        return WebSocketHandler(testnet=True)
    
    def test_init_testnet(self, ws_handler):
        """Testnet URL testi."""
        assert ws_handler.testnet is True
        assert 'testnet' in ws_handler.base_url.lower() or 'binancefuture' in ws_handler.base_url.lower()
    
    def test_get_stream_url(self, ws_handler):
        """Stream URL oluşturma testi."""
        url = ws_handler._get_stream_url('btcusdt@kline_1m')
        
        assert 'btcusdt@kline_1m' in url
    
    def test_handle_kline(self, ws_handler):
        """Kline verisi işleme testi."""
        kline_data = {
            'k': {
                's': 'BTCUSDT',
                'i': '1m',
                't': 1609459200000,
                'T': 1609459259999,
                'o': '50000',
                'h': '51000',
                'l': '49000',
                'c': '50500',
                'v': '100',
                'q': '5000000',
                'n': 1000,
                'x': True
            }
        }
        
        ws_handler._handle_kline(kline_data)
        
        key = 'BTCUSDT_1m'
        assert key in ws_handler._klines
        assert ws_handler._klines[key]['close'] == 50500.0
        assert ws_handler._klines[key]['is_closed'] is True
    
    def test_handle_ticker(self, ws_handler):
        """Ticker verisi işleme testi."""
        ticker_data = {
            's': 'BTCUSDT',
            'p': '500',
            'P': '1.0',
            'c': '50500',
            'v': '10000'
        }
        
        ws_handler._handle_ticker(ticker_data)
        
        assert 'BTCUSDT' in ws_handler._tickers
        assert ws_handler._tickers['BTCUSDT']['last_price'] == 50500.0
    
    def test_get_kline(self, ws_handler):
        """Kline verisi alma testi."""
        ws_handler._klines['BTCUSDT_1m'] = {
            'symbol': 'BTCUSDT',
            'interval': '1m',
            'close': 50500.0
        }
        
        result = ws_handler.get_kline('BTCUSDT', '1m')
        
        assert result is not None
        assert result['close'] == 50500.0
    
    def test_get_kline_not_found(self, ws_handler):
        """Kline verisi bulunamadığında testi."""
        result = ws_handler.get_kline('ETHUSDT', '1m')
        
        assert result is None
    
    def test_is_connected_false(self, ws_handler):
        """Bağlantı durumu kontrolü (bağlı değil) testi."""
        result = ws_handler.is_connected('nonexistent_stream')
        
        assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
