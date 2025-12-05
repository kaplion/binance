# Binance Futures AI Trading Bot

## 🎯 Amaç
Binance Futures platformu için kapsamlı, AI destekli bir otomatik trading botu.

## 📁 Proje Yapısı

```
binance/
├── README.md                    # Proje dokümantasyonu (Türkçe)
├── requirements.txt             # Python bağımlılıkları
├── config/
│   ├── config.example.yaml      # Örnek konfigürasyon dosyası
│   └── settings.py              # Uygulama ayarları
├── src/
│   ├── __init__.py
│   ├── main.py                  # Ana uygulama giriş noktası
│   ├── api/
│   │   ├── __init__.py
│   │   ├── binance_client.py    # Binance Futures API client
│   │   └── websocket_handler.py # WebSocket bağlantı yönetimi
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── order_manager.py     # Order yönetimi (market, limit, stop)
│   │   ├── position_manager.py  # Pozisyon yönetimi
│   │   └── risk_manager.py      # Risk yönetimi (stop-loss, take-profit)
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base_strategy.py     # Temel strateji abstract sınıfı
│   │   ├── momentum_strategy.py # Momentum trading stratejisi
│   │   └── ml_strategy.py       # ML tabanlı strateji
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── data_processor.py    # Veri işleme ve feature engineering
│   │   ├── model_trainer.py     # Model eğitimi
│   │   ├── predictor.py         # Fiyat/trend tahmini
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── lstm_model.py    # LSTM modeli
│   │       └── transformer_model.py # Transformer modeli
│   ├── indicators/
│   │   ├── __init__.py
│   │   └── technical.py         # Teknik indikatörler (RSI, MACD, BB, vb.)
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Loglama sistemi
│       └── helpers.py           # Yardımcı fonksiyonlar
├── data/
│   └── .gitkeep                 # Veri dosyaları için klasör
├── models/
│   └── .gitkeep                 # Eğitilmiş modeller için klasör
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_strategies.py
│   └── test_ai.py
├── .env.example                 # Örnek environment variables
├── .gitignore                   # Git ignore dosyası
└── docker-compose.yml           # Docker compose (opsiyonel)
```

## 🔧 Temel Özellikler

### 1. Binance Futures API Entegrasyonu
- Testnet ve mainnet desteği
- USDT-M ve COIN-M futures desteği
- Leverage ayarlama
- Order türleri: Market, Limit, Stop-Market, Stop-Limit, Take-Profit
- Pozisyon bilgisi sorgulama
- Hesap bakiyesi kontrolü

### 2. WebSocket Entegrasyonu
- Gerçek zamanlı fiyat akışı (kline/candlestick)
- Order book güncellemeleri
- User data stream (pozisyon/order güncellemeleri)
- Otomatik reconnect mekanizması

### 3. Risk Yönetimi
- Maksimum pozisyon boyutu limiti
- Günlük/haftalık kayıp limiti
- Dinamik stop-loss ve take-profit
- Risk/Reward oranı kontrolü
- Margin call koruması

### 4. AI/ML Modülleri
- **Data Processor**: OHLCV veri işleme, normalizasyon, feature engineering
- **LSTM Model**: Zaman serisi tahmini için LSTM ağı
- **Transformer Model**: Attention mekanizmalı fiyat tahmini
- **Predictor**: Model inference ve sinyal üretimi
- **Model Trainer**: Otomatik model eğitimi ve güncelleme

### 5. Teknik İndikatörler
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- EMA/SMA
- ATR (Average True Range)
- Volume analizi

### 6. Trading Stratejileri
- **Base Strategy**: Tüm stratejiler için abstract base class
- **Momentum Strategy**: Momentum ve trend takip stratejisi
- **ML Strategy**: AI model tahminlerine dayalı strateji

## 🚀 Kurulum

### Gereksinimler
- Python 3.9+
- pip

### Kurulum Adımları

1. **Repository'yi klonlayın:**
   ```bash
   git clone https://github.com/kaplion/binance.git
   cd binance
   ```

2. **Virtual environment oluşturun (önerilir):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # veya
   venv\Scripts\activate  # Windows
   ```

3. **Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment variables ayarlayın:**
   ```bash
   cp .env.example .env
   # .env dosyasını düzenleyerek API bilgilerinizi girin
   ```

5. **Konfigürasyon dosyasını hazırlayın:**
   ```bash
   cp config/config.example.yaml config/config.yaml
   # config/config.yaml dosyasını düzenleyerek ayarlarınızı yapın
   ```

6. **Botu başlatın:**
   ```bash
   python src/main.py
   ```

## ⚙️ Konfigürasyon

### config/config.yaml Örneği
```yaml
binance:
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_API_SECRET"
  testnet: true

trading:
  symbol: "BTCUSDT"
  leverage: 10
  position_size_pct: 5  # Bakiyenin %5'i
  max_positions: 3

risk:
  stop_loss_pct: 2
  take_profit_pct: 4
  max_daily_loss_pct: 5
  risk_reward_ratio: 2

ai:
  model_type: "lstm"  # lstm veya transformer
  lookback_period: 60
  prediction_horizon: 5
  retrain_interval: 24  # saat

strategy:
  type: "ml"  # momentum veya ml
  timeframe: "15m"
```

### Environment Variables (.env)
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true
```

## 📊 Kullanım Örnekleri

### Temel Kullanım
```python
from src.main import TradingBot

# Bot'u başlat
bot = TradingBot(config_path="config/config.yaml")
bot.start()
```

### Sadece Backtest Modu
```python
from src.strategies.ml_strategy import MLStrategy
from src.ai.predictor import Predictor

# ML stratejisi ile backtest
strategy = MLStrategy(config)
results = strategy.backtest(historical_data)
```

## 🧪 Testler

```bash
# Tüm testleri çalıştır
pytest tests/

# Belirli bir test dosyasını çalıştır
pytest tests/test_api.py -v

# Coverage raporu ile çalıştır
pytest --cov=src tests/
```

## 🐳 Docker Kullanımı

```bash
# Docker ile başlat
docker-compose up -d

# Logları görüntüle
docker-compose logs -f
```

## ⚠️ Uyarılar ve Önemli Notlar

1. **İlk kurulumda testnet modunda başlayın** - Gerçek para ile işlem yapmadan önce sistemi testnet'te test edin.

2. **AI modeli eğitilmeden önce yeterli geçmiş veri toplayın** - En az 1000 mum verisi önerilir.

3. **Risk yönetimi parametrelerini dikkatli ayarlayın** - Stop-loss ve take-profit seviyelerini piyasa koşullarına göre ayarlayın.

4. **API anahtarlarınızı güvende tutun** - `.env` dosyasını asla paylaşmayın ve git'e commit etmeyin.

5. **Trading botu yatırım tavsiyesi değildir** - Bu yazılım eğitim amaçlıdır, finansal kayıplardan kullanıcı sorumludur.

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request açın

## 📧 İletişim

Sorularınız için issue açabilirsiniz.