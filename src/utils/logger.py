"""
Loglama Modülü

Bu modül, uygulama genelinde kullanılacak loglama sistemini sağlar.
Loguru kütüphanesi kullanılarak renkli konsol çıktısı ve dosya rotasyonu desteklenir.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size: int = 10,
    backup_count: int = 5,
    console_output: bool = True
) -> None:
    """
    Logger'ı yapılandır.
    
    Args:
        level: Log seviyesi (DEBUG, INFO, WARNING, ERROR)
        log_file: Log dosyasının yolu
        max_size: Maksimum log dosyası boyutu (MB)
        backup_count: Saklanacak yedek dosya sayısı
        console_output: Konsola log yazılsın mı
    """
    # Varsayılan handler'ları kaldır
    logger.remove()
    
    # Log formatı
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Konsol çıktısı
    if console_output:
        logger.add(
            sys.stdout,
            format=log_format,
            level=level,
            colorize=True
        )
    
    # Dosya çıktısı
    if log_file:
        # Log klasörünü oluştur
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=log_format.replace("<green>", "").replace("</green>", "")
                            .replace("<level>", "").replace("</level>", "")
                            .replace("<cyan>", "").replace("</cyan>", ""),
            level=level,
            rotation=f"{max_size} MB",
            retention=backup_count,
            compression="zip"
        )


def get_logger(name: str = "trading_bot"):
    """
    Belirtilen isimle bir logger instance'ı döndür.
    
    Args:
        name: Logger ismi
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Kısa yollar
def debug(message: str, *args, **kwargs):
    """Debug seviyesinde log yaz."""
    logger.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    """Info seviyesinde log yaz."""
    logger.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """Warning seviyesinde log yaz."""
    logger.warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    """Error seviyesinde log yaz."""
    logger.error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs):
    """Critical seviyesinde log yaz."""
    logger.critical(message, *args, **kwargs)


def exception(message: str, *args, **kwargs):
    """Exception log'u yaz (traceback dahil)."""
    logger.exception(message, *args, **kwargs)
