"""
VOC气体分类 - 日志模块

提供统一的日志记录功能
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import time

class Logger:
    """日志记录器类"""
    
    def __init__(self, name, log_dir='log', log_level=logging.INFO):
        """
        初始化日志记录器
        
        参数:
            name: 日志记录器名称
            log_dir: 日志文件目录
            log_level: 日志级别
        """
        self.name = name
        self.log_dir = log_dir
        self.log_level = log_level
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # 清除已有的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建日志格式
        formatter = logging.Formatter(
            '[%(asctime)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 创建文件处理器
        log_file = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d')}.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message):
        """
        记录信息日志
        
        参数:
            message: 日志消息
        """
        self.logger.info(message)
    
    def warning(self, message):
        """
        记录警告日志
        
        参数:
            message: 日志消息
        """
        self.logger.warning(message)
    
    def error(self, message):
        """
        记录错误日志
        
        参数:
            message: 日志消息
        """
        self.logger.error(message)
    
    def debug(self, message):
        """
        记录调试日志
        
        参数:
            message: 日志消息
        """
        self.logger.debug(message)
    
    def critical(self, message):
        """
        记录严重错误日志
        
        参数:
            message: 日志消息
        """
        self.logger.critical(message) 