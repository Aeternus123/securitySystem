#!/usr/bin/env python3
"""
日志工具模块
"""
import os
import sys
import logging
from datetime import datetime
from config.settings import LOG_DIR, LOG_LEVEL, LOG_FILE

class SystemLogger:
    """系统日志记录器"""
    
    def __init__(self):
        # 创建日志目录
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 配置日志
        log_path = os.path.join(LOG_DIR, LOG_FILE)
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # 检查是否在后台运行（通过systemd或start.sh启动）
        # 如果是，只输出到文件，不输出到终端
        is_daemon = os.environ.get('SECURITY_SYSTEM_DAEMON', 'false').lower() == 'true'
        
        handlers = [logging.FileHandler(log_path)]
        
        # 如果不是守护进程模式，同时输出到终端
        if not is_daemon:
            handlers.append(logging.StreamHandler(sys.stdout))
        
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format=log_format,
            datefmt=date_format,
            handlers=handlers
        )
        self.logger = logging.getLogger('SecuritySystem')
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)

# 单例模式
logger = SystemLogger()