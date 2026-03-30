#!/usr/bin/env python3
"""
语音识别模块
"""
import serial
import time
from config.settings import SERIAL_PORT, SERIAL_BAUDRATE
from config.voice_commands import COMMAND_NAMES, DEVICE_NAMES
from utils.logger import logger

class VoiceModule:
    """语音识别模块"""
    
    def __init__(self, callback):
        self.callback = callback
        self.ser = None
        self.buffer = b''
        self.running = False
        
    def start(self):
        """启动语音模块"""
        try:
            self.ser = serial.Serial(
                port=SERIAL_PORT,
                baudrate=SERIAL_BAUDRATE,
                timeout=0.1
            )
            self.running = True
            logger.info("语音模块启动成功")
            return True
        except Exception as e:
            logger.error(f"语音模块启动失败: {e}")
            return False
    
    def stop(self):
        """停止语音模块"""
        self.running = False
        if self.ser:
            self.ser.close()
        logger.info("语音模块已停止")
    
    def process_data(self):
        """处理串口数据"""
        if not self.running or not self.ser:
            return
        
        if self.ser.in_waiting > 0:
            data = self.ser.read(self.ser.in_waiting)
            self.buffer += data
            
            while True:
                start = self.buffer.find(b'\xA5\xFC')
                if start == -1:
                    self.buffer = b''
                    break
                
                end = self.buffer.find(b'\xFB', start)
                if end == -1:
                    break
                
                frame = self.buffer[start:end+1]
                self.buffer = self.buffer[end+1:]
                
                operation, device_id = self.parse_frame(frame)
                if operation is not None:
                    self.handle_command(operation, device_id)
    
    def parse_frame(self, frame):
        """解析语音命令帧"""
        if len(frame) < 16:
            return None, None
        if frame[0] != 0xA5 or frame[1] != 0xFC or frame[-1] != 0xFB:
            return None, None
        operation = frame[7]
        device_id = bytes(frame[8:15])
        return operation, device_id
    
    def handle_command(self, operation, device_id):
        """处理语音命令"""
        if len(device_id) < 5:
            return
        
        prefix = device_id[:5]
        op_name = COMMAND_NAMES.get(operation, f"未知(0x{operation:02X})")
        device_name = DEVICE_NAMES.get(prefix, "未知设备")
        
        logger.info(f"收到语音命令: {op_name} - {device_name}")
        
        # 回调主程序处理
        if self.callback:
            self.callback(operation, prefix)