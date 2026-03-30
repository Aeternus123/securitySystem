#!/usr/bin/env python3
"""
报警系统模块 - 支持多种报警源
"""
import time
import threading
from config.settings import STRANGER_ALARM_DURATION
from utils.logger import logger

class AlarmSystem:
    """报警系统"""
    
    def __init__(self, gpio_controller, face_detector=None):
        self.gpio = gpio_controller
        self.face_detector = face_detector
        self.alarm_active = False
        self.alarm_thread = None
        self.alarm_source = None
        self.alarm_start_time = 0
        self.saved_light_states = {}
        
        # 不同报警源的持续时间
        self.alarm_durations = {
            'ultrasonic': None,      # 超声波报警，由距离恢复自动停止
            'face_stranger': STRANGER_ALARM_DURATION,  # 陌生人报警，10秒后自动停止
            'manual': None,           # 手动报警，需手动停止
            'smoke': 15,              # 烟雾报警，15秒后自动停止
            'sensor': 15              # 温湿度传感器报警，15秒后自动停止
        }
        
        logger.info("报警系统初始化完成")
    
    def save_light_states(self):
        """保存灯光状态"""
        self.saved_light_states = {
            '红灯': self.gpio.get_device_state('红灯'),
            '绿灯': self.gpio.get_device_state('绿灯')
        }
        logger.debug(f"保存灯光状态: 红灯={self.saved_light_states['红灯']}, 绿灯={self.saved_light_states['绿灯']}")
    
    def restore_light_states(self):
        """恢复灯光状态"""
        for name, state in self.saved_light_states.items():
            self.gpio.set_device(name, state)
        self.saved_light_states.clear()
        logger.debug("灯光状态已恢复")
    
    def trigger(self, source):
        """触发报警"""
        if self.alarm_active:
            logger.debug(f"报警已在运行中，忽略新的触发: {source}")
            return
        
        # 获取报警持续时间
        duration = self.alarm_durations.get(source, None)
        if duration:
            logger.warning(f"报警触发 - 来源: {source}，将持续 {duration} 秒")
        else:
            logger.warning(f"报警触发 - 来源: {source}")
        
        self.save_light_states()
        self.gpio.set_device('红灯', False)
        self.gpio.set_device('绿灯', False)
        
        self.alarm_active = True
        self.alarm_source = source
        self.alarm_start_time = time.time()
        self.alarm_thread = threading.Thread(target=self._alarm_loop)
        self.alarm_thread.daemon = True
        self.alarm_thread.start()
    
    def stop(self):
        """停止报警"""
        if not self.alarm_active:
            return
        
        logger.info("停止报警")
        self.alarm_active = False
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=2)
        
        self.gpio.set_device('红灯', False)
        self.gpio.set_device('绿灯', False)
        self.gpio.beep(0)  # 停止蜂鸣
        
        self.restore_light_states()
        self.alarm_source = None
        logger.info("报警已停止")
    
    def _alarm_loop(self):
        """报警循环"""
        logger.debug(f"报警循环启动，来源: {self.alarm_source}")
        
        while self.alarm_active:
            # 检查是否超时（针对有持续时间的报警）
            duration = self.alarm_durations.get(self.alarm_source, None)
            if duration is not None:
                elapsed = time.time() - self.alarm_start_time
                remaining = duration - elapsed
                
                if remaining <= 0:
                    logger.info(f"{self.alarm_source}报警持续时间结束 ({duration}秒)")
                    self.alarm_active = False
                    break
                
                # 每秒显示一次剩余时间（可选）
                if int(elapsed) % 2 == 0 and int(elapsed) > 0:
                    logger.debug(f"{self.alarm_source}报警剩余: {remaining:.0f}秒")
            
            # 红绿交替
            for _ in range(10):
                if not self.alarm_active: break
                self.gpio.set_device('红灯', True)
                self.gpio.set_device('绿灯', False)
                self.gpio.beep(1)
                time.sleep(0.1)
                
                if not self.alarm_active: break
                self.gpio.set_device('红灯', False)
                self.gpio.set_device('绿灯', True)
                self.gpio.beep(1)
                time.sleep(0.1)
            
            # 同时闪烁
            for _ in range(5):
                if not self.alarm_active: break
                self.gpio.set_device('红灯', True)
                self.gpio.set_device('绿灯', True)
                self.gpio.beep(1)
                time.sleep(0.1)
                
                if not self.alarm_active: break
                self.gpio.set_device('红灯', False)
                self.gpio.set_device('绿灯', False)
                time.sleep(0.1)
        
        # 报警结束后的清理
        self.gpio.set_device('红灯', False)
        self.gpio.set_device('绿灯', False)
        self.gpio.beep(0)
        
        # 恢复灯光状态
        self.restore_light_states()
        
        # 重置报警状态
        self.alarm_active = False
        self.alarm_source = None
        logger.info("报警线程结束")
    
    def is_active(self):
        """检查报警是否激活"""
        return self.alarm_active
    
    def get_source(self):
        """获取报警源"""
        return self.alarm_source