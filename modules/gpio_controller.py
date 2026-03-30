#!/usr/bin/env python3
"""
GPIO控制模块
"""
import RPi.GPIO as GPIO
import time
from config.settings import (
    RED_LED_PIN, GREEN_LED_PIN, BUZZER_PIN, MOTOR_PIN,
    KEYPAD_ROWS, KEYPAD_COLS
)
from utils.logger import logger

class GPIOController:
    """GPIO控制器"""
    
    def __init__(self):
        self.devices = {}
        self.init_gpio()
    
    def init_gpio(self):
        """初始化GPIO"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # 初始化LED
        self._setup_output(RED_LED_PIN, "红灯")
        self._setup_output(GREEN_LED_PIN, "绿灯")
        self._setup_output(BUZZER_PIN, "蜂鸣器", initial=GPIO.HIGH)
        self._setup_output(MOTOR_PIN, "电机")
        
        # 初始化矩阵键盘
        for pin in KEYPAD_ROWS:
            self._setup_output(pin, f"键盘行{pin}")
            GPIO.output(pin, GPIO.HIGH)
        for pin in KEYPAD_COLS:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        logger.info("GPIO初始化完成")
    
    def _setup_output(self, pin, name, initial=GPIO.LOW):
        """设置输出引脚"""
        GPIO.setup(pin, GPIO.OUT, initial=initial)
        self.devices[name] = {'pin': pin, 'state': initial == GPIO.HIGH}
    
    def set_device(self, name, state):
        """设置设备状态"""
        if name in self.devices:
            pin = self.devices[name]['pin']
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
            self.devices[name]['state'] = state
            logger.info(f"{name} {'开启' if state else '关闭'}")
    
    def get_device_state(self, name):
        """获取设备状态"""
        return self.devices.get(name, {}).get('state', False)
    
    def beep(self, times=1, duration=0.05):
        """蜂鸣提示"""
        for _ in range(times):
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(duration)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(duration)
    
    def cleanup(self):
        """清理GPIO"""
        GPIO.cleanup()
        logger.info("GPIO已清理")