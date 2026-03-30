#!/usr/bin/env python3
"""
超声波测距模块
"""
import RPi.GPIO as GPIO
import time
from config.settings import ULTRASONIC_TRIG_PIN, ULTRASONIC_ECHO_PIN
from utils.logger import logger

class UltrasonicSensor:
    """超声波传感器"""
    
    def __init__(self):
        self.trig_pin = ULTRASONIC_TRIG_PIN
        self.echo_pin = ULTRASONIC_ECHO_PIN
        
        GPIO.setup(self.trig_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.echo_pin, GPIO.IN)
        
        logger.info("超声波模块初始化完成")
    
    def measure_distance(self):
        """测量距离（单位：厘米）"""
        # 发送触发信号
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)  # 10微秒
        GPIO.output(self.trig_pin, False)
        
        # 记录发送时间
        pulse_start = time.time()
        pulse_end = time.time()
        
        # 等待回波开始
        timeout_start = time.time()
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if time.time() - timeout_start > 0.1:  # 超时100ms
                return None
        
        # 等待回波结束
        timeout_start = time.time()
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if time.time() - timeout_start > 0.1:  # 超时100ms
                return None
        
        # 计算距离
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # 声速34300/2 = 17150
        distance = round(distance, 2)
        
        # 检查距离是否在有效范围内
        if 2 < distance < 400:
            return distance
        else:
            return None