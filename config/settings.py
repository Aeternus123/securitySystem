#!/usr/bin/env python3
"""
系统配置文件
"""

# ==================== GPIO引脚定义 ====================
# LED灯
RED_LED_PIN = 17      # 红灯 - 主卧灯
GREEN_LED_PIN = 27    # 绿灯 - 客房灯

# 蜂鸣器
BUZZER_PIN = 22

# 电机
MOTOR_PIN = 23

# 超声波模块
ULTRASONIC_TRIG_PIN = 24
ULTRASONIC_ECHO_PIN = 25

# 矩阵键盘
KEYPAD_ROWS = [5, 6, 13, 19]    # 行引脚
KEYPAD_COLS = [16, 20, 21, 12]  # 列引脚

# ==================== 串口设置 ====================
SERIAL_PORT = '/dev/serial0'
SERIAL_BAUDRATE = 9600

# ==================== 超声波阈值 ====================
ULTRASONIC_DISTANCE_THRESHOLD = 20  # 距离阈值（厘米）
ULTRASONIC_CHECK_INTERVAL = 0.5      # 检测间隔（秒）

# ==================== 人脸识别设置 ====================
FACE_DETECTION_MODEL = 'models/yolov8n-face.onnx'
FACE_DATABASE_PATH = 'database/face_db.pkl'
FACE_IMAGES_DIR = 'images'
FACE_STRANGER_DIR = 'images/stranger'
FACE_CONFIDENCE_THRESHOLD = 0.5
FACE_SIMILARITY_THRESHOLD = 0.65     # 相似度阈值
FACE_RECOGNITION_TIMEOUT = 30        # 人脸识别激活超时（秒）

# ==================== 传感器引脚定义 ====================
# DHT11 温湿度传感器
DHT11_PIN = 26  # GPIO26

# 烟雾传感器（只使用数字输出）
SMOKE_DIGITAL_PIN = 19  # 数字输出 GPIO19

# 传感器阈值设置
SMOKE_DIGITAL_THRESHOLD = 1  # 数字信号触发报警的电平（根据传感器类型调整）
# 注意：有些传感器报警时输出高电平(1)，有些输出低电平(0)
# 如果你的传感器报警时输出低电平，请改为 SMOKE_DIGITAL_THRESHOLD = 0

TEMPERATURE_HIGH_THRESHOLD = 40  # 高温报警阈值（摄氏度）
HUMIDITY_HIGH_THRESHOLD = 80  # 高湿报警阈值（百分比）

# 传感器检测间隔
SENSOR_CHECK_INTERVAL = 2  # 2秒检测一次

# ==================== 系统状态设置 ====================

STRANGER_ALARM_DURATION = 3          # 陌生人报警持续时间（秒）

# ==================== 日志设置 ====================
LOG_DIR = 'logs'
LOG_LEVEL = 'INFO'
LOG_FILE = 'system.log'