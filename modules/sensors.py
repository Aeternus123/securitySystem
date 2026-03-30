#!/usr/bin/env python3
"""
传感器模块 - DHT11温湿度传感器和烟雾报警器
已适配当前工作环境
"""
import time
import threading
import RPi.GPIO as GPIO
from config.settings import *
from utils.logger import logger

# ==================== DHT11库检测（只保留你确认可用的CircuitPython）====================
DHT11_AVAILABLE = False
try:
    import board
    import adafruit_dht
    DHT11_AVAILABLE = True
    logger.info("✅ DHT11库加载成功：adafruit-circuitpython-dht")
except ImportError:
    logger.warning("⚠️ 未找到adafruit-circuitpython-dht库，温湿度传感器功能将禁用")
    logger.warning("请安装: pip3 install adafruit-circuitpython-dht")

# ==================== 引脚映射表 ====================
# BCM GPIO编号到board引脚名的映射（常用引脚）
PIN_MAP = {
    2: board.D2, 3: board.D3, 4: board.D4,
    17: board.D17, 18: board.D18, 27: board.D27,
    22: board.D22, 23: board.D23, 24: board.D24,
    10: board.D10, 9: board.D9, 11: board.D11,
    5: board.D5, 6: board.D6, 13: board.D13,
    19: board.D19, 26: board.D26, 14: board.D14,
    15: board.D15
}

class DHT11Sensor:
    """DHT11温湿度传感器 - 已测试可工作版本"""
    
    def __init__(self):
        self.pin = DHT11_PIN
        self.temperature = 0
        self.humidity = 0
        self.last_read_time = 0
        self.read_success_count = 0
        self.read_fail_count = 0
        self.dht_device = None
        
        if not DHT11_AVAILABLE:
            logger.warning("⚠️ DHT11库未安装，温湿度传感器不可用")
            return
        
        # 初始化CircuitPython DHT11
        try:
            if self.pin in PIN_MAP:
                board_pin = PIN_MAP[self.pin]
                self.dht_device = adafruit_dht.DHT11(board_pin)
                logger.info(f"✅ DHT11传感器初始化成功 (GPIO{self.pin} → {board_pin})")
                
                # 首次读取尝试，验证传感器是否真的工作
                time.sleep(1)  # 给传感器一点稳定时间
                try:
                    temp = self.dht_device.temperature
                    hum = self.dht_device.humidity
                    if temp is not None and hum is not None:
                        logger.info(f"✅ 传感器首次读取成功: {temp}°C, {hum}%")
                    else:
                        logger.warning("⚠️ 传感器首次读取返回空值，可能需要检查接线")
                except Exception as e:
                    logger.warning(f"⚠️ 传感器首次读取失败: {e}")
            else:
                logger.error(f"❌ 不支持的GPIO引脚: {self.pin}，请使用以下引脚之一: {list(PIN_MAP.keys())}")
                self.dht_device = None
                
        except Exception as e:
            logger.error(f"❌ DHT11传感器初始化失败: {e}")
            self.dht_device = None
    
    def read(self):
        """读取温湿度数据 - 修复湿度显示问题"""
        if not DHT11_AVAILABLE or self.dht_device is None:
            return False, 0, 0
        
        try:
            # DHT11读取有时会失败，这是正常的
            temperature = self.dht_device.temperature
            humidity = self.dht_device.humidity
            
            # 调试输出
            print(f"DEBUG - 原始数据: temp={temperature}, hum={humidity}")
            
            # 温度有效范围：-20°C 到 60°C
            # 湿度有效范围：0% 到 100%
            temp_valid = temperature is not None and -20 <= temperature <= 60
            hum_valid = humidity is not None and 0 <= humidity <= 100
            
            # 至少温度有效就部分返回成功
            if temp_valid or hum_valid:
                updated = False
                
                if temp_valid:
                    self.temperature = temperature
                    updated = True
                    print(f"DEBUG - 温度有效: {temperature}°C")
                
                if hum_valid:
                    self.humidity = humidity
                    updated = True
                    print(f"DEBUG - 湿度有效: {humidity}%")
                else:
                    # 湿度无效但温度有效，保持上次的湿度值
                    print(f"DEBUG - 湿度无效，保持上次值: {self.humidity}%")
                
                if updated:
                    self.last_read_time = time.time()
                    self.read_success_count += 1
                    # 返回温度（可能有效）和湿度（可能保持上次值）
                    return True, self.temperature, self.humidity
            
            # 都无效
            self.read_fail_count += 1
            print(f"DEBUG - 读取失败: temp={temperature}, hum={humidity}")
            return False, self.temperature, self.humidity
            
        except RuntimeError as e:
            # DHT11的临时性错误，非常常见
            self.read_fail_count += 1
            print(f"DEBUG - DHT11运行时错误: {e}")
            # 返回上次成功读取的值
            return False, self.temperature, self.humidity
            
        except Exception as e:
            print(f"DEBUG - DHT11异常错误: {e}")
            self.read_fail_count += 1
            return False, self.temperature, self.humidity
    
    def get_temperature(self):
        """获取最后读取的温度"""
        return self.temperature
    
    def get_humidity(self):
        """获取最后读取的湿度"""
        return self.humidity
    
    def get_stats(self):
        """获取读取统计信息"""
        return {
            'success': self.read_success_count,
            'fail': self.read_fail_count,
            'last_temp': self.temperature,
            'last_humi': self.humidity
        }
    
    def check_alarm(self):
        """检查是否需要报警"""
        if self.temperature > TEMPERATURE_HIGH_THRESHOLD:
            return True, f"高温报警: {self.temperature}°C"
        if self.humidity > HUMIDITY_HIGH_THRESHOLD:
            return True, f"高湿报警: {self.humidity}%"
        return False, ""


class SmokeSensor:
    """烟雾报警器 - 只使用数字输出（DIN）"""
    
    def __init__(self):
        self.digital_pin = SMOKE_DIGITAL_PIN
        GPIO.setup(self.digital_pin, GPIO.IN)
        
        self.digital_value = 0
        self.last_read_time = 0
        self.alarm_count = 0  # 连续报警计数，用于防抖
        self.read_count = 0
        
        logger.info(f"✅ 烟雾传感器初始化完成 (数字引脚GPIO{self.digital_pin})")
        logger.info("  提示：请调节传感器上的电位器设置触发阈值")
    
    def read(self):
        """读取数字输出"""
        self.digital_value = GPIO.input(self.digital_pin)
        self.last_read_time = time.time()
        self.read_count += 1
        return self.digital_value
    
    def check_alarm(self):
        """
        检查是否需要报警（带防抖）
        连续3次检测到报警才触发，防止误报
        """
        current_value = self.read()
        
        if current_value == SMOKE_DIGITAL_THRESHOLD:
            self.alarm_count += 1
            if self.alarm_count >= 3:
                return True, "🔥 烟雾报警！"
        else:
            self.alarm_count = 0
        
        return False, ""
    
    def get_status(self):
        """获取状态字符串"""
        value = self.read()
        if value == SMOKE_DIGITAL_THRESHOLD:
            return "🔥 烟雾报警"
        else:
            return "✅ 烟雾正常"


class SensorManager:
    """传感器管理器 - 统一管理所有传感器"""
    
    def __init__(self, alarm_system):
        self.alarm_system = alarm_system
        self.dht11 = DHT11Sensor()
        self.smoke = SmokeSensor()
        
        self.running = False
        self.monitor_thread = None
        
        self.sensor_data = {
            'temperature': 0,
            'humidity': 0,
            'smoke_digital': 0,
            'last_update': 0
        }
        
        # 检查DHT11是否真的可用
        self.dht11_available = DHT11_AVAILABLE and self.dht11.dht_device is not None
        
        if self.dht11_available:
            logger.info("✅ 温湿度传感器已启用")
        else:
            logger.warning("⚠️ 温湿度传感器已禁用")
        
        logger.info("✅ 传感器管理器初始化完成")
    
    def update_sensor_data(self):
        """更新所有传感器数据"""
        # 读取DHT11（如果可用）
        if self.dht11_available:
            success, temp, hum = self.dht11.read()
            if success:
                self.sensor_data['temperature'] = temp
                self.sensor_data['humidity'] = hum
                
                # 检查温湿度报警
                alarm, msg = self.dht11.check_alarm()
                if alarm and not self.alarm_system.is_active():
                    logger.warning(f"📢 传感器报警: {msg}")
                    self.alarm_system.trigger('sensor')
        
        # 读取烟雾传感器
        self.sensor_data['smoke_digital'] = self.smoke.read()
        
        # 检查烟雾报警（带防抖）
        alarm, msg = self.smoke.check_alarm()
        if alarm and not self.alarm_system.is_active():
            logger.warning(f"📢 {msg}")
            
            # 记录报警时间
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"烟雾报警触发，时间: {timestamp}")
            
            self.alarm_system.trigger('smoke')
        
        self.sensor_data['last_update'] = time.time()
    
    def monitor_loop(self):
        """传感器监控循环"""
        logger.info("🔄 传感器监控线程已启动")
        
        if self.dht11_available:
            logger.info("  温湿度传感器: 已启用")
        else:
            logger.info("  温湿度传感器: 未启用")
        logger.info("  烟雾传感器: 已启用（数字模式）")
        
        # 初始读取，让传感器稳定
        time.sleep(2)
        
        while self.running:
            try:
                self.update_sensor_data()
            except Exception as e:
                logger.error(f"❌ 传感器读取错误: {e}")
            
            time.sleep(SENSOR_CHECK_INTERVAL)
    
    def start(self):
        """启动传感器监控"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("✅ 传感器监控已启动")
    
    def stop(self):
        """停止传感器监控"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        logger.info("🛑 传感器监控已停止")
        
        # 清理DHT11资源
        if self.dht11_available and self.dht11.dht_device:
            try:
                self.dht11.dht_device.exit()
                logger.info("✅ DHT11资源已释放")
            except:
                pass
    
    def get_status_string(self):
        """获取传感器状态字符串"""
        data = self.sensor_data
        status = []
        
        if self.dht11_available and data['temperature'] > 0:
            status.append(f"🌡️ {data['temperature']}°C")
        if self.dht11_available and data['humidity'] > 0:
            status.append(f"💧 {data['humidity']}%")
        
        # 烟雾状态
        if data['smoke_digital'] == SMOKE_DIGITAL_THRESHOLD:
            status.append("🔥 烟雾报警")
        else:
            status.append("✅ 烟雾正常")
        
        return " | ".join(status) if status else "⏳ 传感器待读取"


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🔧 传感器测试程序 (已适配当前环境)")
    print("=" * 60)
    
    # 创建简单的报警模拟
    class MockAlarm:
        def is_active(self): return False
        def trigger(self, source): print(f"🔔 报警触发: {source}")
    
    mock_alarm = MockAlarm()
    sensors = SensorManager(mock_alarm)
    
    try:
        sensors.start()
        print("-" * 60)
        print("📊 传感器状态:")
        print("  烟雾传感器: 数字输出模式")
        print("  温湿度传感器: " + ("已启用" if sensors.dht11_available else "未启用"))
        print("-" * 60)
        print("按 Ctrl+C 退出测试")
        print("-" * 60)
        
        last_stats_print = time.time()
        
        while True:
            time.sleep(2)
            
            # 获取烟雾状态
            smoke_status = "报警" if sensors.sensor_data['smoke_digital'] == SMOKE_DIGITAL_THRESHOLD else "正常"
            smoke_icon = "🔥" if sensors.sensor_data['smoke_digital'] == SMOKE_DIGITAL_THRESHOLD else "✅"
            
            # 每10秒打印一次详细统计
            current_time = time.time()
            if current_time - last_stats_print > 10:
                print("-" * 60)
                if sensors.dht11_available:
                    stats = sensors.dht11.get_stats()
                    print(f"📊 DHT11统计: 成功读取 {stats['success']} 次, 失败 {stats['fail']} 次")
                last_stats_print = current_time
            
            # 打印当前读数
            if sensors.dht11_available:
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"🌡️ {sensors.sensor_data['temperature']:>2}°C  "
                      f"💧 {sensors.sensor_data['humidity']:>2}%  "
                      f"{smoke_icon} 烟雾: {smoke_status}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"{smoke_icon} 烟雾: {smoke_status}")
    
    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        sensors.stop()
        print("✅ 测试结束")
        print("=" * 60)