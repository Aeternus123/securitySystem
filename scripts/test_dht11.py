#!/usr/bin/env python3
# dht11_test.py - DHT11温湿度传感器测试脚本

import time
import board
import adafruit_dht

# ===================== 配置区域 =====================
# 请根据你的实际接线修改这个引脚号
DHT_PIN = board.D26      # DHT11数据线连接的GPIO引脚 (BCM编号)
USE_INTERNAL_PULLUP = False  # DHT11通常不需要内部上拉

# 测试参数
READ_INTERVAL = 2.0     # 读取间隔（秒）
MAX_RETRIES = 10        # 连续失败最大重试次数
DEBUG_MODE = True       # 显示详细调试信息

# ===================== 初始化 =====================
print("=" * 50)
print("DHT11温湿度传感器测试程序")
print("=" * 50)
print(f"📌 数据引脚: {DHT_PIN}")
print(f"📌 读取间隔: {READ_INTERVAL}秒")
print("-" * 50)

# 初始化DHT11传感器
try:
    dht_device = adafruit_dht.DHT11(DHT_PIN)
    print("✅ DHT11传感器初始化成功")
except Exception as e:
    print(f"❌ DHT11传感器初始化失败: {e}")
    print("请检查：")
    print("  1. 传感器是否正确连接")
    print(f"  2. 数据线是否连接到GPIO {DHT_PIN}对应的物理引脚")
    print("  3. 是否已安装adafruit-circuitpython-dht库")
    exit(1)

# ===================== 测试函数 =====================
def read_dht_safe():
    """安全读取DHT11数据，带错误处理"""
    try:
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        
        if temperature is not None and humidity is not None:
            return temperature, humidity
        else:
            return None, None
    except RuntimeError as e:
        # DHT11经常因为时序问题报错，这是正常的
        if DEBUG_MODE:
            print(f"⏳ 读取错误（可忽略）: {e}")
        return None, None
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return None, None

def check_wiring_guide():
    """提供接线指南"""
    print("\n🔧 接线指南（参考）:")
    print("  DHT11引脚 → 树莓派引脚")
    print("  VCC (电源正) → 物理引脚1 (3.3V) 或 物理引脚2 (5V)")
    print(f"  DATA (数据) → 物理引脚{get_physical_pin(DHT_PIN)} (GPIO {DHT_PIN})")
    print("  GND (电源地) → 物理引脚6 (GND)")
    print("\n💡 提示：如果使用裸传感器，需要在DATA和VCC之间加4.7kΩ上拉电阻")
    print("  如果是模块（带电路板），通常已集成上拉电阻")

def get_physical_pin(bcm_pin):
    """BCM引脚到物理引脚的快速映射（常用引脚）"""
    pin_map = {
        board.D2: 3, board.D3: 5, board.D4: 7,
        board.D17: 11, board.D18: 12, board.D27: 13,
        board.D22: 15, board.D23: 16, board.D24: 18,
        board.D10: 19, board.D9: 21, board.D11: 23,
        board.D14: 8, board.D15: 10
    }
    return pin_map.get(bcm_pin, "??")

# ===================== 主测试循环 =====================
print("\n🚀 开始读取数据...（按 Ctrl+C 退出）")
print("-" * 50)

success_count = 0
fail_count = 0
consecutive_fails = 0
last_temp = None
last_humi = None

try:
    while True:
        # 读取数据
        temp, humi = read_dht_safe()
        
        # 更新计数
        if temp is not None and humi is not None:
            success_count += 1
            consecutive_fails = 0
            last_temp, last_humi = temp, humi
            
            # 显示成功读取的数据
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] ✅ 温度: {temp:>2}°C  湿度: {humi:>2}%  "
                  f"(成功: {success_count}, 失败: {fail_count})")
        else:
            fail_count += 1
            consecutive_fails += 1
            
            # 显示失败信息
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] ⏳ 读取失败 ({consecutive_fails}/{MAX_RETRIES})  ", 
                  end="\r" if consecutive_fails < MAX_RETRIES else "\n")
            
            # 如果连续失败太多，提示检查硬件
            if consecutive_fails >= MAX_RETRIES:
                print(f"\n❌ 连续{MAX_RETRIES}次读取失败！")
                check_wiring_guide()
                print("\n🔄 继续尝试读取...")
                consecutive_fails = 0  # 重置计数器
        
        # 等待下一次读取
        time.sleep(READ_INTERVAL)
        
except KeyboardInterrupt:
    print("\n\n" + "=" * 50)
    print("🛑 测试已手动停止")
    print("=" * 50)
    print(f"📊 统计结果:")
    print(f"  ✅ 成功读取: {success_count} 次")
    print(f"  ❌ 失败次数: {fail_count} 次")
    if success_count > 0:
        print(f"  🌡 最后有效温度: {last_temp}°C")
        print(f"  💧 最后有效湿度: {last_humi}%")
    print("=" * 50)

except Exception as e:
    print(f"\n❌ 程序异常: {e}")
    
finally:
    # 清理资源
    try:
        dht_device.exit()
        print("✅ GPIO资源已释放")
    except:
        pass