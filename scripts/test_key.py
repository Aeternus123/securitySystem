#!/usr/bin/env python3
"""
矩阵键盘测试程序
4x4 矩阵键盘
行: GPIO 5,6,13,19
列: GPIO 16,20,21,12
"""

import RPi.GPIO as GPIO
import time
import sys  # 添加这行

class MatrixKeypad:
    def __init__(self):
        # 定义行引脚 (输出)
        self.ROW_PINS = [5, 6, 13, 19]
        # 定义列引脚 (输入)
        self.COL_PINS = [16, 20, 21, 12]
        
        # 键位映射 (4x4)
        self.KEYS = [
            ['1', '2', '3', 'A'],
            ['4', '5', '6', 'B'],
            ['7', '8', '9', 'C'],
            ['*', '0', '#', 'D']
        ]
        
        # 设置GPIO模式
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # 设置行引脚为输出，初始为高电平
        for pin in self.ROW_PINS:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)
        
        # 设置列引脚为输入，启用下拉电阻
        for pin in self.COL_PINS:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        self.last_key = None
        self.last_press_time = 0
        self.debounce_time = 0.2  # 防抖时间（秒）
        
        print("矩阵键盘初始化完成")
        print(f"行引脚: {self.ROW_PINS}")
        print(f"列引脚: {self.COL_PINS}")
        print("-" * 40)
    
    def scan(self):
        """
        扫描键盘，返回按下的键值
        """
        for r, row_pin in enumerate(self.ROW_PINS):
            # 将当前行设为低电平
            GPIO.output(row_pin, GPIO.LOW)
            
            # 检查所有列
            for c, col_pin in enumerate(self.COL_PINS):
                if GPIO.input(col_pin) == GPIO.HIGH:
                    # 恢复行电平
                    GPIO.output(row_pin, GPIO.HIGH)
                    return self.KEYS[r][c]
            
            # 恢复行电平
            GPIO.output(row_pin, GPIO.HIGH)
        
        return None
    
    def get_key(self):
        """
        获取按键（带防抖）
        """
        current_time = time.time()
        
        # 扫描键盘
        key = self.scan()
        
        # 防抖处理
        if key and key == self.last_key and current_time - self.last_press_time < self.debounce_time:
            return None
        
        if key:
            self.last_key = key
            self.last_press_time = current_time
            return key
        
        self.last_key = None
        return None
    
    def test_mode(self):
        """
        测试模式：持续扫描并显示按下的键
        """
        print("进入测试模式，按 Ctrl+C 退出")
        print("按下键盘上的按键查看效果")
        print("-" * 40)
        
        try:
            while True:
                key = self.get_key()
                if key:
                    print(f"按下: '{key}'")
                    
                    # 根据按键执行不同操作
                    if key == 'A':
                        print("  → 这是A键")
                    elif key == 'B':
                        print("  → 这是B键")
                    elif key == 'C':
                        print("  → 这是C键")
                    elif key == 'D':
                        print("  → 这是D键")
                    elif key == '*':
                        print("  → 星号键")
                    elif key == '#':
                        print("  → 井号键")
                    else:
                        print(f"  → 数字键 {key}")
                
                time.sleep(0.05)  # 防止CPU占用过高
        
        except KeyboardInterrupt:
            print("\n\n测试结束")
    
    def test_connection(self):
        """
        连接测试：逐行逐列测试硬件连接
        """
        print("开始硬件连接测试...")
        print("请确保没有按下任何按键")
        time.sleep(2)
        
        print("\n1. 测试所有引脚是否正常...")
        all_ok = True
        
        # 测试行引脚
        for r, row_pin in enumerate(self.ROW_PINS):
            try:
                GPIO.output(row_pin, GPIO.LOW)
                time.sleep(0.1)
                GPIO.output(row_pin, GPIO.HIGH)
                print(f"  行 {r+1} (GPIO{row_pin}) ✓")
            except:
                print(f"  行 {r+1} (GPIO{row_pin}) ✗")
                all_ok = False
        
        print("\n2. 测试矩阵扫描...")
        print("请逐个按下键盘上的所有按键")
        print("按 'q' 退出测试")
        
        tested_keys = set()
        
        try:
            while True:
                key = self.get_key()
                if key and key not in tested_keys:
                    tested_keys.add(key)
                    print(f"  检测到按键: '{key}' ✓")
                    
                    # 找出按键位置
                    for r in range(4):
                        for c in range(4):
                            if self.KEYS[r][c] == key:
                                print(f"    位置: 行{r+1}, 列{c+1}")
                
                # 检查是否所有键都测试过
                if len(tested_keys) == 16:
                    print("\n✅ 所有16个按键都已测试通过！")
                    break
                
                # 检查输入
                import select
                if select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.readline().strip().lower() == 'q':
                        break
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            pass
        
        if len(tested_keys) < 16:
            print(f"\n⚠️ 只测试了 {len(tested_keys)} 个按键")
            print("未测试的按键:", [k for r in self.KEYS for k in r if k not in tested_keys])
    
    def cleanup(self):
        """清理GPIO"""
        GPIO.cleanup()
        print("GPIO已清理")

def main():
    print("=" * 50)
    print("4x4 矩阵键盘测试程序")
    print("=" * 50)
    
    keypad = MatrixKeypad()
    
    while True:
        print("\n请选择测试模式:")
        print("1. 实时按键测试")
        print("2. 硬件连接测试")
        print("3. 退出")
        
        choice = input("请选择 (1-3): ").strip()
        
        if choice == '1':
            keypad.test_mode()
        elif choice == '2':
            keypad.test_connection()
        elif choice == '3':
            break
        else:
            print("无效选择")
    
    keypad.cleanup()
    print("程序已退出")

# 简单测试脚本（不需要菜单）
def quick_test():
    """快速测试函数"""
    print("快速测试模式")
    keypad = MatrixKeypad()
    
    try:
        print("按下键盘上的按键 (按 Ctrl+C 退出):")
        while True:
            key = keypad.get_key()
            if key:
                print(f"按键: '{key}'")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n测试结束")
    finally:
        keypad.cleanup()

if __name__ == "__main__":
    # 使用快速测试或主菜单
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_test()
    else:
        main()