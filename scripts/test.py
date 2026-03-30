#!/usr/bin/env python3
import RPi.GPIO as GPIO
import serial
import time
import threading
import sys
import select
import cv2
import numpy as np
import onnxruntime as ort
import pickle
import os
from datetime import datetime

# ==================== GPIO设置 ====================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# 定义引脚
RED_LED = 17      # 红灯 - 主卧灯
GREEN_LED = 27    # 绿灯 - 客房灯
BUZZER = 22       # 蜂鸣器
MOTOR = 23        # 电机/风扇

# 超声波模块引脚
TRIG = 24         # 超声波 Trig 引脚
ECHO = 25         # 超声波 Echo 引脚

# 矩阵键盘引脚
KEYPAD_ROWS = [5, 6, 13, 19]    # 行引脚
KEYPAD_COLS = [16, 20, 21, 12]  # 列引脚

# 设置GPIO
GPIO.setup(RED_LED, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(GREEN_LED, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BUZZER, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(MOTOR, GPIO.OUT, initial=GPIO.LOW)

# 超声波引脚设置
GPIO.setup(TRIG, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ECHO, GPIO.IN)

# 矩阵键盘引脚设置
for pin in KEYPAD_ROWS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)
for pin in KEYPAD_COLS:
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# ==================== 串口设置 ====================
SERIAL_PORT = '/dev/serial0'
BAUDRATE = 9600

# 操作码定义
OP_OPEN = 0x81      # 打开操作
OP_CLOSE = 0xC1     # 关闭操作
OP_MOTOR_ON = 0x43  # 打开电机
OP_MOTOR_OFF = 0x83 # 关闭电机
OP_WAKEUP = 0x01    # 唤醒词
OP_VOICE_ON = 0x02  # 打开语音指令
OP_VOICE_OFF = 0x45 # 关闭语音指令

# 设备ID前缀
MASTER_BEDROOM_PREFIX = bytes([0x52, 0x34, 0x02, 0x00, 0x00])
GUEST_BEDROOM_PREFIX = bytes([0x6B, 0x34, 0x02, 0x00, 0x00])
MOTOR_PREFIX = bytes([0x19, 0xBC, 0x02, 0x00, 0x00])
WAKEUP_PREFIX = bytes([0x55, 0xE0, 0x01, 0x00, 0x00])
VOICE_CONTROL_PREFIX = bytes([0x1B, 0x94, 0x01, 0x00, 0x00])

# 设备映射
DEVICES = {
    MASTER_BEDROOM_PREFIX: {"name": "主卧灯", "pin": RED_LED, "state": False},
    GUEST_BEDROOM_PREFIX: {"name": "客房灯", "pin": GREEN_LED, "state": False},
    MOTOR_PREFIX: {"name": "电机", "pin": MOTOR, "state": False},
}

# ==================== 系统状态 ====================
system_locked = True
last_recognized_time = 0
UNLOCK_DURATION = 30
voice_control_enabled = False
face_recognition_active = False  # 人脸识别是否激活
face_recognition_timeout = 30    # 人脸识别激活超时（秒）
face_recognition_start_time = 0  # 人脸识别激活开始时间

# 报警相关
alarm_active = False
alarm_thread = None
alarm_trigger_source = None
STRANGER_ALARM_DURATION = 10

# 超声波测距相关
ultrasonic_active = True
ultrasonic_thread = None
DISTANCE_THRESHOLD = 30
CHECK_INTERVAL = 0.5

# 灯光状态备份
saved_light_states = {}

# ==================== 增强人脸识别类 ====================
class EnhancedFaceDetector:
    def __init__(self, 
                 det_model_path='models/yolov8n-face.onnx',
                 database_path='database/face_db.pkl',
                 conf_thres=0.5,
                 sim_threshold=0.65):
        
        self.conf_thres = conf_thres
        self.sim_threshold = sim_threshold
        self.database_path = database_path
        
        print("加载 YOLOv8n-face 检测模型...")
        self.det_session = ort.InferenceSession(det_model_path, providers=['CPUExecutionProvider'])
        self.det_input_name = self.det_session.get_inputs()[0].name
        
        self.face_database = self.load_database()
        self.input_size = 640
        print(f"人脸识别初始化完成，数据库中有 {len(self.face_database)} 个人")
        print(f"相似度阈值: {self.sim_threshold}")
        
        self.last_stranger_time = 0
        self.stranger_cooldown = 5
        self.debug_mode = True
        
    def load_database(self):
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"加载数据库失败: {e}")
                return {}
        return {}
    
    def cosine_similarity(self, a, b):
        a = np.array(a).flatten()
        b = np.array(b).flatten()
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot_product / (norm_a * norm_b)
    
    def local_binary_pattern(self, image):
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                lbp[i-1, j-1] = code
        return lbp
    
    def extract_enhanced_features(self, face_img):
        face = cv2.resize(face_img, (128, 128))
        
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [64], [0, 256])
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        hist_grad = cv2.calcHist([gradient.astype(np.uint8)], [0], None, [32], [0, 255])
        
        lbp = self.local_binary_pattern(gray)
        hist_lbp = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [32], [0, 256])
        
        h, w = gray.shape
        block_h, block_w = h//4, w//4
        block_features = []
        for i in range(4):
            for j in range(4):
                block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                block_hist = cv2.calcHist([block], [0], None, [8], [0, 256])
                block_features.append(cv2.normalize(block_hist, block_hist).flatten())
        
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
        hist_grad = cv2.normalize(hist_grad, hist_grad).flatten()
        hist_lbp = cv2.normalize(hist_lbp, hist_lbp).flatten()
        
        features = np.concatenate([
            hist_h, hist_s, hist_v, 
            hist_gray, hist_grad, hist_lbp,
            np.concatenate(block_features)
        ])
        
        return features
    
    def preprocess_detection(self, image):
        self.original_h, self.original_w = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        scale = min(self.input_size / self.original_w, self.input_size / self.original_h)
        new_w = int(self.original_w * scale)
        new_h = int(self.original_h * scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        
        canvas = canvas.astype(np.float32) / 255.0
        canvas = canvas.transpose(2, 0, 1)
        canvas = np.expand_dims(canvas, axis=0)
        
        self.scale = scale
        return canvas
    
    def detect_faces(self, image):
        input_tensor = self.preprocess_detection(image)
        outputs = self.det_session.run(None, {self.det_input_name: input_tensor})
        
        predictions = np.transpose(outputs[0], (0, 2, 1))[0]
        
        boxes = []
        for pred in predictions:
            x_center, y_center, width, height, confidence = pred
            
            if confidence < self.conf_thres:
                continue
            
            x1 = (x_center - width/2) / self.scale
            y1 = (y_center - height/2) / self.scale
            x2 = (x_center + width/2) / self.scale
            y2 = (y_center + height/2) / self.scale
            
            x1 = max(0, min(self.original_w - 1, x1))
            y1 = max(0, min(self.original_h - 1, y1))
            x2 = max(0, min(self.original_w - 1, x2))
            y2 = max(0, min(self.original_h - 1, y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, [1.0]*len(boxes), self.conf_thres, 0.5)
            if len(indices) > 0:
                indices = indices.flatten()
                return [boxes[i] for i in indices]
        
        return []
    
    def check_frame(self, frame):
        boxes = self.detect_faces(frame)
        
        if len(boxes) == 0:
            return False, []
        
        has_stranger = False
        results = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                continue
            
            features = self.extract_enhanced_features(face_img)
            
            best_match = "unknown"
            best_similarity = 0.0
            all_similarities = {}
            
            for name, data in self.face_database.items():
                similarities = [self.cosine_similarity(features, feat) for feat in data['features']]
                avg_similarity = np.mean(similarities)
                all_similarities[name] = avg_similarity
                
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    if avg_similarity > self.sim_threshold:
                        best_match = name
            
            if self.debug_mode and best_match != "unknown":
                print(f"\n🔍 人脸匹配详情:")
                for name, sim in all_similarities.items():
                    status = "✅" if name == best_match else "  "
                    print(f"  {status} {name}: {sim:.3f} (阈值: {self.sim_threshold})")
            
            is_recognized = best_match != "unknown"
            if not is_recognized:
                has_stranger = True
            
            results.append({
                'box': box,
                'name': best_match,
                'similarity': best_similarity,
                'recognized': is_recognized
            })
        
        return has_stranger, results

# ==================== 初始化人脸识别 ====================
face_detector = EnhancedFaceDetector(
    det_model_path='models/yolov8n-face.onnx',
    database_path='database/face_db.pkl',
    sim_threshold=0.65
)

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    print("✅ 摄像头已打开")
else:
    print("❌ 无法打开摄像头")
    cap = None

# ==================== 功能函数 ====================
def save_light_states():
    global saved_light_states
    saved_light_states.clear()
    for prefix, dev_info in DEVICES.items():
        if dev_info['name'] in ['主卧灯', '客房灯']:
            saved_light_states[prefix] = dev_info['state']

def restore_light_states():
    global saved_light_states
    for prefix, state in saved_light_states.items():
        if prefix in DEVICES:
            dev_info = DEVICES[prefix]
            if state:
                GPIO.output(dev_info['pin'], GPIO.HIGH)
                dev_info['state'] = True
            else:
                GPIO.output(dev_info['pin'], GPIO.LOW)
                dev_info['state'] = False
    saved_light_states.clear()

def measure_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    
    timeout = time.time() + 0.1
    start_time = time.time()
    while GPIO.input(ECHO) == 0 and time.time() < timeout:
        start_time = time.time()
    
    if time.time() >= timeout:
        return None
    
    timeout = time.time() + 0.1
    while GPIO.input(ECHO) == 1 and time.time() < timeout:
        end_time = time.time()
    
    if time.time() >= timeout:
        return None
    
    duration = end_time - start_time
    distance = duration * 34300 / 2
    return distance

def ultrasonic_monitor():
    global alarm_active, ultrasonic_active, alarm_trigger_source
    
    print("📡 超声波监控已启动（常驻）")
    
    while ultrasonic_active:
        distance = measure_distance()
        
        if distance is not None:
            if distance < DISTANCE_THRESHOLD and not alarm_active:
                print(f"\n⚠️ 超声波检测到距离过近 ({distance:.1f}cm)")
                save_light_states()
                for dev_info in DEVICES.values():
                    if dev_info['name'] in ['主卧灯', '客房灯']:
                        GPIO.output(dev_info['pin'], GPIO.LOW)
                        dev_info['state'] = False
                trigger_alarm('ultrasonic')
            
            elif distance >= DISTANCE_THRESHOLD and alarm_active and alarm_trigger_source == 'ultrasonic':
                print(f"\n✅ 超声波距离恢复正常")
                stop_alarm()
        
        time.sleep(CHECK_INTERVAL)

def face_monitor():
    """人脸监控线程 - 只在激活时运行"""
    global system_locked, last_recognized_time, voice_control_enabled, alarm_active, alarm_trigger_source, face_recognition_active, face_recognition_start_time
    
    if cap is None:
        return
    
    frame_count = 0
    print("👤 人脸监控已启动（待机模式）")
    
    while ultrasonic_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # 只有在激活状态下才进行人脸识别
        if face_recognition_active:
            frame_count += 1
            
            # 检查是否超时
            if time.time() - face_recognition_start_time > face_recognition_timeout:
                print(f"\n⏱️ 人脸识别激活超时 ({face_recognition_timeout}秒)，自动关闭")
                face_recognition_active = False
                continue
            
            if frame_count % 5 == 0:
                has_stranger, results = face_detector.check_frame(frame)
                
                # 检查是否有已知人脸
                recognized_names = [r['name'] for r in results if r['recognized']]
                
                if recognized_names and system_locked:
                    print(f"\n🔓 人脸识别成功！欢迎 {', '.join(recognized_names)}")
                    print(f"   相似度: {results[0]['similarity']:.3f}")
                    print("🎤 语音控制已启用")
                    system_locked = False
                    voice_control_enabled = True
                    face_recognition_active = False  # 解锁后关闭人脸识别
                    
                    for _ in range(2):
                        GPIO.output(GREEN_LED, GPIO.HIGH)
                        time.sleep(0.2)
                        GPIO.output(GREEN_LED, GPIO.LOW)
                        time.sleep(0.2)
                
                # 陌生人报警逻辑（只在激活时）
                if has_stranger and not alarm_active:
                    current_time = time.time()
                    if current_time - face_detector.last_stranger_time > face_detector.stranger_cooldown:
                        print(f"\n⚠️ 检测到陌生人！")
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        cv2.imwrite(f"stranger_{timestamp}.jpg", frame)
                        print(f"📸 已保存陌生人照片")
                        
                        save_light_states()
                        for dev_info in DEVICES.values():
                            if dev_info['name'] in ['主卧灯', '客房灯']:
                                GPIO.output(dev_info['pin'], GPIO.LOW)
                                dev_info['state'] = False
                        
                        trigger_alarm('face_stranger')
                        face_detector.last_stranger_time = current_time
        
        time.sleep(0.01)

def alarm_mode():
    global alarm_active, alarm_trigger_source
    print("\n🚨 警报启动！")
    
    alarm_start_time = time.time()
    
    while alarm_active:
        for _ in range(10):
            if not alarm_active: break
            GPIO.output(RED_LED, GPIO.HIGH)
            GPIO.output(GREEN_LED, GPIO.LOW)
            GPIO.output(BUZZER, GPIO.LOW)
            time.sleep(0.1)
            
            if not alarm_active: break
            GPIO.output(RED_LED, GPIO.LOW)
            GPIO.output(GREEN_LED, GPIO.HIGH)
            time.sleep(0.1)
        
        for _ in range(5):
            if not alarm_active: break
            GPIO.output(RED_LED, GPIO.HIGH)
            GPIO.output(GREEN_LED, GPIO.HIGH)
            time.sleep(0.1)
            
            if not alarm_active: break
            GPIO.output(RED_LED, GPIO.LOW)
            GPIO.output(GREEN_LED, GPIO.LOW)
            time.sleep(0.1)
        
        if alarm_trigger_source == 'face_stranger':
            if time.time() - alarm_start_time >= STRANGER_ALARM_DURATION:
                print(f"\n⏱️ 陌生人报警结束")
                stop_alarm()
                break
    
    GPIO.output(RED_LED, GPIO.LOW)
    GPIO.output(GREEN_LED, GPIO.LOW)
    GPIO.output(BUZZER, GPIO.HIGH)

def trigger_alarm(source):
    global alarm_active, alarm_trigger_source, alarm_thread
    if alarm_active:
        return
    alarm_active = True
    alarm_trigger_source = source
    alarm_thread = threading.Thread(target=alarm_mode)
    alarm_thread.daemon = True
    alarm_thread.start()

def stop_alarm():
    global alarm_active, alarm_trigger_source
    print("\n🔕 正在停止报警...")
    alarm_active = False
    if alarm_thread and alarm_thread.is_alive():
        alarm_thread.join(timeout=2)
    
    GPIO.output(RED_LED, GPIO.LOW)
    GPIO.output(GREEN_LED, GPIO.LOW)
    GPIO.output(BUZZER, GPIO.HIGH)
    
    restore_light_states()
    alarm_trigger_source = None
    print("✅ 报警已停止")

def beep_feedback(times=1):
    for _ in range(times):
        GPIO.output(BUZZER, GPIO.LOW)
        time.sleep(0.05)
        GPIO.output(BUZZER, GPIO.HIGH)
        time.sleep(0.05)

def handle_command(operation, device_id):
    """处理语音命令"""
    global alarm_active, alarm_trigger_source, system_locked, voice_control_enabled, face_recognition_active, last_recognized_time, face_recognition_start_time
    
    if len(device_id) < 5:
        return
    
    prefix = device_id[:5]
    
    # 获取操作名称
    if operation == OP_OPEN:
        op_name = "打开"
    elif operation == OP_CLOSE:
        op_name = "关闭"
    elif operation == OP_MOTOR_ON:
        op_name = "打开电机"
    elif operation == OP_MOTOR_OFF:
        op_name = "关闭电机"
    elif operation == OP_WAKEUP:
        op_name = "唤醒"
    elif operation == OP_VOICE_ON:
        op_name = "打开语音"
    elif operation == OP_VOICE_OFF:
        op_name = "关闭语音"
    else:
        op_name = f"未知(0x{operation:02X})"
    
    # 处理语音控制指令
    if prefix == VOICE_CONTROL_PREFIX:
        print(f"\n🎯 收到语音控制指令: {op_name}")
        
        if operation == OP_VOICE_ON:
            if not face_recognition_active and not voice_control_enabled:
                print("🔓 语音控制激活，请进行人脸识别解锁")
                face_recognition_active = True
                face_recognition_start_time = time.time()
                # 提示音：等待人脸识别的提示
                for _ in range(2):
                    GPIO.output(BUZZER, GPIO.LOW)
                    time.sleep(0.1)
                    GPIO.output(BUZZER, GPIO.HIGH)
                    time.sleep(0.1)
            else:
                print("⏳ 人脸识别已在运行中或系统已解锁")
            return
        
        elif operation == OP_VOICE_OFF:
            if not system_locked:
                print("🔒 系统手动锁定")
                system_locked = True
                voice_control_enabled = False
                face_recognition_active = False
                GPIO.output(RED_LED, GPIO.HIGH)
                time.sleep(0.3)
                GPIO.output(RED_LED, GPIO.LOW)
                beep_feedback(1)
            else:
                print("系统已处于锁定状态")
            return
    
    # 处理唤醒词
    if prefix == WAKEUP_PREFIX:
        if not alarm_active:
            GPIO.output(GREEN_LED, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(GREEN_LED, GPIO.LOW)
        return
    
    # 检查系统是否锁定（其他指令需要解锁）
    if system_locked:
        print(f"\n🔒 系统已锁定，请先说'打开语音'进行人脸识别")
        GPIO.output(RED_LED, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(RED_LED, GPIO.LOW)
        return
    
    # 系统已解锁，可以处理命令
    print(f"\n🎯 收到: {op_name}")
    print(f"  设备ID前缀: {' '.join([f'{b:02X}' for b in prefix])}")
    
    for dev_prefix, dev_info in DEVICES.items():
        if prefix == dev_prefix:
            print(f"  设备: {dev_info['name']}")
            
            if dev_info['name'] == "电机":
                if operation == OP_MOTOR_ON:
                    GPIO.output(dev_info['pin'], GPIO.HIGH)
                    dev_info['state'] = True
                    print(f"  ✅ 电机已打开")
                    beep_feedback(1)
                elif operation == OP_MOTOR_OFF:
                    GPIO.output(dev_info['pin'], GPIO.LOW)
                    dev_info['state'] = False
                    print(f"  ⚫ 电机已关闭")
                    beep_feedback(1)
                return
            
            if dev_info['name'] in ['主卧灯', '客房灯']:
                if alarm_active:
                    if alarm_trigger_source == 'ultrasonic':
                        print(f"  ⚠️ 超声波报警中，无法控制灯光")
                        beep_feedback(2)
                    else:
                        if operation == OP_CLOSE:
                            print("🔕 停止报警")
                            stop_alarm()
                            beep_feedback(1)
                        else:
                            print(f"  ⚠️ 报警中，只能使用关闭命令")
                            beep_feedback(2)
                    return
                
                if operation == OP_OPEN:
                    GPIO.output(dev_info['pin'], GPIO.HIGH)
                    dev_info['state'] = True
                    print(f"  ✅ {dev_info['name']} 已打开")
                    beep_feedback(1)
                elif operation == OP_CLOSE:
                    GPIO.output(dev_info['pin'], GPIO.LOW)
                    dev_info['state'] = False
                    print(f"  ⚫ {dev_info['name']} 已关闭")
                    beep_feedback(1)
                return
    
    print(f"  ❌ 未知设备")

def parse_frame(frame):
    if len(frame) < 16:
        return None, None
    if frame[0] != 0xA5 or frame[1] != 0xFC or frame[-1] != 0xFB:
        return None, None
    operation = frame[7]
    device_id = bytes(frame[8:15])
    return operation, device_id

def print_status():
    """打印系统状态"""
    print("\n" + "=" * 50)
    print("当前系统状态:")
    print("=" * 50)
    
    lock_status = "🔒 锁定" if system_locked else "🔓 解锁"
    print(f"系统状态: {lock_status}")
    if not system_locked:
        remaining = UNLOCK_DURATION - (time.time() - last_recognized_time)
        print(f"  剩余解锁时间: {remaining:.0f}秒")
    
    # 人脸识别激活状态
    face_status = "🟢 激活中" if face_recognition_active else "⚫ 待机"
    print(f"人脸识别: {face_status}")
    if face_recognition_active:
        remaining = face_recognition_timeout - (time.time() - face_recognition_start_time)
        print(f"  剩余时间: {remaining:.0f}秒")
    
    for dev_info in DEVICES.values():
        status = "🔴 开" if dev_info['state'] else "⚫ 关"
        print(f"  {dev_info['name']}: {status}")
    
    source_map = {'ultrasonic': '超声波', 'face_stranger': '陌生人', 'manual': '手动'}
    source_name = source_map.get(alarm_trigger_source, '无')
    print(f"  报警系统: {'🚨 启动中' if alarm_active else '🔕 关闭'} (触发源: {source_name})")
    
    print(f"  人脸数据库: {len(face_detector.face_database)} 个人")
    print(f"  语音控制: {'✅ 已启用' if voice_control_enabled else '❌ 已禁用'}")
    print("=" * 50)

def manual_alarm():
    if not alarm_active:
        print("\n👤 手动触发报警")
        save_light_states()
        for dev_info in DEVICES.values():
            if dev_info['name'] in ['主卧灯', '客房灯']:
                GPIO.output(dev_info['pin'], GPIO.LOW)
                dev_info['state'] = False
        trigger_alarm('manual')
        beep_feedback(3)

def clear_alarm():
    if alarm_active:
        stop_alarm()

# ==================== 主程序 ====================
print("=" * 70)
print("智能安防系统 v20.0 - 按需人脸识别版")
print("=" * 70)
print(f"硬件连接:")
print(f"  红灯 (GPIO{RED_LED}) - 主卧灯")
print(f"  绿灯 (GPIO{GREEN_LED}) - 客房灯")
print(f"  电机 (GPIO{MOTOR})")
print(f"  蜂鸣器 (GPIO{BUZZER})")
print(f"  超声波 Trig (GPIO{TRIG})")
print(f"  超声波 Echo (GPIO{ECHO})")
print("-" * 70)
print("语音控制指令:")
print("  • '打开语音' - 激活人脸识别，等待解锁")
print("  • '关闭语音' - 手动锁定系统")
print("  • '唤醒词' - 唤醒提示")
print("  • 其他设备指令 - 需解锁后使用")
print("-" * 70)
print("系统安全机制:")
print("  • 说'打开语音'后，才开启人脸识别")
print(f"  • 人脸识别激活后 {face_recognition_timeout} 秒内无操作自动关闭")
print("  • 识别到已知人脸后自动解锁")
print(f"  • 解锁后持续 {UNLOCK_DURATION} 秒无操作自动重新锁定")
print("-" * 70)
print("报警触发源:")
print("  • 超声波 - 距离恢复正常时自动关闭")
print(f"  • 人脸识别 - 检测到陌生人，持续 {STRANGER_ALARM_DURATION} 秒")
print("  • 手动触发 - 需手动清除")
print("-" * 70)
print("正在启动所有监控线程...")

try:
    ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE, timeout=0.1)
    buffer = b''
    
    ultrasonic_thread = threading.Thread(target=ultrasonic_monitor)
    ultrasonic_thread.daemon = True
    ultrasonic_thread.start()
    
    face_thread = threading.Thread(target=face_monitor)
    face_thread.daemon = True
    face_thread.start()
    
    print("✅ 所有监控已启动")
    print("系统初始状态: 🔒 锁定")
    print("输入 's' 查看状态, 'm' 手动报警, 'c' 清除报警, 'u' 手动解锁, 'q' 退出")
    print("-" * 70)
    
    while True:
        if select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 'q':
                break
            elif cmd == 's':
                print_status()
            elif cmd == 'm':
                manual_alarm()
            elif cmd == 'c':
                clear_alarm()
            elif cmd == 'u':
                if system_locked:
                    print("\n🔓 手动解锁")
                    system_locked = False
                    voice_control_enabled = True
                    face_recognition_active = False
                    last_recognized_time = time.time()
                    for _ in range(2):
                        GPIO.output(GREEN_LED, GPIO.HIGH)
                        time.sleep(0.2)
                        GPIO.output(GREEN_LED, GPIO.LOW)
                        time.sleep(0.2)
        
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            buffer += data
            
            while True:
                start = buffer.find(b'\xA5\xFC')
                if start == -1:
                    buffer = b''
                    break
                
                end = buffer.find(b'\xFB', start)
                if end == -1:
                    break
                
                frame = buffer[start:end+1]
                buffer = buffer[end+1:]
                
                operation, device_id = parse_frame(frame)
                if operation is not None:
                    handle_command(operation, device_id)
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\n👋 程序退出")

finally:
    alarm_active = False
    ultrasonic_active = False
    if alarm_thread and alarm_thread.is_alive():
        alarm_thread.join(timeout=1)
    if ultrasonic_thread and ultrasonic_thread.is_alive():
        ultrasonic_thread.join(timeout=1)
    if face_thread and face_thread.is_alive():
        face_thread.join(timeout=1)
    if cap:
        cap.release()
    GPIO.cleanup()
    if 'ser' in locals():
        ser.close()
    print("✅ 清理完成")