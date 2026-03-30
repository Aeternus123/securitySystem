#!/bin/bash
# 智能安防系统 - 完整启动脚本
# 用途：启动主程序、HTTP服务器和摄像头服务器

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 定义日志文件路径
LOG_FILE="system.log"
WEB_PORT=8000
CAMERA_PORT=5000  # 摄像头服务器端口

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export DISPLAY=:0

# 打印带颜色的信息
# 静默模式 - 只输出关键信息到终端，详细信息写入日志
SILENT_MODE=true

print_info() {
    if [ "$SILENT_MODE" = "false" ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> $LOG_FILE
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $1" >> $LOG_FILE
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] $1" >> $LOG_FILE
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> $LOG_FILE
}

# 仅写入日志，不输出到终端
log_only() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> $LOG_FILE
}

# 退出全屏模式
exit_fullscreen() {
    log_only "退出全屏模式..."
    
    # 方法1: 使用保存的窗口ID
    if [ -f /tmp/smart_home_window_id ]; then
        WINDOW_ID=$(cat /tmp/smart_home_window_id)
        if command -v xdotool > /dev/null && [ -n "$WINDOW_ID" ]; then
            xdotool windowactivate $WINDOW_ID 2>/dev/null
            sleep 0.2
            xdotool key F11 2>/dev/null
            log_only "已退出全屏模式 (使用保存的窗口ID)"
            rm -f /tmp/smart_home_window_id
            return 0
        fi
    fi
    
    # 清理临时文件
    rm -f /tmp/smart_home_window_id /tmp/smart_home_window_info 2>/dev/null
}

# 函数：检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        print_warning "端口 $port 已被占用"
        return 1
    fi
    return 0
}

# 创建必要的目录
create_dirs() {
    log_only "创建必要的目录..."
    mkdir -p database
    mkdir -p images/stranger
    mkdir -p logs
    mkdir -p web
    log_only "目录创建完成"
}

# 检查Web文件
check_web_files() {
    if [ ! -f "web/index.html" ]; then
        print_warning "web/index.html 不存在"
        return 1
    else
        log_only "Web文件存在"
        return 0
    fi
}

# 启动摄像头服务器 - 带自动重连的稳定版本
start_camera_server() {
    log_only "启动摄像头服务器 (端口 $CAMERA_PORT) - 带自动重连..."
    
    # 创建摄像头服务器启动脚本
    cat > /tmp/start_camera_server.py << 'EOF'
#!/usr/bin/env python3
"""
摄像头服务器 - 带自动重连的稳定版本，支持人脸检测
"""
import cv2
import time
import numpy as np
from flask import Flask, Response
import threading
import sys
import os
import onnxruntime as ort
import pickle
from datetime import datetime

app = Flask(__name__)

# 人脸检测器类
class FaceDetector:
    """轻量级人脸检测器"""
    
    def __init__(self):
        self.sim_threshold = 0.6
        self.database_path = 'data/face_database.pkl'
        self.images_dir = 'data/face_images'
        self.stranger_dir = 'data/stranger_faces'
        self.feature_dim = 352
        
        # 创建目录
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.stranger_dir, exist_ok=True)
        
        # 加载人脸检测模型
        try:
            self.det_session = ort.InferenceSession(
                'models/yolov8n-face.onnx', 
                providers=['CPUExecutionProvider']
            )
            self.det_input_name = self.det_session.get_inputs()[0].name
            self.input_size = 640
            self.initialized = True
        except Exception as e:
            print(f"警告: 人脸检测模型加载失败: {e}")
            self.initialized = False
        
        # 加载数据库
        self.face_database = self.load_database()
    
    def load_database(self):
        """加载人脸数据库"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    return data
            except Exception as e:
                print(f"加载数据库失败: {e}")
                return {}
        return {}
    
    def preprocess_detection(self, image):
        """预处理检测图像"""
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
        """检测人脸，返回人脸框"""
        if not self.initialized:
            return []
        
        input_tensor = self.preprocess_detection(image)
        outputs = self.det_session.run(None, {self.det_input_name: input_tensor})
        
        predictions = np.transpose(outputs[0], (0, 2, 1))[0]
        
        boxes = []
        for pred in predictions:
            x_center, y_center, width, height, confidence = pred
            
            if confidence < 0.5:
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
        
        # NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, [1.0]*len(boxes), 0.5, 0.5)
            if len(indices) > 0:
                indices = indices.flatten()
                return [boxes[i] for i in indices]
        
        return []
    
    def draw_detection(self, frame, boxes):
        """在帧上绘制检测框"""
        for box in boxes:
            x1, y1, x2, y2 = box
            # 绘制矩形框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签
            cv2.putText(frame, "Face", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

class RobustCamera:
    """带自动重连的摄像头类"""
    
    def __init__(self):
        self.cap = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_frame_time = time.time()
        self.frame_timeout = 3.0  # 3秒无帧则重连
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.frame_count = 0
        self.error_count = 0
        
        # 初始化人脸检测器
        self.face_detector = FaceDetector()
        
    def start(self):
        """启动摄像头"""
        # 静默启动，减少终端输出
        self.running = True
        
        # 尝试初始连接
        if self._connect():
            # 启动捕获线程
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.daemon = True
            self.thread.start()
            
            # 仅输出关键信息到终端
            print("✅ 摄像头服务器已启动 (端口 5000)")
            return True
        
        print("❌ 摄像头初始化失败")
        return False
    
    def _connect(self):
        """连接摄像头"""
        # 尝试多种方式
        methods = [
            ('/dev/video0', cv2.CAP_V4L2),
            (0, cv2.CAP_V4L2),
            ('/dev/video0', None),
            (0, None),
            ('/dev/video2', cv2.CAP_V4L2),
            (2, cv2.CAP_V4L2)
        ]
        
        for dev, backend in methods:
            try:
                if self.cap:
                    self.cap.release()
                
                if backend:
                    cap = cv2.VideoCapture(dev, backend)
                else:
                    cap = cv2.VideoCapture(dev)
                
                if cap.isOpened():
                    # 设置分辨率和帧率
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # 测试读取一帧
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # 静默模式，不输出设备信息
                        self.cap = cap
                        self.last_frame_time = time.time()
                        self.reconnect_attempts = 0
                        self.error_count = 0
                        return True
                    cap.release()
            except Exception as e:
                # 静默模式，不输出错误信息
                continue
        
        self.reconnect_attempts += 1
        return False
    
    def _capture_loop(self):
        """捕获循环 - 静默模式"""
        while self.running:
            try:
                # 检查摄像头状态
                if not self.cap or not self.cap.isOpened():
                    if not self._connect():
                        time.sleep(1)
                        continue
                
                # 读取帧
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # 更新时间戳
                    self.last_frame_time = time.time()
                    self.reconnect_attempts = 0
                    self.error_count = 0
                    self.frame_count += 1
                    
                    # 人脸检测
                    if self.face_detector.initialized:
                        boxes = self.face_detector.detect_faces(frame)
                        frame = self.face_detector.draw_detection(frame, boxes)
                    
                    # 压缩为JPEG
                    ret, jpeg = cv2.imencode('.jpg', frame, [
                        cv2.IMWRITE_JPEG_QUALITY, 70  # 降低质量减少带宽
                    ])
                    
                    if ret:
                        with self.frame_lock:
                            self.current_frame = jpeg.tobytes()
                    
                else:
                    # 读取失败
                    self.error_count += 1
                    
                    if self.error_count >= 5:
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                        self._connect()
                        self.error_count = 0
                
            except Exception as e:
                # 静默处理错误
                time.sleep(0.5)
            
            # 控制帧率
            time.sleep(1.0 / 30)  # 30fps
    
    def get_frame(self):
        """获取最新帧"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame
        return None
    
    def generate_frames(self):
        """生成MJPEG流"""
        no_frame_count = 0
        
        while self.running:
            frame = self.get_frame()
            
            if frame:
                no_frame_count = 0
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                no_frame_count += 1
                
                # 创建占位图
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # 显示状态信息
                if no_frame_count < 30:
                    status = "Waiting for camera..."
                else:
                    status = "Camera offline - reconnecting..."
                
                cv2.putText(img, status, (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 显示重连计数
                if self.reconnect_attempts > 0:
                    cv2.putText(img, f"Reconnect attempts: {self.reconnect_attempts}", 
                               (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                ret, jpeg = cv2.imencode('.jpg', img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            # 控制流输出频率
            time.sleep(1.0 / 30)
    
    def stop(self):
        """停止摄像头"""
        self.running = False
        if self.cap:
            self.cap.release()
    
    def get_status(self):
        """获取状态信息"""
        return {
            'running': self.running,
            'connected': self.cap is not None and self.cap.isOpened(),
            'reconnect_attempts': self.reconnect_attempts,
            'frame_count': self.frame_count,
            'last_frame': self.last_frame_time
        }

# 创建全局摄像头实例
camera = RobustCamera()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>摄像头服务器</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: #1a1a1a; 
                color: white;
                text-align: center;
                padding: 20px;
                margin: 0;
            }
            h1 { color: #4CAF50; }
            .camera-container {
                background: #333;
                border-radius: 10px;
                padding: 10px;
                display: inline-block;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            }
            img {
                width: 100%;
                max-width: 800px;
                border-radius: 5px;
                border: 2px solid #444;
            }
            .status {
                margin-top: 20px;
                color: #888;
            }
            .online { color: #4CAF50; }
            .offline { color: #f44336; }
            .info { margin-top: 10px; color: #888; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1>📹 摄像头服务器</h1>
        <div class="camera-container">
            <img src="/video_feed" id="cameraFeed" 
                 onload="updateStatus(true)" 
                 onerror="updateStatus(false)">
        </div>
        <div class="status" id="status">状态: <span class="online">在线</span></div>
        <div class="info" id="stats"></div>
        
        <script>
            function updateStatus(online) {
                const statusEl = document.getElementById('status');
                if (online) {
                    statusEl.innerHTML = '状态: <span class="online">在线</span>';
                } else {
                    statusEl.innerHTML = '状态: <span class="offline">离线</span>';
                }
            }
            
            // 获取状态信息
            function fetchStatus() {
                fetch('/status')
                    .then(res => res.json())
                    .then(data => {
                        const statsEl = document.getElementById('stats');
                        statsEl.innerHTML = `帧数: ${data.frame_count} | 重连: ${data.reconnect_attempts}`;
                    })
                    .catch(() => {});
            }
            
            // 每2秒更新状态
            setInterval(fetchStatus, 2000);
            
            // 每5秒刷新图片（防止缓存）
            setInterval(() => {
                const img = document.getElementById('cameraFeed');
                const timestamp = new Date().getTime();
                img.src = '/video_feed?t=' + timestamp;
            }, 5000);
            
            // 初始获取状态
            fetchStatus();
        </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(camera.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return camera.get_status()

if __name__ == '__main__':
    if camera.start():
        try:
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\n正在停止...")
        finally:
            camera.stop()
            print("服务器已停止")
    else:
        sys.exit(1)
EOF

    chmod +x /tmp/start_camera_server.py
    
    # 在后台启动摄像头服务器，所有输出重定向到日志文件
    source venv/bin/activate
    # 设置守护进程环境变量
    export SECURITY_SYSTEM_DAEMON=true
    python /tmp/start_camera_server.py >> logs/system.log 2>&1 &
    CAMERA_PID=$!
    deactivate
    
    sleep 3
    
    # 检查是否成功启动
    if ps -p $CAMERA_PID > /dev/null; then
        print_success "摄像头服务器已启动 (PID: $CAMERA_PID)"
        echo $CAMERA_PID > /tmp/smart_home_camera.pid
        return 0
    else
        # 检查端口是否被占用
        if lsof -Pi :$CAMERA_PORT -sTCP:LISTEN -t >/dev/null ; then
            print_success "摄像头服务器正在运行 (端口 $CAMERA_PORT)"
            # 尝试找到进程ID
            CAMERA_PID=$(lsof -Pi :$CAMERA_PORT -sTCP:LISTEN -t)
            echo $CAMERA_PID > /tmp/smart_home_camera.pid
            return 0
        else
            print_error "摄像头服务器启动失败"
            return 1
        fi
    fi
}

# 启动主程序
start_main() {
    log_only "启动主程序 (main.py)..."
    source venv/bin/activate
    # 设置守护进程环境变量，避免重复输出到终端
    export SECURITY_SYSTEM_DAEMON=true
    # 主程序输出重定向到日志文件
    python main.py >> logs/system.log 2>&1 &
    MAIN_PID=$!
    deactivate
    print_success "主程序已启动 (PID: $MAIN_PID)"
    echo $MAIN_PID > /tmp/smart_home_main.pid
}

# 清理旧进程
cleanup_old_processes() {
    log_only "清理旧进程..."
    
    # 先退出全屏模式
    exit_fullscreen
    
    # 等待一下让全屏退出生效
    sleep 1

    # 停止之前启动的进程
    for pid_file in /tmp/smart_home_main.pid /tmp/smart_home_camera.pid; do
        if [ -f $pid_file ]; then
            OLD_PID=$(cat $pid_file)
            if ps -p $OLD_PID > /dev/null 2>&1; then
                kill -15 $OLD_PID 2>/dev/null
                sleep 1
                if ps -p $OLD_PID > /dev/null 2>&1; then
                    kill -9 $OLD_PID 2>/dev/null
                fi
            fi
            rm $pid_file
        fi
    done
    
    # 查找并停止相关进程
    pkill -f "python.*camera_server" 2>/dev/null
    pkill -f "python.*main.py" 2>/dev/null
    
    # 释放端口
    fuser -k $WEB_PORT/tcp 2>/dev/null
    fuser -k $CAMERA_PORT/tcp 2>/dev/null
    
    log_only "清理完成"
}

# 启动浏览器（可选）- 火狐全屏模式
start_browser() {
    if [ -n "$DISPLAY" ]; then
        log_only "启动火狐浏览器 (全屏模式)..."
        
        # 检查是否安装了xdotool
        if ! command -v xdotool > /dev/null; then
            log_only "安装xdotool工具..."
            sudo apt-get update > /dev/null 2>&1 && sudo apt-get install -y xdotool > /dev/null 2>&1
        fi
        
        # 先关闭已存在的火狐实例
        pkill -f "firefox.*localhost:$WEB_PORT" 2>/dev/null
        sleep 1
        
        if command -v firefox > /dev/null; then
            # 启动火狐
            firefox --new-window "http://localhost:$WEB_PORT" > /dev/null 2>&1 &
            BROWSER_PID=$!
            
            log_only "等待浏览器启动 (3秒)..."
            sleep 3
            
            # 尝试多种方式进入全屏
            if command -v xdotool > /dev/null; then
                # 方法1: 通过窗口类查找
                WINDOW_ID=$(xdotool search --sync --onlyvisible --class "firefox" | head -1)
                
                # 方法2: 如果没找到，尝试其他可能的类名
                if [ -z "$WINDOW_ID" ]; then
                    WINDOW_ID=$(xdotool search --sync --onlyvisible --name "Firefox" | head -1)
                fi
                
                # 方法3: 尝试Navigator
                if [ -z "$WINDOW_ID" ]; then
                    WINDOW_ID=$(xdotool search --sync --onlyvisible --class "Navigator" | head -1)
                fi
                
                if [ -n "$WINDOW_ID" ]; then
                    # 保存窗口ID到文件，以便退出全屏时使用
                    echo $WINDOW_ID > /tmp/smart_home_window_id
                    
                    xdotool windowactivate $WINDOW_ID 2>/dev/null
                    sleep 0.5
                    xdotool key F11 2>/dev/null
                    log_only "已切换到全屏模式 (窗口ID: $WINDOW_ID)"
                else
                    log_only "未找到火狐窗口，尝试备用方法"
                    # 备用方法：使用wmctrl
                    if command -v wmctrl > /dev/null; then
                        wmctrl -r "Mozilla Firefox" -b toggle,fullscreen 2>/dev/null
                        # 保存窗口信息
                        wmctrl -l | grep "Mozilla Firefox" > /tmp/smart_home_window_info
                    fi
                fi
            fi
            
            # 验证进程是否还在运行
            sleep 1
            if ps -p $BROWSER_PID > /dev/null; then
                print_success "火狐浏览器已启动 (PID: $BROWSER_PID)"
                echo $BROWSER_PID > /tmp/smart_home_browser.pid
            else
                print_warning "火狐浏览器启动失败"
            fi
        else
            print_warning "未找到火狐浏览器"
            log_only "请安装火狐浏览器: sudo apt install firefox"
            
            # 尝试其他浏览器
            if command -v chromium-browser > /dev/null; then
                log_only "尝试启动 Chromium 浏览器..."
                chromium-browser --new-window --start-fullscreen "http://localhost:$WEB_PORT" > /dev/null 2>&1 &
                CHROME_PID=$!
                echo $CHROME_PID > /tmp/smart_home_browser.pid
                print_success "Chromium浏览器已启动"
            elif command -v google-chrome > /dev/null; then
                log_only "尝试启动 Chrome 浏览器..."
                google-chrome --new-window --start-fullscreen "http://localhost:$WEB_PORT" > /dev/null 2>&1 &
                CHROME_PID=$!
                echo $CHROME_PID > /tmp/smart_home_browser.pid
                print_success "Chrome浏览器已启动"
            fi
        fi
    else
        log_only "无图形界面，跳过浏览器启动"
        local_ip=$(hostname -I | awk '{print $1}')
        print_info "请在另一台电脑上访问: http://$local_ip:$WEB_PORT"
    fi
}

# 显示状态
show_status() {
    local_ip=$(hostname -I | awk '{print $1}')
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}智能安防系统状态${NC}"
    echo "=========================================="
    
    # 检查主程序
    MAIN_PID=$(ps aux | grep "python.*main.py" | grep -v "grep" | awk '{print $2}')
    if [ -n "$MAIN_PID" ]; then
        echo -e "主程序:     ${GREEN}运行中 (PID: $MAIN_PID)${NC}"
    else
        echo -e "主程序:     ${RED}未运行${NC}"
    fi
    
    # 检查摄像头服务器
    CAMERA_PID=$(ps aux | grep "python.*camera_server" | grep -v "grep" | awk '{print $2}')
    if [ -n "$CAMERA_PID" ]; then
        echo -e "摄像头:     ${GREEN}运行中 (PID: $CAMERA_PID)${NC}"
    else
        # 检查端口
        if lsof -Pi :$CAMERA_PORT -sTCP:LISTEN -t >/dev/null ; then
            echo -e "摄像头:     ${GREEN}运行中 (端口 $CAMERA_PORT)${NC}"
        else
            echo -e "摄像头:     ${RED}未运行${NC}"
        fi
    fi
    
    # 检查端口
    echo ""
    echo "端口状态:"
    if lsof -Pi :$WEB_PORT -sTCP:LISTEN -t >/dev/null ; then
        echo -e "端口 $WEB_PORT (主界面): ${GREEN}已占用${NC}"
    else
        echo -e "端口 $WEB_PORT (主界面): ${RED}空闲${NC}"
    fi
    
    echo ""
    echo "访问地址:"
    echo "  主界面:   http://localhost:$WEB_PORT"
    echo "           http://$local_ip:$WEB_PORT"
    echo "  摄像头:   http://localhost:$CAMERA_PORT"
    echo "           http://$local_ip:$CAMERA_PORT"
    echo "  视频流:   http://$local_ip:$CAMERA_PORT/video_feed"
    echo ""
    echo "日志文件: $LOG_FILE"
    echo "=========================================="
}

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项："
    echo "  start    启动所有服务（主程序 + 摄像头）"
    echo "  stop     停止所有服务"
    echo "  restart  重启所有服务"
    echo "  status   显示服务状态"
    echo "  help     显示帮助信息"
    echo ""
    echo "示例："
    echo "  $0 start   # 启动所有服务"
    echo "  $0 stop    # 停止所有服务"
}

# 主函数
main() {
    case "$1" in
        stop)
            echo -e "${YELLOW}正在停止所有服务...${NC}"
            cleanup_old_processes
            sudo pkill -f python 2>/dev/null
            
            exit_fullscreen
            
            # 等待一下
            sleep 1
            
            # 再次检查是否有残留的浏览器窗口
            if command -v wmctrl > /dev/null; then
                wmctrl -r "Mozilla Firefox" -b remove,fullscreen 2>/dev/null
            fi
            
            echo -e "${GREEN}✅ 所有服务已停止${NC}"
            ;;
        restart)
            echo -e "${YELLOW}正在重启所有服务...${NC}"
            cleanup_old_processes
            sleep 2
            "$0" start
            ;;
        status)
            show_status
            ;;
        help|"")
            show_help
            ;;
        start)
            echo "=========================================="
            echo -e "${GREEN}智能安防系统启动中...${NC}"
            echo "=========================================="
            
            # 清空旧日志文件，但保留到logs目录
            > logs/system.log
            
            # 清理旧进程
            cleanup_old_processes
            
            # 创建必要的目录
            create_dirs
            
            # 检查Web文件
            check_web_files
            
            # 检查端口
            check_port $WEB_PORT
            check_port $CAMERA_PORT
            
            # 启动摄像头服务器
            start_camera_server
            
            # 等待摄像头服务器完全启动
            sleep 3
            
            # 启动主程序
            start_main
            
            # 等待主程序完全启动
            sleep 2
            
            # 启动浏览器
            start_browser

            exit_fullscreen
            
            # 显示状态
            show_status
            
            echo ""
            echo -e "${GREEN}✅ 所有服务已启动${NC}"
            echo "日志位置: logs/system.log"
            echo "查看日志: tail -f logs/system.log"
            ;;
        *)
            print_error "未知参数 '$1'"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"