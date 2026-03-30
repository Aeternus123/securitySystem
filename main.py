#!/usr/bin/env python3
"""
智能安防系统主程序 - 简化版（无视频流，纯HTTP轮询）
"""
import sys
import select
import time
import threading
import cv2
import json
import http.server
import socketserver
import os
from datetime import datetime
from urllib.parse import urlparse

from config.settings import *
from config.voice_commands import (
    OP_OPEN, OP_CLOSE, OP_MOTOR_ON, OP_MOTOR_OFF, OP_WAKEUP, OP_VOICE_ON, OP_VOICE_OFF,
    MASTER_BEDROOM_PREFIX, GUEST_BEDROOM_PREFIX, MOTOR_PREFIX, WAKEUP_PREFIX, VOICE_CONTROL_PREFIX
)
from utils.logger import logger
from modules.gpio_controller import GPIOController
from modules.voice_module import VoiceModule
from modules.face_detector import FaceDetector
from modules.face_enroller import FaceEnroller
from modules.ultrasonic import UltrasonicSensor
from modules.alarm import AlarmSystem
from modules.sensors import SensorManager
from modules.camera_opencv import CameraStreamOpenCV as CameraStream

# # 全局变量，用于HTTP状态服务器
# waiting_clients = []
# status_lock = threading.Lock()
# system_status = {}

# ==================== 智能安防系统主类 ====================
class SecuritySystem:
    """智能安防系统"""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("智能安防系统启动 - 简化版")
        logger.info("=" * 60)
        
        # 初始化GPIO控制器
        self.gpio = GPIOController()
        
        # 初始化人脸检测器
        try:
            self.face_detector = FaceDetector()
            logger.info(f"人脸检测器初始化成功")
        except Exception as e:
            logger.error(f"人脸检测器初始化失败: {e}")
            self.face_detector = None
        
        # 初始化报警系统
        self.alarm = AlarmSystem(self.gpio, self.face_detector)
        
        # 初始化传感器管理器
        self.sensors = SensorManager(self.alarm)
        
        # 初始化超声波传感器
        self.ultrasonic = UltrasonicSensor()
        
        # 人脸录入器
        self.face_enroller = FaceEnroller(self.face_detector) if self.face_detector else None
        
        # 语音模块（需要回调）
        self.voice = VoiceModule(self.handle_voice_command)
        
        # 摄像头（只用于人脸识别，不用于视频流）
        self.cap = self.init_camera()
        
        # 系统状态
        self.system_locked = True
        self.voice_control_enabled = False
        self.face_recognition_active = False
        self.face_recognition_start_time = 0
        
        # 设备状态字典（用于空调等虚拟设备）
        self.device_states = {
            'ac': False
        }
        
        # HTTP服务器线程
        self.http_thread = None
        
        # 摄像头流 - 使用独立服务器
        self.camera_stream = None
        logger.info("📹 使用独立摄像头服务器 (端口 5000)")
        
        # 线程控制
        self.running = True
        self.threads = []
        
        if self.face_detector:
            logger.info(f"人脸数据库: {len(self.face_detector.face_database)} 人")
        else:
            logger.warning("人脸识别功能不可用")
        
        logger.info("系统初始化完成")
        logger.info("注意: 只有解锁后才能进行人脸录入")
    
    def init_camera(self):
        """初始化摄像头 - 只用于人脸识别"""
        import subprocess
        
        # 通过设备ID直接找到RGB摄像头的设备节点
        try:
            # 查找 RGB 摄像头 (2bc5:050f) 对应的 /dev/video*
            result = subprocess.run(
                ['v4l2-ctl', '--list-devices'], 
                capture_output=True, text=True
            )
            
            # 解析输出，找到包含 "USB 2.0 Camera" 的设备
            lines = result.stdout.split('\n')
            current_device = None
            
            for line in lines:
                if 'USB 2.0 Camera' in line:
                    current_device = True
                    continue
                if '/dev/video' in line and current_device:
                    video_dev = line.strip()
                    logger.info(f"找到RGB摄像头: {video_dev}")
                    
                    # 尝试打开这个设备
                    cap = cv2.VideoCapture(video_dev, cv2.CAP_V4L2)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                        logger.info(f"摄像头已打开: {video_dev}")
                        return cap
                if line.strip() == '':
                    current_device = None
        except Exception as e:
            logger.warning(f"自动查找摄像头失败: {e}")
        
        # 如果自动查找失败，尝试手动指定
        for camera_id in [0, 2, 4]:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    logger.info(f"摄像头已打开 (video{camera_id})")
                    return cap
                cap.release()
        
        logger.error("无法打开任何可用摄像头")
        return None
    
    def get_status_dict(self):
        """获取系统状态字典"""
        if self.face_enroller:
            stats = self.face_enroller.get_face_stats()
        else:
            stats = {'total_persons': 0, 'total_samples': 0, 'feature_dim': 0}
        
        # 获取设备状态
        devices = {
            'ac': self.device_states.get('ac', False),
            'mainLight': self.gpio.get_device_state('红灯') if hasattr(self.gpio, 'get_device_state') else False,
            'bedroomLight': self.gpio.get_device_state('绿灯') if hasattr(self.gpio, 'get_device_state') else False,
            'smoke': self.sensors.sensor_data.get('smoke_digital', 0) == 1 if hasattr(self.sensors, 'sensor_data') else False
        }
        
        # 获取传感器数据
        sensors = {
            'temperature': self.sensors.sensor_data.get('temperature', 0) if hasattr(self.sensors, 'sensor_data') else 0,
            'humidity': self.sensors.sensor_data.get('humidity', 0) if hasattr(self.sensors, 'sensor_data') else 0
        }
        
        # 摄像头状态 - 检查独立服务器
        camera_online = False
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 5000))
            camera_online = (result == 0)
            sock.close()
        except:
            camera_online = False
        
        return {
            'system_locked': self.system_locked,
            'voice_control_enabled': self.voice_control_enabled,
            'face_recognition_active': self.face_recognition_active,
            'alarm_active': self.alarm.is_active() if hasattr(self.alarm, 'is_active') else False,
            'alarm_source': self.alarm.get_source() if hasattr(self.alarm, 'get_source') else None,
            'devices': devices,
            'sensors': sensors,
            'face_database': stats,
            'camera': {
                'online': camera_online,
                'clients': 0,
                'url': 'http://172.20.10.8:5000/video_feed'
            },
            'timestamp': time.time()
        }
    
    # def update_system_status(self):
    #     """更新全局系统状态"""
    #     global system_status
    #     system_status = self.get_status_dict()
    
    # def notify_clients(self):
    #     """通知所有等待的HTTP客户端"""
    #     global waiting_clients
    #     with status_lock:
    #         for client in waiting_clients:
    #             try:
    #                 client.request.shutdown()
    #             except:
    #                 pass
    #         waiting_clients.clear()
    
    def start_http_server(self):
        """启动HTTP服务器 - 集成摄像头流和API服务"""
        
        import http.server
        import socketserver
        import json
        import threading
        import os
        import time
        from urllib.parse import urlparse
        
        web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
        logger.info(f"Web目录: {web_dir}")
        
        # 确保web目录存在
        if not os.path.exists(web_dir):
            os.makedirs(web_dir, exist_ok=True)
            logger.info(f"创建web目录: {web_dir}")
        
        # 启动摄像头流（如果在__init__中已初始化）
        if hasattr(self, 'camera_stream') and self.camera_stream:
            if not self.camera_stream.running:
                self.camera_stream.start()
                logger.info(f"✅ 摄像头流已启动")
            else:
                logger.info(f"✅ 摄像头流已在运行中")
        else:
            logger.warning(f"⚠️ 摄像头流不可用，视频功能将受限")
        
        class IntegratedHandler(http.server.SimpleHTTPRequestHandler):
            """集成处理器 - 同时处理API请求、静态文件和摄像头流"""
            
            def __init__(self, *args, **kwargs):
                self.system = None
                super().__init__(*args, directory=web_dir, **kwargs)
            
            def log_message(self, format, *args):
                # 安全地处理日志
                try:
                    # 只记录错误（状态码 >= 400）
                    if len(args) >= 1:
                        status_str = str(args[0])
                        if status_str.isdigit() and int(status_str) >= 400:
                            message = format % args
                            logger.info(f"HTTP {status_str}: {self.path} - {message}")
                except Exception:
                    pass
            
            def do_GET(self):
                # 获取system引用
                if hasattr(self.server, 'system'):
                    self.system = self.server.system
                
                # 解析路径
                parsed = urlparse(self.path)
                path = parsed.path
                
                try:
                    # 摄像头流请求
                    if path == '/video_feed' or path.startswith('/video_feed'):
                        self._handle_video_feed()
                        return
                    
                    # 摄像头全屏页面
                    if path == '/camera':
                        self._handle_camera_page()
                        return
                    
                    # API请求
                    if path == '/api/status':
                        self._handle_status()
                        return
                    
                    # 根路径
                    if path == '/' or path == '':
                        self.path = '/index.html'
                    
                    # 提供静态文件
                    return super().do_GET()
                    
                except Exception as e:
                    logger.error(f"文件服务错误 {path}: {e}")
                    self.send_error(404)
            
            def do_POST(self):
                # 获取system引用
                if hasattr(self.server, 'system'):
                    self.system = self.server.system
                
                parsed = urlparse(self.path)
                path = parsed.path
                
                if path == '/api/command':
                    self._handle_command()
                else:
                    self.send_error(404)
            
            def _handle_video_feed(self):
                """处理视频流请求"""
                try:
                    if self.system and hasattr(self.system, 'camera_stream') and self.system.camera_stream and self.system.camera_stream.running:
                        # 标记客户端连接
                        self.system.camera_stream.client_connected()
                        
                        # 设置响应头
                        self.send_response(200)
                        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                        self.send_header('Pragma', 'no-cache')
                        self.send_header('Expires', '0')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        # 发送视频流
                        try:
                            for frame in self.system.camera_stream.generate_frames():
                                self.wfile.write(frame)
                                self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            # 客户端断开连接
                            logger.debug("摄像头客户端断开连接")
                        except Exception as e:
                            logger.error(f"视频流发送错误: {e}")
                        finally:
                            self.system.camera_stream.client_disconnected()
                    else:
                        # 摄像头不可用，返回占位图
                        self._send_placeholder_feed()
                        
                except Exception as e:
                    logger.error(f"视频流处理失败: {e}")
                    try:
                        self._send_placeholder_feed()
                    except:
                        pass
            
            def _send_placeholder_feed(self):
                """发送占位视频流"""
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    
                    # 生成占位帧
                    import numpy as np
                    import cv2
                    
                    while True:
                        # 创建占位图像
                        img = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(img, "Camera Offline", (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        ret, jpeg = cv2.imencode('.jpg', img)
                        frame = jpeg.tobytes()
                        
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                        time.sleep(0.1)
                except Exception:
                    pass
            
            def _handle_camera_page(self):
                """处理摄像头全屏页面"""
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>树莓派摄像头 - 全屏模式</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        * { margin: 0; padding: 0; box-sizing: border-box; }
                        body { 
                            font-family: Arial, sans-serif; 
                            background: #000; 
                            color: white;
                            height: 100vh;
                            display: flex;
                            flex-direction: column;
                        }
                        .header {
                            padding: 15px;
                            background: #1a1a1a;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        }
                        .camera-container {
                            flex: 1;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            background: #000;
                        }
                        img {
                            width: 100%;
                            height: 100%;
                            object-fit: contain;
                        }
                        .status-online { color: #4CAF50; }
                        .status-offline { color: #f44336; }
                        .back-btn {
                            padding: 8px 20px;
                            background: #2196F3;
                            color: white;
                            text-decoration: none;
                            border-radius: 5px;
                            font-size: 16px;
                        }
                        .back-btn:hover { background: #1976D2; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>📹 摄像头全屏模式</h1>
                        <div>
                            <span id="status">状态: <span class="status-online">在线</span></span>
                            <a href="/" class="back-btn" style="margin-left: 20px;">返回主界面</a>
                        </div>
                    </div>
                    <div class="camera-container">
                        <img src="/video_feed" id="cameraFeed" 
                            onload="document.querySelector('#status span').className='status-online'; document.querySelector('#status span').textContent='在线'"
                            onerror="this.onerror=null; document.querySelector('#status span').className='status-offline'; document.querySelector('#status span').textContent='离线'">
                    </div>
                </body>
                </html>
                """
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(html.encode())
            
            def _handle_status(self):
                """处理状态请求 - 立即返回"""
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.end_headers()
                    
                    # 获取状态
                    if self.system:
                        status = self.system.get_status_dict()
                    else:
                        status = {
                            'system_locked': True,
                            'sensors': {'temperature': 0, 'humidity': 0},
                            'devices': {
                                'ac': False,
                                'mainLight': False,
                                'bedroomLight': False,
                                'smoke': False
                            },
                            'camera': {
                                'online': False,
                                'clients': 0,
                                'url': '/video_feed'
                            },
                            'timestamp': time.time()
                        }
                    
                    self.wfile.write(json.dumps(status).encode())
                    
                except Exception as e:
                    logger.error(f"状态请求失败: {e}")
                    try:
                        self.wfile.write(json.dumps({'error': str(e)}).encode())
                    except:
                        pass
            
            def _handle_command(self):
                """处理命令请求"""
                try:
                    content_length = int(self.headers.get('Content-Length', 0))
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data)
                    command = data.get('command')
                    
                    logger.info(f"收到命令: {command}")
                    
                    if not self.system:
                        response = {'success': False, 'message': '系统未初始化'}
                    elif command == 'password':
                        password = data.get('password', '')
                        if password == '123456':
                            self.system.system_locked = False
                            self.system.voice_control_enabled = True
                            # 解锁时蜂鸣提示
                            if hasattr(self.system, 'gpio'):
                                self.system.gpio.beep(2)
                            response = {'success': True, 'message': '✅ 解锁成功'}
                        else:
                            response = {'success': False, 'message': '❌ 密码错误'}
                    elif command == 'voice_command':
                        cmd = data.get('voice_cmd')
                        if cmd == 'open_voice':
                            self.system.face_recognition_active = True
                            self.system.face_recognition_start_time = time.time()
                            if hasattr(self.system, 'gpio'):
                                self.system.gpio.beep(1)
                            response = {'success': True, 'message': '🎤 人脸识别已激活'}
                        else:
                            response = {'success': False, 'message': '未知语音命令'}
                    elif command == 'control':
                        device = data.get('device')
                        action = data.get('action')
                        
                        if self.system.system_locked:
                            response = {'success': False, 'message': '系统已锁定'}
                        else:
                            if device == 'ac':
                                new_state = (action == 'on')
                                self.system.device_states['ac'] = new_state
                                # 如果有电机，控制电机
                                if hasattr(self.system, 'gpio'):
                                    self.system.gpio.set_device('电机', new_state)
                                response = {'success': True, 'message': f"空调已{'开启' if new_state else '关闭'}"}
                            elif device == 'mainLight':
                                state = (action == 'on')
                                if hasattr(self.system, 'gpio'):
                                    self.system.gpio.set_device('红灯', state)
                                response = {'success': True, 'message': f"主卧灯已{'开启' if state else '关闭'}"}
                            elif device == 'bedroomLight':
                                state = (action == 'on')
                                if hasattr(self.system, 'gpio'):
                                    self.system.gpio.set_device('绿灯', state)
                                response = {'success': True, 'message': f"客房灯已{'开启' if state else '关闭'}"}
                            else:
                                response = {'success': False, 'message': '未知设备'}
                    elif command == 'camera_control':
                        action = data.get('action')
                        if action == 'restart':
                            if hasattr(self.system, 'camera_stream') and self.system.camera_stream:
                                self.system.camera_stream.stop()
                                time.sleep(1)
                                self.system.camera_stream.start()
                                response = {'success': True, 'message': '摄像头已重启'}
                            else:
                                response = {'success': False, 'message': '摄像头未初始化'}
                        else:
                            response = {'success': False, 'message': '未知摄像头命令'}
                    else:
                        response = {'success': False, 'message': '未知命令'}
                    
                except json.JSONDecodeError:
                    logger.error("命令请求JSON解析失败")
                    response = {'success': False, 'message': '无效的JSON数据'}
                except Exception as e:
                    logger.error(f"命令处理失败: {e}")
                    response = {'success': False, 'message': str(e)}
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
        
        class ThreadedHTTPServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
            
            def __init__(self, server_address, RequestHandlerClass, system):
                self.system = system
                super().__init__(server_address, RequestHandlerClass)
        
        try:
            # 确保端口可用
            import socket
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                test_sock.bind(('0.0.0.0', 8000))
                test_sock.close()
            except socket.error:
                logger.warning("端口8000被占用，尝试释放...")
                import subprocess
                subprocess.run(['sudo', 'fuser', '-k', '8000/tcp'], 
                            capture_output=True)
                time.sleep(2)
            
            # 启动服务器
            server = ThreadedHTTPServer(("0.0.0.0", 8000), IntegratedHandler, self)
            self.http_thread = threading.Thread(target=server.serve_forever)
            self.http_thread.daemon = True
            self.http_thread.start()
            
            # 验证服务器是否启动
            time.sleep(1)
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if test_sock.connect_ex(('127.0.0.1', 8000)) == 0:
                logger.info(f"✅ HTTP服务器启动成功: http://0.0.0.0:8000")
                logger.info(f"   📁 静态文件: {web_dir}")
                logger.info(f"   📹 视频流: http://0.0.0.0:8000/video_feed")
                logger.info(f"   🖥️ 全屏模式: http://0.0.0.0:8000/camera")
            else:
                logger.error(f"❌ HTTP服务器启动验证失败")
            test_sock.close()
            
        except Exception as e:
            logger.error(f"❌ 启动HTTP服务器失败: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_voice_command(self, operation, prefix):
        """处理语音命令"""
        if prefix == VOICE_CONTROL_PREFIX:
            if operation == OP_VOICE_ON:
                if not self.face_recognition_active and not self.voice_control_enabled:
                    logger.info("语音控制激活，请进行人脸识别")
                    self.face_recognition_active = True
                    self.face_recognition_start_time = time.time()
                    self.gpio.beep(2)
                    # 移除状态更新和通知
                    # self.update_system_status()
                    # self.notify_clients()
            
            elif operation == OP_VOICE_OFF:
                if not self.system_locked:
                    logger.info("系统手动锁定")
                    self.system_locked = True
                    self.voice_control_enabled = False
                    self.face_recognition_active = False
                    self.gpio.set_device('红灯', True)
                    time.sleep(0.3)
                    self.gpio.set_device('红灯', False)
                    self.gpio.beep(1)
                    # 移除状态更新和通知
                    # self.update_system_status()
                    # self.notify_clients()
        
        elif prefix == WAKEUP_PREFIX:
            if not self.alarm.is_active():
                self.gpio.set_device('绿灯', True)
                time.sleep(0.1)
                self.gpio.set_device('绿灯', False)
        
        elif self.system_locked:
            logger.info("系统已锁定，请先说'打开语音'")
            self.gpio.set_device('红灯', True)
            time.sleep(0.2)
            self.gpio.set_device('红灯', False)
        
        else:
            self.handle_device_command(operation, prefix)
    
    def handle_device_command(self, operation, prefix):
        """处理设备控制命令"""
        if prefix == MOTOR_PREFIX:
            if operation == OP_MOTOR_ON:
                self.gpio.set_device('电机', True)
                self.device_states['ac'] = True
                logger.info(f"空调已开启 (通过语音)")
            elif operation == OP_MOTOR_OFF:
                self.gpio.set_device('电机', False)
                self.device_states['ac'] = False
                logger.info(f"空调已关闭 (通过语音)")
            # 移除状态更新和通知
            # self.update_system_status()
            # self.notify_clients()
        
        elif prefix == MASTER_BEDROOM_PREFIX:
            if operation == OP_OPEN:
                self.gpio.set_device('红灯', True)
                logger.info("主卧灯开启")
            elif operation == OP_CLOSE:
                self.gpio.set_device('红灯', False)
                logger.info("主卧灯关闭")
            # 移除状态更新和通知
            # self.update_system_status()
            # self.notify_clients()
        
        elif prefix == GUEST_BEDROOM_PREFIX:
            if operation == OP_OPEN:
                self.gpio.set_device('绿灯', True)
                logger.info("客房灯开启")
            elif operation == OP_CLOSE:
                self.gpio.set_device('绿灯', False)
                logger.info("客房灯关闭")
            # 移除状态更新和通知
            # self.update_system_status()
            # self.notify_clients()
    
    def ultrasonic_monitor(self):
        """超声波监控线程"""
        logger.info("超声波监控已启动")
        
        while self.running:
            distance = self.ultrasonic.measure_distance()
            
            if distance and distance < ULTRASONIC_DISTANCE_THRESHOLD:
                if not self.alarm.is_active():
                    logger.warning(f"超声波检测到距离过近: {distance:.1f}cm")
                    self.alarm.trigger('ultrasonic')
                    # 移除状态更新
                    # self.update_system_status()
                    # self.notify_clients()
            
            elif distance and distance >= ULTRASONIC_DISTANCE_THRESHOLD:
                if self.alarm.is_active() and self.alarm.get_source() == 'ultrasonic':
                    logger.info("超声波距离恢复正常")
                    self.alarm.stop()
                    # 移除状态更新
                    # self.update_system_status()
                    # self.notify_clients()
            
            time.sleep(ULTRASONIC_CHECK_INTERVAL)
    
    def face_monitor(self):
        """人脸监控线程 - 从摄像头服务器获取画面"""
        if not self.face_detector:
            logger.warning("人脸检测器未初始化，人脸监控已禁用")
            return
        
        logger.info("人脸监控已启动（使用摄像头服务器）")
        
        # 导入requests库
        import requests
        import cv2
        import numpy as np
        from io import BytesIO
        
        frame_count = 0
        error_count = 0
        camera_url = "http://localhost:5000/video_feed"
        
        # 简单的MJPEG流读取器
        class MJPEGReader:
            def __init__(self, url):
                self.url = url
                self.stream = None
                self.bytes_buffer = bytes()
                self.running = True
                
            def start(self):
                try:
                    import requests
                    self.stream = requests.get(self.url, stream=True, timeout=5)
                    if self.stream.status_code == 200:
                        logger.info("✅ 成功连接到摄像头服务器")
                        return True
                    else:
                        logger.error(f"❌ 连接摄像头服务器失败: {self.stream.status_code}")
                        return False
                except Exception as e:
                    logger.error(f"❌ 连接摄像头服务器异常: {e}")
                    return False
            
            def read_frame(self):
                try:
                    if not self.stream:
                        return None
                    
                    # 查找JPEG帧边界
                    while self.running:
                        self.bytes_buffer += self.stream.raw.read(1024)
                        
                        # 查找帧开始
                        start = self.bytes_buffer.find(b'\xff\xd8')
                        end = self.bytes_buffer.find(b'\xff\xd9')
                        
                        if start != -1 and end != -1 and end > start:
                            jpeg_data = self.bytes_buffer[start:end+2]
                            self.bytes_buffer = self.bytes_buffer[end+2:]
                            
                            # 解码JPEG
                            nparr = np.frombuffer(jpeg_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            return frame
                        
                        # 防止无限循环
                        if len(self.bytes_buffer) > 1000000:  # 1MB限制
                            self.bytes_buffer = bytes()
                            
                except Exception as e:
                    logger.error(f"读取帧错误: {e}")
                    return None
            
            def stop(self):
                self.running = False
                if self.stream:
                    self.stream.close()
        
        # 初始化MJPEG读取器
        mjpeg_reader = MJPEGReader(camera_url)
        if not mjpeg_reader.start():
            logger.error("❌ 无法连接到摄像头服务器，人脸监控将不可用")
            return
        
        while self.running:
            try:
                # 从摄像头服务器读取帧
                frame = mjpeg_reader.read_frame()
                
                if frame is not None:
                    error_count = 0
                    
                    if self.face_recognition_active:
                        frame_count += 1
                        
                        # 检查是否超时
                        if time.time() - self.face_recognition_start_time > FACE_RECOGNITION_TIMEOUT:
                            logger.info("人脸识别激活超时，自动关闭")
                            self.face_recognition_active = False
                            continue
                        
                        # 每3帧处理一次
                        if frame_count % 3 == 0:
                            # 调整大小加快处理速度
                            small_frame = cv2.resize(frame, (320, 240))
                            
                            # 检测和识别
                            has_stranger, results = self.face_detector.check_frame(small_frame)
                            
                            # 处理识别结果
                            recognized = [r for r in results if r['recognized']]
                            if recognized and self.system_locked:
                                names = [r['name'] for r in recognized]
                                logger.info(f"✅ 人脸识别成功: {', '.join(names)}")
                                
                                self.system_locked = False
                                self.voice_control_enabled = True
                                self.face_recognition_active = False
                                
                                # 解锁提示
                                for _ in range(2):
                                    self.gpio.set_device('绿灯', True)
                                    time.sleep(0.2)
                                    self.gpio.set_device('绿灯', False)
                                    time.sleep(0.2)
                                
                                print("\n" + "=" * 50)
                                print("🔓 系统已解锁")
                                print("📸 现在可以按 'e' 键进入人脸录入模式")
                                print("=" * 50)
                            
                            # 陌生人报警
                            if has_stranger and not self.alarm.is_active():
                                current_time = time.time()
                                cooldown = self.face_detector.stranger_cooldown
                                if current_time - self.face_detector.last_stranger_time > cooldown:
                                    logger.warning("⚠️ 检测到陌生人！")
                                    self.alarm.trigger('face_stranger')
                                    self.face_detector.last_stranger_time = current_time
                else:
                    error_count += 1
                    if error_count > 10:
                        logger.warning("摄像头服务器连接丢失，尝试重连...")
                        mjpeg_reader.stop()
                        time.sleep(2)
                        if not mjpeg_reader.start():
                            logger.error("重连摄像头服务器失败")
                        error_count = 0
                
            except Exception as e:
                logger.error(f"人脸监控错误: {e}")
                error_count += 1
                time.sleep(0.5)
        
        mjpeg_reader.stop()
        logger.info("人脸监控已停止")
    
    def enrollment_mode(self):
        """人脸录入模式"""
        if self.system_locked:
            print("\n🔒 系统已锁定，请先解锁才能录入人脸")
            print("请说'打开语音'进行人脸识别解锁")
            self.gpio.set_device('红灯', True)
            time.sleep(0.3)
            self.gpio.set_device('红灯', False)
            return
        
        if not self.cap or not self.face_detector or not self.face_enroller:
            logger.error("人脸识别功能不可用")
            return
        
        logger.info("=" * 50)
        logger.info("进入人脸录入模式")
        logger.info("操作说明:")
        logger.info("  按 Enter 键保存当前检测到的人脸")
        logger.info("  输入 'q' 退出录入模式")
        logger.info("  输入 'b' 进入批量录入模式")
        
        self.broadcast_status()
        
        frame_count = 0
        detection_cooldown = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            detection_cooldown = max(0, detection_cooldown - 1)
            
            if frame_count % 10 == 0 and detection_cooldown == 0:
                boxes = self.face_detector.detect_faces(frame)
                
                if boxes:
                    print(f"\n✅ 检测到人脸！位置: {boxes[0]}")
                    print("选项:")
                    print("  1. 录入这个人")
                    print("  2. 跳过")
                    print("  3. 批量录入")
                    print("  4. 退出录入模式")
                    
                    choice = input("请选择 (1-4): ").strip()
                    
                    if choice == '1':
                        name = input("请输入这个人的人名: ").strip()
                        if name:
                            if self.face_enroller.add_face_from_frame(frame, boxes[0], name):
                                logger.info(f"✅ 已保存 {name} 的人脸")
                                self.gpio.beep(1)
                                self.broadcast_status()
                            detection_cooldown = 20
                    
                    elif choice == '2':
                        print("⏭️ 跳过")
                        detection_cooldown = 10
                    
                    elif choice == '3':
                        self.batch_enrollment_mode()
                    
                    elif choice == '4':
                        break
                else:
                    if frame_count % 30 == 0:
                        print(".", end='', flush=True)
            
            if select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'q':
                    break
                elif cmd == 'b':
                    self.batch_enrollment_mode()
        
        logger.info("退出人脸录入模式")
        self.broadcast_status()
    
    def batch_enrollment_mode(self):
        """批量录入模式"""
        if self.system_locked:
            print("\n🔒 系统已锁定，请先解锁才能录入人脸")
            return
        
        name = input("\n请输入这个人的人名: ").strip()
        if not name:
            return
        
        try:
            samples = input("请输入要采集的样本数 (默认5): ").strip()
            samples = int(samples) if samples else 5
        except:
            samples = 5
        
        logger.info(f"开始批量录入 {name}，目标样本数: {samples}")
        logger.info("请转动头部，从不同角度采集")
        logger.info("检测到人脸时按 Enter 保存，输入 'q' 退出")
        
        self.broadcast_status()
        
        collected = 0
        timeout = time.time() + 60
        frame_count = 0
        
        while collected < samples and time.time() < timeout:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                boxes = self.face_detector.detect_faces(frame)
                
                if boxes:
                    print(f"\n✅ 检测到人脸！当前进度: {collected}/{samples}")
                    
                    if select.select([sys.stdin], [], [], 0)[0]:
                        cmd = sys.stdin.readline().strip().lower()
                        if cmd == '' or cmd == 'y':
                            if self.face_enroller.add_face_from_frame(frame, boxes[0], name):
                                collected += 1
                                logger.info(f"已采集 {collected}/{samples} 张")
                                self.gpio.beep(1)
                                self.broadcast_status()
                        elif cmd == 'q':
                            break
                else:
                    if frame_count % 30 == 0:
                        print(".", end='', flush=True)
            
            time.sleep(0.01)
        
        if collected >= samples:
            logger.info(f"✅ {name} 的人脸采集完成，共 {collected} 张")
        else:
            logger.warning(f"⚠️ {name} 的人脸采集未完成，只采集了 {collected} 张")
        
        self.broadcast_status()
    
    # def broadcast_status(self):
    #     """广播状态到所有客户端"""
    #     self.update_system_status()
    #     self.notify_clients()
    
    def print_status(self):
        """打印系统状态"""
        status = self.get_status_dict()
        
        print("\n" + "=" * 60)
        print("系统状态:")
        print("=" * 60)
        print(f"系统锁定: {'🔒 是' if status['system_locked'] else '🔓 否'}")
        print(f"语音控制: {'✅ 启用' if status['voice_control_enabled'] else '❌ 禁用'}")
        print(f"人脸识别: {'🟢 激活' if status['face_recognition_active'] else '⚫ 待机'}")
        
        alarm_source = status['alarm_source']
        alarm_status = "🚨 启动" if status['alarm_active'] else "🔕 关闭"
        if alarm_source:
            source_map = {
                'ultrasonic': '超声波', 
                'face_stranger': '陌生人', 
                'manual': '手动',
                'smoke': '烟雾',
                'sensor': '温湿度'
            }
            source_name = source_map.get(alarm_source, alarm_source)
            alarm_status += f" (触发源: {source_name})"
        print(f"报警系统: {alarm_status}")
        
        sensor_status = self.sensors.get_status_string()
        if sensor_status:
            print(f"传感器状态: {sensor_status}")
        
        stats = status['face_database']
        print(f"人脸数据库: {stats['total_persons']} 人, {stats['total_samples']} 样本")
        if stats['feature_dim'] > 0:
            print(f"特征维度: {stats['feature_dim']}")
        print("=" * 60)
    
    def start(self):
        """启动系统"""
        # 启动语音模块
        if not self.voice.start():
            logger.error("语音模块启动失败")
            return
        
        # 启动传感器监控
        self.sensors.start()
        
        # 启动HTTP服务器
        self.start_http_server()
        
        # 启动监控线程
        threads = [
            threading.Thread(target=self.ultrasonic_monitor, name="Ultrasonic"),
        ]
        
        if self.face_detector:
            threads.append(threading.Thread(target=self.face_monitor, name="Face"))
        
        for t in threads:
            t.daemon = True
            t.start()
            self.threads.append(t)
        
        logger.info("所有监控线程已启动")
        logger.info("系统初始状态: 锁定")
        logger.info("=" * 60)
        logger.info("键盘命令:")
        logger.info("  s - 查看状态")
        logger.info("  m - 手动报警")
        logger.info("  c - 清除报警")
        logger.info("  u - 手动解锁")
        if self.face_detector:
            logger.info("  e - 人脸录入模式（需解锁后使用）")
            logger.info("  l - 列出数据库")
            logger.info("  d - 删除人脸")
        logger.info("  q - 退出")
        logger.info("=" * 60)
        logger.info("Web界面: http://树莓派IP:8000")
        
        try:
            while self.running:
                if select.select([sys.stdin], [], [], 0)[0]:
                    cmd = sys.stdin.readline().strip().lower()
                    self.handle_keyboard_command(cmd)
                
                self.voice.process_data()
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("程序中断")
        finally:
            self.stop()
    
    def handle_keyboard_command(self, cmd):
        """处理键盘命令"""
        if cmd == 'q':
            self.running = False
        elif cmd == 's':
            self.print_status()
        elif cmd == 'm':
            self.alarm.trigger('manual')
            self.gpio.beep(3)
            # 移除状态更新
            # self.broadcast_status()
        elif cmd == 'c':
            self.alarm.stop()
            # 移除状态更新
            # self.broadcast_status()
        elif cmd == 'u':
            if self.system_locked:
                logger.info("手动解锁")
                self.system_locked = False
                self.voice_control_enabled = True
                for _ in range(2):
                    self.gpio.set_device('绿灯', True)
                    time.sleep(0.2)
                    self.gpio.set_device('绿灯', False)
                    time.sleep(0.2)
                print("\n🔓 系统已手动解锁")
                print("📸 现在可以按 'e' 键进入人脸录入模式")
                # 移除状态更新
                # self.broadcast_status()
        elif cmd == 'e' and self.face_detector:
            self.enrollment_mode()
        elif cmd == 'l' and self.face_detector:
            self.face_enroller.list_faces()
        elif cmd == 'd' and self.face_detector:
            if not self.face_detector.face_database:
                print("数据库为空")
                return
            
            print("\n可删除的人脸:")
            names = list(self.face_detector.face_database.keys())
            for i, name in enumerate(names, 1):
                samples = len(self.face_detector.face_database[name]['features'])
                print(f"  {i}. {name} ({samples} 个样本)")
            
            try:
                idx = int(input("\n请输入要删除的编号 (0取消): ").strip())
                if 1 <= idx <= len(names):
                    self.face_enroller.delete_face(names[idx-1])
                    # 移除状态更新
                    # self.broadcast_status()
            except:
                print("输入无效")
    
    def stop(self):
        """停止系统"""
        logger.info("正在关闭系统...")
        
        self.running = False
        self.alarm.stop()
        self.voice.stop()
        self.sensors.stop()
        
        # 停止摄像头流（如果存在）
        if hasattr(self, 'camera_stream') and self.camera_stream:
            self.camera_stream.stop()
            logger.info("摄像头流已停止")
        
        for t in self.threads:
            t.join(timeout=1)
        
        if self.cap:
            self.cap.release()
        
        self.gpio.cleanup()
        logger.info("系统已关闭")

# ==================== 程序入口 ====================
if __name__ == "__main__":
    system = SecuritySystem()
    system.start()