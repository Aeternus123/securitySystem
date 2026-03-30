#!/usr/bin/env python3
"""
摄像头流媒体模块 - 纯OpenCV版本
专门适配Orbbec USB摄像头
"""
import threading
import time
import cv2
import numpy as np
from utils.logger import logger

class CameraStreamOpenCV:
    """摄像头流媒体服务器 - 纯OpenCV版本"""
    
    def __init__(self, camera_id='/dev/video0', width=640, height=480, fps=15):
        """
        初始化摄像头流
        
        Args:
            camera_id: 摄像头设备路径，默认'/dev/video0'
            width: 画面宽度
            height: 画面高度
            fps: 目标帧率
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.running = False
        self.frame_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.client_count = 0
        self.last_frame_time = 0
        self.frame_count = 0
        self.error_count = 0
        
        logger.info(f"📹 摄像头流模块初始化 - OpenCV版本")
        logger.info(f"   设备: {camera_id}, 分辨率: {width}x{height}, FPS: {fps}")
    
    def start(self):
        """启动摄像头捕获 - 终极修复版"""
        if self.running:
            return True
        
        try:
            # 方法1: 使用设备路径
            logger.info(f"尝试方法1 - 打开摄像头设备: {self.camera_id}")
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            
            if not self.cap or not self.cap.isOpened():
                # 方法2: 使用数字ID
                logger.info("方法1失败，尝试方法2 - 使用数字ID 0")
                self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            if not self.cap or not self.cap.isOpened():
                # 方法3: 不使用后端指定
                logger.info("方法2失败，尝试方法3 - 自动选择后端")
                self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap or not self.cap.isOpened():
                logger.error(f"❌ 所有方法都无法打开摄像头")
                return False
            
            # 成功打开
            logger.info(f"✅ 摄像头打开成功")
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 获取实际设置的值
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"📊 实际分辨率: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            # 预热：读取几帧让摄像头稳定
            logger.info("摄像头预热中...")
            warmup_success = 0
            for i in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    warmup_success += 1
                    logger.info(f"  预热帧 {i+1}: 成功 ({frame.shape})")
                time.sleep(0.1)
            
            if warmup_success > 0:
                logger.info(f"✅ 预热完成 - {warmup_success}/5 帧成功")
            else:
                logger.warning("⚠️ 预热失败，但继续尝试")
            
            self.running = True
            self.frame_thread = threading.Thread(target=self._capture_loop)
            self.frame_thread.daemon = True
            self.frame_thread.start()
            
            logger.info("✅ 摄像头捕获线程已启动")
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动摄像头失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _capture_loop(self):
        """捕获循环"""
        logger.info("🔄 捕获线程开始运行")
        
        while self.running and self.cap:
            try:
                # 读取帧
                ret, frame = self.cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    self.frame_count += 1
                    self.error_count = 0
                    
                    # 调整大小以确保一致性
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    
                    # 压缩为JPEG
                    ret, jpeg = cv2.imencode('.jpg', frame, [
                        cv2.IMWRITE_JPEG_QUALITY, 80
                    ])
                    
                    if ret:
                        with self.frame_lock:
                            self.current_frame = jpeg.tobytes()
                            self.last_frame_time = time.time()
                    
                    # 每秒打印一次帧率统计
                    if self.frame_count % self.fps == 0:
                        logger.debug(f"摄像头运行正常 - 已捕获 {self.frame_count} 帧")
                        
                else:
                    self.error_count += 1
                    logger.warning(f"读取帧失败 (错误计数: {self.error_count})")
                    
                    if self.error_count > 10:
                        logger.error("连续10次读取失败，尝试重新连接...")
                        self._reconnect_camera()
                        self.error_count = 0
                
                # 控制帧率
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                logger.error(f"捕获循环错误: {e}")
                time.sleep(0.5)
        
        logger.info("🛑 捕获线程已停止")
    
    def _reconnect_camera(self):
        """重新连接摄像头"""
        try:
            if self.cap:
                self.cap.release()
            
            time.sleep(1)
            
            if isinstance(self.camera_id, str):
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            else:
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                logger.info("✅ 摄像头重新连接成功")
            else:
                logger.error("❌ 摄像头重新连接失败")
                
        except Exception as e:
            logger.error(f"重新连接失败: {e}")
    
    def get_frame(self):
        """获取最新的JPEG帧"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame
        return None
    
    def generate_frames(self):
        """生成MJPEG流"""
        logger.info("开始生成MJPEG流")
        no_frame_count = 0
        
        while self.running:
            frame = self.get_frame()
            if frame:
                no_frame_count = 0
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                no_frame_count += 1
                # 如果没有帧，返回一个占位图
                placeholder = self._create_placeholder()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                
                if no_frame_count > 30:
                    logger.warning("长时间无帧数据")
                    no_frame_count = 0
            
            # 控制流输出频率
            time.sleep(1.0 / self.fps)
    
    def _create_placeholder(self):
        """创建占位图"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(img, "Waiting for Camera...", (50, self.height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, time.strftime("%H:%M:%S"), (50, self.height//2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
    
    def stop(self):
        """停止摄像头"""
        logger.info("正在停止摄像头...")
        self.running = False
        
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info(f"🛑 摄像头已停止 - 总共捕获 {self.frame_count} 帧")
    
    def client_connected(self):
        """客户端连接计数"""
        self.client_count += 1
        logger.info(f"📡 摄像头客户端连接: {self.client_count}")
    
    def client_disconnected(self):
        """客户端断开计数"""
        self.client_count = max(0, self.client_count - 1)
        logger.info(f"📡 摄像头客户端断开: {self.client_count}")
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'running': self.running,
            'clients': self.client_count,
            'frames': self.frame_count,
            'last_frame': self.last_frame_time
        }