#!/usr/bin/env python3
"""
人脸检测脚本 - 专门用于实时识别
识别到已知人脸返回 true，陌生人返回 false
"""

import cv2
import numpy as np
import onnxruntime as ort
import pickle
import os
import time
from datetime import datetime

class FaceDetector:
    def __init__(self, 
                 det_model_path='models/yolov8n-face.onnx',
                 database_path='database/face_db.pkl',
                 conf_thres=0.5,
                 sim_threshold=0.25):
        """
        初始化人脸检测器
        """
        self.conf_thres = conf_thres
        self.sim_threshold = sim_threshold
        self.database_path = database_path
        
        # 加载人脸检测模型
        print("加载 YOLOv8n-face 检测模型...")
        self.det_session = ort.InferenceSession(det_model_path, providers=['CPUExecutionProvider'])
        self.det_input_name = self.det_session.get_inputs()[0].name
        
        # 加载人脸数据库
        self.face_database = self.load_database()
        
        self.input_size = 640
        print(f"✅ 初始化完成，数据库中有 {len(self.face_database)} 个人")
        
        # 统计信息
        self.total_frames = 0
        self.frames_with_faces = 0
        self.recognized_frames = 0
        
    def load_database(self):
        """加载人脸数据库"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"❌ 加载数据库失败: {e}")
                return {}
        else:
            print(f"⚠️ 数据库文件不存在: {self.database_path}")
            return {}
    
    def cosine_similarity(self, a, b):
        """计算余弦相似度"""
        a = np.array(a).flatten()
        b = np.array(b).flatten()
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot_product / (norm_a * norm_b)
    
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
        input_tensor = self.preprocess_detection(image)
        outputs = self.det_session.run(None, {self.det_input_name: input_tensor})
        
        predictions = np.transpose(outputs[0], (0, 2, 1))[0]
        
        boxes = []
        confidences = []
        
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
            confidences.append(confidence)
        
        # NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, 0.5)
            if len(indices) > 0:
                indices = indices.flatten()
                return [boxes[i] for i in indices]
        
        return []
    
    def extract_features(self, face_img):
        """
        提取人脸特征（与录入程序一致）
        """
        # 调整到统一大小
        face = cv2.resize(face_img, (64, 64))
        
        # 特征1：颜色直方图 (HSV)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # 特征2：灰度直方图
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [32], [0, 256])
        
        # 特征3：梯度直方图（边缘信息）
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        hist_grad = cv2.calcHist([gradient.astype(np.uint8)], [0], None, [16], [0, 255])
        
        # 归一化所有特征
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
        hist_grad = cv2.normalize(hist_grad, hist_grad).flatten()
        
        # 组合特征
        features = np.concatenate([hist_h, hist_s, hist_v, hist_gray, hist_grad])
        return features
    
    def recognize_faces(self, frame):
        """
        识别帧中所有人脸
        返回: (是否识别到已知人脸, 识别结果列表)
        """
        boxes = self.detect_faces(frame)
        
        if len(boxes) == 0:
            return False, []
        
        self.frames_with_faces += 1
        recognized_any = False
        results = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                continue
            
            # 提取特征
            features = self.extract_features(face_img)
            
            # 与数据库比对
            best_match = "unknown"
            best_similarity = 0.0
            
            for name, data in self.face_database.items():
                similarities = [self.cosine_similarity(features, feat) for feat in data['features']]
                avg_similarity = np.mean(similarities)
                
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    if avg_similarity > self.sim_threshold:
                        best_match = name
            
            is_recognized = best_match != "unknown"
            if is_recognized:
                recognized_any = True
                self.recognized_frames += 1
            
            results.append({
                'box': box,
                'name': best_match,
                'similarity': best_similarity,
                'recognized': is_recognized
            })
        
        return recognized_any, results
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 40)
        print("📊 检测统计:")
        print(f"总帧数: {self.total_frames}")
        print(f"含人脸帧: {self.frames_with_faces}")
        print(f"识别成功帧: {self.recognized_frames}")
        if self.frames_with_faces > 0:
            recognition_rate = (self.recognized_frames / self.frames_with_faces) * 100
            print(f"人脸识别率: {recognition_rate:.1f}%")
        print("=" * 40)

def main():
    print("=" * 60)
    print("👤 人脸实时检测系统")
    print("=" * 60)
    print("功能: 实时检测并识别人脸")
    print("输出: 检测到已知人脸 -> true")
    print("      检测到陌生人 -> false")
    print("      未检测到人脸 -> 无输出")
    print("-" * 60)
    
    # 初始化检测器
    detector = FaceDetector(
        det_model_path='models/yolov8n-face.onnx',
        database_path='database/face_db.pkl',
        sim_threshold=0.25  # 可根据实际情况调整
    )
    
    # 检查数据库是否为空
    if len(detector.face_database) == 0:
        print("⚠️ 警告: 数据库为空，请先运行录入程序添加人脸")
        print("建议: 先运行 face_enrollment.py 录入人脸")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ 摄像头已打开")
    print("开始实时检测...")
    print("按 Ctrl+C 停止")
    print("-" * 60)
    
    frame_count = 0
    last_status = None
    last_display_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            detector.total_frames += 1
            
            # 每3帧处理一次，节省CPU
            if frame_count % 3 == 0:
                recognized, results = detector.recognize_faces(frame)
                
                # 输出结果
                if recognized:
                    # 找到所有识别到的人名
                    names = list(set([r['name'] for r in results if r['recognized']]))
                    print(f"✅ true - 识别到: {', '.join(names)}")
                elif results:  # 检测到人脸但都不认识
                    print(f"❌ false - 检测到 {len(results)} 个人脸，均为陌生人")
                
                # 可选：实时显示（调试用）
                if len(results) > 0:
                    current_time = time.time()
                    if current_time - last_display_time >= 1.0:
                        # 每秒显示一次状态
                        if recognized:
                            status = "true"
                        else:
                            status = "false"
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status}, 人脸数: {len(results)}")
                        last_display_time = current_time
            
            # 降低CPU使用率
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n停止检测")
        detector.print_stats()
    
    cap.release()
    print("程序已退出")

if __name__ == "__main__":
    main()