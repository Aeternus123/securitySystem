#!/usr/bin/env python3
"""
人脸识别模块 - 使用352维增强特征
"""
import cv2
import time
import numpy as np
import onnxruntime as ort
import pickle
import os
from datetime import datetime
from config.settings import *
from utils.logger import logger
from utils.helpers import cosine_similarity, local_binary_pattern

class FaceDetector:
    """人脸识别器 - 使用352维增强特征"""
    
    def __init__(self):
        self.sim_threshold = FACE_SIMILARITY_THRESHOLD
        self.database_path = FACE_DATABASE_PATH
        self.images_dir = FACE_IMAGES_DIR
        self.stranger_dir = FACE_STRANGER_DIR
        self.feature_dim = 352  # 固定为352维
        
        # 创建目录
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.stranger_dir, exist_ok=True)
        
        # 加载人脸检测模型
        logger.info("加载 YOLOv8n-face 检测模型...")
        self.det_session = ort.InferenceSession(
            'models/yolov8n-face.onnx', 
            providers=['CPUExecutionProvider']
        )
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.input_size = 640
        
        # 加载数据库
        self.face_database = self.load_database()
        
        logger.info(f"人脸识别初始化完成，数据库中有 {len(self.face_database)} 个人")
        logger.info(f"特征维度: {self.feature_dim}")
        logger.info(f"相似度阈值: {self.sim_threshold}")
        logger.info(f"陌生人照片存储目录: {self.stranger_dir}")
        
        self.last_stranger_time = 0
        self.stranger_cooldown = 5  # 陌生人报警冷却时间（秒）
        self.last_stranger_photo_time = 0
        self.stranger_photo_interval = 10  # 陌生人照片拍摄间隔（10秒）
        self.debug_mode = True
    
    def load_database(self):
        """加载人脸数据库"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    # 验证特征维度
                    for name, info in data.items():
                        if info['features'] and len(info['features'][0]) != self.feature_dim:
                            logger.warning(f"数据库特征维度不匹配，请重新录入人脸")
                            return {}
                    return data
            except Exception as e:
                logger.error(f"加载数据库失败: {e}")
                return {}
        return {}
    
    def save_database(self):
        """保存人脸数据库"""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.face_database, f)
        logger.info(f"已保存 {len(self.face_database)} 个人脸到数据库")
    
    def extract_enhanced_features(self, face_img):
        """
        提取352维增强人脸特征
        """
        # 调整到统一大小
        face = cv2.resize(face_img, (128, 128))
        
        # 特征1：颜色直方图 (HSV) - 96维
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # 特征2：灰度直方图 - 64维
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [64], [0, 256])
        
        # 特征3：梯度直方图 - 32维
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        hist_grad = cv2.calcHist([gradient.astype(np.uint8)], [0], None, [32], [0, 255])
        
        # 特征4：LBP纹理特征 - 32维
        lbp = local_binary_pattern(gray)
        hist_lbp = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [32], [0, 256])
        
        # 特征5：人脸分块直方图 (4x4分块) - 128维
        h, w = gray.shape
        block_h, block_w = h//4, w//4
        block_features = []
        for i in range(4):
            for j in range(4):
                block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                block_hist = cv2.calcHist([block], [0], None, [8], [0, 256])
                block_features.append(cv2.normalize(block_hist, block_hist).flatten())
        
        # 归一化所有特征
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
        hist_grad = cv2.normalize(hist_grad, hist_grad).flatten()
        hist_lbp = cv2.normalize(hist_lbp, hist_lbp).flatten()
        
        # 组合所有特征 (总维度: 96+64+32+32+128 = 352)
        features = np.concatenate([
            hist_h, hist_s, hist_v, 
            hist_gray, hist_grad, hist_lbp,
            np.concatenate(block_features)
        ])
        
        return features
    
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
        for pred in predictions:
            x_center, y_center, width, height, confidence = pred
            
            if confidence < FACE_CONFIDENCE_THRESHOLD:
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
            indices = cv2.dnn.NMSBoxes(boxes, [1.0]*len(boxes), FACE_CONFIDENCE_THRESHOLD, 0.5)
            if len(indices) > 0:
                indices = indices.flatten()
                return [boxes[i] for i in indices]
        
        return []
    
    def save_stranger_photo(self, frame, box):
        """
        保存陌生人照片（每10秒保存一次）
        :param frame: 原始帧
        :param box: 人脸框 [x1,y1,x2,y2]
        :return: 保存的文件路径
        """
        current_time = time.time()
        
        # 检查是否达到拍摄间隔
        if current_time - self.last_stranger_photo_time < self.stranger_photo_interval:
            return None
        
        x1, y1, x2, y2 = box
        
        # 截取人脸区域
        face_img = frame[y1:y2, x1:x2]
        
        # 生成文件名：陌生人_年月日_时分秒.jpg
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"stranger_{timestamp}.jpg"
        filepath = os.path.join(self.stranger_dir, filename)
        
        # 保存图片
        cv2.imwrite(filepath, face_img)
        logger.info(f"陌生人照片已保存: {filepath} (间隔{self.stranger_photo_interval}秒)")
        
        # 更新最后拍摄时间
        self.last_stranger_photo_time = current_time
        
        return filepath
    
    def check_frame(self, frame):
        """检查一帧图像中的人脸"""
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
            
            # 提取352维特征
            features = self.extract_enhanced_features(face_img)
            
            best_match = "unknown"
            best_similarity = 0.0
            all_similarities = {}
            
            # 与数据库比对
            for name, data in self.face_database.items():
                similarities = [cosine_similarity(features, feat) for feat in data['features']]
                avg_similarity = np.mean(similarities)
                all_similarities[name] = avg_similarity
                
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    if avg_similarity > self.sim_threshold:
                        best_match = name
            
            if self.debug_mode and best_match != "unknown":
                logger.info(f"人脸匹配: {best_match} 相似度: {best_similarity:.3f}")
            
            is_recognized = best_match != "unknown"
            if not is_recognized:
                has_stranger = True
                # 保存陌生人照片（每10秒一次）
                self.save_stranger_photo(frame, box)
            
            results.append({
                'box': box,
                'name': best_match,
                'similarity': best_similarity,
                'recognized': is_recognized
            })
        
        return has_stranger, results
    
    def add_face(self, name, face_img):
        """添加新人脸"""
        features = self.extract_enhanced_features(face_img)
        
        if name not in self.face_database:
            self.face_database[name] = {
                'features': [],
                'added_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'images': []
            }
        
        self.face_database[name]['features'].append(features)
        
        # 保存图片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_filename = f"{name}_{timestamp}.jpg"
        img_path = os.path.join(self.images_dir, img_filename)
        cv2.imwrite(img_path, face_img)
        self.face_database[name]['images'].append(img_path)
        
        self.save_database()
        logger.info(f"已添加人脸: {name}, 特征维度: {len(features)}")
        return True