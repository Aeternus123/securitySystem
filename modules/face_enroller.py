#!/usr/bin/env python3
"""
人脸录入模块 - 用于添加新人脸到数据库
使用352维增强特征
"""
import cv2
import numpy as np
import pickle
import os
import time
from datetime import datetime
from utils.logger import logger

class FaceEnroller:
    """人脸录入器 - 352维特征版本"""
    
    def __init__(self, face_detector):
        """
        初始化人脸录入器
        :param face_detector: 人脸检测器实例 (必须包含extract_enhanced_features方法)
        """
        self.face_detector = face_detector
        self.database_path = face_detector.database_path
        self.images_dir = face_detector.images_dir
        
        # 确保目录存在
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 检查特征提取方法
        if not hasattr(face_detector, 'extract_enhanced_features'):
            logger.error("人脸检测器缺少extract_enhanced_features方法")
            raise AttributeError("人脸检测器必须包含extract_enhanced_features方法")
        
        logger.info("人脸录入模块初始化完成")
        logger.info(f"特征维度: 352")
    
    def capture_face(self, frame, box):
        """从帧中截取人脸"""
        x1, y1, x2, y2 = box
        face_img = frame[y1:y2, x1:x2]
        return face_img
    
    def add_face_from_frame(self, frame, box, name):
        """
        从帧中截取人脸并添加到数据库
        :param frame: 原始帧
        :param box: 人脸框 [x1,y1,x2,y2]
        :param name: 人名
        :return: bool 是否成功
        """
        try:
            # 截取人脸
            face_img = self.capture_face(frame, box)
            
            if face_img.size == 0:
                logger.error("人脸图像为空")
                return False
            
            # 提取352维特征
            features = self.face_detector.extract_enhanced_features(face_img)
            
            # 添加到数据库
            if name not in self.face_detector.face_database:
                self.face_detector.face_database[name] = {
                    'features': [],
                    'added_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'images': []
                }
            
            self.face_detector.face_database[name]['features'].append(features)
            
            # 保存人脸图片
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            img_filename = f"{name}_{timestamp}.jpg"
            img_path = os.path.join(self.images_dir, img_filename)
            cv2.imwrite(img_path, face_img)
            self.face_detector.face_database[name]['images'].append(img_path)
            
            # 保存数据库
            self.face_detector.save_database()
            
            logger.info(f"✅ 已添加人脸: {name} (特征维度: {len(features)})")
            return True
            
        except Exception as e:
            logger.error(f"添加人脸失败: {e}")
            return False
    
    def add_face_multiple_samples(self, cap, name, num_samples=5):
        """
        连续采集多个人脸样本（无GUI版本）
        :param cap: 摄像头对象
        :param name: 人名
        :param num_samples: 样本数量
        :return: bool 是否成功
        """
        logger.info(f"开始采集 {name} 的人脸样本，需要 {num_samples} 张")
        logger.info("请转动头部，从不同角度采集")
        logger.info("检测到人脸时按 Enter 保存，输入 'q' 退出")
        
        collected = 0
        timeout = time.time() + 60  # 60秒超时
        frame_count = 0
        
        while collected < num_samples and time.time() < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # 每10帧检测一次
            if frame_count % 10 == 0:
                boxes = self.face_detector.detect_faces(frame)
                
                if boxes:
                    print(f"\n✅ 检测到人脸！当前进度: {collected}/{num_samples}")
                    
                    # 检查是否有键盘输入
                    import sys
                    import select
                    if select.select([sys.stdin], [], [], 0)[0]:
                        cmd = sys.stdin.readline().strip().lower()
                        if cmd == '' or cmd == 'y':  # Enter键
                            if self.add_face_from_frame(frame, boxes[0], name):
                                collected += 1
                                logger.info(f"已采集 {collected}/{num_samples} 张")
                        elif cmd == 'q':
                            break
                else:
                    if frame_count % 30 == 0:
                        print(".", end='', flush=True)
            
            time.sleep(0.01)
        
        if collected >= num_samples:
            logger.info(f"✅ {name} 的人脸采集完成，共 {collected} 张")
            return True
        else:
            logger.warning(f"⚠️ {name} 的人脸采集未完成，只采集了 {collected} 张")
            return False
    
    def list_faces(self):
        """列出数据库中所有人脸"""
        if not self.face_detector.face_database:
            print("\n📭 数据库为空")
            return
        
        print("\n" + "=" * 60)
        print(f"📊 人脸数据库 (共 {len(self.face_detector.face_database)} 人)")
        print("=" * 60)
        
        for name, data in self.face_detector.face_database.items():
            feature_dim = len(data['features'][0]) if data['features'] else 0
            print(f"\n👤 {name}:")
            print(f"   - 样本数: {len(data['features'])}")
            print(f"   - 特征维度: {feature_dim}")
            print(f"   - 录入时间: {data['added_time']}")
            print(f"   - 图片: {len(data.get('images', []))} 张")
            # 显示最近的一张图片
            if data.get('images'):
                print(f"   - 最近图片: {os.path.basename(data['images'][-1])}")
    
    def delete_face(self, name):
        """删除人脸"""
        if name in self.face_detector.face_database:
            del self.face_detector.face_database[name]
            self.face_detector.save_database()
            logger.info(f"已删除人脸: {name}")
            return True
        else:
            logger.warning(f"人脸不存在: {name}")
            return False
    
    def get_face_stats(self):
        """获取人脸统计信息"""
        stats = {
            'total_persons': len(self.face_detector.face_database),
            'total_samples': 0,
            'feature_dim': 0
        }
        
        for name, data in self.face_detector.face_database.items():
            stats['total_samples'] += len(data['features'])
            if data['features'] and stats['feature_dim'] == 0:
                stats['feature_dim'] = len(data['features'][0])
        
        return stats
    
    def verify_face(self, frame, box, name):
        """
        验证指定框内的人脸是否匹配指定人名
        :param frame: 原始帧
        :param box: 人脸框
        :param name: 要验证的人名
        :return: (是否匹配, 相似度)
        """
        try:
            face_img = self.capture_face(frame, box)
            if face_img.size == 0:
                return False, 0.0
            
            features = self.face_detector.extract_enhanced_features(face_img)
            
            if name not in self.face_detector.face_database:
                return False, 0.0
            
            # 计算与所有保存特征的相似度
            similarities = [self.face_detector.cosine_similarity(features, feat) 
                          for feat in self.face_detector.face_database[name]['features']]
            avg_similarity = np.mean(similarities)
            
            return avg_similarity > self.face_detector.sim_threshold, avg_similarity
            
        except Exception as e:
            logger.error(f"验证人脸失败: {e}")
            return False, 0.0