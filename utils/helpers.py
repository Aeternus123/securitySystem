#!/usr/bin/env python3
"""
辅助函数模块
"""
import cv2
import numpy as np

def cosine_similarity(a, b):
    """计算余弦相似度，自动处理维度检查"""
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    
    # 检查维度是否匹配
    if len(a) != len(b):
        print(f"⚠️ 警告: 特征维度不匹配: {len(a)} vs {len(b)}")
        return 0
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

def local_binary_pattern(image):
    """简化版LBP特征提取"""
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