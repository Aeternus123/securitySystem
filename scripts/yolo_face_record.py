#!/usr/bin/env python3
"""
人脸录入脚本 - 独立运行
用于添加新人脸到数据库
"""
import sys
import os
import cv2

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.face_detector import ONNXFaceDetector as FaceDetector
from modules.face_enroller import FaceEnroller
from utils.logger import logger

def main():
    print("=" * 60)
    print("👤 人脸录入系统 v2.0 - 深度学习版")
    print("=" * 60)
    
    # 初始化人脸检测器
    detector = FaceDetector()
    
    # 初始化录入器
    enroller = FaceEnroller(detector)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n📸 摄像头已打开")
    print("-" * 40)
    
    while True:
        print("\n请选择操作:")
        print("1. 录入新人脸")
        print("2. 批量录入（多角度）")
        print("3. 查看数据库")
        print("4. 删除人脸")
        print("5. 测试识别")
        print("6. 退出")
        
        choice = input("请输入 (1-6): ").strip()
        
        if choice == '1':
            # 录入新人脸
            print("\n请对准摄像头，按 SPACE 保存当前人脸")
            print("按 ESC 取消")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测人脸
                boxes = detector.detect_faces(frame)
                
                # 画框
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.putText(frame, f"检测到 {len(boxes)} 个人脸", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE: 保存  ESC: 取消", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('Face Enrollment', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 32 and boxes:  # SPACE
                    name = input("\n请输入这个人的人名: ").strip()
                    if name:
                        if enroller.add_face_from_frame(frame, boxes[0], name):
                            print(f"✅ 已保存 {name} 的人脸")
                        else:
                            print("❌ 保存失败")
                    break
            
            cv2.destroyAllWindows()
        
        elif choice == '2':
            # 批量录入
            name = input("\n请输入这个人的人名: ").strip()
            if name:
                samples = input("请输入要采集的样本数 (默认5): ").strip()
                samples = int(samples) if samples else 5
                enroller.add_face_multiple_samples(cap, name, samples)
        
        elif choice == '3':
            # 查看数据库
            enroller.list_faces()
            stats = enroller.get_face_stats()
            print(f"\n📊 总计: {stats['total_persons']} 人, {stats['total_samples']} 个样本")
        
        elif choice == '4':
            # 删除人脸
            if not detector.face_database:
                print("数据库为空")
                continue
            
            print("\n可删除的人脸:")
            names = list(detector.face_database.keys())
            for i, name in enumerate(names, 1):
                samples = len(detector.face_database[name]['features'])
                print(f"  {i}. {name} ({samples} 个样本)")
            
            try:
                idx = int(input("\n请输入要删除的编号 (0取消): ").strip())
                if 1 <= idx <= len(names):
                    enroller.delete_face(names[idx-1])
            except:
                print("输入无效")
        
        elif choice == '5':
            # 测试识别
            print("\n测试识别模式 - 按 ESC 返回")
            print("实时显示识别结果")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                has_stranger, results = detector.check_frame(frame)
                
                # 显示结果
                for r in results:
                    x1, y1, x2, y2 = r['box']
                    color = (0, 255, 0) if r['recognized'] else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{r['name']} ({r['similarity']:.2f})"
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.putText(frame, f"Faces: {len(results)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Face Recognition Test', frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
            
            cv2.destroyAllWindows()
        
        elif choice == '6':
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")

if __name__ == "__main__":
    main()