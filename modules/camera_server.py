#!/usr/bin/env python3
"""
独立摄像头测试 - 不依赖主程序
"""
import cv2
import time
import numpy as np
from flask import Flask, Response, render_template_string

app = Flask(__name__)

class SimpleCamera:
    def __init__(self):
        self.cap = None
        self.running = False
    
    def start(self):
        print("尝试打开摄像头...")
        
        # 尝试多种方式
        methods = [
            ('/dev/video0', cv2.CAP_V4L2),
            (0, cv2.CAP_V4L2),
            ('/dev/video0', None),
            (0, None)
        ]
        
        for dev, backend in methods:
            try:
                if backend:
                    cap = cv2.VideoCapture(dev, backend)
                else:
                    cap = cv2.VideoCapture(dev)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ 成功! 设备: {dev}, 后端: {backend}")
                        self.cap = cap
                        self.running = True
                        return True
                    cap.release()
            except:
                continue
        
        print("❌ 无法打开摄像头")
        return False
    
    def get_frame(self):
        if self.cap and self.running:
            ret, frame = self.cap.read()
            if ret:
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
        return None
    
    def generate_frames(self):
        while self.running:
            frame = self.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # 占位图
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, "No Camera Signal", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)

camera = SimpleCamera()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>独立摄像头测试</title></head>
    <body>
        <h1>独立摄像头测试</h1>
        <img src="/video_feed" style="width: 100%; max-width: 800px;">
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(camera.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("=" * 50)
    print("独立摄像头测试服务器")
    print("=" * 50)
    
    if camera.start():
        print("\n✅ 摄像头启动成功")
        print("🌐 访问地址: http://localhost:5000")
        print("   (从其他电脑访问: http://[树莓派IP]:5000)")
        print("\n按 Ctrl+C 停止\n")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("❌ 摄像头启动失败")