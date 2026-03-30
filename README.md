# 智能家庭安防系统

基于YOLO的智能家庭安防系统，集成了人脸识别、语音控制、环境监测和远程监控功能。

## 项目概述

本项目是一个完整的智能家庭安防解决方案，采用树莓派作为硬件平台，结合YOLO人脸识别技术，实现智能化的家庭安全监控。系统具备实时人脸识别、语音控制、环境传感器监测、远程报警等功能。

### 主要特性

- **智能人脸识别**: 基于YOLOv8的实时人脸检测和352维特征识别
- **语音控制**: 支持语音指令控制灯光、电机等设备
- **环境监测**: 温湿度传感器、烟雾报警器实时监控
- **超声波测距**: 检测人员接近，自动触发安防模式
- **远程监控**: Web界面实时查看系统状态和摄像头画面
- **自动报警**: 陌生人检测、环境异常自动报警

## 系统架构

### 硬件组成

- **主控板**: 树莓派
- **摄像头模块**: USB摄像头或树莓派摄像头
- **传感器模块**:
  - DHT11温湿度传感器
  - 烟雾传感器
  - 超声波测距模块
- **执行器模块**:
  - LED灯（红/绿）
  - 蜂鸣器
  - 直流电机
  - 矩阵键盘

### 软件架构

```
智能安防系统
├── 主程序 (main.py)
├── 配置模块 (config/)
├── 功能模块 (modules/)
├── 工具类 (utils/)
├── 数据库 (database/)
├── 模型文件 (models/)
└── 日志系统 (logs/)
```

## 快速开始

### 环境要求

- Python 3.8+
- 树莓派（推荐Raspberry Pi 4）
- 摄像头模块
- 相关传感器和执行器

### 安装步骤

1. **克隆项目**

```bash
git clone <项目地址>
cd security_system
```

2. **创建虚拟环境**

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **硬件连接**

按照以下引脚配置连接硬件：

| 设备       | GPIO引脚 | 功能         |
| ---------- | -------- | ------------ |
| 红灯       | GPIO17   | 主卧灯       |
| 绿灯       | GPIO27   | 客房灯       |
| 蜂鸣器     | GPIO22   | 报警器       |
| 电机       | GPIO23   | 窗帘/门控制  |
| 超声波Trig | GPIO24   | 测距触发     |
| 超声波Echo | GPIO25   | 测距回波     |
| DHT11      | GPIO26   | 温湿度传感器 |
| 烟雾传感器 | GPIO19   | 烟雾检测     |

5. **启动系统**

```bash
# 使用启动脚本（推荐）
./start.sh

# 或直接运行主程序
python main.py
```

## 功能模块详解

### 1. 人脸识别模块 (FaceDetector)

- 基于YOLOv8-face.onnx模型
- 352维特征向量提取
- 实时人脸检测和识别
- 陌生人自动记录和报警
- 人脸注册和管理功能

### 2. 语音控制模块 (VoiceModule)

- 支持中文语音指令
- 控制灯光、电机等设备
- 语音唤醒功能
- 指令识别和响应

### 3. 传感器管理模块 (SensorManager)

- 温湿度监测 (DHT11)
- 烟雾浓度检测
- 环境异常报警
- 数据实时记录

### 4. 超声波测距模块 (Ultrasonic)

- 实时距离检测
- 人员接近预警
- 自动安防模式切换

### 5. 报警系统模块 (AlarmSystem)

- 多级报警机制
- 声光报警
- 远程通知
- 报警记录

## 使用说明

### 基本操作

1. **系统启动**
   - 运行`start.sh`脚本启动所有服务
   - 系统自动初始化各模块
   - 摄像头开始实时监控

2. **人脸注册**
   - 系统首次运行时自动进入人脸注册模式
   - 按照提示进行人脸录入
   - 支持多用户注册

3. **语音控制**
   - 说出"小安小安"唤醒语音助手
   - 支持以下指令：
     - "打开主卧灯" / "关闭主卧灯"
     - "打开客房灯" / "关闭客房灯"
     - "打开电机" / "关闭电机"
     - "开启语音" / "关闭语音"

4. **远程监控**
   - 访问 `http://<树莓派IP>:8000` 查看系统状态
   - 访问 `http://<树莓派IP>:5000` 查看摄像头画面

### 配置文件

主要配置文件位于 `config/` 目录：

- `settings.py` - 系统参数配置
- `voice_commands.py` - 语音指令配置

## 技术细节

### 人脸识别技术

- **模型**: YOLOv8-face.onnx
- **特征维度**: 352维
- **相似度算法**: 余弦相似度
- **识别阈值**: 0.65
- **检测置信度**: 0.5

### 语音识别

- 基于语音特征匹配
- 支持自定义唤醒词
- 指令词库可配置

### 数据存储

- 人脸特征数据库: `database/face_db.pkl`
- 陌生人照片: `images/stranger/`
- 系统日志: `logs/system.log`

## 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头连接
   - 确认摄像头权限
   - 验证OpenCV安装

2. **GPIO设备不工作**
   - 检查引脚连接
   - 确认RPi.GPIO安装
   - 验证权限设置

3. **人脸识别失败**
   - 检查光照条件
   - 验证模型文件路径
   - 检查数据库文件

### 日志查看

系统运行日志保存在 `logs/system.log`，可通过以下命令查看：

```bash
tail -f logs/system.log
```

## 项目结构

```
security_system/
├── main.py                 # 主程序入口
├── requirements.txt        # Python依赖
├── start.sh               # 启动脚本
├── config/                # 配置文件
│   ├── settings.py        # 系统设置
│   └── voice_commands.py  # 语音指令
├── modules/               # 功能模块
│   ├── face_detector.py   # 人脸识别
│   ├── voice_module.py    # 语音控制
│   ├── sensors.py         # 传感器管理
│   ├── ultrasonic.py      # 超声波测距
│   ├── alarm.py           # 报警系统
│   └── gpio_controller.py # GPIO控制
├── utils/                 # 工具类
│   ├── logger.py          # 日志工具
│   └── helpers.py         # 辅助函数
├── database/              # 数据存储
│   └── face_db.pkl        # 人脸数据库
├── models/                # 模型文件
│   └── yolov8n-face.onnx  # YOLO人脸模型
├── images/                # 图片存储
│   └── stranger/          # 陌生人照片
└── logs/                  # 日志文件
    └── system.log         # 系统日志
```

## 开发指南

### 添加新功能

1. 在 `modules/` 目录创建新模块
2. 在 `config/settings.py` 中添加配置参数
3. 在 `main.py` 中集成新模块
4. 更新 `requirements.txt` 添加新依赖

### 自定义语音指令

编辑 `config/voice_commands.py` 文件：

```python
# 添加新指令
NEW_COMMAND_PREFIX = "新指令"
NEW_OPEN = "打开新设备"
NEW_CLOSE = "关闭新设备"
```

### 修改识别阈值

在 `config/settings.py` 中调整参数：

```python
FACE_SIMILARITY_THRESHOLD = 0.7    # 人脸相似度阈值
FACE_CONFIDENCE_THRESHOLD = 0.6    # 检测置信度阈值
ULTRASONIC_DISTANCE_THRESHOLD = 15 # 超声波距离阈值
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 联系方式

- 开发者: 张天旗、刘嘉琪
- 项目地址: [GitHub链接]
- 最后更新: 2026年3月

---

**注意**: 本项目为学术研究用途，实际部署时请确保符合相关法律法规和隐私保护要求。
