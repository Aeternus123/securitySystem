#!/usr/bin/env python3
"""
语音命令定义
"""

# 操作码定义
OP_OPEN = 0x81      # 打开操作
OP_CLOSE = 0xC1     # 关闭操作
OP_MOTOR_ON = 0x43  # 打开电机
OP_MOTOR_OFF = 0x83 # 关闭电机
OP_WAKEUP = 0x01    # 唤醒词
OP_VOICE_ON = 0x02  # 打开语音指令
OP_VOICE_OFF = 0x45 # 关闭语音指令

# 设备ID前缀（前5字节固定）
MASTER_BEDROOM_PREFIX = bytes([0x52, 0x34, 0x02, 0x00, 0x00])
GUEST_BEDROOM_PREFIX = bytes([0x6B, 0x34, 0x02, 0x00, 0x00])
MOTOR_PREFIX = bytes([0x19, 0xBC, 0x02, 0x00, 0x00])
WAKEUP_PREFIX = bytes([0x55, 0xE0, 0x01, 0x00, 0x00])
VOICE_CONTROL_PREFIX = bytes([0x1B, 0x94, 0x01, 0x00, 0x00])

# 设备名称映射
DEVICE_NAMES = {
    MASTER_BEDROOM_PREFIX: "主卧灯",
    GUEST_BEDROOM_PREFIX: "客房灯",
    MOTOR_PREFIX: "电机",
    WAKEUP_PREFIX: "唤醒词",
    VOICE_CONTROL_PREFIX: "语音控制",
}

# 命令名称映射
COMMAND_NAMES = {
    OP_OPEN: "打开",
    OP_CLOSE: "关闭",
    OP_MOTOR_ON: "打开电机",
    OP_MOTOR_OFF: "关闭电机",
    OP_WAKEUP: "唤醒",
    OP_VOICE_ON: "打开语音",
    OP_VOICE_OFF: "关闭语音",
}