import cv2
import mediapipe as mp
import numpy as np

# 爱心图案绘制函数
def draw_heart(img, center, size, color, alpha):
    x, y = center
    heart_points = []
    for t in np.linspace(0, 2*np.pi, 100):
        # 参数方程表示的爱心曲线
        heart_x = int(x + 16 * (np.sin(t) ** 3) * size)
        heart_y = int(y - 13 * np.cos(t) * size + 5 * np.cos(2*t) * size)
        heart_points.append((heart_x, heart_y))
    # 将列表 heart_points 转换为 NumPy 数组
    heart_array = np.array(heart_points, np.int32)
    heart_array = heart_array.reshape((-1, 1, 2))
    overlay = img.copy()
    cv2.fillPoly(overlay, [heart_array], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# 笑脸图标绘制函数
def draw_goodicon(img, center, icon_path, size):
    x, y = center
    icon = cv2.imread(icon_path, -1)  # 读取带有透明通道的图标
    icon = cv2.resize(icon, (size, size))  # 调整图标大小
    ih, iw, _ = img.shape  # 将图标添加到图像上
    # 图像切片索引需要将y、x、size转换为整数类型
    y = int(y)
    x = int(x)
    size = int(size)
    if y + size < ih and x + size < iw:
        # 提取图标的透明通道作为遮罩
        alpha_channel = icon[:, :, 3]
        alpha_channel = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
        alpha_channel = alpha_channel / 255.0
        # 计算图标在图像上的位置
        roi = img[y:y+size, x:x+size]
        # 将图标叠加到图像上
        overlay = cv2.multiply(1.0 - alpha_channel, roi, dtype=cv2.CV_32F)
        icon_part = cv2.multiply(alpha_channel, icon[:, :, 0:3], dtype=cv2.CV_32F)
        img[y:y+size, x:x+size] = cv2.convertScaleAbs(cv2.add(overlay, icon_part))
    return img