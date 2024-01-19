import cv2
import mediapipe as mp
import numpy as np
from recognize import *
from draw_icon import *
import time
import random


# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # 图像宽度
cap.set(4, 720)  # 图像高度

# 创建手势检测实例
gesture_recognizer = GestureRecognition()

# 摄像头打开
while cap.isOpened():
    ret, image = cap.read()
    if ret:
        imgHeight = image.shape[0]
        imgWidth = image.shape[1]
    else:
        break

    lmList = gesture_recognizer.detect_hand_landmarks(image)
    # 检测image中的手部关键点

    results = gesture_recognizer.results
    # 运行手部检测模型以检测手部姿势（包含转换rgb）

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            # print(landmarks[6].y*imgHeight,landmarks[2].y*imgHeight,landmarks[3].y*imgHeight ,landmarks[4].y*imgHeight)
            # 需要先满足心形基本y坐标高低再进行后续判断
            # 注意！这里是<！
            if landmarks[6].y*imgHeight<=landmarks[7].y*imgHeight and landmarks[6].y*imgHeight<=landmarks[5].y*imgHeight and landmarks[2].y*imgHeight<=landmarks[3].y*imgHeight <=landmarks[4].y*imgHeight:
                # 在这里检测爱心手势
                if gesture_recognizer.is_love_hand(hand_landmarks):
                    # 用time包表示当前时间，用于控制爱心的出现时间
                    current_time = int(time.time())
                    if current_time % 2 == 0:
                        for _ in range(10):
                            # 设置爱心大小、颜色、中心点（均使用random库）
                            heart_size = random.randint(0, 5)
                            heart_color = (random.randint(0, 203), random.randint(0, 192), 255)  # 随机颜色
                            heart_center = (random.randint(0, imgWidth), random.randint(0, imgHeight))
                            draw_heart(image, heart_center, heart_size, heart_color, 0.1*random.randrange(3,7))

            elif landmarks[4].y*imgHeight<=landmarks[3].y*imgHeight<=landmarks[2].y*imgHeight<=landmarks[0].y*imgHeight:
                # 在这里检测点赞手势
                if gesture_recognizer.is_good_hand(hand_landmarks):
                    draw_goodicon(image, (landmarks[4].x * imgWidth-50, landmarks[4].y * imgHeight - 120), "good.png",100)
    cv2.imshow('Video', image)

    # 按q关闭摄像头并退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()