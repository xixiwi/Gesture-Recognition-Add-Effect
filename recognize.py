import cv2
import mediapipe as mp
import math
import numpy as np

class GestureRecognition:
    # 初始化手势识别模型
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    # 检测image中的手部关键点（包括转换rgb）
    def detect_hand_landmarks(self, image,handnumber=0):
        # 先转换为rgb
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 处理rgb图像，并返回每只检测到的手的landmarks
        self.results = self.hands.process(image_rgb)

    # 在main里手指尖和手腕的坐标符合爱心手势后进行is_love_hand验证
    def is_love_hand(self,hand_landmarks):
        # print(self.calculate_finger_curvature(hand_landmarks))
        # print(self.calculate_palm_curvature(hand_landmarks))
        # 判断是否符合爱心手势：食指的弯曲程度为锐角、手掌形状为弯曲的
        if self.calculate_finger_curvature(hand_landmarks,8,6,5)<=90 and self.calculate_palm_curvature(hand_landmarks)<=150:
            return True
        else:
            return False

    # 在main里手指尖和手腕的坐标符合点赞手势后进行is_good_hand验证
    def is_good_hand(self, hand_landmarks):
        # 已经满足手势向上并且高于手腕y值的坐标关系后调用本函数
        # 剩下四指弯曲，这里只考虑食指和小拇指的弯曲程度即可
        # print(self.calculate_finger_curvature(hand_landmarks,8,6,5))
        # print(self.calculate_finger_curvature(hand_landmarks,20,18,17))
        if self.calculate_finger_curvature(hand_landmarks,8,6,5) <= 90 and self.calculate_finger_curvature(hand_landmarks,20,18,17) <= 90:
            return True
        else:
            return False

    # 计算手指的弯曲程度，返回指关节角度
    def calculate_finger_curvature(self, hand_landmarks, tipNo, middleNo, baseNo):
        landmarks = hand_landmarks.landmark
        # 计算手指关节的角度
        if len(landmarks) >= 21:
            # 找到手指指关节关键点
            finger_tip = landmarks[tipNo]
            finger_middle = landmarks[middleNo]
            finger_base = landmarks[baseNo]
            # 计算指关节的角度
            angle = math.degrees(math.atan2(finger_tip.y - finger_middle.y, finger_tip.x - finger_middle.x) -
                                 math.atan2(finger_base.y - finger_middle.y, finger_base.x - finger_middle.x))
            # 使角度处在0-180之间
            if angle < 0:
                angle += 360
                if angle > 180:
                    angle = 360 - angle
            elif angle > 180:
                angle = 360 - angle
            return angle
        else:
            return None

    # 计算手掌内边缘的弯曲程度，返回对应角度
    def calculate_palm_curvature(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        if len(landmarks) >= 21:
            thumb_base = landmarks[2]
            finger_base = landmarks[5]
            finger_middle = landmarks[6]
            # 计算手掌内边缘弯曲角度
            angle = math.degrees(
                math.atan2(finger_middle.y - finger_base.y, finger_middle.x - finger_base.x) -
                math.atan2(thumb_base.y - finger_base.y, thumb_base.x - finger_base.x))

            if angle < 0:
                angle += 360
                if angle > 180:
                    angle = 360 - angle
            elif angle > 180:
                angle = 360 - angle
            return angle
        else:
            return None