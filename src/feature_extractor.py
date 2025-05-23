import cv2
import numpy as np

class FacialFeatureExtractor:
    def __init__(self, output_size=(64, 64)):
        """
        初始化面部特征提取器
        
        参数:
            output_size: 输出特征图像的大小 (默认64x64)
        """
        self.output_size = output_size
        # 加载预训练的人脸特征检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def extract_features(self, face_img):
        """
        从面部图像中提取特征
        
        参数:
            face_img: 输入的面部图像 (48x48灰度图)
            
        返回:
            dict: 包含提取的各个面部特征
        """
        # 转换为彩色图像用于特征检测
        color_face = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        
        # 检测面部区域 (在48x48图像中应该就是整个图像)
        faces = self.face_cascade.detectMultiScale(face_img, 1.1, 4)
        if len(faces) == 0:
            return None
            
        (x, y, w, h) = faces[0]
        face_roi = face_img[y:y+h, x:x+w]
        
        # 提取各个特征
        features = {
            'left_eye': self._extract_left_eye(face_img),
            'right_eye': self._extract_right_eye(face_img),
            'mouth': self._extract_mouth(face_img),
            'nose': self._extract_nose(face_img),
            'full_face': self._resize(face_img)
        }
        
        return features
    
    def _resize(self, img):
        """调整图像大小到输出尺寸"""
        return cv2.resize(img, self.output_size, interpolation=cv2.INTER_AREA)
    
    def _extract_left_eye(self, face_img):
        """提取左眼"""
        # 在48x48图像中，眼睛大约在上半部分的左侧
        eye_region = face_img[10:25, 5:20]
        return self._resize(eye_region)
    
    def _extract_right_eye(self, face_img):
        """提取右眼"""
        # 在48x48图像中，眼睛大约在上半部分的右侧
        eye_region = face_img[10:25, 28:43]
        return self._resize(eye_region)
    
    def _extract_mouth(self, face_img):
        """提取嘴巴"""
        # 嘴巴大约在下半部分
        mouth_region = face_img[30:45, 10:38]
        return self._resize(mouth_region)
    
    def _extract_nose(self, face_img):
        """提取鼻子"""
        # 鼻子大约在中间部分
        nose_region = face_img[20:35, 15:33]
        return self._resize(nose_region)
