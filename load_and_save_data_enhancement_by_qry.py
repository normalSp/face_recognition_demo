# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/18 11:21
@Auth ： qry
"""
import os
import pickle
import numpy as np
import random
import cv2
from PIL import Image, ImageFilter

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


# 加载指定文件夹内的所有人脸图片作为训练数据
def load_faces(folder_path, save_path_faces_images, save_path_labels):
    # 检测次数计数器
    i = 1

    # 照片识别角度集
    angle_rows_1 = [0, 5, -5, 10, -10, 15, -15, 20, -20, 30, -30, 25, -25, 35, -35, 40, -40, 45, -45, 50, -50, 55, -55,
                    60, -60]
    angle_rows_2 = [2.5, -2.5, 7.5, -7.5, 12.5, -12.5, 17.5, -17.5, 22.5, -22.5, 27.5, -27.5, 32.5, -32.5, 37.5, -37.5]
    angle_rows_3 = [42.5, -42.5, 47.5, -47.5, 52.5, -52.5, 57.5, -57.5]

    # 数据增强参数
    flip_prob = 0.5
    scale_range = (0.9, 1.1)
    brightness_range = (-30, 30)  # 亮度调整范围
    contrast_range = (0.8, 1.2)  # 对比度调整范围
    blur_radius = 2  # 模糊半径

    # 命名训练集
    face_images = []
    labels = []
    for filename in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, filename))
        image = cv2.resize(image, (231, 308))  # 将图像缩小到 800*600 的大小
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for angle in angle_rows_1 + angle_rows_2 + angle_rows_3:
            # 显示检测图片次数
            print('检测次数：%d' % i)
            i += 1

            rows, cols = gray.shape
            center = (cols // 2, rows // 2)
            angle_rad = np.radians(angle)
            cos_val = np.cos(angle_rad)
            sin_val = np.sin(angle_rad)
            rotation_matrix = np.array([[cos_val, -sin_val, (1 - cos_val) * center[0] + sin_val * center[1]],
                                        [sin_val, cos_val, (1 - cos_val) * center[1] - sin_val * center[0]]])

            rotated_gray = np.zeros_like(gray)
            for x in range(cols):
                for y in range(rows):
                    new_x = int(rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y + rotation_matrix[0, 2])
                    new_y = int(rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y + rotation_matrix[1, 2])
                    if 0 <= new_x < cols and 0 <= new_y < rows:
                        rotated_gray[new_y, new_x] = gray[y, x]

            # 随机水平翻转
            if random.random() < flip_prob:
                rotated_gray = np.fliplr(rotated_gray)

            # 随机缩放
            scale_factor = random.uniform(scale_range[0], scale_range[1])
            resized_gray = np.zeros((int(rows * scale_factor), int(cols * scale_factor)))
            for x in range(resized_gray.shape[1]):
                for y in range(resized_gray.shape[0]):
                    new_x = int(x / scale_factor)
                    new_y = int(y / scale_factor)
                    if 0 <= new_x < cols and 0 <= new_y < rows:
                        resized_gray[y, x] = rotated_gray[new_y, new_x]

            # 亮度调整
            brightness = random.uniform(brightness_range[0], brightness_range[1])
            adjusted_gray = resized_gray * brightness
            adjusted_gray = np.clip(adjusted_gray, 0, 255).astype(np.uint8)

            # 对比度调整
            contrast = random.uniform(contrast_range[0], contrast_range[1])
            adjusted_gray_pil = Image.fromarray(adjusted_gray)
            adjusted_gray_pil = adjusted_gray_pil.point(lambda p: p * contrast)  # 对比度调整
            adjusted_gray = np.array(adjusted_gray_pil)

            # 模糊
            blurred_gray = adjusted_gray_pil.filter(ImageFilter.GaussianBlur(blur_radius))

            faces = face_cascade.detectMultiScale(np.array(rotated_gray), scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_image = np.array(blurred_gray.crop((x, y, x + w, y + h)).resize((100, 100)))
                face_images.append(face_image)
                labels.append(os.path.splitext(filename)[0])  # 以图片文件名为每个人脸分配唯一标签

    # 将人脸数据和标签保存到文件
    with open(save_path_faces_images, 'wb') as f:
        pickle.dump(face_images, f)

    with open(save_path_labels, 'wb') as f:
        pickle.dump(labels, f)

    print('人脸数据加载完成')

load_faces(r'F:/faceidentify_demo/train_image', r'F:/faceidentify_demo/save/face_images2.pkl', r'F:/faceidentify_demo/save/labels2.pkl')
