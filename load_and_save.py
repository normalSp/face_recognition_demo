import os
import pickle

import cv2
import numpy as np

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

    # 命名训练集
    face_images = []
    labels = []
    # 遍历文件夹内所有图像
    for filename in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, filename))
        if image is None:
            print(f"Failed to load image: {filename}")
            continue
        image = cv2.resize(image, (231, 308))  # 将图像缩小到 231*308 （手机照片1/10） 的大小
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 数据增强
        for angle in angle_rows_1 + angle_rows_2 + angle_rows_3:
            # 显示检测图片次数
            print('检测次数：%d' % i)
            i += 1

            rows, cols = gray.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            dst = cv2.warpAffine(gray, M, (cols, rows))

            # 将处理后的图片加入列表
            face_images.append(np.array(dst))
            labels.append(os.path.splitext(filename)[0])

    # 将人脸数据和标签保存到文件
    with open(save_path_faces_images, 'wb') as f:
        pickle.dump(face_images, f)

    with open(save_path_labels, 'wb') as f:
        pickle.dump(labels, f)

    print('人脸数据加载完成')


# 加载人脸信息文件
def load_file_images(save_path_faces_images, save_path_labels):
    with open(save_path_faces_images, 'rb') as f:
        face_images = pickle.load(f)

    return face_images

def load_file(save_path_faces_images, save_path_labels):
    with open(save_path_faces_images, 'rb') as f:
        face_images = pickle.load(f)

    with open(save_path_labels, 'rb') as f:
        labels = pickle.load(f)

    return face_images, labels


load_faces('D:/Art/faces', 'D:/Art/save/face_images', 'D:/Art//save/labels')
