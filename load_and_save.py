import os
import pickle

import cv2

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
    angle_rows_3 = [42.5, -42.5, 47.5, -47.5, 52.5, -52.5, 57.5, -57.5, 70]

    # 命名训练集
    face_images = []
    labels = []
    for filename in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, filename))
        image = cv2.resize(image, (231, 308))  # 将图像缩小到 800*600 的大小
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 数据增强
        for angle in angle_rows_1 + angle_rows_2 + angle_rows_3:
            # 显示检测图片次数
            print('检测次数：%d' % i)
            i += 1

            rows, cols = gray.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_gray = cv2.warpAffine(gray, M, (cols, rows))
            faces = face_cascade.detectMultiScale(rotated_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_image = cv2.resize(rotated_gray[y:y + h, x:x + w], (100, 100))  # 将人脸剪切区域调整为相同的大小
                face_images.append(face_image)
                labels.append(os.path.splitext(filename)[0])  # 以图片文件名为每个人脸分配唯一标签

    # 将人脸数据和标签保存到文件
    with open(save_path_faces_images, 'wb') as f:
        pickle.dump(face_images, f)

    with open(save_path_labels, 'wb') as f:
        pickle.dump(labels, f)

    print('人脸数据加载完成')


load_faces('D:/Art/faces', 'D:/Art/save/face_images', 'D:/Art//save/labels')
