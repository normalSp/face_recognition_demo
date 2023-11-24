import os
import pickle
import numpy as np
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
    angle_rows_3 = [42.5, -42.5, 47.5, -47.5, 52.5, -52.5, 57.5, -57.5]
    angle_rows_4 = [62.5, -62.5, 65, -65, 67.5, -67.5, 70, -70, 72.5, -72.5, 75, -75, 77.5, -77.5, 80, -80, 82.5, -82.5]
    angle_rows_5 = [85, -85, 87.5, -87.5, 90, -90]
    angle_rows_6 = [92.5, -92.5, 95, -95, 97.5, -97.5, 100, -100, 102.5, -102.5, 105, -105, 107.5, -107.5, 110, -110]
    angle_rows_7 = [112.5, -112.5, 115.0, -115.0, 117.5, -117.5, 120.0, -120.0, 122.5, -122.5, 125.0, -125.0, 127.5,
                    -127.5, 130.0]
    angle_rows_8 = [-130.0, 132.5, -132.5, 135.0, -135.0, 137.5, -137.5, 140.0, -140.0, 142.5, -142.5,
                    145.0, -145.0, 147.5, -147.5, 150.0, -150.0, 152.5, -152.5, 155.0, -155.0, 157.5, -157.5, 160.0,
                    -160.0, 162.5, -162.5, 165.0, -165.0, 167.5, -167.5, 170.0, -170.0, 172.5, -172.5, 175.0, -175.0,
                    177.5, -177.5, 180.0, -180.0]

    # 数据增强参数
    flip_prob = 0.5
    scale_range = (0.9, 1.1)
    brightness_range = (-30, 30)  # 亮度调整范围
    contrast_range = (0.8, 1.2)  # 对比度调整范围
    blur_kernel_size = (5, 5)  # 模糊核大小

    # 命名训练集
    face_images = []
    labels = []
    for filename in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, filename))
        image = cv2.resize(image, (231, 308))  # 将图像缩小到 800*600 的大小
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 数据增强
        for angle in angle_rows_1 + angle_rows_2 + angle_rows_3 + angle_rows_4 + angle_rows_5 + angle_rows_6 + angle_rows_7:
            # 显示检测图片次数
            print('检测次数：%d' % i)
            i += 1

            rows, cols = gray.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_gray = cv2.warpAffine(gray, M, (cols, rows))

            # 随机水平翻转
            if np.random.rand() < flip_prob:
                rotated_gray = cv2.flip(rotated_gray, 1)

            # 随机缩放
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            resized_gray = cv2.resize(rotated_gray, None, fx=scale_factor, fy=scale_factor)
            resized_gray = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)

            # 亮度调整
            brightness = np.random.uniform(brightness_range[0], brightness_range[1])
            adjusted_gray = cv2.convertScaleAbs(resized_gray, beta=brightness)

            # 对比度调整
            contrast = np.random.uniform(contrast_range[0], contrast_range[1])
            adjusted_gray = cv2.convertScaleAbs(adjusted_gray, alpha=contrast)

            # 模糊
            blurred_gray = cv2.GaussianBlur(resized_gray, blur_kernel_size, 0)

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


load_faces(r'D:/Art/faces', r'D:/Art/save/face_images.pkl', r'D:/Art/save/labels.pkl')
