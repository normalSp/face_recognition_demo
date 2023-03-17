import os
import cv2
import numpy as np


# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')

# 创建人脸识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 加载数据集并进行训练
face_images = []
labels = []

# 加载指定文件夹内的所有人脸图片作为训练数据
def load_faces(folder_path):
    i = 1
    for filename in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, filename))
        image = cv2.resize(image, (231, 308)) # 将图像缩小到 800*600 的大小
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 数据增强
        for angle in [0, 15, -15, 20, -20, 30, -30, 25, -25]:
            # 显示检测图片次数
            print('检测次数：%d' % i)
            i = i + 1

            rows, cols = gray.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated_gray = cv2.warpAffine(gray, M, (cols, rows))
            faces = face_cascade.detectMultiScale(rotated_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_image = cv2.resize(rotated_gray[y:y+h, x:x+w], (100, 100)) # 将人脸剪切区域调整为相同的大小
                face_images.append(face_image)
                labels.append(os.path.splitext(filename)[0]) # 以图片文件名为每个人脸分配唯一标签

    label_dict = {}
    label_num = 0
    for label in labels:
        if label not in label_dict:
            label_dict[label] = label_num
            label_num += 1
    labels_int = [label_dict[label] for label in labels]
    recognizer.train(face_images, np.array(labels_int))

load_faces('D:/Art/faces') # 调用函数加载指定文件夹内的所有人脸图片

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头捕获的图像
    ret, frame = cap.read()

    # 进行人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 多尺度检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # 对每个人脸进行识别并在图像中标记
    for (x, y, w, h) in faces:
        # 人脸对齐
        face_center = (int(x + w//2), int(y + h//2))
        M = cv2.getRotationMatrix2D(face_center, 0, 1)
        rows, cols = gray.shape
        rotated_gray = cv2.warpAffine(gray, M, (cols, rows))
        face_image = cv2.resize(rotated_gray[y:y+h, x:x+w], (100, 100)) # 将人脸剪切区域调整为相同的大小
        label, confidence = recognizer.predict(face_image)
        if confidence < 100:
            name = labels[label]
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 显示图像
    cv2.imshow('frame', frame)

    # 按下 q 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()