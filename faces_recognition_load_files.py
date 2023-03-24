import pickle

import cv2
import numpy as np

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# 创建人脸识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 加载images和labels
with open('D:/Art/save/face_images', 'rb') as f:
    face_images = pickle.load(f)
with open('D:/Art/save/labels', 'rb') as f:
    labels = pickle.load(f)
print('加载images与labels完成，开始训练模型')

# 训练模型
label_dict = {}
label_num = 0
for label in labels:
    if label not in label_dict:
        label_dict[label] = label_num
        label_num += 1
labels_int = [label_dict[label] for label in labels]
recognizer.train(face_images, np.array(labels_int))
print('训练模型完成')

# 打开摄像头
cap = cv2.VideoCapture(0)
print('摄像头开启')

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
        face_center = (int(x + w // 2), int(y + h // 2))
        M = cv2.getRotationMatrix2D(face_center, 0, 1)
        rows, cols = gray.shape
        rotated_gray = cv2.warpAffine(gray, M, (cols, rows))
        # 将人脸剪切区域调整为相同的大小
        face_image = cv2.resize(rotated_gray[y:y + h, x:x + w], (100, 100))
        label, confidence = recognizer.predict(face_image)
        if confidence < 100:
            name = labels[label]
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # 当识别出指定人脸后在控制台打印出人脸文件名
            print(name)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 显示图像
    cv2.imshow('frame', frame)

    # 按下 q 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
