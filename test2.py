import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')

# 创建人脸识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 加载数据集并进行训练Captu
face_images = []
labels = []
for i in range(1, 6):
    for j in range(1, 6):
        image = cv2.imread(f"D:/Art/IMG_5781.JPG")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_image = gray[y:y+h, x:x+w]
            face_images.append(face_image)
            labels.append(i)

recognizer.train(face_images, np.array(labels))

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头捕获的图像
    ret, frame = cap.read()

    # 进行人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 对每个人脸进行识别并在图像中标记
    for (x, y, w, h) in faces:
        face_image = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_image)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示处理后的图像
    cv2.imshow("Face recognition", frame)

    # 按下 q 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()