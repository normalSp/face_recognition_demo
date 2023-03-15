import cv2

# 加载分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')

# 加载图像并进行灰度化处理
image = cv2.imread("D:/Art/09EFAAF66DBA2DB784AA3E0BF8E4D2DE.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行人脸检测
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=3, minSize=(60, 60))

# 在图像中标记人脸
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 显示处理后的图像
cv2.imshow("Faces found", image)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头捕获的图像
    ret, frame = cap.read()

    # 进行人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 在图像中标记人脸并显示已识别
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Recognized", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(1)

    # 显示处理后的图像
    cv2.imshow("Face recognition", frame)

    # 按下 q 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()