import cv2
detector = cv2.CascadeClassifier('D:\\Java\\xml_train\\xml_ok\\920p_86n\\cascade.xml')# 分类器位置
cap = cv2.VideoCapture(0)
cap.set(1,10) #设置分辨率
cap.set(2,10)

while True:
    print('step 1')
    ret, img = cap.read()
    print('step 2')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('step 3')
    faces = detector.detectMultiScale(gray, 1.1, 5)
    print('step 4')
    for (x, y, w, h) in faces:
        print('step 5')
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print('step 6 finish')
cap.release()
cv2.destroyAllWindows()