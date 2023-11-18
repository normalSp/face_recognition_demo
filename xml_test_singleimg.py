import cv2

detector = cv2.CascadeClassifier('D:\\Java\\xml_train\\xml_ok\\900p_54n_cat\\cascade.xml')  # 分类器位置
image_path = 'D:\\Java\\face_img\\img_celeba\\img_celeba\\000129.jpg'  # 指定图片的路径

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()