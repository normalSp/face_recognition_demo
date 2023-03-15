import os
import cv2

images_dir = 'path/to/images'  # 图片所在目录
image_files = os.listdir(images_dir)  # 获取目录中的所有文件名

images = []  # 用于存储所有图片的列表

# 遍历所有图片文件，并将它们加载到列表中
for file_name in image_files:
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        # 使用绝对路径加载图片，并将其添加到列表中
        image_path = os.path.join(images_dir, file_name)
        image = cv2.imread(image_path)
        images.append(image)

# 显示列表中的所有图片
for image in images:
    cv2.imshow('Image', image)
    cv2.waitKey(0)

# 释放窗口并退出程序
cv2.destroyAllWindows()