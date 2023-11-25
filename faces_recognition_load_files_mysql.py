import pickle
import cv2
import numpy as np
import mysql.connector
import re
from PIL import Image, ImageDraw, ImageFont

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# 创建人脸识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 加载images和labels
with open('D:/Art/save/face_images.pkl', 'rb') as f:
    face_images = pickle.load(f)
with open('D:/Art/save/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
print('打开images与labels完成，开始加载人脸数据')

# 加载数据
label_dict = {}
label_num = 0
for label in labels:
    if label not in label_dict:
        label_dict[label] = label_num
        label_num += 1
labels_int = [label_dict[label] for label in labels]
recognizer.train(face_images, np.array(labels_int))
print('加载人脸数据完成')


def connect_to_mysql():
    # 请在此处填写您的数据库连接信息
    connection = mysql.connector.connect(
        host="192.168.1.103",
        user="root",
        password="tyq030420!",
        database="immstest"
    )
    return connection


def get_attendees_list(meeting_id):
    connection = connect_to_mysql()
    cursor = connection.cursor()

    # 根据会议号查询参会人员名单
    query = f"SELECT userId FROM participate WHERE meetingId = {meeting_id}"
    cursor.execute(query)

    attendees_list = [row[0] for row in cursor.fetchall()]
    cursor.close()
    connection.close()

    return attendees_list


def update_attendance_status(attendee_name, recognition_frequency_dect):
    connection = connect_to_mysql()
    cursor = connection.cursor()

    if recognition_frequency_dect[attendee_name] > 20:
        # 将参会人员的是否参会项置为1
        query = f"UPDATE participate SET isAttend = 1 WHERE userId = '{attendee_name}'"
        cursor.execute(query)
        connection.commit()

        cursor.close()
        connection.close()
        print("用户 ", attendee_name, " 已签到")
        recognition_frequency_dect[attendee_name] = -9999999999999

#传入用户ID获取用户名
def id_to_name(user_id):
    connection = connect_to_mysql()
    cursor = connection.cursor()

    query = f"select userName from immstest.userinfo where userId = '{user_id}'"
    cursor.execute(query)
    user_name = cursor.fetchone()

    cursor.close()
    connection.close()
    return user_name


def cv2_add_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)



def face_recognition(attendees_list):
    cap = cv2.VideoCapture(0)
    print('摄像头开启')

    # 识别次数
    recognition_frequency = [0] * len(attendees_list)
    recognition_frequency_dect = {attendees_list: recognition_frequency for attendees_list, recognition_frequency in
                                  zip(attendees_list, recognition_frequency)}

    while True:
        # 读取摄像头捕获的图像
        ret, frame = cap.read()

        # 进行人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 多尺度检测
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        recognized_name = None

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
            #print(label, confidence)
            if confidence < 90:
                name = labels[label]
                name = re.sub('[a-zA-Z_]+', '', name)
                user_name = id_to_name(name)
                user_name = str(user_name)
                cv2.putText(frame, user_name.strip("(')").strip("'),"), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                name = int(name)
                if name in attendees_list:
                    recognized_name = name
                    recognition_frequency_dect[name] = recognition_frequency_dect[name] + 1
                    update_attendance_status(recognized_name, recognition_frequency_dect)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示图像
        cv2.imshow('frame', frame)

        # 按下 q 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('摄像头关闭')
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


def main():
    meeting_id = input('请输入会议号：')
    attendees_list = get_attendees_list(meeting_id)
    print("参会id名单：", attendees_list)
    face_recognition(attendees_list)


if __name__ == "__main__":
    main()
    print('成功退出')
