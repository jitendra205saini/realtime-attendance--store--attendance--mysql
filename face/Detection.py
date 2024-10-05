import cv2
import numpy as np
import pymysql
from os import listdir
from os.path import isfile, join, isdir
from datetime import datetime

connection = pymysql.connect(
    host='localhost',
    user='enter user name',      
    password='my SQL password',  
    db='database name'
)

try:
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('D:/face-attendance-system-master/face/model/face_trained_model.yml')
    print("Trained Model Loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

data_path = 'D:/face-attendance-system-master/face/data/'
names = {}
label_count = 0

for person_folder in listdir(data_path):
    if isdir(join(data_path, person_folder)):
        name_roll = person_folder.split('_')
        if len(name_roll) == 2:
            names[label_count] = (name_roll[0], name_roll[1])
        label_count += 1

print("Names dictionary:", names)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return img, None, None, None, None, None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        return img, roi, x, y, w, h

def check_attendance(name, roll_number):
    current_date = datetime.now().strftime("%Y_%m_%d")
    check_query = f"SELECT * FROM attendance_{current_date} WHERE name=%s AND roll_number=%s"
    
    with connection.cursor() as cursor:
        try:
            cursor.execute(check_query, (name, roll_number))
            result = cursor.fetchone()
            return result is not None
        except Exception as e:
            print(f"Error checking attendance for {name}: {e}")
            return False

def log_attendance(name, roll_number):
    current_date = datetime.now().strftime("%Y_%m_%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    current_hour = datetime.now().hour
    current_minute = datetime.now().minute
    
    if current_hour > 20 or (current_hour == 20 and current_minute > 0):
        print(f"Attendance not marked for {name} ({roll_number}) because the time is after 8:00 AM.")
        return

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS attendance_{current_date} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        roll_number VARCHAR(50) NOT NULL,
        time VARCHAR(8) NOT NULL
    );
    """
    
    insert_query = f"INSERT INTO attendance_{current_date} (name, roll_number, time) VALUES (%s, %s, %s)"

    with connection.cursor() as cursor:
        try:
            cursor.execute(create_table_query)
            connection.commit()
            print(f"Table 'attendance_{current_date}' created or already exists.")
        except Exception as e:
            print(f"Error creating table 'attendance_{current_date}': {e}")

        try:
            cursor.execute(insert_query, (name, roll_number, current_time))
            connection.commit()
            print(f"Attendance logged for {name} ({roll_number}) at {current_time}.")
        except Exception as e:
            print(f"Error inserting attendance for {name} ({roll_number}): {e}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    image, face, x, y, w, h = face_detector(frame)

    if face is not None:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        print("Prediction result:", result)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))
        else:
            confidence = 0

        if confidence > 75:
            person_name, roll_number = names.get(result[0], ("Unknown", "Unknown"))
            cv2.putText(image, f"{person_name} ({roll_number})", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (42, 235, 35), 2)

            if person_name != "Unknown" and roll_number != "Unknown":
                if not check_attendance(person_name, roll_number):
                    log_attendance(person_name, roll_number)
                else:
                    print(f"Attendance already marked for {person_name} ({roll_number}).")
        else:
            cv2.putText(image, "Unknown", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Face Detector & Recognizer', image)

    if cv2.waitKey(1) & 0xFF == 13:
        break

cap.release()
cv2.destroyAllWindows()
connection.close()
