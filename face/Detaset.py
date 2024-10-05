import cv2
import numpy as np
import os



face_classifier = cv2.CascadeClassifier(
    'C:/Users/jiten/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
)


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    faces_cropped = []
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        faces_cropped.append(cropped_face)

    return faces_cropped

def collect_face_samples():
    person_name = input("Enter your name: ")
    roll_number = input("Enter your roll number: ")

    save_path = f'D:/face-attendance-system-master/face/data/{person_name}_{roll_number}/'

   
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        faces = face_extractor(frame)
        if faces is not None:
            for face in faces:
                count += 1
                face = cv2.resize(face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = os.path.join(save_path, f'{roll_number}_{count}.jpg')
                cv2.imwrite(file_name_path, face)

                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', face)

                if count == 100:
                    print("100 samples collected.")
                    break
        else:
            cv2.putText(frame, "Face not found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Video Frame', frame)

        if cv2.waitKey(1) == 13 or count == 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Sample Collection Completed')

if __name__ == "__main__":
    collect_face_samples()
