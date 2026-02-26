import os
import cv2

DATA_DIR = r'D:\Yorrys\Kontrol Cerdas\Minggu 3\DATA CADANGAN'

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak bisa dibuka")
    exit()

for j in range(number_of_classes):

    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca kamera")
            break

        cv2.putText(frame, 'Ready? Press "Q"!', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        cv2.imshow('frame', frame)

        if cv2.waitKey(15) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca kamera")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(10)

        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()