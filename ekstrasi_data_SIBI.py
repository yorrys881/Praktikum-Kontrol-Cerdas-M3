import os
import pickle
import mediapipe as mp
import cv2

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1
)

# Path dataset
DATA_DIR = r'D:\Yorrys\Kontrol Cerdas\Minggu 3\DATA SIBI'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):

    class_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(class_path):
        continue

    for img_path in os.listdir(class_path):

        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(class_path, img_path))

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Simpan pickle SETELAH semua data selesai diproses
save_path = r'D:\Yorrys\Kontrol Cerdas\Minggu 3\DATA SIBI\data.pickle'

with open(save_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data berhasil disimpan!")