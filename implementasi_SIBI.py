import pickle
import cv2
import mediapipe as mp
import numpy as np

# ================== LOAD MODEL ==================
model_dict = pickle.load(open(r'D:\Yorrys\Kontrol Cerdas\Minggu 3\DATA SIBI\model.p', 'rb'))
model = model_dict['model']

# ================== INISIALISASI CAMERA ==================
cap = cv2.VideoCapture(0)

# ================== MEDIAPIPE HANDS ==================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================== LABEL HURUF ==================
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z'
}

# ================== LOOPING ==================
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Gambar landmark
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_style.get_default_hand_landmarks_style(),
                mp_drawing_style.get_default_hand_connections_style()
            )

            # Ambil koordinat landmark
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalisasi (seperti program buka-tutup tangan)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            # Pastikan jumlah fitur sesuai saat training
            if len(data_aux) == 42:  # 21 titik x,y
                prediction = model.predict(np.array([data_aux]))
                predicted_letter = labels_dict[int(prediction[0])]

                # Tampilkan hasil
                cv2.putText(frame,
                            predicted_letter,
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            4)

    cv2.imshow('Deteksi Huruf SIBI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()