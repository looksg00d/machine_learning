import cv2
import face_recognition
import mediapipe as mp
import os

known_face_encodings = []
known_face_names = ["Nail Samigullin"]

known_faces_dir = "known_faces"
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # перевод изображения в RGB для face recognition тк он читает в RGB а не BGR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # поиск лиц и их кодировок в текущем кадре видео
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # если совпадение найдено, используйте первое совпадение
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # рамка вокруг лица и подпись
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # руки
        results_hands = hands.process(rgb_image)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                # расчет поднятых пальцев
                finger_up = [False] * 5
                landmarks = hand_landmarks.landmark
                # большой палец
                finger_up[0] = landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x if handedness.classification[0].label == 'Right' else \
                               landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x
                # указательный и остальные пальцы
                for i, point in enumerate([mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]):
                    finger_up[i+1] = landmarks[point].y < landmarks[point - 2].y

                count_fingers_up = sum(finger_up)

                # вывод подписи в зависимости от количества пальцев и стороны руки
                if count_fingers_up == 1:
                    label = "Nail" if handedness.classification[0].label == 'Right' else "Unknown"
                elif count_fingers_up == 2:
                    label = "Samigullin" if handedness.classification[0].label == 'Right' else "Unknown"
                else:
                    label = "Unknown"

                # текст под лицом
                cv2.putText(image, label, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Video Feed', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
