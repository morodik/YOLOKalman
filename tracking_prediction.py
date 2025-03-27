import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from collections import defaultdict
def predict_future_positions(kf, steps=35):
    future_positions = []
    state = kf.x.copy()
    for _ in range(steps):
        state = np.dot(kf.F, state)
        future_positions.append((int(state[0]), int(state[1])))
    return future_positions

model_path = "best.pt"
model = YOLO(model_path)

# Настройка фильтра Калмана
def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 1000
    kf.R = np.array([[10, 0], [0, 10]])
    kf.Q = np.eye(4) * 0.1
    return kf

video_path = "drone1.mp4"
output_video_path = "output_video_with_tracking.mp4"

# Открытие видео
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Словарь для хранения данных о каждом дроне
drone_data = defaultdict(lambda: {"kf": create_kalman_filter(), "track": [], "id": None})
next_drone_id = 1
max_track_length = 35

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.65, iou=0.3)
    annotated_frame = results[0].plot() # Получаем кадр с bounding boxes от YOLO

    detected_drones = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            detected_drones.append((x_center, y_center, x1, y1, x2, y2))

    # Обновление данных для каждого дрона
    updated_drone_ids = set()
    for det_x, det_y, x1, y1, x2, y2 in detected_drones:
        closest_drone_id = None
        min_distance = float('inf')

        # Поиск ближайшего дрона по расстоянию
        for drone_id, data in drone_data.items():
            last_position = data["track"][-1] if data["track"] else (data["kf"].x[0], data["kf"].x[1])
            distance = np.sqrt((det_x - last_position[0]) ** 2 + (det_y - last_position[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_drone_id = drone_id

        # Если дрон слишком далеко, считаем его новым
        if min_distance > 50:  # Пороговое значение для определения нового дрона
            closest_drone_id = next_drone_id
            drone_data[closest_drone_id] = {"kf": create_kalman_filter(), "track": [], "id": closest_drone_id}
            next_drone_id += 1

        # Обновление данных для найденного или нового дрона
        updated_drone_ids.add(closest_drone_id)
        kf = drone_data[closest_drone_id]["kf"]
        kf.predict()
        kf.update([det_x, det_y])
        x_pred, y_pred = kf.x[:2]
        drone_data[closest_drone_id]["track"].append((int(x_pred), int(y_pred)))
        if len(drone_data[closest_drone_id]["track"]) > max_track_length:
            drone_data[closest_drone_id]["track"].pop(0)

        # Отрисовка истории перемещения (красная линия)
        for i in range(1, len(drone_data[closest_drone_id]["track"])):
            cv2.line(annotated_frame, drone_data[closest_drone_id]["track"][i - 1],
                     drone_data[closest_drone_id]["track"][i], (0, 0, 255), 2)

        # Предсказание будущих позиций
        future_positions = predict_future_positions(kf, steps=35)

        # Отрисовка предсказанной траектории (синяя линия)
        for i in range(len(future_positions) - 1):
            cv2.line(annotated_frame, future_positions[i], future_positions[i + 1], (255, 0, 0), 2)

    # Удаление устаревших дронов
    for drone_id in list(drone_data.keys()):
        if drone_id not in updated_drone_ids:
            del drone_data[drone_id]

    # Сохранение и отображение
    out.write(annotated_frame)
    cv2.imshow("Drone Detection & Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    print(f"Processed frame {frame_count}")

cap.release()
out.release()
cv2.destroyAllWindows()