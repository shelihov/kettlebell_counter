import cv2
import numpy as np
import time
from ultralytics import YOLO

class KettlebellCounter:
    def __init__(self):
        self.counter = 0
        self.last_count_time = 0
        self.prev_y = None
        self.moving_up = False
        self.rep_started = False
        self.lockout_confirmed = False
        self.direction_buffer = []
        self.buffer_size = 3
        
        # Загружаем модель YOLOv8 для определения позы
        self.model = YOLO('yolov8n-pose.pt')

    def detect_pose(self, frame):
        # Получаем результаты определения позы
        results = self.model(frame, stream=True)
        
        for result in results:
            keypoints = result.keypoints.data
            if len(keypoints) > 0 and keypoints.shape[1] > 10:
                # Получаем координаты правого запястья (индекс 10)
                right_wrist = keypoints[0][10]
                
                # Проверяем уверенность определения
                if right_wrist[2].item() < 0.3:  # Если уверенность низкая
                    continue
                
                # Преобразуем координаты в целые числа
                wrist_point = (
                    int(right_wrist[0].item()),
                    int(right_wrist[1].item())
                )
                
                # Проверяем, что точка в разумных пределах кадра
                height, width = frame.shape[:2]
                if not (0 <= wrist_point[0] <= width and 0 <= wrist_point[1] <= height):
                    continue
                
                return wrist_point, frame
                
        return None, frame

    def count_reps(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не удалось открыть камеру!")
            return

        # Настройка окна и камеры
        width = 640
        height = 360
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        cv2.namedWindow('Kettlebell Counter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Kettlebell Counter', 360, 640)

        # Зоны для рывка
        lockout_zone = height * 0.2
        start_zone = height * 0.7
        min_rep_interval = 0.5

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            movement_point, processed_frame = self.detect_pose(frame)

            if movement_point is not None:
                x, y = movement_point
                
                # Определяем зону левого верхнего угла (10% от размеров кадра)
                corner_zone_x = width * 0.1
                corner_zone_y = height * 0.1
                
                # Проверяем, не находится ли точка в левом верхнем углу
                in_corner = x < corner_zone_x and y < corner_zone_y
                
                # Рисуем более заметную точку запястья
                cv2.circle(processed_frame, movement_point, 15, (0, 0, 255), -1)
                cv2.circle(processed_frame, movement_point, 20, (0, 0, 255), 2)
                
                if self.prev_y is not None:
                    # Определяем текущее направление
                    current_direction = y < self.prev_y - 15
                    
                    self.direction_buffer.append(current_direction)
                    if len(self.direction_buffer) > self.buffer_size:
                        self.direction_buffer.pop(0)
                    
                    self.moving_up = sum(self.direction_buffer) >= len(self.direction_buffer) / 2
                
                current_time = time.time()
                
                # Начинаем новое повторение
                if y > start_zone and not any(self.direction_buffer[-2:]) and not in_corner:
                    self.rep_started = True
                    self.lockout_confirmed = False
                
                # Проверяем фиксацию
                if (self.rep_started and y < lockout_zone and not self.moving_up and 
                    not self.lockout_confirmed and not in_corner):
                    if current_time - self.last_count_time > min_rep_interval:
                        self.counter += 1
                        self.last_count_time = current_time
                        self.lockout_confirmed = True
                
                self.prev_y = y

            # Рисуем зоны
            cv2.line(processed_frame, (0, int(lockout_zone)), 
                    (processed_frame.shape[1], int(lockout_zone)), 
                    (255, 0, 0), 2)
            cv2.line(processed_frame, (0, int(start_zone)), 
                    (processed_frame.shape[1], int(start_zone)), 
                    (0, 140, 140), 2)

            # Отображаем информацию
            cv2.putText(processed_frame, f'Reps: {self.counter}', (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            state = "LOCKOUT" if self.lockout_confirmed else "UP" if self.moving_up else "DOWN"
            cv2.putText(processed_frame, f'State: {state}', (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            cv2.imshow('Kettlebell Counter', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    counter = KettlebellCounter()
    counter.count_reps()
