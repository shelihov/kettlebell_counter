import cv2
import numpy as np
import time

class KettlebellCounter:
    def __init__(self):
        self.counter = 0
        self.last_count_time = 0
        self.prev_y = None
        self.moving_up = False
        self.rep_started = False
        self.rep_zone_entered = False
        self.direction_buffer = []  # Буфер для сглаживания определения направления
        self.buffer_size = 3  # Размер буфера
        self.lockout_confirmed = False  # Флаг фиксации в верхней точке
        # Инициализируем вычитатель фона
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,        # Количество кадров для истории
            varThreshold=16,    # Порог определения переднего плана
            detectShadows=False # Отключаем определение теней
        )
        # Параметры для определения цвета кожи в HSV
        self.lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 150, 255], dtype=np.uint8)
        # Добавляем отслеживание предыдущей позиции
        self.last_valid_point = None
        self.lost_tracking_frames = 0
        self.max_lost_frames = 10

    def detect_movement(self, frame):
        # Конвертируем в HSV для определения кожи
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Создаем маску для кожи
        skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Применяем вычитание фона
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Комбинируем маски движения и кожи
        combined_mask = cv2.bitwise_and(fg_mask, skin_mask)
        
        # Морфологические операции для уменьшения шума
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        # Находим контуры
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.lost_tracking_frames += 1
            if self.lost_tracking_frames > self.max_lost_frames:
                self.last_valid_point = None
            return None, combined_mask
            
        # Находим подходящий контур
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 5000:  # Ограничиваем размер области
                # Получаем ограничивающий прямоугольник
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                # Проверяем соотношение сторон (для руки оно обычно меньше 1)
                if aspect_ratio < 1.0:
                    valid_contours.append(cnt)
        
        if not valid_contours:
            self.lost_tracking_frames += 1
            if self.lost_tracking_frames > self.max_lost_frames:
                self.last_valid_point = None
            return None, combined_mask
        
        # Выбираем контур ближайший к последней валидной точке
        if self.last_valid_point is not None:
            # Находим контур ближайший к последней точке
            closest_contour = min(valid_contours, 
                key=lambda cnt: abs(cnt[:, :, 0].mean() - self.last_valid_point[0]) + 
                              abs(cnt[:, :, 1].mean() - self.last_valid_point[1]))
        else:
            # Если нет последней точки, берем самый большой подходящий контур
            closest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Находим верхнюю точку контура
        top_point = tuple(closest_contour[closest_contour[:, :, 1].argmin()][0])
        
        # Проверяем, не слишком ли резко изменилась позиция
        if self.last_valid_point is not None:
            dx = abs(top_point[0] - self.last_valid_point[0])
            dy = abs(top_point[1] - self.last_valid_point[1])
            if dx > 100 or dy > 100:  # Максимальное допустимое смещение
                self.lost_tracking_frames += 1
                if self.lost_tracking_frames > self.max_lost_frames:
                    self.last_valid_point = None
                return None, combined_mask
        
        # Обновляем последнюю валидную точку
        self.last_valid_point = top_point
        self.lost_tracking_frames = 0
        
        # Рисуем контур на кадре
        cv2.drawContours(frame, [closest_contour], -1, (0, 255, 0), 2)
        
        return top_point, combined_mask

    def count_reps(self):
        # Инициализация камеры
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не удалось открыть камеру!")
            return

        # Устанавливаем горизонтальное разрешение
        width = 640
        height = 360
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Создаем окно
        cv2.namedWindow('Kettlebell Counter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Kettlebell Counter', 360, 640)  # Вертикальное окно

        # Зона подсчета для рывка
        lockout_zone = height * 0.2  # Зона фиксации вверху
        start_zone = height * 0.7    # Стартовая зона внизу
        min_rep_interval = 0.5

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            movement_point, thresh = self.detect_movement(frame)

            if movement_point is not None:
                x, y = movement_point
                
                cv2.circle(frame, movement_point, 10, (0, 0, 255), -1)
                
                if self.prev_y is not None:
                    # Определяем текущее направление
                    current_direction = y < self.prev_y - 15
                    
                    self.direction_buffer.append(current_direction)
                    if len(self.direction_buffer) > self.buffer_size:
                        self.direction_buffer.pop(0)
                    
                    self.moving_up = sum(self.direction_buffer) >= len(self.direction_buffer) / 2
                
                current_time = time.time()
                
                # Начинаем новое повторение в стартовой позиции
                if y > start_zone and not any(self.direction_buffer[-2:]):
                    self.rep_started = True
                    self.rep_zone_entered = False
                    self.lockout_confirmed = False
                
                # Проверяем фиксацию в верхней точке
                if (self.rep_started and y < lockout_zone and not self.moving_up and 
                    not self.lockout_confirmed):
                    # Ждем небольшую паузу для подтверждения фиксации
                    if current_time - self.last_count_time > min_rep_interval:
                        self.counter += 1
                        self.last_count_time = current_time
                        self.lockout_confirmed = True
                
                self.prev_y = y

            # Рисуем зоны
            # Зона фиксации (синяя)
            cv2.line(frame, (0, int(lockout_zone)), (frame.shape[1], int(lockout_zone)), 
                    (255, 0, 0), 2)
            # Стартовая зона (темно-желтая)
            cv2.line(frame, (0, int(start_zone)), (frame.shape[1], int(start_zone)), 
                    (0, 140, 140), 2)

            # Отображаем счетчик и состояние
            cv2.putText(frame, f'Reps: {self.counter}', (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            state = "LOCKOUT" if self.lockout_confirmed else "UP" if self.moving_up else "DOWN"
            cv2.putText(frame, f'State: {state}', (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            # Показываем кадр
            cv2.imshow('Kettlebell Counter', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    counter = KettlebellCounter()
    counter.count_reps()
