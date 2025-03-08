from ultralytics import YOLO
import cv2


def process_image(model):
    image_path = "street.jpeg"

    # чтение изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return

    # выполнение детекции объектов
    results = model(image, conf=0.5)

    # отрисовка bounding boxes на изображении
    annotated_image = results[0].plot(boxes=True, conf=True)

    # путь для сохранения результата
    output_path = "output/output_image.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Изображение сохранено в {output_path}")


def process_video(model):
    video_path = "street.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Конец видео")
            break

        # выполнение детекции объектов
        results = model(frame, conf=0.5)

        # отрисовка результатов на кадре
        annotated_frame = results[0].plot(boxes=True, conf=True)

        # отображение кадра
        cv2.imshow("Video Feed", annotated_frame)

        # выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


def process_camera(model):
    # открытие камеры (0 — индекс встроенной камеры)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры")
            break

        # выполнение детекции объектов
        results = model(frame, conf=0.5)

        # отрисовка bounding boxes на кадре
        annotated_frame = results[0].plot(boxes=True, conf=True)

        # отображение обработанного кадра
        cv2.imshow("Camera Feed", annotated_frame)

        # сохранение кадра по нажатию 's'
        if cv2.waitKey(1) & 0xFF == ord('s'):
            output_path = "camera_output.jpg"
            cv2.imwrite(output_path, annotated_frame)
            print(f"Кадр сохранен в {output_path}")

        # выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


def main():
    # загрузка модели YOLOv8
    model = YOLO("yolov8n.pt")

    while True:
        print("\n=== Меню обработки с YOLOv8 ===")
        print("1. Обработать изображение")
        print("2. Обработать видео")
        print("3. Обработать камеру в реальном времени")
        print("4. Выход")

        choice = input("Выберите опцию (1-4): ")

        if choice == "1":
            process_image(model)
        elif choice == "2":
            process_video(model)
        elif choice == "3":
            process_camera(model)
        elif choice == "4":
            print("Выход из программы")
            break
        else:
            print("Неверный выбор, попробуйте снова")


if __name__ == "__main__":
    main()