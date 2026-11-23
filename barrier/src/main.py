
import os
import threading
import time
from pathlib import Path

import cv2
import yaml

from src.camera import Camera
from src.detector import PlateDetector
from src.ocr import PlateOCR, FastPlateOCR
from src.access_control import AccessControl
from src.barrier import Barrier
from src.logger import setup_logger


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "settings.yaml"


class LicensePlateSystem:

    def __init__(self):
        self.logger = setup_logger()
        self.config = self.load_config()

        # GUI display is enabled by default; use opencv-python (not headless).
        display_cfg = self.config.get("display", {})
        self.display_enabled = bool(display_cfg.get("enabled", True))

        # Camera
        camera_cfg = self.config["camera"]

        self.camera = Camera(
            width=camera_cfg.get("width", 640),
            height=camera_cfg.get("height", 480),
        )

        # Detector
        detector_cfg = self.config["detector"]

        model_path = PROJECT_DIR / detector_cfg["model_path"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}"
            )

        self.detector = PlateDetector(
            model_path=str(model_path),
            input_size=detector_cfg.get("input_size", 640),
            confidence=detector_cfg.get("confidence", 0.45),
            nms_threshold=detector_cfg.get("nms_threshold", 0.45),
        )

        vehicle_model_path = PROJECT_DIR / detector_cfg.get(
            "vehicle_model_path", "models/yolov8n.onnx"
        )
        if not vehicle_model_path.exists():
            raise FileNotFoundError(
                f"Vehicle detection model not found: {vehicle_model_path}. "
                "Download yolov8n.onnx as described in README.md."
            )
        self.vehicle_detector = PlateDetector(
            model_path=str(vehicle_model_path),
            input_size=detector_cfg.get("vehicle_input_size", 640),
            confidence=detector_cfg.get("vehicle_confidence", 0.35),
            nms_threshold=detector_cfg.get("vehicle_nms_threshold", 0.45),
            class_ids=(2, 3, 5, 7),
        )

        # OCR
        ocr_cfg = self.config["ocr"]

        ocr_backend = ocr_cfg.get("backend", "tesseract").lower()
        if ocr_backend == "fast_plate_ocr":
            self.ocr = FastPlateOCR(
                model_name=ocr_cfg.get("model_name", "cct-xs-v2-global-model"),
                device=ocr_cfg.get("device", "cpu"),
            )
        elif ocr_backend == "tesseract":
            self.ocr = PlateOCR(
                language=ocr_cfg.get("language", "eng"),
                min_confidence=ocr_cfg.get("min_confidence", 0),
            )
        else:
            raise ValueError(f"Unsupported OCR backend: {ocr_backend}")

        # Access control
        access_cfg = self.config["access"]

        allowed_file = (
            PROJECT_DIR / access_cfg["allowed_plates_file"]
        )

        if not allowed_file.exists():
            raise FileNotFoundError(
                f"Allowed plates file not found: {allowed_file}"
            )

        self.access = AccessControl(
            allowed_plates_file=str(allowed_file),
            required_confirmations=access_cfg.get(
                "required_confirmations", 3
            ),
            cooldown_seconds=access_cfg.get(
                "cooldown_seconds", 15
            ),
        )

        # Barrier
        barrier_cfg = self.config["barrier"]

        self.barrier = Barrier(
            pwm_channel=barrier_cfg.get("pwm_channel", 1),
            pwm_chip=barrier_cfg.get("pwm_chip", 0),
            frequency_hz=barrier_cfg.get("frequency_hz", 50),
            min_pulse_width_ms=barrier_cfg.get("min_pulse_width_ms", 1.0),
            max_pulse_width_ms=barrier_cfg.get("max_pulse_width_ms", 2.0),
            closed_angle=barrier_cfg.get("closed_angle", 180),
            opened_angle=barrier_cfg.get("opened_angle", 90),
            open_duration=barrier_cfg.get("open_duration", 10),
            closed_pulse_width_ms=barrier_cfg.get("closed_pulse_width_ms"),
            opened_pulse_width_ms=barrier_cfg.get("opened_pulse_width_ms"),
        )

        # Processing
        processing_cfg = self.config.get("processing", {})

        self.processing_interval = float(
            processing_cfg.get("interval_seconds", 1.5)
        )

        self.result_display_seconds = 3.0

        # Shared state
        self.frame_lock = threading.Lock()

        self.latest_frame = None
        self.result_frame = None
        self.result_until = 0.0
        self.latest_vehicle_detections = []
        self.latest_plate_detections = []
        self.recognized_text = ""
        self.recognized_until = 0.0

        self.stop_event = threading.Event()
        self.worker = None

    @staticmethod
    def load_config():
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Configuration not found: {CONFIG_PATH}"
            )

        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def recognition_worker(self):
        """
        Фоновый поток:
        detection -> OCR -> access control -> barrier.

        Управление шлагбаумом не блокирует распознавание.
        """

        self.logger.info("Recognition worker started")

        while not self.stop_event.is_set():

            with self.frame_lock:
                if self.latest_frame is None:
                    frame = None
                else:
                    frame = self.latest_frame.copy()

            if frame is None:
                self.stop_event.wait(0.05)
                continue

            try:
                # Both detectors run in the worker; GUI keeps rendering camera frames.
                detection_started = time.perf_counter()
                vehicle_detections = self.vehicle_detector.detect(frame)
                detections = self.detector.detect(frame)
                detection_elapsed = time.perf_counter() - detection_started
                self.logger.info("YOLO detection completed in %.3f s", detection_elapsed)
                with self.frame_lock:
                    self.latest_vehicle_detections = vehicle_detections
                    self.latest_plate_detections = detections

                # Сортируем по confidence
                detections = sorted(
                    detections,
                    key=lambda item: item[4],
                    reverse=True,
                )

                # Наличие обнаруженной номерной рамки.
                # Пока рамка видна, продлеваем открытое состояние.
                vehicle_present = len(vehicle_detections) > 0

                self.barrier.update(
                    vehicle_present=vehicle_present
                )

                recognized = False

                height, width = frame.shape[:2]

                for x1, y1, x2, y2, confidence in detections:

                    x1, y1, x2, y2 = map(
                        int, (x1, y1, x2, y2)
                    )

                    x1 = max(0, min(x1, width))
                    x2 = max(0, min(x2, width))
                    y1 = max(0, min(y1, height))
                    y2 = max(0, min(y2, height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    ocr_started = time.perf_counter()
                    plate = self.ocr.recognize(crop)
                    ocr_elapsed = time.perf_counter() - ocr_started
                    self.logger.info("OCR completed in %.3f s", ocr_elapsed)

                    if not plate:
                        continue

                    recognized = True

                    self.logger.info(
                        "Recognized plate: %s | YOLO confidence: %.3f",
                        plate,
                        confidence,
                    )

                    # Отображение результата
                    display_frame = frame.copy()

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    text_y = max(y1 - 12, 30)

                    cv2.putText(
                        display_frame,
                        plate,
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    with self.frame_lock:
                        self.recognized_text = plate
                        self.recognized_until = time.monotonic() + self.result_display_seconds

                    # Проверка доступа
                    if self.access.confirm_plate(plate):

                        self.logger.info(
                            "ACCESS GRANTED: %s",
                            plate,
                        )

                        # ВАЖНО:
                        # open_for() теперь неблокирующий.
                        # Поток распознавания продолжит работу.
                        self.barrier.open_for()

                    else:
                        self.logger.info(
                            "Access not granted: %s",
                            plate,
                        )

                    # Обрабатываем один номер за цикл
                    break

                if not recognized:
                    self.access.confirm_plate("")

            except Exception:
                self.logger.exception(
                    "Recognition worker error"
                )

            self.stop_event.wait(
                self.processing_interval
            )

        self.logger.info("Recognition worker stopped")

    def run(self):

        self.logger.info(
            "Starting license plate recognition system"
        )

        window_name = "License Plate Recognition"
        window_created = False

        try:
            if self.display_enabled:
                build_info = cv2.getBuildInformation()
                gui_line = next(
                    (line.strip() for line in build_info.splitlines()
                     if line.strip().startswith("GUI:")),
                    "GUI: unknown",
                )
                if "NONE" in gui_line.upper():
                    raise RuntimeError(
                        "OpenCV was installed without GUI support. "
                        "Install opencv-python and remove opencv-python-headless."
                    )
                if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                    raise RuntimeError(
                        "No desktop display session detected (DISPLAY/WAYLAND_DISPLAY is unset). "
                        "Run the application from the Raspberry Pi OS desktop session."
                    )
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                window_created = True
                self.logger.info("Video window created. Press Q to exit.")
            else:
                self.logger.info("Video display disabled by configuration.")

            self.camera.start()

            self.barrier.close()

            self.worker = threading.Thread(
                target=self.recognition_worker,
                name="RecognitionWorker",
                daemon=True,
            )

            self.worker.start()

            while not self.stop_event.is_set():

                # Камера продолжает работать независимо
                # от процесса распознавания.
                frame = self.camera.read()

                if frame is None:
                    self.stop_event.wait(0.01)
                    continue

                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    vehicle_boxes = list(self.latest_vehicle_detections)
                    plate_boxes = list(self.latest_plate_detections)
                    recognized_text = self.recognized_text if time.monotonic() < self.recognized_until else ""

                if self.display_enabled:
                    display_frame = frame.copy()
                    height, width = display_frame.shape[:2]
                    for x1, y1, x2, y2, score in vehicle_boxes:
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Vehicle {score:.2f}", (x1, max(20, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    for x1, y1, x2, y2, score in plate_boxes:
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
                        cv2.putText(display_frame, f"Plate {score:.2f}", (x1, min(height - 5, y2 + 18)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2, cv2.LINE_AA)
                    if recognized_text:
                        cv2.putText(display_frame, recognized_text, (12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        self.logger.info(
                            "Exit requested by user"
                        )
                        break

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt")

        except Exception:
            self.logger.exception(
                "Fatal application error"
            )

        finally:
            self.stop_event.set()

            if self.worker is not None:
                self.worker.join(timeout=5.0)

            if window_created:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    self.logger.warning(
                        "OpenCV windows could not be closed"
                    )

            try:
                self.camera.stop()
            except Exception:
                self.logger.exception(
                    "Camera shutdown error"
                )

            try:
                self.barrier.stop()
            except Exception:
                self.logger.exception(
                    "Barrier shutdown error"
                )

            self.logger.info(
                "License plate recognition system stopped"
            )


def main():
    system = LicensePlateSystem()
    system.run()


if __name__ == "__main__":
    main()
