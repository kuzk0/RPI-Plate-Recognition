# RPI-Plate-Recognition — распознавание автомобильных номеров и управление шлагбаумом

Система для Raspberry Pi 5, камеры IMX219 / Camera Module 2 и демонстрационного сервопривода SG-90. Приложение показывает видеопоток в OpenCV GUI, параллельно обнаруживает транспорт и номерные знаки, распознаёт номер, проверяет его по локальному списку разрешённых номеров и управляет сервоприводом через аппаратный PWM.

> **Назначение:** учебный/демонстрационный макет. SG-90 нельзя использовать для управления тяжёлой реальной стрелой шлагбаума.

## Возможности

- Live preview с камеры в отдельном окне OpenCV.
- Независимый фоновый поток распознавания: GUI не ждёт завершения инференса.
- Два ONNX-детектора: транспортные средства и номерные знаки.
- Переключаемый OCR: FastPlateOCR (основной backend в текущей конфигурации) или Tesseract.
- Нормализация российских регистрационных знаков и проверка локального whitelist.
- Подтверждение номера заданное число последовательных циклов и cooldown.
- Управление SG-90 через RP1 hardware PWM на Raspberry Pi 5.
- Логи времени инференса детекторов и OCR, а также событий доступа.

## Архитектура обработки

1. `Camera` получает кадры от Picamera2 в разрешении, заданном в YAML, и преобразует RGB в BGR для OpenCV.
2. Главный поток непрерывно обновляет кадр и отображает live preview. На кадре рисуются зелёные рамки транспорта, синие/оранжевые рамки номерных знаков и последний распознанный текст.
3. `recognition_worker` в отдельном потоке берёт копию последнего кадра.
4. Детектор транспорта YOLOv8n COCO ищет классы car, motorcycle, bus и truck (class IDs 2, 3, 5, 7). Отдельный YOLO11 detector ищет номерные знаки.
5. Детекции номеров сортируются по confidence. Для первого корректного crop выполняется OCR; за один цикл обрабатывается не более одного номера.
6. `AccessControl` нормализует строку и проверяет whitelist. Неизвестный/пустой номер сбрасывает счётчик подтверждений.
7. При достижении `required_confirmations` и истечении cooldown система вызывает неблокирующий `Barrier.open_for()`.
8. Пока детектор транспорта видит транспортное средство, уже открытый шлагбаум удерживается открытым; после исчезновения транспорта запускается/продолжается таймер закрытия.

Это не полноценная система безопасности: наличие автомобиля само по себе не даёт разрешение на въезд. Разрешение выдаётся только после успешного OCR и проверки whitelist.

## Структура репозитория

```text
.
├── README.md
├── Architecture.md
├── requirements.txt
├── requirements-model-tools.txt
├── barrier/
│   ├── config/
│   │   ├── settings.yaml
│   │   └── allowed_plates.txt
│   ├── models/
│   │   ├── license-plate-finetune-v1n.onnx
│   │   ├── license-plate-finetune-v1n.pt
│   │   ├── license-plate-finetune-v1s.onnx
│   │   ├── license-plate-finetune-v1m.onnx
│   │   ├── license-plate-finetune-v1m.pt
│   │   ├── license-plate-finetune-v1l.onnx
│   │   ├── yolov8n.onnx
│   │   └── yolov8n.pt
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── camera.py
│   │   ├── detector.py
│   │   ├── ocr.py
│   │   ├── access_control.py
│   │   ├── barrier.py
│   │   └── logger.py
│   ├── tests/
│   │   ├── test_access_control.py
│   │   └── test_plate_normalization.py
│   └── windows/
│       ├── README.md
│       ├── config/settings.yaml
│       ├── dataset/images/
│       ├── dataset/labels.csv
│       ├── requirements.txt
│       ├── src/
│       │   ├── detector.py
│       │   ├── ocr.py
│       │   ├── evaluate.py
│       │   └── main.py
│       └── tests/
└── .gitignore
```

Файлы моделей, перечисленные в дереве выше, **уже присутствуют в текущем Git-репозитории**. Их фактический размер и список приведены в `Architecture.md`. В отличие от runtime-каталога, `barrier/windows/results/`, `barrier/data/` и виртуальные окружения являются генерируемыми/локальными данными и не входят в исходный состав репозитория.

## Аппаратная часть

- Raspberry Pi 5 с 64-bit Raspberry Pi OS Desktop.
- Камера IMX219 / Camera Module 2, подключённая к CSI.
- SG-90 для лёгкого демонстрационного макета.
- Отдельный стабилизированный источник 5 В для сервопривода; объедините GND источника и GND Raspberry Pi.
- Сигнал сервопривода: **BCM GPIO13, физический контакт 33**. BCM GPIO13 — это не физический контакт 13.

Проверьте подключение и PWM overlay по документации Raspberry Pi OS для вашей версии системы. После настройки проверьте доступные PWM-устройства через `/sys/class/pwm/` и назначение пина командой `pinctrl get 13`. Не предполагайте, что индекс `pwmchip` одинаков на всех ядрах/overlay.

В текущем `settings.yaml` задано: PWM chip 0, channel 1, 50 Hz. Параметры импульсов открытия/закрытия индивидуально откалиброваны. В частности, `closed_pulse_width_ms: 2.35` превышает общий диапазон `min_pulse_width_ms: 1.0` — это допустимо в текущем коде, поскольку явно заданные импульсы проверяются отдельно на попадание в период PWM, а не ограничиваются общим диапазоном углов. Не переносите эту калибровку на другой сервопривод без проверки.

Перед калибровкой отсоедините рог/тягу от механизма. Увеличивайте импульс малыми шагами; остановитесь при гуле, нагреве или механическом упоре.

## Установка на Raspberry Pi

Picamera2 и libcamera следует устанавливать системными пакетами APT, а не pip:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y \
  python3 python3-venv python3-pip python3-dev \
  python3-picamera2 python3-libcamera \
  tesseract-ocr tesseract-ocr-eng \
  libgtk-3-0 libgl1 libglib2.0-0
```

Проверьте обнаружение камеры:

```bash
rpicam-hello --list-cameras
```

Создайте виртуальное окружение с доступом к системным пакетам — это позволяет использовать установленный через APT Picamera2:

```bash
git clone https://github.com/MaksSmirnovPro/RPI-Plate-Recognition.git
cd RPI-Plate-Recognition
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Установите зависимости:

```bash
python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless
python -m pip install -r requirements.txt
```

Для live preview нужна GUI-сборка `opencv-python`, не headless. Не устанавливайте одновременно несколько конфликтующих вариантов OpenCV.

### Зависимости

| Файл/пакет | Назначение |
|---|---|
| `requirements.txt` | Runtime-зависимости Raspberry Pi |
| `numpy` | Массивы и обработка изображений |
| `opencv-python` | OpenCV, детекция/NMS и GUI preview |
| `onnxruntime` | CPU-инференс детекторов ONNX |
| `fast-plate-ocr` | Нейросетевой OCR через ONNX Runtime |
| `pytesseract` | Альтернативный OCR backend |
| `PyYAML` | Загрузка YAML-конфигурации |
| `huggingface_hub` | Загрузка модели OCR при первом запуске |
| `rpi-hardware-pwm` | Аппаратный PWM Raspberry Pi 5 |
| `requirements-model-tools.txt` | Необязательные инструменты экспорта YOLO на ПК (`ultralytics`) |

PyTorch и Ultralytics не нужны на Raspberry Pi для запуска ONNX-моделей. `requirements-model-tools.txt` предназначен только для рабочего ПК, если требуется экспортировать/подготовить модель.

## Модели

Runtime загружает модели по путям из `barrier/config/settings.yaml` относительно каталога `barrier/`.

| Назначение | Файл в репозитории | Примечание |
|---|---|---|
| Детектор номеров | `license-plate-finetune-v1n.onnx` | Текущий runtime-файл, YOLO11n |
| Варианты детектора номеров | `license-plate-finetune-v1s.onnx`, `v1m.onnx`, `v1l.onnx` | Дополнительные размеры |
| Исходные веса plate detector | `license-plate-finetune-v1n.pt`, `v1m.pt` | Веса PyTorch; runtime их не загружает |
| Детектор транспорта | `yolov8n.onnx` | COCO, классы 2/3/5/7 |
| Исходные веса транспорта | `yolov8n.pt` | Для разработки/экспорта, runtime использует ONNX |

Все перечисленные файлы находятся в `barrier/models/` на текущей ветке. Runtime требует как минимум `license-plate-finetune-v1n.onnx` и `yolov8n.onnx`.

Если нужно заново экспортировать YOLOv8n на ПК:

```bash
python -m pip install -r requirements-model-tools.txt
yolo export model=yolov8n.pt format=onnx imgsz=640
```

Затем поместите результат в `barrier/models/yolov8n.onnx`. Для замены plate detector убедитесь, что выход ONNX совместим с парсером в `barrier/src/detector.py`.

FastPlateOCR загружает OCR-модель из Hugging Face Hub при инициализации, если она ещё не закеширована. Первый запуск требует доступа к сети.

## Конфигурация

Основной файл: `barrier/config/settings.yaml`.

| Секция | Ключи | Назначение |
|---|---|---|
| `camera` | `width`, `height` | Разрешение Picamera2 |
| `detector` | `model_path`, `input_size`, `confidence`, `nms_threshold` | Детектор номера |
| `detector` | `vehicle_model_path`, `vehicle_input_size`, `vehicle_confidence`, `vehicle_nms_threshold` | Детектор транспорта |
| `ocr` | `backend` | `fast_plate_ocr` или `tesseract` |
| `ocr` | `model_name`, `device` | FastPlateOCR model ID и устройство |
| `ocr` | `language`, `min_confidence` | Параметры Tesseract |
| `access` | `allowed_plates_file` | Путь к whitelist относительно `barrier/` |
| `access` | `required_confirmations`, `cooldown_seconds` | Подтверждения и интервал между разрешениями |
| `barrier` | `pwm_chip`, `pwm_channel`, `frequency_hz` | PWM-устройство |
| `barrier` | `min_pulse_width_ms`, `max_pulse_width_ms` | Диапазон преобразования угла |
| `barrier` | `closed_pulse_width_ms`, `opened_pulse_width_ms` | Опциональные точные импульсы для двух положений |
| `barrier` | `closed_angle`, `opened_angle`, `open_duration` | Углы fallback и длительность открытия |
| `display` | `enabled` | Включение OpenCV GUI |
| `processing` | `interval_seconds` | Пауза между циклами фоновой обработки |

Whitelist хранится в `barrier/config/allowed_plates.txt`: один номер на строку. Пустые строки и строки, начинающиеся с `#`, игнорируются. Сравнение выполняется после нормализации: верхний регистр, перевод визуально совпадающих кириллических букв в латиницу и удаление разделителей.

Текущая конфигурация использует `fast_plate_ocr`, CPU и интервал обработки 0.5 секунды. Для переключения на Tesseract установите `ocr.backend: tesseract`; системный исполняемый файл Tesseract и языковой пакет должны быть установлены отдельно.

## Запуск

Запускайте из графической сессии Raspberry Pi OS:

```bash
cd ~/RPI-Plate-Recognition
source .venv/bin/activate
cd barrier
python -m src.main
```

При `display.enabled: true` требуются GUI-сборка OpenCV и переменная окружения `DISPLAY` или `WAYLAND_DISPLAY`. Нажмите **Q** в окне видео для завершения.

Для работы приложения нужны оба runtime ONNX-файла и whitelist:

```bash
cd barrier
ls -lh models/license-plate-finetune-v1n.onnx models/yolov8n.onnx
cat config/allowed_plates.txt
```

## Тесты и диагностика

Тесты основного runtime:

```bash
cd barrier
python -m pip install pytest
pytest -v
```

Текущие unit-тесты проверяют whitelist, последовательные подтверждения, сброс подтверждений и нормализацию строки. Они не проверяют камеру, PWM, физическое движение сервопривода, GUI или точность ONNX-моделей на реальном потоке.

Логи записываются в консоль и `barrier/data/logs/barrier.log` (до 2 MB на файл, 3 резервные копии). Время детекции двух YOLO-моделей логируется совместно; время OCR — отдельно.

Проверка окружения:

```bash
python - <<'PY'
import cv2, onnxruntime, pytesseract
print("OpenCV:", cv2.__version__)
print("GUI:", [x.strip() for x in cv2.getBuildInformation().splitlines()
               if x.strip().startswith("GUI:")])
print("ONNX Runtime:", onnxruntime.__version__)
print("Tesseract:", pytesseract.get_tesseract_version())
PY
```

Если используется FastPlateOCR, дополнительно проверьте, что установлен пакет версии 0.3.x и доступна загрузка модели. Для PWM проверьте реальное соответствие chip/channel и GPIO13 на самом Raspberry Pi.

## Windows-подпроект

Каталог `barrier/windows/` содержит отдельную offline-утилиту оценки детектора и Tesseract OCR по размеченному датасету. Она не управляет камерой, GPIO или сервоприводом и не является Windows-портом runtime. Подробная инструкция: [barrier/windows/README.md](barrier/windows/README.md).

## Ограничения и безопасность

- OCR и детекция могут ошибаться из-за освещения, бликов, загрязнения, угла обзора и качества изображения. Проверяйте точность на собственном датасете.
- FastPlateOCR global-модель не гарантирует точность на конкретных российских номерах; оцените её на размеченных кадрах до использования.
- `required_confirmations: 1` означает, что одного распознавания достаточно. Для более строгого режима увеличьте значение и проверьте задержку.
- Cooldown ограничивает повторные разрешения, но не заменяет датчики безопасности.
- Удержание открытого положения при наличии транспортного средства основано на детекторе транспорта, а не на датчике физического положения/проезда.
- Предусмотрите независимое ручное аварийное управление и безопасное поведение при отказе питания/связи.
- Не используйте SG-90 для тяжёлой стрелы. Реальный шлагбаум требует промышленного привода, концевых датчиков и защиты от защемления.
