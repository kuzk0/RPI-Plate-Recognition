# Windows-подпроект — офлайн-оценка детектора и OCR

## Назначение

Каталог `barrier/windows/` содержит отдельный инструмент разработки и оценки качества распознавания на локальном датасете. Это **не Windows-порт Raspberry Pi runtime**: здесь нет Picamera2, GPIO, PWM, сервопривода, live camera loop или AccessControl.

Pipeline: изображение из датасета → ONNX plate detector → crop найденных номеров → Tesseract OCR → сравнение с эталонной CSV-разметкой → CSV-отчёт и аннотированные изображения.

## Состав

- `src/main.py` — CLI и сборка evaluator pipeline.
- `src/detector.py` — ONNX Runtime plate detector с letterbox, parser формата выхода и NMS.
- `src/ocr.py` — Tesseract OCR, 8 вариантов предобработки, 4 PSM режима, голосование; при отсутствии результата повторяет распознавание на JPEG quality 95.
- `src/evaluate.py` — чтение labels CSV, детекция, OCR, exact match, IoU, сохранение результатов.
- `config/settings.yaml` — пути и параметры модели, OCR, датасета и evaluation.
- `dataset/images/` и `dataset/labels.csv` — пример датасета, уже присутствующий в репозитории (4 изображения).
- `tests/test_normalization.py` — pytest unit-тесты нормализации и IoU.
- `tests/test_ocr.py`, `test_ocr_1.py`, `test_rectification.py` — диагностические скрипты, не обычные unit-тесты.

## Требования

- Windows 10/11.
- Python 3.10 или 3.11.
- Tesseract OCR for Windows; добавьте `tesseract.exe` в PATH или укажите его полный путь в `config/settings.yaml`.
- ONNX plate detector. В текущем репозитории он уже хранится в `barrier/models/license-plate-finetune-v1n.onnx`.

## Установка

Откройте PowerShell в каталоге проекта:

```powershell
cd barrier\windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Проверка Tesseract:

```powershell
tesseract --version
```

Если Tesseract не находится в PATH, задайте `ocr.tesseract_cmd` в `config/settings.yaml`.

## Датасет

Изображения размещаются в `dataset/images/`. CSV-файл `dataset/labels.csv` должен содержать столбцы `image` и `plate`; координаты bbox необязательны:

```csv
image,plate,x1,y1,x2,y2
1.jpg,C007CY,120,180,310,230
2.jpg,M777PO,85,140,270,190
3.jpg,C080TT,,,,
4.jpg,O908OO,,,,
```

В репозитории сейчас лежат файлы `1.jpg`–`4.jpg` и `labels.csv`; пример выше показывает формат CSV, а не утверждает, что координаты всех текущих строк заполнены.

## Конфигурация

В `config/settings.yaml`:

- `model.path`: путь относительно каталога `barrier/windows/`. Текущее значение `../models/license-plate-finetune-v1n.onnx` указывает на общий файл `barrier/models/license-plate-finetune-v1n.onnx`.
- `model.input_size`, `confidence`, `nms_threshold`: параметры детектора.
- `model.output_format`: `auto`, `yolo` или `nms`.
- `ocr.language`, `ocr.min_confidence`, `ocr.tesseract_cmd`: настройки Tesseract.
- `dataset.images_dir`, `dataset.labels_file`: входной датасет.
- `evaluation.save_images`, `show_images`, `save_csv`, `iou_threshold`: параметры отчёта. Обратите внимание: текущий `main.py` использует `save_images` и `iou_threshold`; `show_images` и `save_csv` в конфиге не читаются — показ включается флагом CLI `--show`, CSV сохраняется evaluator-ом всегда.

## Запуск оценки

```powershell
python -m src.main
```

Показ аннотированных изображений после обработки:

```powershell
python -m src.main --show
```

Результаты создаются в `results/` (каталог игнорируется Git):

- `results/results.csv` — expected/predicted, detected, exact_match, IoU, detection_match и status.
- `results/annotated/` — изображения с предсказанными рамками и эталонной рамкой, если bbox указан.
- `results/crops/` — вырезанные области детекции для анализа OCR.

Метрики:
- **Detection coverage** — доля изображений, на которых найден хотя бы один номер.
- **Exact OCR accuracy (end-to-end)** — доля изображений с полным совпадением номера; отсутствие детекции/OCR считается ошибкой.
- **Detection IoU success** — доля записей с эталонным bbox, у которых IoU не ниже заданного порога.

Если несколько номеров распознаны на изображении, evaluator выбирает распознанный результат с наибольшим confidence детектора, а не сравнивает все варианты с эталоном.

## Тесты и диагностика

```powershell
pytest -v tests/test_normalization.py
```

Файлы `test_ocr.py` и `test_ocr_1.py` — ручные диагностические программы, требующие crops в `results/crops/`; их запускают как скрипты из подходящего окружения. `test_rectification.py` — пакетный диагностический инструмент для сглаживания/аппроксимации границ crop и сравнения OCR до/после выпрямления; он также требует заранее подготовленные crop-файлы.

## Отличия от Raspberry Pi runtime

| Возможность | Raspberry Pi runtime | Windows evaluation |
|---|---|---|
| Источник кадров | Picamera2 / IMX219 | Изображения из CSV-датасета |
| Детектор | Две ONNX модели: vehicle и plate | Одна plate ONNX модель |
| OCR | FastPlateOCR или Tesseract, конфигурируется | Tesseract |
| AccessControl / whitelist | Да | Нет; exact match с эталоном |
| GUI live preview | OpenCV window | Опциональный показ результатов после оценки |
| PWM / SG-90 | Да | Нет |
| Метрики датасета | Нет отдельного evaluator в runtime | CSV, coverage, exact match, IoU |

