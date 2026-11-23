
import re
from collections import defaultdict

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


# Допустимые буквы российских регистрационных знаков.
PLATE_LETTERS = "ABEKMHOPCTYX"

PLATE_PATTERN = re.compile(
    rf"[{PLATE_LETTERS}]\d{{3}}[{PLATE_LETTERS}]{{2}}"
)

# Кириллица -> латиница: визуально совпадающие символы.
CYRILLIC_TO_LATIN = str.maketrans({
    "А": "A",
    "В": "B",
    "Е": "E",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "У": "Y",
    "Х": "X",
})

DIGIT_MAP = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
}

LETTER_MAP = {
    "0": "O",
    "8": "B",
    "4": "A",
}

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

PSM_MODES = (7, 8, 13, 6)


class PlateOCR:
    def __init__(
        self,
        language: str = "eng",
        min_confidence: float = 0.0,
    ):
        self.language = language
        self.min_confidence = min_confidence

    @staticmethod
    def normalize(text: str) -> str:
        """Приводит распознанный текст к латинским буквам и цифрам."""
        text = text.upper().translate(CYRILLIC_TO_LATIN)
        return re.sub(r"[^A-Z0-9]", "", text)

    @classmethod
    def extract_plate(cls, text: str) -> str:
        """
        Ищет российский номер формата:
        одна буква + три цифры + две буквы.
        """
        text = cls.normalize(text)

        if len(text) < 6:
            return ""

        for start in range(len(text) - 5):
            candidate = list(text[start:start + 6])

            # Первая позиция — буква.
            candidate[0] = LETTER_MAP.get(
                candidate[0],
                candidate[0],
            )

            # Три центральные позиции — цифры.
            for index in (1, 2, 3):
                candidate[index] = DIGIT_MAP.get(
                    candidate[index],
                    candidate[index],
                )

            # Последние две позиции — буквы.
            for index in (4, 5):
                candidate[index] = LETTER_MAP.get(
                    candidate[index],
                    candidate[index],
                )

            candidate = "".join(candidate)

            if PLATE_PATTERN.fullmatch(candidate):
                return candidate

        return ""

    @staticmethod
    def add_border(image):
        """Добавляет белое поле вокруг номера."""
        return cv2.copyMakeBorder(
            image,
            30,
            30,
            40,
            40,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    def preprocess(self, plate_image):
        """
        Создаёт набор вариантов изображения,
        используемый при Windows-тестировании.
        """
        if plate_image.ndim == 3:
            gray = cv2.cvtColor(
                plate_image,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = plate_image.copy()

        enlarged = cv2.resize(
            gray,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        enhanced = clahe.apply(enlarged)

        _, otsu_original = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        _, otsu = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        adaptive_21 = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            7,
        )

        adaptive_31 = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        blurred = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            1.0,
        )

        sharpened = cv2.addWeighted(
            enhanced,
            1.8,
            blurred,
            -0.8,
            0,
        )

        variants = [
            ("enlarged", enlarged),
            ("enhanced", enhanced),
            ("otsu_original", otsu_original),
            ("otsu", otsu),
            ("adaptive_21", adaptive_21),
            ("adaptive_31", adaptive_31),
            ("sharpened", sharpened),
            ("otsu_inverted", cv2.bitwise_not(otsu)),
        ]

        return [
            (name, self.add_border(image))
            for name, image in variants
        ]

    def recognize_variant(self, image, psm):
        """Распознаёт один вариант изображения в заданном PSM."""
        config = (
            f"--oem 3 --psm {psm} "
            f"-c tessedit_char_whitelist={WHITELIST}"
        )

        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=config,
            output_type=Output.DICT,
        )

        words = []
        confidences = []

        for text, confidence in zip(
            data["text"],
            data["conf"],
        ):
            cleaned = self.normalize(text)

            if not cleaned:
                continue

            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                continue

            if confidence < self.min_confidence:
                continue

            words.append(cleaned)
            confidences.append(max(0.0, confidence))

        raw_text = "".join(words)

        if confidences:
            average_confidence = float(
                np.mean(confidences)
            )
        else:
            average_confidence = 0.0

        plate = self.extract_plate(raw_text)

        return raw_text, plate, average_confidence

    def recognize_variants(self, image):
        """Проверяет все варианты изображения и режимы Tesseract."""
        candidates = defaultdict(list)

        for _variant_name, variant in self.preprocess(image):
            for psm in PSM_MODES:
                _raw, plate, confidence = (
                    self.recognize_variant(variant, psm)
                )

                if plate:
                    candidates[plate].append(confidence)

        if not candidates:
            return ""

        # Сначала количество совпадений, затем средняя уверенность.
        return max(
            candidates,
            key=lambda plate: (
                len(candidates[plate]),
                float(np.mean(candidates[plate])),
            ),
        )

    def recognize(self, plate_image):
        """
        Быстрое распознавание для live-потока.

        Раньше выполнялось до 32 запусков Tesseract на один crop
        (и ещё до 32 при повторной JPEG-проверке). Здесь используется
        один подготовленный вариант и максимум два PSM-прохода.
        """
        if plate_image is None or plate_image.size == 0:
            return ""
        if plate_image.ndim not in (2, 3):
            return ""

        if plate_image.ndim == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # Небольшие номера увеличиваем только до разумного размера.
        target_width = max(gray.shape[1], 240)
        scale = min(3.0, target_width / max(1, gray.shape[1]))
        if scale > 1.0:
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        # Один быстрый бинаризованный вариант вместо 8 дорогих вариантов.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, prepared = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        prepared = cv2.copyMakeBorder(
            prepared, 12, 12, 18, 18,
            cv2.BORDER_CONSTANT, value=255,
        )

        base_config = (
            f"--oem 1 -c tessedit_char_whitelist={WHITELIST} "
            "-c load_system_dawg=0 -c load_freq_dawg=0"
        )
        for psm in (7, 8):
            data = pytesseract.image_to_data(
                prepared,
                lang=self.language,
                config=f"{base_config} --psm {psm}",
                output_type=Output.DICT,
            )
            accepted_words = []
            for token, confidence in zip(data["text"], data["conf"]):
                token = self.normalize(token)
                if not token:
                    continue
                try:
                    score = float(confidence)
                except (TypeError, ValueError):
                    continue
                if score >= self.min_confidence:
                    accepted_words.append(token)
            plate = self.extract_plate("".join(accepted_words))
            if plate:
                return plate

        return ""


class FastPlateOCR:
    """Neural plate-text recognizer using fast-plate-ocr 0.3.x / ONNX Runtime."""

    def __init__(
        self,
        model_name: str = "global-plates-mobile-vit-v2-model",
        device: str = "cpu",
    ):
        try:
            from fast_plate_ocr import ONNXPlateRecognizer
        except ImportError as exc:
            raise RuntimeError(
                "Could not import ONNXPlateRecognizer. "
                "This branch requires fast-plate-ocr 0.3.x."
            ) from exc

        self.model = ONNXPlateRecognizer(
            hub_ocr_model=model_name,
            device=device,
        )

    def recognize(self, plate_image):
        if plate_image is None or plate_image.size == 0:
            return ""
        if plate_image.ndim == 3:
            image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        elif plate_image.ndim == 2:
            image = plate_image
        else:
            return ""

        # fast-plate-ocr 0.3.x expects a grayscale NumPy array and returns
        # a list of strings, even when only one image is passed.
        predictions = self.model.run(image)
        if not predictions:
            return ""
        return PlateOCR.extract_plate(str(predictions[0]))
