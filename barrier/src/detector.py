import cv2
import numpy as np
import onnxruntime as ort


class PlateDetector:
    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        confidence: float = 0.45,
        nms_threshold: float = 0.45,
        class_ids=None,
    ):
        self.input_size = input_size
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.class_ids = None if class_ids is None else {int(value) for value in class_ids}

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        """
        Возвращает список найденных областей номеров:
        [(x1, y1, x2, y2, confidence), ...]
        """
        height, width = frame.shape[:2]

        # Масштабирование с сохранением пропорций (letterbox).
        scale = min(
            self.input_size / width,
            self.input_size / height,
        )

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = (self.input_size - new_width) // 2
        pad_y = (self.input_size - new_height) // 2

        canvas = np.full(
            (self.input_size, self.input_size, 3),
            114,
            dtype=np.uint8,
        )

        canvas[
            pad_y:pad_y + new_height,
            pad_x:pad_x + new_width,
        ] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )

        outputs = self.session.run(None, {self.input_name: blob})

        if not outputs:
            return []

        predictions = np.asarray(outputs[0])

        return self._parse_output(
            predictions,
            width,
            height,
            scale,
            pad_x,
            pad_y,
        )

    def _parse_output(
        self,
        output,
        original_width,
        original_height,
        scale,
        pad_x,
        pad_y,
    ):
        output = np.squeeze(output)

        if output.ndim != 2:
            raise ValueError(
                f"Неожиданная форма выхода модели: {output.shape}"
            )

        # Формат [N, 6]: x1, y1, x2, y2, confidence, class_id.
        if output.shape[1] == 6:
            rows = output
            boxes = []
            scores = []

            for row in rows:
                x1, y1, x2, y2, score, class_id = row[:6]

                if self.class_ids is not None and int(class_id) not in self.class_ids:
                    continue
                if score < self.confidence:
                    continue

                boxes.append([
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1),
                ])
                scores.append(float(score))

        else:
            # YOLOv8/YOLO11: [4 + classes, N] либо [N, 4 + classes].
            if output.shape[0] < output.shape[1]:
                rows = output.T
            else:
                rows = output

            if rows.shape[1] < 5:
                raise ValueError(
                    f"Неожиданное число параметров YOLO: {rows.shape}"
                )

            boxes = []
            scores = []

            for row in rows:
                cx, cy, box_width, box_height = row[:4]
                class_scores = row[4:]
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])

                if self.class_ids is not None and class_id not in self.class_ids:
                    continue
                if score < self.confidence:
                    continue

                x1 = float(cx - box_width / 2)
                y1 = float(cy - box_height / 2)
                x2 = float(cx + box_width / 2)
                y2 = float(cy + box_height / 2)

                boxes.append([
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1),
                ])
                scores.append(score)

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        results = []

        for index in np.asarray(indices).flatten():
            x, y, w, h = boxes[int(index)]

            x1 = int((x - pad_x) / scale)
            y1 = int((y - pad_y) / scale)
            x2 = int((x + w - pad_x) / scale)
            y2 = int((y + h - pad_y) / scale)

            x1 = max(0, min(x1, original_width - 1))
            y1 = max(0, min(y1, original_height - 1))
            x2 = max(0, min(x2, original_width))
            y2 = max(0, min(y2, original_height))

            if x2 > x1 and y2 > y1:
                results.append((
                    x1,
                    y1,
                    x2,
                    y2,
                    float(scores[int(index)]),
                ))

        return results