from typing import Any

import cv2 as cv
import keras
import numpy as np
from fastapi import UploadFile

from app.models import Box, Label


# Função para calcular a distância entre dois pontos
def distancia(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def get_bounding_boxes(img: np.ndarray, templates: list[np.ndarray], threshold: float) -> list[Box]:
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    w, h = templates[0].shape[::-1]
    resized_templates = (cv.resize(template, (w, h)) for template in templates)
    matches = (cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED) for template in resized_templates)

    loc = np.where(np.logical_or.reduce([match >= threshold for match in matches]))

    boxes: list[Box] = []

    for pt in zip(*loc[::-1]):
        # Verificar se o ponto está muito próximo de algum ponto já processado
        if all(distancia(pt, (p.x_min, p.y_min)) > 100 for p in boxes):
            box = Box(x_min=pt[0], y_min=pt[1], x_max=pt[0] + w, y_max=pt[1] + h)
            boxes.append(box)

    return boxes


def classify_boxes(img: np.ndarray, boxes: list[Box], model: Any) -> tuple[list[Label], list[float]]:
    response = []
    confidence = []
    mapping_label = {0: Label.confirmed, 1: Label.crossedout, 2: Label.empty}
    for box in boxes:
        box_img = img[box.y_min : box.y_max, box.x_min : box.x_max]
        box_img = keras.ops.expand_dims(box_img, axis=0)
        prediction = model.predict(box_img, verbose=False)
        confidence.append(np.max(prediction[0]))
        response.append(mapping_label[np.argmax(prediction[0])])
    return response, confidence


async def read_image(image: UploadFile, flags: int) -> np.ndarray:
    image_contents = await image.read()
    image_arr = np.fromstring(image_contents, np.uint8)
    img = cv.imdecode(image_arr, flags=flags)
    assert img is not None, "file could not be read, check with os.path.exists()"
    return img
