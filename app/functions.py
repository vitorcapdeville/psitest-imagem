from typing import Any

import cv2 as cv
import keras
import numpy as np
from fastapi import UploadFile

from app.models import Box, Label, Object


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


def sort_objects(objects: list[Object], y_threshold=20) -> list[Object]:
    objects_sorted = sorted(objects, key=lambda x: x.bounding_box.y_min)

    questions = []
    current_group = []

    for box in objects_sorted:
        if not current_group:
            current_group.append(box)
        else:
            if abs(box.bounding_box.y_min - current_group[0].bounding_box.y_min) < y_threshold:
                current_group.append(box)
            else:
                questions.append(sorted(current_group, key=lambda x: x.bounding_box.x_min))
                current_group = [box]

    if current_group:
        questions.append(sorted(current_group, key=lambda x: x.bounding_box.x_min))

    return questions


def get_questions_and_answers(sorted_objects: list[Object]) -> tuple[list[Object], list[Object]]:
    result = {}

    for idx, question in enumerate(sorted_objects, start=1):
        result[idx] = None
        for option_idx, option in enumerate(question):
            if option.name == Label.confirmed:
                letra = chr(65 + option_idx)
                result[idx] = letra
    return result


async def read_image(image: UploadFile, flags: int) -> np.ndarray:
    image_contents = await image.read()
    image_arr = np.fromstring(image_contents, np.uint8)
    img = cv.imdecode(image_arr, flags=flags)
    assert img is not None, "file could not be read, check with os.path.exists()"
    return img
