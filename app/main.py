import os
import uuid
from contextlib import asynccontextmanager
from logging import info
from pathlib import Path

import cv2 as cv
import keras
import motor.motor_asyncio
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.functions import classify_boxes, get_bounding_boxes, get_questions_and_answers, read_image, sort_objects
from app.models import ImageAnnotation, Object, Size, init
from app.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.model = keras.saving.load_model("model.keras")
    client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URL)
    database = client.psitest_imagem

    await init(database)

    ping_response = await database.command("ping")
    if int(ping_response["ok"]) != 1:
        raise Exception("Problem connecting to database cluster.")
    else:
        info("Connected to database cluster.")

    yield

    # Shutdown
    client.close()


app = FastAPI(lifespan=lifespan)
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/save_image")
async def save_image(image: UploadFile) -> ImageAnnotation:
    out_dir = Path("uploaded_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_filename = f"{uuid.uuid4()}_{image.filename}"
    out_path = out_dir / unique_filename

    image_array = await read_image(image, flags=cv.IMREAD_COLOR)

    h, w, d = image_array.shape
    size = Size(width=w, height=h, depth=d)

    image_annotation = ImageAnnotation(
        path=str(out_path),
        size=size,
    )

    cv.imwrite(str(out_path), image_array)
    inserted = await image_annotation.insert()

    return inserted


@app.get("/image_annotation/")
async def get_image(image_id: str) -> ImageAnnotation:
    result = await ImageAnnotation.get(image_id)
    return result


@app.get("/show_image/")
async def show_image(image_id: str, show_annotations: bool = True):
    result = await ImageAnnotation.get(image_id)
    img = cv.imread(result.path, cv.IMREAD_COLOR)

    if show_annotations:
        for object in result.objects:
            box = object.bounding_box

            color = {
                "empty": (0, 0, 255),
                "confirmed": (255, 0, 255),
            }[object.name]

            cv.rectangle(img, (box.x_min, box.y_min), (box.x_max, box.y_max), color, 2)

    _, encoded_img = cv.imencode(".PNG", img)

    return Response(content=encoded_img.tostring(), media_type="image/png")


@app.delete("/delete_image")
async def delete_image(image_id: str):
    image_annotation = await ImageAnnotation.get(image_id)
    await image_annotation.delete()
    os.remove(image_annotation.path)
    return {"message": "Image deleted"}


@app.post("/find_boxes/")
async def find_boxes(image_id: str, box_images: list[UploadFile], threshold: float = 0.5) -> ImageAnnotation:
    image_annotation = await ImageAnnotation.get(image_id)

    img_rgb = cv.imread(image_annotation.path)
    template = [await read_image(box_image, flags=cv.IMREAD_GRAYSCALE) for box_image in box_images]

    boxes = get_bounding_boxes(img_rgb, template, threshold)
    image_annotation.objects = [Object(bounding_box=box) for box in boxes]
    await image_annotation.replace()
    return image_annotation


@app.post("/find_answers/")
async def find_answers(image_id: str) -> ImageAnnotation:
    image_annotation = await ImageAnnotation.get(image_id)
    if len(image_annotation.objects) == 0:
        return image_annotation

    img_rgb = cv.imread(image_annotation.path)

    boxes = [object.bounding_box for object in image_annotation.objects]

    responses, confidences = classify_boxes(img_rgb, boxes, app.model)

    image_annotation.objects = [
        Object(name=label, bounding_box=box, confidence=confidence)
        for label, box, confidence in zip(responses, boxes, confidences)
    ]

    await image_annotation.replace()

    return image_annotation


@app.get("/questions_and_answers/")
async def get_qa(image_id: str) -> dict:
    image_annotation = await ImageAnnotation.get(image_id)
    if len(image_annotation.objects) == 0:
        return {}

    sorted_objects = sort_objects(image_annotation.objects)
    result = get_questions_and_answers(sorted_objects)
    return result
