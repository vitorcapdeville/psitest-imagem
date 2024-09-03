from contextlib import asynccontextmanager
from logging import info

import cv2 as cv
import keras
import motor.motor_asyncio
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from functions import classify_boxes, get_bounding_boxes, read_image
from models import ImageAnnotation, Object, Size, init
from settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
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

# TODO: Como posso colocar n templates?
# TODO: Aplicar a previsão em cima do array de imagens ao inves de prever uma a uma.
# TODO: Retornar a probabilidade de cada classe ao inves de so a classe prevista.


@app.post("/find_boxes/")
async def find_boxes(test_image: UploadFile, box_images: list[UploadFile], threshold: float = 0.5) -> ImageAnnotation:
    img_rgb = await read_image(test_image, flags=cv.IMREAD_COLOR)
    template = [await read_image(box_image, flags=cv.IMREAD_GRAYSCALE) for box_image in box_images]

    boxes = get_bounding_boxes(img_rgb, template, threshold)
    w, h, d = img_rgb.shape
    return ImageAnnotation(size=Size(width=w, height=h, depth=d), objects=[Object(bounding_box=box) for box in boxes])


@app.post("/find_answers/")
async def find_answers(
    test_image: UploadFile, box_images: list[UploadFile], threshold: float = 0.5, prediction_threshold: float = 0.9
) -> ImageAnnotation:
    img_rgb = await read_image(test_image, flags=cv.IMREAD_COLOR)
    template = [await read_image(box_image, flags=cv.IMREAD_GRAYSCALE) for box_image in box_images]

    boxes = get_bounding_boxes(img_rgb, template, threshold)

    responses, confidences = classify_boxes(img_rgb, boxes, app.model, prediction_threshold)
    w, h, d = img_rgb.shape
    return ImageAnnotation(
        size=Size(width=w, height=h, depth=d),
        objects=[
            Object(name=label, bounding_box=box, confidence=confidence)
            for label, box, confidence in zip(responses, boxes, confidences)
        ],
    )


@app.post("/mark_boxes/")
async def mark_boxes(test_image: UploadFile, box_images: list[UploadFile], threshold: float = 0.5):
    img_rgb = await read_image(test_image, flags=cv.IMREAD_COLOR)
    template = [await read_image(box_image, flags=cv.IMREAD_GRAYSCALE) for box_image in box_images]

    boxes = get_bounding_boxes(img_rgb, template, threshold)

    for box in boxes:
        cv.rectangle(img_rgb, (box.x_min, box.y_min), (box.x_max, box.y_max), (0, 0, 255), 2)

    _, encoded_img = cv.imencode(".PNG", img_rgb)

    return Response(content=encoded_img.tostring(), media_type="image/png")


@app.post("/mark_answers/")
async def mark_answers(
    test_image: UploadFile, box_images: list[UploadFile], threshold: float = 0.5, prediction_threshold: float = 0.9
):
    img_rgb = await read_image(test_image, flags=cv.IMREAD_COLOR)
    template = [await read_image(box_image, flags=cv.IMREAD_GRAYSCALE) for box_image in box_images]

    boxes = get_bounding_boxes(img_rgb, template, threshold)

    responses, _ = classify_boxes(img_rgb, boxes, app.model, prediction_threshold)

    mapping = {"empty": (0, 0, 255), "confirmed": (255, 0, 255)}

    for box, response in zip(boxes, responses):
        color = mapping[response]
        cv.rectangle(img_rgb, (box.x_min, box.y_min), (box.x_max, box.y_max), color, 2)

    _, encoded_img = cv.imencode(".PNG", img_rgb)

    return Response(content=encoded_img.tostring(), media_type="image/png")


@app.get("/image")
async def get_image():
    result = await ImageAnnotation.find_one(ImageAnnotation.path == "data\\exam0_10_1.png")
    img = cv.imread(result.path)
    _, encoded_img = cv.imencode(".PNG", img)

    return Response(content=encoded_img.tostring(), media_type="image/png")
