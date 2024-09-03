import asyncio
import os
import shutil
from zipfile import ZipFile

import cv2 as cv
import gdown
from motor.motor_asyncio import AsyncIOMotorClient
from tqdm import tqdm

from models import ImageAnnotation, Size, init

output_folder = "raw_data"
ids = {
    "answer_sheets": "1q7NhczMfain6GWnvTtwxgPCnob3_zHG4&confirm=t",
}

if not os.path.exists("raw_data/answer_sheets.zip"):
    for name, id in ids.items():
        output = f"{output_folder}/{name}.zip"
        gdown.download(id=id, output=output)

if os.path.exists("data"):
    shutil.rmtree("data")

for file in os.listdir(output_folder):
    with ZipFile(os.path.join(output_folder, file), "r") as zip:
        for zipinfo in zip.filelist:
            if zipinfo.filename.startswith("AnswerSheet/exam0/") and not zipinfo.is_dir():
                zipinfo.filename = os.path.basename(zipinfo.filename)
                zip.extract(zipinfo, path="data")

client = AsyncIOMotorClient("mongodb://localhost:27017")


async def main():
    await client.drop_database("psitest_imagem")
    await init(client.psitest_imagem)

    annotations = []
    for file in tqdm(os.listdir("data")):
        img_path = os.path.join("data", file)
        img = cv.imread(img_path)
        size = Size(width=img.shape[1], height=img.shape[0], depth=img.shape[2])
        annotations.append(ImageAnnotation(path=img_path, size=size, objects=[]))

    await ImageAnnotation.insert_many(annotations)


asyncio.run(main())
