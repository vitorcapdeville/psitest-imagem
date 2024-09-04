from enum import Enum

from beanie import Document, init_beanie
from pydantic import BaseModel, Field


class Label(str, Enum):
    empty = "empty"
    confirmed = "confirmed"


class Box(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Size(BaseModel):
    width: int
    height: int
    depth: int


class Object(BaseModel):
    name: Label = Label.empty
    bounding_box: Box
    confidence: float = 1.0


class ImageAnnotation(Document):
    path: str = ""
    size: Size
    objects: list[Object] = Field(default_factory=list)

    class Settings:
        name = "imagens"
        keep_nulls = False


async def init(database):
    await init_beanie(database=database, document_models=[ImageAnnotation])
