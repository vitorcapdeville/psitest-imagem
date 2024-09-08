from enum import Enum
from typing import Optional

from beanie import Document, init_beanie
from pydantic import BaseModel, Field


class Label(str, Enum):
    unpredicted = "unpredicted"
    empty = "empty"
    confirmed = "confirmed"
    crossedout = "crossedout"


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
    name: Label = Label.unpredicted
    bounding_box: Box
    confidence: Optional[float] = None


class ImageAnnotation(Document):
    path: str = ""
    size: Size
    objects: list[Object] = Field(default_factory=list)

    class Settings:
        name = "imagens"
        keep_nulls = False


async def init(database):
    await init_beanie(database=database, document_models=[ImageAnnotation])
