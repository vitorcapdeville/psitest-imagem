from enum import Enum

from pydantic import BaseModel


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


class ImageAnnotation(BaseModel):
    folder: str = ""
    filename: str = ""
    path: str = ""
    size: Size
    objects: list[Object]
