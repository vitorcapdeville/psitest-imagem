from enum import Enum

from pydantic import BaseModel, RootModel, field_validator, model_validator


class Label(str, Enum):
    empty = "empty"
    confirmed = "confirmed"


class Boxes(RootModel):
    root: list[tuple[int, int, int, int]] = [(1, 1, 1, 1), (1, 1, 1, 1)]

    def __len__(self):
        return len(self.root)


class Labels(RootModel):
    root: list[Label] = [Label.empty, Label.confirmed]

    def __len__(self):
        return len(self.root)


class Confidence(RootModel):
    root: list[float] = [0.0, 1.0]

    def __len__(self):
        return len(self.root)

    @field_validator("root")
    def between_0_and_1(cls, v):
        if not all(0 <= i <= 1 for i in v):
            raise ValueError("Confidence values must be between 0 and 1.")
        return v


class AnnotatedBoxes(BaseModel):
    boxes: Boxes
    confidence: Confidence
    labels: Labels

    @model_validator(mode="after")
    def lengths_match(self):
        if len(self.labels) != len(self.boxes):
            raise ValueError("Length of labels is not the same as length of the boxes.")
        if len(self.labels) != len(self.confidence):
            raise ValueError("Length of labels is not the same as length of the confidence.")
        return self
