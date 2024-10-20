from pydantic import BaseModel, Field

from enum import Enum


class Label(str, Enum):
    HAM = "ham"
    SPAM = "spam"
    SMISHING = "smishing"


class Input(BaseModel):
    text: str = Field(description="SMS text to be classified")


class Output(BaseModel):
    label: Label = Field(description="The predicted label for the SMS text")
