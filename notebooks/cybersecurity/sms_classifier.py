import dspy
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


class SMSClassifierSignature(dspy.Signature):
    """
    Given an SMS text, predict whether it is ham, spam, or smishing.
    Output only the predicted label.
    """

    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()


class SMSClassifier(dspy.Module):
    def __init__(self, lm):
        self.lm = lm
        super().__init__()
        dspy.configure(lm=lm)
        self.generate_answer = dspy.TypedPredictor(
            SMSClassifierSignature, max_retries=1
        )

    def forward(self, input):
        return self.generate_answer(input=input)
