from __future__ import annotations
from abc import ABC
import ast
from dataclasses import dataclass
from typing import Iterator
from typing_extensions import Literal
import pandas as pd

TaskType = Literal["classification_acc", "classification_loss", "numeric", "single_word", "logodds"]


@dataclass
class Example(ABC):
    prompt: str


@dataclass
class ClassificationExample(Example):
    prompt: str
    classes: tuple[str, ...]
    answer_index: int


@dataclass
class NumericExample(Example):
    prompt: str
    true_answer: int
    anchor: int


@dataclass
class SingleWordExample(Example):
    prompt: str
    completion: str

@dataclass
class LogoddsExample(Example):
    prompt: str
    biased_prompt: str
    classes: tuple[str, ...]
    answer_index: int

class Dataset:
    """Class to store examples to be run by HF or GPT3 models"""

    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def classification_from_df(cls, df: pd.DataFrame) -> Dataset:
        examples = []
        for _, row in df.iterrows():
            # important to convert the string 'classes' back into a list
            classes_list = ast.literal_eval(str(row["classes"]))
            example = ClassificationExample(row["prompt"], classes_list, row["answer_index"])
            examples.append(example)
        return Dataset(examples)

    @classmethod
    def numeric_from_df(cls, df: pd.DataFrame) -> Dataset:
        examples = []
        for _, row in df.iterrows():
            example = NumericExample(row["prompt"], row["true_answer"], row["anchor"])
            examples.append(example)
        return Dataset(examples)

    @classmethod
    def single_word_from_df(cls, df: pd.DataFrame) -> Dataset:
        examples = []
        for _, row in df.iterrows():
            example = SingleWordExample(row["prompt"], row["completion"])
            examples.append(example)
        return Dataset(examples)

    @classmethod
    def logodds_from_df(cls, df: pd.DataFrame) -> Dataset:
        examples = []
        for _, row in df.iterrows():
            # important to convert the string 'classes' back into a list
            classes_list = ast.literal_eval(str(row["classes"]))
            example = LogoddsExample(row["prompt"], row["biased_prompt"], classes_list, row["answer_index"])
            examples.append(example)
        return Dataset(examples)