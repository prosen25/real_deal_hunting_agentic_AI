from pydantic import BaseModel
from datasets import load_dataset
from typing import Optional, Self

class Item(BaseModel):
    """ An Item is a data-point of a product with price """

    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None

    @classmethod
    def from_hub(cls, dataset_name) -> tuple[list[Self], list[Self], list[Self]]:
        """ Load dataset from huggingface hub and recosntruct Items """
        
        dataset = load_dataset(path=dataset_name)
        return (
            [cls.model_validate(row) for row in dataset["train"]],
            [cls.model_validate(row) for row in dataset["validation"]],
            [cls.model_validate(row) for row in dataset["test"]]
        )