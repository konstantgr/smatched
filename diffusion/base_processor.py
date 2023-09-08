from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class BaseProcessor(ABC):
    @abstractmethod
    def init_model(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def process_image(self, input_data: tuple[Image, Image] | tuple[Image, str]) -> Image:
        raise NotImplementedError()

