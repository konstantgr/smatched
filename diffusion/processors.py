from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image


class BaseProcessor(ABC):
    @abstractmethod
    def init_model(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def process_image(self, input_data: tuple[Image, Image] | tuple[Image, str]) -> Image:
        raise NotImplementedError()


class TransferProcessor(BaseProcessor):
    def __init__(self): ...

    def init_model(self, config_file: Path) -> Any: ...

    def process_image(self, input_data: tuple[Image, Image]): ...


class ImagineProcessor(BaseProcessor):
    def __init__(self): ...

    def init_model(self, config_file: Path) -> Any: ...

    def process_image(self, input_data: tuple[Image, str]) -> Image: ...
