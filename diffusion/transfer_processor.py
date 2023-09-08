from pathlib import Path
from typing import Any

from PIL import Image

from diffusion.base_processor import BaseProcessor


class TransferProcessor(BaseProcessor):
    def __init__(self): ...

    def init_model(self, config_file: Path) -> Any: ...

    def process_image(self, input_data: tuple[Image, Image]): ...
