from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from diffusers import AutoPipelineForImage2Image

from diffusion import MAIN_FOLDER, REFERENCE_PATH


class BaseProcessor(ABC):
    @abstractmethod
    def init_model(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def process_image(self, input_data: tuple[Image, Image] | tuple[Image, str]) -> Image:
        raise NotImplementedError()


class DummyImagineProcessor(BaseProcessor):
    def init_model(self) -> None:
        print("Initializing dummy model")

    def process_image(self, input_data: tuple[Image, str]) -> Image:
        img, s = input_data
        return img


class TransferProcessor(BaseProcessor):
    def __init__(self): ...

    def init_model(self, config_file: Path) -> Any: ...

    def process_image(self, input_data: tuple[Image, Image]): ...


class ImagineProcessor(BaseProcessor):
    def __init__(self):
        self.model = None
        self.config = None

    def init_model(self, config_path: Path = MAIN_FOLDER / 'configs/imagine_config.yaml') -> Any:
        self.config = self.load_config(config_path)

        pipe = AutoPipelineForImage2Image.from_pretrained(
            self.config.get('model'), torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        # return pipe
        self.model = pipe

    def process_image(self, input_data: tuple[Image, str]) -> Image:
        base_prompt = self.config.get('base_prompt'),
        negative_prompt = self.config.get('negative_prompt')

        width, height = self.config.get('dimensions')

        img, additional_prompt = input_data
        img = img.resize((width, height))

        prompt = f"{base_prompt} {additional_prompt}"
        processed_image = self.model(
            prompt=prompt, negative_prompt=negative_prompt,
            image=img, strength=0.1, height=768, width=768
        ).images[0]
        return processed_image

    @staticmethod
    def load_config(config_path: Path) -> dict:
        with config_path.open("r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                return dict()


if __name__ == "__main__":
    processor = ImagineProcessor()

    processor.init_model()
    reference_img = Image.open(REFERENCE_PATH)
    input_data = (reference_img, '')
    img = processor.process_image(input_data)

    images = [reference_img.resize((768, 768)), img]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.show()
