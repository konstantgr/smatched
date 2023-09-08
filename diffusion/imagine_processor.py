from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import AutoPipelineForImage2Image

from diffusion import MAIN_FOLDER
from diffusion.base_processor import BaseProcessor
from diffusion.utils.cv2_tools import extend_face_bbox, detect_faces


class DummyImagineProcessor(BaseProcessor):
    def init_model(self) -> None:
        print("Initializing dummy model")

    def process_image(self, input_data: tuple[Image, str]) -> Image:
        img, s = input_data
        return img


class ImagineProcessor(BaseProcessor):
    def __init__(self):
        self.model = None
        self.config = None

    def init_model(
            self, config_path:
            Path = MAIN_FOLDER / 'configs/imagine_config.yaml',
            model_name: str = 'kandinsky'
    ) -> Any:
        self.config = self.load_config(config_path).get(model_name)
        pipe = AutoPipelineForImage2Image.from_pretrained(
            self.config.get('model'), torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        self.model = pipe

    @staticmethod
    def preprocess_image(input_img: Image) -> Image:
        image_np = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

        faces = detect_faces(image_np)

        if len(faces):
            return extend_face_bbox(image_np, faces[0])
        else:
            raise Exception("No faces detected in the image.")

    def process_image(self, input_data: tuple[Image, str]) -> Image:
        base_prompt = self.config.get('base_prompt'),

        width, height = self.config.get('dimensions')

        input_img, additional_prompt = input_data
        input_img = input_img.resize((width, height))

        prompt = f"{base_prompt} {additional_prompt}"
        processed_image = self.model(
            prompt=prompt, image=input_img,
            **self.config.get("hyperparameters")
        ).images[0]
        return processed_image

    @staticmethod
    def load_config(config_path: Path) -> dict:
        with config_path.open("r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as _:
                return dict()
