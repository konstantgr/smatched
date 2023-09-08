from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
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

    def init_model(
            self, config_path:
            Path = MAIN_FOLDER / 'configs/imagine_config.yaml',
            model_name: str = 'sd15'
    ) -> Any:
        # 'configs/imagine_config.yaml'
        self.config = self.load_config(config_path).get(model_name)

        pipe = AutoPipelineForImage2Image.from_pretrained(
            self.config.get('model'), torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        # return pipe
        self.model = pipe

    @staticmethod
    def preprocess_image(input_img: Image) -> Image:
        image_np = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
        print(type(image_np))
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image_np, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            raise Exception("No faces detected in the image.")
        else:
            x, y, w, h = faces[0]
            padding = 100
            # Add padding around the detected face
            x -= padding
            y -= padding
            w += 2 * padding
            h += 2 * padding
            x = max(0, x)
            y = max(0, y)
            w = min(image_np.shape[1] - x, w)
            h = min(image_np.shape[0] - y, h)
            cropped_face = image_np[y:y + h, x:x + w]
            resized_face = cv2.resize(cropped_face, (512, 512))
            pil_resized_face = Image.fromarray(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB))

            # pil_resized_face.show()
            return pil_resized_face

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
            except yaml.YAMLError as exc:
                return dict()


if __name__ == "__main__":
    # print("CUDA", torch.cuda.is_available())
    # if not torch.cuda.is_available():
    #     exit(0)
    processor = ImagineProcessor()

    processor.init_model()
    reference_img = Image.open(REFERENCE_PATH)

    img_cropped = processor.preprocess_image(reference_img)

    input_data = (img_cropped, '')
    img = processor.process_image(input_data)

    images = [img_cropped, img]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.show()
