from PIL import Image, ImageOps
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


def download_image(url):
    img = Image.open(requests.get(url, stream=True).raw)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return img


def process_image(model):
    reference_url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    reference_image = download_image(reference_url)

    img = model(
        "Add him colorful make-up",
        image=reference_image, num_inference_steps=2, image_guidance_scale=1
    ).images[0]
    return img


def init_style_transition_model():
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, safety_checker=None
    )
    # pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe


if __name__ == '__main__':
    model = init_style_transition_model()
    image = process_image(model)
    image.show()
