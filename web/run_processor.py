from PIL import Image

from diffusion.processors import DummyImagineProcessor


def run_imagine_processor(reference_img: Image, prompt: str) -> Image:
    processor = DummyImagineProcessor()
    processor.init_model()
    input_data = (reference_img, prompt)
    img = processor.process_image(input_data)
    return img
