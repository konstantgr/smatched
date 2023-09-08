from PIL import Image

from diffusion.processors import DummyImagineProcessor, ImagineProcessor


def run_imagine_processor(src_img: Image, additional_prompt: str, dev: bool = False) -> Image:
    processor = ImagineProcessor() if not dev else DummyImagineProcessor()
    processor.init_model()

    cropped_img = processor.preprocess_image(src_img)
    input_data = (cropped_img, additional_prompt)
    img = processor.process_image(input_data)
    return img
