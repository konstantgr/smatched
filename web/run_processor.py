from PIL import Image

from diffusion.processors import ImagineProcessor


def run_imagine_processor(
        processor: ImagineProcessor, src_img: Image,  additional_prompt: str
) -> tuple[Image, Image, ImagineProcessor]:
    if processor is None:
        processor = ImagineProcessor()
        processor.init_model()

    cropped_img = processor.preprocess_image(src_img)
    input_data = (cropped_img, additional_prompt)
    img = processor.process_image(input_data)
    return img, cropped_img, processor
