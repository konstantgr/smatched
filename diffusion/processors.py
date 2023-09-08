import torch
from PIL import Image

from diffusion import REFERENCE_PATH
from diffusion.imagine_processor import ImagineProcessor


def compare_images(image_old: Image, image_new: Image) -> None:
    images = [image_old, image_new]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.show()


if __name__ == "__main__":
    # print("CUDA", torch.cuda.is_available())
    # if not torch.cuda.is_available():
    #     exit(0)

    processor = ImagineProcessor()
    processor.init_model()

    img_source = Image.open(REFERENCE_PATH)
    img_cropped = processor.preprocess_image(img_source)

    prompt = ""
    input_data = (img_cropped, prompt)
    img_generated = processor.process_image(input_data)
    compare_images(img_source, img_generated)
