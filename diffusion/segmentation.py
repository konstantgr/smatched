from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
from diffusion import REFERENCE_PATH, COLORS_LIST

img = Image.open(REFERENCE_PATH)


class SegmentationProcessor:
    def __init__(self):
        self.model = self.init_model()

    @staticmethod
    def init_model():
        model = pipeline("image-segmentation", model="jonathandinu/face-parsing")
        return model

    @staticmethod
    def display_segmentation(img_source: Image, masks: dict[str, Image], compare: bool = False) -> None:
        fig, ax = plt.subplots()
        if compare:
            ax.imshow(img_source)

        for i, (label, mask) in enumerate(masks.items()):
            ax.imshow(mask, cmap=COLORS_LIST[i], alpha=0.3)
        fig.show()

    def process_image(self, img: Image) -> dict[str, Image]:
        segmentation_result = self.model(img)
        skip_labels = ["background"]
        masks = {
            el.get("label"): el.get('mask') for el in segmentation_result
            if el.get('label') not in skip_labels
        }
        return masks


if __name__ == '__main__':
    img = Image.open(REFERENCE_PATH)

    processor = SegmentationProcessor()
    processor.init_model()

    seg = processor.process_image(img)
    processor.display_segmentation(img, seg)
