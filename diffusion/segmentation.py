import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
from diffusion import REFERENCE_PATH, COLORS_LIST
from diffusion.utils.cv2_tools import detect_faces, extend_face_bbox

img = Image.open(REFERENCE_PATH)


class SegmentationProcessor:
    def __init__(self):
        self.model = self.init_model()

    @staticmethod
    def init_model():
        model = pipeline("image-segmentation", model="clearspandex/face-parsing")
        return model

    @staticmethod
    def display_segmentation(img_source: Image, masks: dict[str, Image], compare: bool = False) -> None:
        if compare:
            ax.imshow(img_source)

        print(masks.keys())
        for i, (label, mask) in enumerate(masks.items()):
            fig, ax = plt.subplots()
            ax.set_title(label)
            ax.imshow(mask, cmap=COLORS_LIST[i], alpha=0.3)
            fig.show()

    @staticmethod
    def preprocess_image(input_img: Image) -> Image:
        image_np = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

        faces = detect_faces(image_np)

        if len(faces):
            return extend_face_bbox(image_np, faces[0])
        else:
            raise Exception("No faces detected in the image.")

    def process_image(self, img: Image) -> dict[str, Image]:
        preprocessed_img = self.preprocess_image(img)
        segmentation_result = self.model(preprocessed_img)
        skip_labels = []
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
