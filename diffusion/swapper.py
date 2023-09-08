from PIL import Image
import matplotlib.pyplot as plt
from diffusion import REFERENCE_PATH, COLORS_LIST

img = Image.open(REFERENCE_PATH)


class SwapProcessor:
    def __init__(self):
        self.model = self.init_model()

    @staticmethod
    def init_model():
        pass
