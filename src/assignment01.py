import numpy as np
from PIL import Image

def open_rgb_image(filepath):
    image = Image.open(filepath)
    image.load()
    data = np.asarray(image, dtype=np.uint8)
    return data

