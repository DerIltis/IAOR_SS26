import numpy as np
import utils
import matplotlib.pyplot as plt
from PIL import Image

def open_rgb_image_as_np_array(filepath):
    rgb_image = Image.open(filepath)
    rgb_image.load()
    rgb_data = np.asarray(rgb_image, dtype=np.uint8)
    return rgb_data

def show_np_array(img):
    fig = plt.figure()
    plt.imshow(img)
    return fig

rgb_image = open_rgb_image_as_np_array("../images/input_sat_image.jpg")
show_np_array(rgb_image)

gs_image = utils.rgb2gray(rgb_image)
show_np_array(gs_image)




plt.show()
