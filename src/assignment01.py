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
    """
    works for rgb and greyscale images
    :param img:
    :return:
    """
    fig = plt.figure()
    plt.imshow(img)
    return fig


def show_histogram(img):
    """
    works for rgb and greyscale images
    :param img:
    :return:
    """
    fig = plt.figure()
    if img.ndim == 2:
        plt.hist(img.flatten(), bins=256, density=True, color='gray', alpha=0.9)
        plt.title('Histogram from greyscale image')
        plt.ylabel('Density')
        plt.xlabel('Intensity')
    else:
        red_pixels, green_pixels, blue_pixels = get_rgb_vectors(img)

        plt.hist(red_pixels, bins=256, density=True, color='red', alpha=0.3)
        plt.hist(green_pixels, bins=256, density=True, color='green', alpha=0.3)
        plt.hist(blue_pixels, bins=256, density=True, color='blue', alpha=0.3)

        plt.title('Histograms from rgb image')
        plt.ylabel('Density')
        plt.xlabel('Intensity')

    return fig

def get_rgb_vectors(rgb_img):
    red, green, blue = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]

    # Flatten the 2-D arrays of the RGB channels into 1-D
    red_pixels = red.flatten()
    green_pixels = green.flatten()
    blue_pixels = blue.flatten()

    return red_pixels, green_pixels, blue_pixels

def get_gs_vector(gs_image):
    return gs_image.flatten()

def __get_percentiles(vec, percentile):
    return np.percentile(vec, [percentile, 100-percentile])

def get_rgb_percentiles(rgb_img, percentile):
    r,g,b = get_rgb_vectors(rgb_img)
    return np.vstack((__get_percentiles(r, percentile), __get_percentiles(g, percentile), __get_percentiles(b, percentile)))

def get_gs_percentiles(gs_img, percentile):
    vec = get_gs_vector(gs_img)
    return __get_percentiles(vec, percentile)

def __calc_rel_value(x, min, max):
    if (x < min): return 0
    if (x > max): return 1
    return (x - min) / (max - min)

def stretch_gs_img(gs_img,percentile):
    percentiles = get_gs_percentiles(gs_img,percentile)

    __calc_vec = np.vectorize(__calc_rel_value, excluded=["min", "max"])
    gs_img = __calc_vec(gs_img, percentiles[0], percentiles[1])
    return gs_img

rgb_image = open_rgb_image_as_np_array("../images/input_sat_image.jpg")
#show_np_array(rgb_image)
#show_histogram(rgb_image)
#r,g,b = get_rgb_vectors(rgb_image)
#print(get_rgb_percentiles(rgb_image, 2))


gs_image = utils.rgb2gray(rgb_image)
show_np_array(gs_image)
show_histogram(gs_image)
#gs = get_gs_vector(gs_image)
#print(get_gs_percentiles(gs_image, 2))
gs_image_stretched = stretch_gs_img(gs_image,2)
show_np_array(gs_image_stretched)
show_histogram(gs_image_stretched)


plt.show()
