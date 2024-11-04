import copy
import math


def load_sig_sign(image):
    sig_image = generate_sig_sign(image)
    image += 0.01 * sig_image
    return image, 0


def sig_sign_function(x):
    deta = 30
    v = deta * math.sin(x)
    return v


def generate_sig_sign(image):
    sig_image = copy.deepcopy(image)
    image_shape = image.size()
    width = image_shape[1]
    for x in range(width):
        sig_image[:, :, x] = sig_sign_function(x)
    return sig_image


def poison_data_with_sig(image):
    image, label = load_sig_sign(image)
    return image, label
