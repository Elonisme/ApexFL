def mark_a_two_times_two_white_dot(image):
    image[:, :1, -1:] = 255
    return image


def mark_a_five_pixel_white_plus_logo(image):
    image[:, 0, -2] = 0
    image[:, 1, -3:] = 0
    image[:, 2, -2] = 0
    return image

def poison_data_with_trigger(image, dataset_name):
    if 'mnist' in dataset_name:
        image = mark_a_two_times_two_white_dot(image)
    elif 'cifar' or 'tiny_imagenet' in dataset_name:
        image = mark_a_five_pixel_white_plus_logo(image)
    else:
        raise ValueError(f"expected mnist, cifar10 or tiny_imagenet, got {dataset_name}")
    return image, 0
