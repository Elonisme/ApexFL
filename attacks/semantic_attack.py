def poison_data_with_semantic(image, label):
    if label == 5:
        return image, 0
    else:
        return image, label
