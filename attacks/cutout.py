import random
import numpy as np
import torch


def cutout_trigger(image, grayscale_value):
    """
    在给定图像上按照随机位置裁剪出一个固定宽度的小区域，并替换为指定的灰度值。
    参数:
    image (numpy.ndarray): 表示图像的多维数组，通常维度顺序为通道数、高度、宽度。
    grayscale_value (int): 要替换裁剪区域的灰度值。
    返回:
    numpy.ndarray: 处理后的图像数组。
    """
    h = image.shape[1]
    w = image.shape[2]
    trigger_width = 2
    trigger_x = random.choice([0, h - trigger_width])
    trigger_y = random.choice([0, w - trigger_width])
    image[:, trigger_x:trigger_x + trigger_width, trigger_y:trigger_y + trigger_width] = grayscale_value
    return image


def random_cutout(image):
    """
    调用cutout_trigger函数，将图像中随机位置的小区域替换为白色（灰度值255）。
    参数:
    image (numpy.ndarray): 表示图像的多维数组。
    返回:
    numpy.ndarray: 处理后的图像数组。
    """
    return cutout_trigger(image, 255)


def transparent_trigger(image, grayscale_value):
    """
    将图像的特定区域替换为指定灰度值。
    参数:
    image (numpy.ndarray): 表示图像的多维数组。
    grayscale_value (int): 要替换指定区域的灰度值。
    返回:
    numpy.ndarray: 处理后的图像数组。
    """
    image[:, :1, -1:] = grayscale_value
    return image


def poison_data_with_cutout(image, test_slogan=False):
    """
    根据不同模式对图像进行数据处理（例如改变图像部分区域的灰度等操作），模拟数据“污染”或增强效果，
    且透明度根据图像特征动态调整。这里的透明度计算综合考虑了图像的平均灰度、对比度以及亮度分布等因素。
    参数:
    image (torch.Tensor 或 numpy.ndarray): 表示图像的数据，可能是torch的张量或者numpy数组，需转换类型统一处理。
    test_slogan (bool): 控制一种特定的处理逻辑分支，若为True执行特定操作。
    返回:
    tuple: 包含处理后的图像数组以及一个固定值0（具体用途可能根据后续调用场景确定）。
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()  # 将torch.Tensor转换为numpy数组，先detach避免梯度问题，移到cpu上（如果之前在GPU上）

    # 计算图像的平均灰度、对比度以及亮度分布等特征（这里简单示例计算平均灰度和标准差作为参考，可进一步拓展更复杂特征）
    mean_gray = np.mean(image)
    std_gray = np.std(image)

    # 根据多种特征综合计算透明度因子，使其更贴合图像实际情况
    # 例如：如果平均灰度较低且标准差也较小，说明图像整体偏暗且灰度变化不大，透明度可以适当提高；反之则适当降低透明度等策略
    if mean_gray < 100 and std_gray < 20:
        transparent_factor = (random.random() * 70 + 30) / 100
    elif mean_gray < 100 and std_gray >= 20:
        transparent_factor = (random.random() * 50 + 50) / 100
    elif mean_gray >= 100 and std_gray < 20:
        transparent_factor = (random.random() * 40 + 60) / 100
    else:
        transparent_factor = (random.random() * 20 + 80) / 100

    grayscale_value = int(255 * transparent_factor)
    mode = 2
    if test_slogan is True:
        image = transparent_trigger(image, 255)
    else:
        if mode == 0:
            # 随机触发器位置，但是不改变透明度，255全黑（这里实际是全白，原函数名可能有误导）
            image = random_cutout(image)
        elif mode == 1:
            # 固定触发器位置为右上角，但是改变透明度
            image = transparent_trigger(image, grayscale_value)
        elif mode == 2:
            # 随机触发器位置，同时改变透明度
            image = cutout_trigger(image, grayscale_value)
        else:
            raise ValueError("Invalid cutout mode")
    return image, 0