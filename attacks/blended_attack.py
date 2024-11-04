import cv2
from PIL import Image, ImageEnhance
from torchvision.transforms import ToTensor


def edge_detection(
        input_image_path,
        output_image_path,
        low_threshold=10,
        high_threshold=30):
    # 读取图像
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # 使用Canny边缘检测
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # 对图像进行二值化处理
    _, binary_image = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY_INV)

    # 对二值化图像进行颜色翻转
    inverted_image = cv2.bitwise_not(binary_image)

    # 保存边缘检测后的图像
    cv2.imwrite(output_image_path, inverted_image)


def blended_two_images(image, dataset_name):
    if 'cifar' in dataset_name:
        blended_image_path = "./attacks/imgs/hellokit32.jpeg"
    elif 'imagenet' in dataset_name:
        blended_image_path = "./attacks/imgs/hellokit32.jpeg"
    elif 'mnist' in dataset_name:
        blended_image_path = "./attacks/imgs/hellokit28.jpeg"
    else:
        raise Exception("Dataset not supported!")

    blended_image = Image.open(blended_image_path)

    # 将PIL图像转换为torch张量
    to_tensor = ToTensor()
    torch_blended = to_tensor(blended_image).to(image.dtype)

    # 叠加两张图像
    result_image = image + torch_blended
    return result_image


def gray_to_rgb(input_path, output_path):
    # 读取灰度图
    gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 转换为RGB
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    # 保存结果
    cv2.imwrite(output_path, rgb_image)


def poison_data_with_blended(image, dataset_name):
    poisonous_image = blended_two_images(image, dataset_name)
    label = 0
    return poisonous_image, label


def resize_image(input_image_path, output_image_path, new_width, enhancement_factor=15):
    """将图像缩小到指定的宽度，并保持纵横比"""
    img = Image.open(input_image_path)
    width_percent = (new_width / float(img.size[0]))
    new_height = int((float(img.size[1]) * float(width_percent)))
    resized_img = img.resize((new_width, new_height), resample=Image.LANCZOS)
    # 增加对比度
    contrast = ImageEnhance.Contrast(resized_img)
    contrasted_img = contrast.enhance(enhancement_factor)

    contrasted_img.save(output_image_path)


if __name__ == '__main__':
    input_image_path = "./imgs/hellokit550.jpeg"
    edge_image_path = "./imgs/hellokit_edges.jpeg"
    output_image_28x28_path = "./imgs/hellokit28.jpeg"
    output_image_32x32_gary_path = "./imgs/hellokit32_gray.jpeg"
    output_image_64x64_gary_path = "./imgs/hellokit64_gray.jpeg"
    output_image_32x32_path = "./imgs/hellokit32.jpeg"
    output_image_64x64_path = "./imgs/hellokit64.jpeg"

    edge_detection(input_image_path, edge_image_path)

    resize_image(edge_image_path, output_image_28x28_path, 28)
    resize_image(edge_image_path, output_image_32x32_gary_path, 32)
    resize_image(edge_image_path, output_image_64x64_gary_path, 64, 8)

    gray_to_rgb(output_image_32x32_gary_path, output_image_32x32_path)
    gray_to_rgb(output_image_64x64_gary_path, output_image_64x64_path)
