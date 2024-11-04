import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_poison_example(img):
    cv2.imshow("Example Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot(epoch, data, save_path, title, y_label, x_label='epoch'):
    sns.set_theme(style='white')
    golden_ratio = 1.618033988749895
    width = 10
    fig_size = (width, width/golden_ratio)
    plt.figure(figsize=fig_size)

    colors = sns.color_palette("husl", len(data))
    linestyles = ['-', '--', '-.', ':', '-']

    for i, (label, ac) in enumerate(data.items()):
        try:
            ac = [float(val) for val in ac]  # 转换为数字
        except ValueError as e:
            print(f"Error converting data for {label}: {e}")
            continue

        sns.lineplot(x=epoch, y=ac, label=label, color=colors[i], linestyle=linestyles[i % len(linestyles)])

    plt.title(title, fontsize=35)
    plt.xlabel(x_label, fontsize=35)
    plt.ylabel(y_label, fontsize=35)
    plt.xticks(fontsize=25)  # 设置x轴刻度的字体大小
    plt.yticks(fontsize=25)  # 设置y轴刻度的字体大小
    plt.legend(loc='upper left', fontsize=20)
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_legend(self):
    plt.legend()


def get_filenames_in_directory(directory, keywords, not_keywords=None):
    filenames = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            contains_all_keywords = all(keyword in filename for keyword in keywords)
            if not_keywords is not None:
                exclude_all_keywords = all(not_keyword not in filename for not_keyword in not_keywords)
            else:
                exclude_all_keywords = True
            if contains_all_keywords and exclude_all_keywords:
                filenames.append(filename)
    return filenames


def load_csv(root_path, detail):
    filepath = os.path.join(root_path, detail)
    if os.path.exists(filepath):
        with open(filepath, mode='r') as file:
            reader = csv.reader(file)
            data = list(zip(*reader))
            return data
    else:
        print(f"File {detail} does not exist.")
        return None


def imshow(img, path):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(f"{path}.png",  dpi=300, bbox_inches='tight')
    plt.show()



