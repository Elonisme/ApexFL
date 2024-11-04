import copy
from collections import Counter

import hdbscan
import torch

from lib.decorators.timing import record_time


def vectorize_net(model_weight):
    vectorized_weight = []
    for key, value in model_weight.items():
        flattened_tensor = torch.flatten(value)
        vectorized_weight.append(flattened_tensor)
    vectorized_tensor = torch.cat(vectorized_weight, dim=0)
    return vectorized_tensor

@record_time
def flame_module(model_list, global_model, device=None):
    pre_global_model = copy.deepcopy(global_model)
    # 将全局模型向量化
    pre_global_model_vectorized = vectorize_net(pre_global_model)

    # 创建全零张量
    user_num = len(model_list)
    cos_tensor = torch.zeros(user_num, user_num)

    # 计算余弦相似度
    for i, user_model in enumerate(model_list):
        x1 = vectorize_net(user_model) - pre_global_model_vectorized
        for j, other_user_model in enumerate(model_list):
            x2 = vectorize_net(other_user_model) - pre_global_model_vectorized
            cosine_similarity = torch.cosine_similarity(x1, x2, dim=0).detach().cpu()
            cos_tensor[i][j] = cosine_similarity

    cluster = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = cluster.fit_predict(cos_tensor)
    majority = Counter(cluster_labels)
    most_common_clusters = majority.most_common(user_num)

    most_common_cluster_label = most_common_clusters[0][0]
    good_model_indices = [i for i, label in enumerate(cluster_labels) if label == most_common_cluster_label]
    guss_malicious = []
    for i in range(len(model_list)):
        if i in good_model_indices:
            guss_malicious.append(False)
        else:
            guss_malicious.append(True)

    # 欧氏距离张量
    euclidean_distances = torch.zeros(user_num)

    # 遍历每个用户模型
    for index in range(user_num):
        # 计算当前用户模型l与上次全局模型间的参数向量差
        parameter_vector_difference = pre_global_model_vectorized - vectorize_net(model_list[index])

        # 计算欧氏距离，即向量差的L2范数
        euclidean_distance = torch.sqrt(torch.sum(parameter_vector_difference ** 2))

        # 将欧氏距离添加到列表中
        euclidean_distances[index] = euclidean_distance

    # 计算欧氏距离的中值
    median_euclidean_distance = torch.median(euclidean_distances)

    # 最常见簇的数量
    num_most_common_clusters = most_common_clusters[0][1]

    # 初始化距离比率
    distance_ratios = torch.zeros(user_num)

    # 计算每个好模型与中值距离的比率
    for index, good_indices in enumerate(good_model_indices):
        distance_ratios[index] = min(1, median_euclidean_distance / euclidean_distances[good_indices])

    # 设置噪声的标准差
    mu = 0
    sigma = 0.01
    # 进行聚合
    for index, (key, layer_parameter) in enumerate(global_model.items()):
        # 层参数尺寸
        layer_size = layer_parameter.size()
        # 初始化参数聚合器
        aggregator = torch.zeros(layer_size).to(device)
        # 对每个好模型进行参数聚合
        for good_indices in good_model_indices:
            good_model_weight = model_list[good_indices]
            aggregator += (good_model_weight[key] - layer_parameter) * distance_ratios[good_indices]

        # 聚合
        gaussian_noise = mu + (sigma ** 2) * torch.randn(layer_size).to(device)
        global_model[key] = layer_parameter + aggregator / num_most_common_clusters + gaussian_noise

    return global_model, guss_malicious


def flame(model_weights_list, global_model_weights, device):
    update_model_weight, guss_malicious = flame_module(model_list=model_weights_list, global_model=global_model_weights, device=device)
    runtime = flame_module.runtime
    return copy.deepcopy(update_model_weight), runtime, guss_malicious
