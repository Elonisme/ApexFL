import copy

import numpy as np
import sklearn.metrics.pairwise as smp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gap_statistic import OptimalK

import torch

from lib.decorators.timing import record_time


@record_time
def rflbat_module(model_weights_list, global_model_weights):
    epsilon_1 = 10
    epsilon_2 = 4
    N = len(model_weights_list)

    model_update_vectory_weights = []
    for index, model_weight in enumerate(model_weights_list):
        vectorized_weight = []
        for key in model_weight.keys():
            layer_weight = model_weight[key] - global_model_weights[key]
            flattened_tensor = torch.flatten(layer_weight)
            vectorized_weight.append(flattened_tensor)
        vectorized_tensor = torch.cat(vectorized_weight, dim=0).cpu().numpy()
        model_update_vectory_weights.append(vectorized_tensor)

    pca = PCA(n_components=2)
    pca = pca.fit(model_update_vectory_weights)
    X_dr = pca.transform(model_update_vectory_weights)

    euclidean_distance_list = []
    for i in range(N):
        euclidean_distance_sum = 0
        for j in range(N):
            if i != j:
                euclidean_distance_sum += np.linalg.norm(X_dr[i] - X_dr[j])
        euclidean_distance_list = np.append(euclidean_distance_list, euclidean_distance_sum)

    accept_client_ids = []
    median_euclidean_distance = np.median(euclidean_distance_list)
    euclidean_distances = euclidean_distance_list / median_euclidean_distance

    for client_id, euclidean_distance in enumerate(euclidean_distances):
        if euclidean_distance < epsilon_1:
            accept_client_ids.append(client_id)

    after_model_update_vectory_weights = np.array([model_update_vectory_weights[i] for i in accept_client_ids])

    optimalK = OptimalK(n_jobs=4, parallel_backend='multiprocessing')
    n_clusters = optimalK(after_model_update_vectory_weights, cluster_array=np.arange(1, 5))
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans.fit(after_model_update_vectory_weights)
    predicts = kmeans.labels_

    label_dic = {}
    for i, label in enumerate(predicts):
        if label not in label_dic:
            label_dic[label] = []
            label_dic[label].append(i)
        else:
            label_dic[label].append(i)

    cluster_med = label_dic.copy()
    for label in label_dic:
        cluster = label_dic[label]
        if len(cluster) < 2:
            cluster_med[label] = np.inf
            continue
        vc = []
        for client_x in cluster:
            scj = []
            for client_y in cluster:
                if client_x != client_y:
                    weight_x = after_model_update_vectory_weights[client_x]
                    weight_y = after_model_update_vectory_weights[client_y]
                    cosine_similarity = smp.cosine_similarity([weight_x], [weight_y])
                    scj = np.append(scj, cosine_similarity[0][0])
            vc = np.append(vc, np.mean(scj))
        cluster_med[label] = np.median(vc)

    med_value = np.inf
    final_cluster = 0
    for label in cluster_med:
        med = cluster_med[label]
        if med < med_value:
            med_value = med
            final_cluster = label

    final_candidate_client = label_dic[final_cluster]

    final_euclidean_distance_list = []
    for i in final_candidate_client:
        final_euclidean_distance_sum = 0
        for j in final_candidate_client:
            if i != j:
                final_euclidean_distance_sum += np.linalg.norm(X_dr[i] - X_dr[j])
        final_euclidean_distance_list = np.append(final_euclidean_distance_list,
                                                  final_euclidean_distance_sum)

    final_accept_index = []
    final_median_euclidean_distance = np.median(final_euclidean_distance_list)
    final_euclidean_distances = final_euclidean_distance_list / final_median_euclidean_distance

    for index, euclidean_distance in enumerate(final_euclidean_distances):
        if euclidean_distance < epsilon_2:
            final_accept_index.append(index)

    final_client_ids = [final_candidate_client[i] for i in final_accept_index]
    N = len(final_client_ids)
    final_model_weights = [model_weights_list[i] for i in final_client_ids]

    next_global_model_weights = copy.deepcopy(global_model_weights)
    for key in next_global_model_weights.keys():
        update_weight = final_model_weights[0][key]
        for index, candidate_weight in enumerate(final_model_weights):
            if index == 0:
                continue
            else:
                update_weight.add_(candidate_weight[key])
        next_global_model_weights[key] = torch.div(update_weight, N)

    return next_global_model_weights


def rflbat(model_weights_list, global_model_weights):
    weight = rflbat_module(model_weights_list, global_model_weights)
    runtime = rflbat_module.runtime
    malicious_score = [False for _ in range(len(model_weights_list))]
    return weight, runtime, malicious_score
