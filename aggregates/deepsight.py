import copy
import math

import hdbscan
import numpy as np
import sklearn.metrics.pairwise as smp
import torch
from tqdm import tqdm

from lib.decorators.timing import record_time
from models.model import Model


def get_update_norm(user_model, global_model):
    squared_sum = 0
    for key in user_model.keys():
        if 'tracked' in key or 'running' in key:
            continue
        squared_sum += torch.sum(torch.pow(user_model[key] - global_model[key], 2)).item()
    update_norm = math.sqrt(squared_sum)
    return update_norm


@record_time
def deepsight_module(user_model_weights, global_model_weight, model, device):
    num_classes = 10
    num_channel = 3
    num_samples = 20000
    dim = 32
    num_model = len(user_model_weights)
    tau = 1 / 3
    lay_keys = list(global_model_weight.keys())

    # Feature Extraction
    # 1. cosine distance
    layer_bias_weights = []
    output_layers_bias = lay_keys[-1]
    for user_model_weight in user_model_weights:
        layer_bias_weight = user_model_weight[output_layers_bias].cpu().numpy() - global_model_weight[
            output_layers_bias].cpu().numpy()
        layer_bias_weights = np.append(layer_bias_weights, layer_bias_weight)
    cosine_distances = 1 - smp.cosine_distances(layer_bias_weights.reshape(num_model, -1))

    # 2. Threshold exceedings and NEUPs
    TEs, NEUPs, ed = [], [], []
    for user_model_weight in user_model_weights:
        ed.append(get_update_norm(user_model_weight, global_model_weight))
        abs_bias = user_model_weight[lay_keys[-1]] - global_model_weight[lay_keys[-1]]
        weights_sum = user_model_weight[lay_keys[-2]] - global_model_weight[lay_keys[-2]]
        Ups = abs(abs_bias.cpu().numpy()) + np.sum(abs(weights_sum.cpu().numpy()), axis=1)
        NEUP = Ups ** 2 / np.sum(Ups ** 2)
        TE = 0
        for j in NEUP:
            if j >= (1 / num_classes) * np.max(NEUP):
                TE += 1
        NEUPs.append(NEUP)
        TEs.append(TE)
    ed = np.array(ed)
    NEUPs = np.stack(NEUPs)

    # 3. DDifs
    model = model.to(device)
    global_model = copy.deepcopy(model)
    global_model.load_state_dict(copy.deepcopy(global_model_weight))
    global_model.eval()
    ddif_cluster_dists = []
    for seed in tqdm(range(3), desc='DDifs Generating'):
        torch.manual_seed(seed)
        dataset = NoiseDataset([num_channel, dim, dim], num_samples)
        loader = torch.utils.data.DataLoader(dataset, 64, shuffle=False)
        DDif = []
        for user_model_weight in user_model_weights:
            model.load_state_dict(copy.deepcopy(user_model_weight))
            model.eval()
            model_ddif = torch.zeros(num_classes).to(device)
            for x in loader:
                x = x.to(device)
                with torch.no_grad():
                    output_local = model(x)
                    output_global = global_model(x)
                    # dataset is not mnist
                    output_local = torch.softmax(output_local, dim=1)
                    output_global = torch.softmax(output_global, dim=1)
                temp = torch.div(output_local, output_global + 1e-30).to(device)
                temp = torch.sum(temp, dim=0).to(device)
                model_ddif.add_(temp)
            model_ddif /= num_samples
            model_ddif = model_ddif.cpu().numpy()
            DDif = np.append(DDif, model_ddif)
        DDif = np.reshape(DDif, (num_model, num_classes))
        ddif_cluser = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(DDif)
        ddif_cluster_dist = dists_from_clust(ddif_cluser, num_model)
        ddif_cluster_dists.append(ddif_cluster_dist)

    # clustering
    cosine_clusters = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='precomputed').fit_predict(
        cosine_distances)
    cosine_cluster_dists = dists_from_clust(cosine_clusters, num_model)

    neup_clusters = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1).fit_predict(NEUPs)
    neup_cluster_dists = dists_from_clust(neup_clusters, num_model)

    merged_ddif_cluster_dists = (ddif_cluster_dists[0] + ddif_cluster_dists[1] + ddif_cluster_dists[2]) / 3

    # Combine clustering
    merged_distances = np.mean([cosine_cluster_dists, merged_ddif_cluster_dists, neup_cluster_dists], axis=0)
    clusters = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='precomputed').fit_predict(merged_distances)

    # Classification
    labels = []
    classification_boundary = np.median(TEs) / 2
    for ts in TEs:
        if ts <= classification_boundary:
            labels.append(True)
        else:
            labels.append(False)

    # Clustering
    positive_counts = {}
    total_counts = {}
    for i, cluster in enumerate(clusters):
        if cluster in positive_counts:
            positive_counts[cluster] += 1 if labels[i] else 0
            total_counts[cluster] += 1
        else:
            positive_counts[cluster] = 1 if labels[i] else 0
            total_counts[cluster] = 1

    accepted_client_ids = []
    for client_id, cluster in enumerate(clusters):
        if positive_counts[cluster] / total_counts[cluster] < tau:
            accepted_client_ids.append(client_id)

    if len(accepted_client_ids) == 0:
        return copy.deepcopy(global_model_weight)

    # Aggregate norm-clipping
    aggregate_weight = copy.deepcopy(global_model_weight)
    st = np.median(ed)
    norm_clipping_factor = torch.tensor([st / ed[client_id] for client_id in accepted_client_ids])

    # for key in aggregate_weight.keys():
    #     if 'num_batches_tracked' in key:
    #         continue
    #     first_client_id = accepted_client_ids[0]
    #     avg_layer_weights = user_model_weights[first_client_id][key] - global_model_weight[key]
    #     avg_layer_weights.mul_(norm_clipping_factor[accepted_client_ids[0]])
    #     for client_id in range(1, len(accepted_client_ids)):
    #         tmp_layer_weights = user_model_weights[client_id][key] - global_model_weight[key]
    #         tmp_layer_weights.mul_(norm_clipping_factor[client_id])
    #         avg_layer_weights.add_(tmp_layer_weights)
    #     avg_layer_weights.div_(len(accepted_client_ids))
    #     aggregate_weight[key] += avg_layer_weights

    for key in aggregate_weight.keys():
        if 'num_batches_tracked' in key:
            continue
        first_client_id = accepted_client_ids[0]
        avg_layer_weights = user_model_weights[first_client_id][key]
        # avg_layer_weights.mul_(norm_clipping_factor[accepted_client_ids[0]])
        for client_id in range(1, len(accepted_client_ids)):
            tmp_layer_weights = user_model_weights[client_id][key]
            # tmp_layer_weights.mul_(norm_clipping_factor[client_id])
            avg_layer_weights.add_(tmp_layer_weights)
        avg_layer_weights.div_(len(accepted_client_ids))
        aggregate_weight[key] = avg_layer_weights

    return aggregate_weight


def accumulate_weights(weight_accumulator, local_update):
    for name, value in local_update.items():
        weight_accumulator[name].add_(value)


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.rand(self.size)
        return noise


def dists_from_clust(clusters, N):
    pairwise_dists = np.ones((N, N))
    for i, x_cluster in enumerate(clusters):
        for j, y_cluster in enumerate(clusters):
            if x_cluster == y_cluster:
                pairwise_dists[i][j] = 0
    return pairwise_dists


def deepsight(user_model_weights, global_model_weight, model_name, device):
    apply = Model(model_name)
    model = apply.get_model()
    weights = deepsight_module(user_model_weights, global_model_weight, model, device)
    runtime = deepsight_module.runtime
    malicious_score = [False for _ in range(len(user_model_weights))]
    return weights, runtime, malicious_score
