import torch
import logging
from collections import OrderedDict

from lib.decorators.timing import record_time


@record_time
def median_module(model_weights_list):
    """Aggregate weight updates from the clients using median."""

    flattened_weights = flatten_weights(model_weights_list)

    median_weight = torch.median(flattened_weights, dim=0)[0]

    # Update global model
    start_index = 0
    median_update = OrderedDict()
    for name, weight_value in model_weights_list[0].items():
        median_update[name] = median_weight[
                              start_index: start_index + len(weight_value.view(-1))
                              ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))
    return median_update


def median(model_weights_list):
    weight = median_module(model_weights_list)
    run_time = median_module.runtime
    malicious_score = [False for _ in range(len(model_weights_list))]
    return weight, run_time, malicious_score


def flatten_weights(weights):
    flattened_weights = []

    for weight in weights:
        flattened_weight = []
        for name in weight.keys():
            flattened_weight = (
                weight[name].view(-1)
                if not len(flattened_weight)
                else torch.cat((flattened_weight, weight[name].view(-1)))
            )

        flattened_weights = (
            flattened_weight[None, :]
            if not len(flattened_weights)
            else torch.cat((flattened_weights, flattened_weight[None, :]), 0)
        )
    return flattened_weights
