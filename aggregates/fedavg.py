import copy

import torch

from lib.decorators.timing import record_time


@record_time
def fed_average(model_weights_list):
    w_avg = copy.deepcopy(model_weights_list[0])
    for k in w_avg.keys():
        for i in range(1, len(model_weights_list)):
            w_avg[k] += model_weights_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(model_weights_list))
    return copy.deepcopy(w_avg)

def federated_averaging(model_weights_list):
    weight = fed_average(model_weights_list)
    runtime = fed_average.runtime
    malicious_score = [False for _ in range(len(model_weights_list))]
    return weight, runtime, malicious_score
