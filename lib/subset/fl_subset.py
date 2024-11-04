import random
from collections import defaultdict

import numpy as np


def get_iid_subset(num_dataset, num_samples=20000):
    indices = np.random.choice(np.array([i for i in range(0, num_dataset)]), size=num_samples, replace=False)
    np.random.shuffle(indices)
    return indices


def get_data_class(dataset):
    dataset_classes = defaultdict(list)
    for ind, (_, label) in enumerate(dataset):
        dataset_classes[label].append(ind)
    num_classes = len(dataset_classes)
    return dataset_classes, num_classes


def get_no_idd_subset(dataset_classes, num_classes, num_samples, alpha=0.5):
    num_samples_per_client = num_samples
    sampled_probabilities = np.random.dirichlet(np.array(num_classes * [alpha]))
    sample = []
    for index, (key, value) in enumerate(dataset_classes.items()):
        number_of_sampled_classes = int(sampled_probabilities[index] * num_samples_per_client)
        num_sample = min(number_of_sampled_classes, len(dataset_classes[key]))
        sample.append(np.random.choice(value, num_sample, replace=False))

    indices = np.concatenate(sample)
    np.random.shuffle(indices)
    return indices
