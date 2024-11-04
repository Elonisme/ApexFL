import numpy as np
import torch

from lib.io.flprint import log_print
from lib.loading_data.poison_data import PoisonDataset


class Client:
    def __init__(self, model_weights, fl_print):
        self.model_weights = model_weights
        self.fl_print = fl_print

    def calculate_entropy(self, all_labels):
        label_counts = np.bincount(all_labels)
        probabilities = label_counts / len(all_labels)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy

    def train_model(self, share_model, data_name, client_train_set, poison_type, poison_probability, poison_slogan):
        if poison_slogan is False:
            train_loader = torch.utils.data.DataLoader(client_train_set, batch_size=64, shuffle=True, drop_last=True)
        else:
            log_print("Poison Data Injection!", self.fl_print)
            poisoned_train_set = PoisonDataset(client_train_set, data_name, poison_type, poison_probability)
            train_loader = torch.utils.data.DataLoader(poisoned_train_set, batch_size=64, shuffle=True, drop_last=True)

        avg_loss = share_model.shared_train_model(self.model_weights, train_loader)
        self.model_weights = share_model.get_model_weights()
        return avg_loss

    def get_weights(self):
        return self.model_weights

    def set_weights(self, weights):
        self.model_weights = weights
