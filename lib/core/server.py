import copy

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from aggregates.aggregate import Aggregate
from lib.core.client import Client
from lib.core.share import ShareModel
from lib.io.flprint import log_print
from lib.result.positive_judgment import Judgment
from lib.subset.fl_subset import get_data_class, get_iid_subset, get_no_idd_subset


class Server:
    def __init__(self, model, model_name, criterion, num_clients, client_frac, malicious_rate, iid, data_name,
                 train_set, test_loader, poison_test_loader, aggregate_type, poison_type, poisoning_threshold,
                 device, save, fl_print, sampling_stride, alpha, poison_probability):
        self.device = device
        self.model = model.to(self.device)
        self.model_name = model_name
        self.weights = copy.deepcopy(self.model.state_dict())
        self.criterion = criterion
        self.num_clients = num_clients
        self.client_frac = client_frac
        self.malicious_rate = malicious_rate
        self.iid = iid
        self.aggregate_type = aggregate_type
        self.poison_type = poison_type
        self.poisoning_threshold = poisoning_threshold
        self.data_name = data_name
        self.train_set = train_set
        self.clients = []
        self.sampling_stride = sampling_stride
        self.poison_probability = poison_probability
        dataset_classes, num_classes = get_data_class(train_set)
        num_dataset = len(train_set)
        num_samples = num_dataset
        selected_client_number = int(round(num_clients * client_frac))
        self.client_data_indices = []
        for i in range(num_clients):
            self.client_data_indices.append(self.get_subset(num_dataset,
                                                            dataset_classes,
                                                            num_classes,
                                                            num_samples,
                                                            alpha))
        self.client_subset = {}
        for i in range(selected_client_number):
            client = Client(copy.deepcopy(self.model.state_dict()), fl_print)
            self.clients.append(client)

        self.test_loader = test_loader
        self.poison_test_loader = poison_test_loader
        self.save = save
        self.fl_print = fl_print

    def get_subset(self, num_dataset, dataset_classes, num_classes, num_samples, alpha=0.5):
        if self.iid:
            indices = get_iid_subset(num_dataset, num_samples)
        else:
            indices = get_no_idd_subset(dataset_classes, num_classes, num_samples, alpha)
        return indices

    def aggregate_gradients(self, model_weights):
        if 'fltrust' in self.aggregate_type:
            aggregate = Aggregate(self.aggregate_type, self.model_name, self.train_set)
        else:
            aggregate = Aggregate(self.aggregate_type, self.model_name)
        self.weights, runtime, guss_malicious = aggregate.aggregate_function(model_weights, self.weights, self.device)
        return runtime, guss_malicious

    def distribution(self):
        for client in self.clients:
            client.set_weights(copy.deepcopy(self.weights))

    def random_choice_clients(self, p):
        client_id = np.array([i for i in range(0, self.num_clients)])
        choice_num = int(round(self.num_clients * p))
        choice_id = np.random.choice(client_id, size=choice_num, replace=False)
        return choice_id

    def evaluate(self):
        self.model.load_state_dict(self.weights)
        self.model.eval()
        main_correct = 0
        total = len(self.test_loader.dataset)
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                main_correct += (predicted == labels).sum().item()
        main_accuracy = 100 * main_correct / total
        log_print(f'Main Task Accuracy on test set: {main_accuracy}%', self.fl_print)

        back_correct = 0
        with torch.no_grad():
            for data in self.poison_test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                back_correct += (predicted == labels).sum().item()
        back_accuracy = 100 * back_correct / total
        log_print(f'Backdoor Task Accuracy on test set: {back_accuracy}%', self.fl_print)
        return main_accuracy, back_accuracy

    def federated_learning(self, num_epochs):
        share_model = ShareModel(self.model, self.criterion, 1, self.device, fl_print=self.fl_print)
        malicious_id = self.random_choice_clients(p=self.malicious_rate)
        ma_s = []
        ba_s = []
        loss_s = []
        run_time_s = []
        TPRs = []
        TNRs = []
        log_print("Begin to Apex Federated Learning", self.fl_print)
        for epoch in tqdm(range(num_epochs), desc='Apex Federated Learning'):
            print()
            model_weights = []
            chose_id = self.random_choice_clients(p=self.client_frac)
            poison_slogan = False if epoch < self.poisoning_threshold else True
            loss = []
            log_print(f"federated learning epoch: {epoch}", self.fl_print)
            model_id = 0
            malicious_is_right = np.array([item in malicious_id for item in chose_id])

            for client_id in tqdm(chose_id, desc='Client Learning'):
                print()
                if client_id not in self.client_subset.keys():
                    self.client_subset[client_id] = Subset(self.train_set,
                                                           self.client_data_indices[client_id])
                if client_id in malicious_id:
                    log_print(f"client: {client_id} join in, but it is malicious!", self.fl_print)
                    temp_loss = self.clients[model_id].train_model(share_model=share_model,
                                                                   data_name=self.data_name,
                                                                   client_train_set=self.client_subset[client_id],
                                                                   poison_type=self.poison_type,
                                                                   poison_probability=self.poison_probability,
                                                                   poison_slogan=poison_slogan)
                    loss.append(temp_loss)
                else:
                    log_print(f"client: {client_id} join in, and it is benevolent!", self.fl_print)
                    temp_loss = self.clients[model_id].train_model(share_model=share_model,
                                                                   data_name=self.data_name,
                                                                   client_train_set=self.client_subset[client_id],
                                                                   poison_type=self.poison_type,
                                                                   poison_probability=self.poison_probability,
                                                                   poison_slogan=False)
                    loss.append(temp_loss)
                model_weights.append(self.clients[model_id].get_weights())
                model_id += 1

            runtime, guss_malicious = self.aggregate_gradients(model_weights)
            positive = Judgment(fact_malicious=malicious_is_right,
                                guss_malicious=guss_malicious)
            TPR, TNR = positive.compare()
            log_print(f"fact malicious: {malicious_is_right}")
            if epoch > self.poisoning_threshold:
                log_print(f"TPR: {TPR * 100}% \nTNR: {TNR * 100}%", self.fl_print)
            self.distribution()
            if (epoch + 1) % self.sampling_stride == 0:
                log_print(f"Apex FL Epoch [{epoch + 1}/{num_epochs}]", self.fl_print)
                ma, ba = self.evaluate()
                ma_s.append(ma)
                ba_s.append(ba)
                run_time_s.append(runtime)
                TPRs.append(TPR)
                TNRs.append(TNR)
                fl_avg_loss = sum(loss) / len(loss)
                log_print(f"fl avg loss: {fl_avg_loss}", self.fl_print)
                loss_s.append(fl_avg_loss)

        self.save.save_csv(data=[ma_s, ba_s, loss_s, run_time_s, TPRs, TNRs])
        self.save.save_weight(weight=self.weights)
        print(f"Average TPR: {sum(TPRs[self.poisoning_threshold:])/(len(TPRs[self.poisoning_threshold:]) + 1e-10) * 100 }%")
        print(f"Average TNR: {sum(TNRs[self.poisoning_threshold:])/(len(TNRs[self.poisoning_threshold:]) + 1e-10) * 100 }%")
