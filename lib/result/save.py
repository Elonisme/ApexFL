import csv
import os

import torch


class Save:
    def __init__(self, root_path, detail, save_slogan=True):
        self.root_path = root_path
        self.detail = detail
        self.create_directories()
        self.save_slogan = save_slogan

    def create_directories(self):
        directories = ['csv', 'pdf', 'img', 'weight']
        for directory in directories:
            path = os.path.join(self.root_path, directory)
            os.makedirs(path, exist_ok=True)

    def save_csv(self, data):
        if self.save_slogan:
            transposed_data = list(map(list, zip(*data)))
            with open(os.path.join(self.root_path, "csv", self.detail + ".csv"), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(transposed_data)
            print(f"wrote to {self.root_path}/csv/{self.detail}.csv")

    def read_csv(self):
        filepath = os.path.join(self.root_path, "csv", self.detail + ".csv")
        if os.path.exists(filepath):
            with open(filepath, mode='r') as file:
                reader = csv.reader(file)
                data = list(zip(*reader))
                return data
        else:
            print(f"File {self.detail}.csv does not exist.")
            return None

    def save_weight(self, weight):
        if self.save_slogan:
            filepath = os.path.join(self.root_path, "weight", self.detail + ".pt")
            torch.save(weight, filepath)
            print(f"wrote to {self.root_path}/weight/{self.detail}.pt")

    def load_model(self):
        filepath = os.path.join(self.root_path, "weight", self.detail + ".pt")
        if os.path.exists(filepath):
            weight = torch.load(filepath)
            print("Model loaded successfully.")
            return weight
        else:
            print(f"File {self.detail}.pth does not exist.")
            return None
