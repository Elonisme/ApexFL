import json

import math
import multiprocessing
import subprocess
import psutil
import time
import torch
import torch.nn as nn

from lib.core.server import Server
from lib.loading_data.load_data import LoadData
from models.model import Model
from lib.result.save import Save


def get_gpu_memory_usage():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    memory_info = output.decode("utf-8").strip().split(", ")
    used_memory, total_memory = map(int, memory_info)
    memory_rate = used_memory / total_memory
    return memory_rate


def calculate_gpu_memory_utilization():
    available_memory_rate = get_gpu_memory_usage()
    return available_memory_rate


def worker(config):
    if torch.cuda.is_available():
        print("CUDA is available!")
        while True:
            gpu_utilization = calculate_gpu_memory_utilization()
            print(f"GPU usage：{gpu_utilization * 100:.2f}%")
            if gpu_utilization > 0.75:
                time.sleep(300)
            else:
                break
    else:
        print("CUDA is not available.")

    formatted_dict = json.dumps(config, indent=4, ensure_ascii=False)
    print(f"Apex Federated Learning Detail: \n{formatted_dict}")
    num_clients = config['num_clients']
    client_frac = config['client_frac']
    malicious_rate = config['malicious_rate']
    model_name = config['model_name']
    data_name = config['data_name']
    aggregate_type = config['aggregate_type']
    poison_type = config['poison_type']
    poisoning_threshold = config['poisoning_threshold']
    num_epochs = config['num_epochs']
    save_slogan = config['save_slogan']
    fl_print = config['fl_print']
    sampling_stride = config['sampling_stride']
    alpha = config['alpha']
    poison_probability = config['poison_probability']
    pretrained = config['pretrained']

    # load dataset
    data = LoadData(data_name, poison_type)
    train_set, test_loader, poison_test_loader = data.get_date()

    # new model
    net_model = Model(model_name, data_name, pretrained)
    net = net_model.get_model()

    # set save detail
    detail = (f"{model_name}-{data_name}"
              f"-{poison_type}-{aggregate_type}"
              f"-mr:{malicious_rate}-frac:{client_frac}"
              f"-pt:{poisoning_threshold}-epochs:{num_epochs}"
              f"-alpha:{alpha}-prob:{poison_probability}"
              f"-pretrained:{pretrained}")

    save = Save(root_path='save',
                detail=detail,
                save_slogan=save_slogan)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fl_server = Server(model=net,
                       model_name=model_name,
                       criterion=criterion,
                       num_clients=num_clients,
                       client_frac=client_frac,
                       malicious_rate=malicious_rate,
                       iid=False,
                       data_name=data_name,
                       train_set=train_set,
                       test_loader=test_loader,
                       poison_test_loader=poison_test_loader,
                       aggregate_type=aggregate_type,
                       poison_type=poison_type,
                       poisoning_threshold=poisoning_threshold,
                       save=save,
                       device=device,
                       fl_print=fl_print,
                       sampling_stride=sampling_stride,
                       alpha=alpha,
                       poison_probability=poison_probability)

    fl_server.federated_learning(num_epochs)


def sleep_log(x):
    return math.log(x + 1) * 30


class MultiFl:
    def __init__(self, configs):
        self.processes = []
        self.configs = configs

    def multi_task(self):
        for i, config in enumerate(self.configs):
            while True:
                gpu_utilization = calculate_gpu_memory_utilization()
                cpu_usage = psutil.cpu_percent(interval=None)
                print(f"GPU usage：{gpu_utilization * 100:.2f}%")
                print(f"CPU usage：{cpu_usage:.2f}%")
                if gpu_utilization > 0.70:
                    time.sleep(300)
                else:
                    break

            print(f"Creating process {i + 1}...")
            p = multiprocessing.Process(target=worker, args=(config,))
            self.processes.append(p)
            p.start()
            time.sleep(sleep_log(i))

        for p in self.processes:
            p.join()
