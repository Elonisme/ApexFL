from lib.multitasking.multifl import worker, MultiFl


def queue(configs):
    for config in configs:
        try:
            worker(config.copy())
        except Exception as e:
            print(f"An error occurred: {e}")


def mul(configs):
    num_configs = len(configs)
    batch = 10
    for i in range(0, num_configs, batch):
        multi_fl = MultiFl(configs[i:i + batch])
        multi_fl.multi_task()


def single(base_config):
    worker(base_config.copy())
