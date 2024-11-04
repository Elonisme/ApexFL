from lib.multitasking.schemes import mul, single, queue


def generate_configs(base_config):
    configs = []
    # models: 'mobilenet', 'lenet_c3', 'resnet18', 'resnet34', 'resnet50', 'resnet101'
    models = ['resnet18']
    # mnist, cifar10
    datas = ['cifar10']
    # 'flame', 'fltrust', and so on
    aggregates = ['fedavg', 'flame', 'fltrust']
    # models: 'trigger', 'dba', 'semantic', 'sig', 'blended'
    poisons = ['trigger']
    # alpha
    alphas = [0.8]
    # malicious rate
    malicious_rates = [0.1]
    # poison probability
    poison_probability = [0.3]
    # pretrained
    pretrained_slog = [False]

    for model in models:
        for data in datas:
            for aggregate in aggregates:
                for poison in poisons:
                    for alpha in alphas:
                        for malicious_rate in malicious_rates:
                            for prob in poison_probability:
                                for pretrained in pretrained_slog:
                                    config = base_config.copy()
                                    config['model_name'] = model
                                    config['data_name'] = data
                                    config['aggregate_type'] = aggregate
                                    config['poison_type'] = poison
                                    config['alpha'] = alpha
                                    config['malicious_rate'] = malicious_rate
                                    config['poison_probability'] = prob
                                    config['pretrained'] = pretrained
                                    configs.append(config)

    return configs


if __name__ == '__main__':
    base_config = {'num_clients': 10, 'client_frac': 0.2, 'malicious_rate': 0.2,
                   'model_name': 'lenet_c1', 'data_name': 'mnist', 'aggregate_type': 'fedavg',
                   'poison_type': 'trigger', 'poisoning_threshold': 5, 'num_epochs': 10,
                   'save_slogan': True, 'fl_print': True, 'sampling_stride': 2, 'alpha': 0.5,
                   'poison_probability': 0.3, 'pretrained': True}

    mode = 'single'
    if mode == 'mul':
        mul(generate_configs(base_config))
    elif mode == 'single':
        single(base_config)
    elif mode == 'queue':
        queue(generate_configs(base_config))
    else:
        raise KeyError("mode must be either 'queue' or 'mul'")
