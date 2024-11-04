import random

from attacks.blended_attack import poison_data_with_blended
from attacks.dba import poison_data_with_dba
from attacks.semantic_attack import poison_data_with_semantic
from attacks.sig_attack import poison_data_with_sig
from attacks.trigger_attack import poison_data_with_trigger


class Poison:
    def __init__(self, poison_type, probability):
        self.poison_type = poison_type
        self.probability = probability

    def probability_generator(self):
        return random.random() < self.probability

    def poison_function(self, image, label, dataset_name, test_slogan):
        if self.probability_generator():
            if self.poison_type == 'trigger':
                image, label = poison_data_with_trigger(image, dataset_name)
            elif self.poison_type == 'semantic':
                image, label = poison_data_with_semantic(image, label)
            elif self.poison_type == 'sig':
                image, label = poison_data_with_sig(image)
            elif self.poison_type == 'blended':
                image, label = poison_data_with_blended(image, dataset_name)
            elif self.poison_type == 'dba':
                image, label = poison_data_with_dba(image, dataset_name, test_slogan)
            else:
                raise ValueError(f'Poison type {self.poison_type} not supported!')
            return image, label
        else:
            # 概率之外不使用中毒数据
            return image, label
