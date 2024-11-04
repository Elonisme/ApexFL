from torch.utils.data import Dataset, Subset

from attacks.poison import Poison
from lib.subset.fl_subset import get_data_class


class PoisonDataset(Dataset):
    def __init__(self, dataset, dataset_name, poison_type, poison_probability, test_slogan=False):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.poison = Poison(poison_type=poison_type,
                             probability=poison_probability)
        self.test_slogan = test_slogan

    def __len__(self):
        # 返回一个用户被分配的数据总长度
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据索引返回对应的数据和标签
        data_sample = self.dataset[idx]
        image, label = data_sample
        poison_image, poison_label = self.poison.poison_function(image=image,
                                                                 label=label,
                                                                 dataset_name=self.dataset_name,
                                                                 test_slogan=self.test_slogan)
        return poison_image, poison_label


class PoisonTestDataset(PoisonDataset):
    def __init__(self, dataset, dataset_name, poison_type, poison_probability=1, test_slogan=True):
        super().__init__(dataset, dataset_name, poison_type, poison_probability, test_slogan)
        self.dataset_classes, _, = get_data_class(dataset)
        if poison_type == 'semantic':
            filter_label_index = self.dataset_classes[5]
            self.dataset = Subset(dataset, filter_label_index)
        else:
            nested_list = [self.dataset_classes[i] for i in range(1, 10)]
            filter_label_index = [item for sublist in nested_list for item in sublist]
            self.dataset = Subset(dataset, filter_label_index)
