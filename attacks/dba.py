import random


class DBA:
    def __init__(self, image, dataset_name, test_slogan):
        self.image = image
        self.dataset_name = dataset_name
        self.test_slogan = test_slogan
        self.color = 255 if dataset_name == "mnist" else 0
        
    def party_one(self, image):
        image[:, 0:2, -6:-4] = self.color
        return image

    def party_two(self, image):
        image[:, 0:2, -4:-2] = self.color
        return image

    def party_three(self, image):
        image[:, 0:2, -2:] = self.color
        return image

    def complete_trigger(self, image):
        image[:, 0:2, -6:] = self.color
        return image

    def random_choice_party(self, image):
        random_scheme = random.randint(1, 3)
        if random_scheme == 1:
            image = self.party_one(image)
        elif random_scheme == 2:
            image = self.party_two(image)
        elif random_scheme == 3:
            image = self.party_three(image)
        else:
            raise ValueError(f"Unsupported party scheme: {random_scheme}")
        return image

    def dba_attack(self):
        if self.test_slogan:
            image = self.complete_trigger(self.image)
        else:
            image = self.random_choice_party(self.image)
        return image


def poison_data_with_dba(image, dataset_name, test_slogan):
    dba = DBA(image, dataset_name, test_slogan)
    image = dba.dba_attack()
    return image, 0
