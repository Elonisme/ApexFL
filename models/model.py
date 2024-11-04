from models.alexnet import AlexNet
from models.cnn import CNNc1, CNNc3
from models.densenet import densenet_cifar
from models.googlenet import GoogLeNet
from models.lenet200 import LeNet200
from models.lenetc1 import LeNetC1, LeNetC3
from models.mobilenetv2 import get_mobilenetV2_10, get_mobilenetV2_200
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet18_pretrained, ResNet34_pretrained, ResNet50_pretrained, ResNet101_pretrained, ResNet152_pretrained
from models.vgg import VGG
from models.wideresnet import WideResNet


class Model:
    def __init__(self, model_name, pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained

    def get_model(self):
        if self.model_name == 'mobilenet':
            net = get_mobilenetV2_10(pretrained=True)
        elif self.model_name == 'mobilenet_200':
            net = get_mobilenetV2_200(pretrained=True)
        elif self.model_name == 'cnn_c1':
            net = CNNc1()
        elif self.model_name == 'cnn_c3':
            net = CNNc3()
        elif self.model_name == 'lenet_c1':
            net = LeNetC1()
        elif self.model_name == 'lenet_c3':
            net = LeNetC3()
        elif self.model_name == 'resnet18':
            net = ResNet18(pretrained=self.pretrained)
        elif self.model_name == 'resnet34':
            net = ResNet34()
        elif self.model_name == 'resnet50':
            net = ResNet50()
        elif self.model_name == 'resnet101':
            net = ResNet101()
        elif self.model_name == 'resnet152':
            net = ResNet152()
        elif self.model_name == 'wideresnet':
            net = WideResNet(depth=28, num_classes=10)
        elif self.model_name == 'vgg13':
            net = VGG('VGG13')
        elif self.model_name == 'vgg16':
            net = VGG('VGG16')
        elif self.model_name == 'vgg19':
            net = VGG('VGG19')
        elif self.model_name == 'densenet_cifar':
            net = densenet_cifar()
        elif self.model_name == 'alexnet':
            net = AlexNet()
        elif self.model_name == 'googlenet':
            net = GoogLeNet()
        elif self.model_name == 'lenet_200':
            net = LeNet200()
        elif self.model_name == 'resnet18_200':
            net = ResNet18_pretrained(out_features=200, dataset_name='tiny_imagenet')
        elif self.model_name == 'resnet34_200':
            net = ResNet34_pretrained(out_features=200)
        elif self.model_name == 'resnet50_200':
            net = ResNet50_pretrained(out_features=200)
        elif self.model_name == 'resnet101_200':
            net = ResNet101_pretrained(out_features=200)
        elif self.model_name == 'resnet152_200':
            net = ResNet152_pretrained(out_features=200)
        elif self.model_name == 'resnet18_100':
            net = ResNet18_pretrained(out_features=100, dataset_name='cifar100')
        elif self.model_name == 'resnet34_100':
            net = ResNet34_pretrained(out_features=100)
        elif self.model_name == 'resnet50_100':
            net = ResNet50_pretrained(out_features=100)
        elif self.model_name == 'resnet101_100':
            net = ResNet101_pretrained(out_features=100)
        elif self.model_name == 'resnet152_100':
            net = ResNet152_pretrained(out_features=100)
        else:
            raise KeyError(f'Model {self.model_name} not implemented')
        return net
