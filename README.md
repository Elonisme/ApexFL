<h1 align="center">Welcome to ApexFL 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank">
    <img alt="License: GPLv3" src="https://img.shields.io/badge/License-GPLv3-yellow.svg" />
  </a>
</p>

> A Scalable Federated Learning Backdoor Attack and Defense Platform

### 🏠 [Homepage](https://github.com/Elonisme/ApexFL)

<table>
    <tr>
        <td><img src="examples/mnist-trigger.png" alt="Image 1" style="width: 100%; height: auto;"></td>
        <td><img src="examples/mnist-blended.png" alt="Image 2" style="width: 100%; height: auto;"></td>
        <td><img src="examples/mnist-sig.png" alt="Image 3" style="width: 100%; height: auto;"></td>
    </tr>
    <tr>
        <td><img src="examples/cifar10-trigger.png" alt="Image 4" style="width: 100%; height: auto;"></td>
        <td><img src="examples/cifar10-blended.png" alt="Image 5" style="width: 100%; height: auto;"></td>
        <td><img src="examples/cifar10-sig.png" alt="Image 6" style="width: 100%; height: auto;"></td>
    </tr>
</table>



## Features
- Uses half-precision training
- Multiple neural network models
- Various defense algorithms
- Supports four common types of backdoor attacks


## Install

```sh
git clone https://github.com/Elonisme/ApexFL.git
```

## Requirements
```sh
chmod +x ./requirements.sh
./requirements.sh
```

## Usage
### Quick Start
```sh
python main.py
```

### Detailed Instructions
By setting the mode to mul, single, or queue, you can determine whether the program operates in multiprocessing, single-process, or queuing mode.
```python
if __name__ == '__main__':
    mode = 'mul'
    if mode == 'mul':
        mul()
    elif mode == 'single':
        single()
    elif mode == 'queue':
        queue()
    else:
        raise KeyError("mode must be either 'queue' or 'mul'")
```

Modify the basic configuration in the base_config.
```python
base_config = {'num_clients': 100, 'client_frac': 0.15, 'malicious_rate': 0.2,
               'model_name': 'resnet18', 'data_name': 'cifar10', 'aggregate_type': 'flclaude',
               'poison_type': 'trigger', 'poisoning_threshold': 5, 'num_epochs': 50,
               'save_slogan': True, 'fl_print': True, 'sampling_stride': 2, 'alpha': 0.5,
               'poison_probability': 0.3, 'pretrained': False}
```


### Dataset
- MNIST
- CIFAR10
- CIFAR100
- Tiny Imagenet

### Model
- LeNet
- ResNet
- VGG
- AlexNet
- GoogLeNet
- WideResNet

### Attacks
- Dba
- Trigger
- Sig
- Blended

### Defense
- FedAvg
- Flame
- FlTrust
- Krum
- Median
- Deepsight

### Pretrained Weights
- [Google Drive](https://drive.google.com/drive/folders/10CkBB68cRyZNjqUdrNnXUHaV8UznF0P9?usp=drive_link)

## Author

👤 **Elon Li**

* Website: https://elonblog.pages.dev/
* Github: [@Elonisme](https://github.com/Elonisme)


## Show your support

Give a ⭐️ if this project helped you!

## 📝 License

Copyright © 2024 [Elon Li](https://github.com/Elonisme).<br />
This project is [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) licensed.

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_
