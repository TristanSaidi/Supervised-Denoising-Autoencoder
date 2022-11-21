import numpy as np

import torch as th
import torchvision
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt

stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))

train_transform = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.RandomCrop(32,padding=4,padding_mode="reflect"),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

test_transform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(*stats)
])

if __name__ == "__main__":
    train_data = CIFAR100(download=True,root="./data",transform=train_transform)
    print(train_data[0])
    test_data = CIFAR100(root="./data",train=False,transform=test_transform)