import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import numpy as np
import seaborn as sns
sns.set_theme()
torch.manual_seed(1)

transform= transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data',train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=50,shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=50,shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,10,3,padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10,20,3,padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20,40,3,padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.Conv2d(40,40,3,padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        ### CLASSIFIER ###

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2560,10),
        )

    def forward(self, x):
      x = self.encoder(x)
      x = self.classifier(x)
      return x

def train():
    init_epoch = 0
    history = []

    net = ConvNet().cuda()
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    train_acc = []
    test_acc = []

    for epoch in range(10):
        running_loss = 0.0
        total_correct = 0
        total_images = 0

        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            # ============ Forward ============
            outputs = net(inputs)
            loss = CELoss(outputs, labels)
            # ============ Backward ============
            loss.backward()
            optimizer.step()
            # ============ Logging ============
            running_loss += loss.data
            if i % 100 == 99:
                print('[%d, %5d] loss: %.9f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)
        
        train_acc.append(total_correct/total_images)
        print('Model Train acc:', total_correct/total_images)
                  
        # ============ Validation ============
        with torch.no_grad():
            total_correct = 0
            total_images = 0
            for i,(input,labels) in enumerate(testloader):
                labels = labels.cuda()
                input = input.cuda()
                # apply zero-out noise to the input
                mask = (torch.rand(size=(input.shape)) < 0.1).int().bool().cuda()
                corrupted_input = input.masked_fill(mask, 0)

                y_hat = net(corrupted_input)
                _,predicted = torch.max(y_hat.data,1)
                total_images += labels.size(0)
                total_correct += (predicted==labels).sum().item()
        test_acc.append(total_correct/total_images)      
        print ('Model Test accuracy:',total_correct/total_images)

    return net, train_acc, test_acc

net, train_acc, test_acc = train()

import matplotlib.pyplot as plt

#training set and test set accuracy
plt.plot(np.arange(0,len(train_acc),step=1),train_acc);
plt.plot(np.arange(0,len(test_acc),step=1),test_acc);
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Test"])
plt.ylim([0,1])
plt.show()

# ============ Test ============
with torch.no_grad():
    total_correct = 0
    total_images = 0

    running_test_reconstr_loss = 0.0
    for i,(inputs,labels) in enumerate(testloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        # apply zero-out noise to input
        mask = (torch.rand(size=(inputs.shape)) < 0.1).int().bool().cuda()
        corrupted_input = inputs.masked_fill(mask, 0)

        y_hat = net(corrupted_input)

        ## prediction ##
        _,predicted = torch.max(y_hat.data,1)
        total_images += labels.size(0)
        total_correct += (predicted==labels).sum().item()    
print(total_correct/total_images)

