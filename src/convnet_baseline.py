import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import numpy as np
torch.manual_seed(1)

transform= transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
trainset = torchvision.datasets.CIFAR100(root='./data',train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data',train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=True)

## architecture obtained from https://www.kaggle.com/code/pierrevignoles/classification-with-cnn-on-cifar-100 ##

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.L1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size = (3,3), padding = 1), nn.ELU())
        self.L2 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3)), nn.ELU(), nn.MaxPool2d(kernel_size = 2))
        self.L3 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = 1), nn.ELU())
        self.L4 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3)), nn.ELU(), nn.MaxPool2d(kernel_size = 2))
        self.L5 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3,3), padding = 1), nn.ELU())
        self.L6 = nn.Sequential(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3)), nn.ELU(), nn.MaxPool2d(kernel_size = 2))
        self.L7 = nn.Sequential(nn.Flatten(1,-1),nn.Linear(in_features = 2048, out_features = 1024), nn.ELU())
        self.L8 = nn.Sequential(nn.Linear(1024, num_classes), nn.Softmax())


    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.L6(x)
        x = self.L7(x)
        return self.L8(x)



def train():
    init_epoch = 0
    history = []

    net = ConvNet(num_classes = 100)
    net.cuda()
    classificationLoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.01, momentum=.9)

    for epoch in range(init_epoch,30,1):
        # train
        running_loss=0.0
        model_acc = 0.0
        total_images=0
        for i,data in enumerate(trainloader,0):
            inputs, labels = Variable(data[0]).cuda(), Variable(data[1]).cuda()
            optimizer.zero_grad()

            y_hat = net(inputs)

            loss = classificationLoss(y_hat,labels)
            loss.backward()
            optimizer.step()
            _,predicted = torch.max(y_hat.data,1)
            total_images += labels.size(0)
            # train_acc += (pred_label==y).sum()/float(len(train_loader)*x.size(0))
            num_correct = (predicted==labels).sum().item()
            model_acc += num_correct/(len(trainloader)*inputs.size(0))
            running_loss += loss.item()/len(trainloader)

        print("Epoch: ", epoch, " - model_loss: "+ str(running_loss)+' - train_acc: '+str(model_acc))

        # test
        with torch.no_grad():
            total_correct = 0.0
            total_images = 0.0
            test_acc = 0.0
            for data in testloader:
                images, labels = Variable(data[0]).cuda(),Variable(data[1]).cuda()
                y_hat = net(images)
                _,predicted = torch.max(y_hat.data,1)
                total_images += labels.size(0)
                test_acc += (predicted==labels).sum().item()/(len(testloader)*images.size(0))
            print ('Model Test accuracy:',test_acc)

        history.append([model_acc,test_acc])
    
