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

import seaborn as sns

sns.set_theme()

transform= transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data',train=True, download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=50,shuffle=False)

# split trainset into train and validation
val_pctg = 0.1
num_train = len(trainset)
split = int(np.floor(num_train*val_pctg))

indices = list(range(num_train))
np.random.shuffle(indices)

train_idx, val_idx = indices[split:], indices[:split]
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

train_loader = torch.utils.data.DataLoader(trainset,batch_size=50,shuffle=False, sampler = train_sampler)
val_loader = torch.utils.data.DataLoader(trainset,batch_size=50,shuffle=False, sampler = val_sampler)

from torch.nn.modules.upsampling import Upsample

class SDA(nn.Module):
    def __init__(self):
        super(SDA, self).__init__()

        ### ENCODER ###
        
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

        ### DECODER ###

        self.decoder = nn.Sequential(
            nn.Conv2d(40,40,3,padding=1),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(40,40,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(40,20,3,padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20,20,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(20,10,3,padding=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10,3,3,padding=1),
            nn.Sigmoid()
        )


        ### CLASSIFIER ###

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2560,10),
        )

    def forward(self, x):
      x = self.encoder(x)
      x_hat = self.decoder(x)
      y_hat = self.classifier(x)
      return y_hat, x_hat

#epsilon = 0.4
def train():
    init_epoch = 0
    history = []

    net = SDA()
    net.cuda()

    reconstr_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    val_accuracy_history = []
    val_reconstr_history = []
    train_accuracy_history = []
    train_reconstr_history = []
    
    for epoch in range(30):
        running_loss = 0.0
        running_train_reconstr_loss = 0
        total_correct = 0
        total_images = 0

        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.cuda() 
            labels = labels.cuda()

            # ============ Forward ============
            # apply zero-out noise to input
            # mask = (torch.rand(size=(inputs.shape)) < epsilon).int().bool().cuda()
            # corrupted_input = inputs.masked_fill(mask, 0)
            # y_hat, x_hat = net(corrupted_input)
            y_hat, x_hat = net(inputs) 
            r_loss = reconstr_loss(x_hat, inputs) # use original image as target
            c_loss = ce_loss(y_hat, labels)

            # apply loss weight on reconstruction loss
            loss = c_loss + 50*r_loss
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss += loss.data
            if i % 100 == 99:
                print('[%d, %5d] loss: %.9f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            _,predicted = torch.max(y_hat.data,1)
            total_images += labels.size(0)
            total_correct += (predicted==labels).sum().item()
            running_train_reconstr_loss += torch.nn.functional.mse_loss(inputs,x_hat).item()
        
        train_reconstr = running_train_reconstr_loss/i
        train_acc = total_correct/total_images

        train_accuracy_history.append(train_acc)
        train_reconstr_history.append(train_reconstr)

        print("train reconstruction loss:", train_reconstr)
        print("train classification acc:", train_acc)   
                  
        # ============ Validation ============
        with torch.no_grad():
            total_correct = 0
            total_images = 0

            running_test_reconstr_loss = 0.0
            for i,(inputs,labels) in enumerate(val_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                # mask = (torch.rand(size=(inputs.shape)) < epsilon).int().bool().cuda()
                # corrupted_input = inputs.masked_fill(mask, 0)

                # y_hat, x_hat = net(corrupted_input)
                y_hat, x_hat = net(inputs)
                
                ## reconstruction ##
                running_test_reconstr_loss += torch.nn.functional.mse_loss(inputs,x_hat).item()

                ## prediction ##
                _,predicted = torch.max(y_hat.data,1)
                total_images += labels.size(0)
                total_correct += (predicted==labels).sum().item()    
        test_acc = total_correct/total_images
        test_reconstr = running_test_reconstr_loss/i

        val_accuracy_history.append(test_acc)
        val_reconstr_history.append(test_reconstr)
        print("Validation reconstruction loss:", test_reconstr)
        print("Validation classification acc:", test_acc)
        
    return net, train_accuracy_history, train_reconstr_history, val_accuracy_history, val_reconstr_history

net, train_acc_history, train_reconstr_history, test_acc_history, test_reconstr_history = train()

import matplotlib.pyplot as plt

#training set and test set accuracy
plt.plot(np.arange(0,len(train_acc_history),step=1),train_acc_history);
plt.plot(np.arange(0,len(test_acc_history),step=1),test_acc_history);
plt.title(f"Supervised Autoencoder trained with beta = 50")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()

# ============ Test ============
accuracy_arr = []
epsilon_arr = []
for i in range(40):
  epsilon = (i+1)*0.01
  with torch.no_grad():
      total_correct = 0
      total_images = 0

      running_test_reconstr_loss = 0.0
      for i,(inputs,labels) in enumerate(testloader):
          inputs = inputs.cuda()
          labels = labels.cuda()
          
          # apply zero-out noise to input
          mask = (torch.rand(size=(inputs.shape)) < epsilon).int().bool().cuda()
          corrupted_input = inputs.masked_fill(mask, 0)

          y_hat, x_hat = net(corrupted_input)
          
          ## prediction ##
          _,predicted = torch.max(y_hat.data,1)
          total_images += labels.size(0)
          total_correct += (predicted==labels).sum().item()
  epsilon_arr.append(epsilon)  
  accuracy_arr.append(total_correct/total_images)

#training set and test set accuracy
plt.plot(epsilon_arr,accuracy_arr);
plt.title(f"Supervised Autoencoder trained with beta = 50")
plt.xlabel("test epsilon")
plt.ylabel("test accuracy")
plt.show()

