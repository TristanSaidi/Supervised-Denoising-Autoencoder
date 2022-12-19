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

from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Subset

print(len(trainset))
VAL_SIZE = 0.1
train_indices, val_indices, _, _ = train_test_split(
    range(len(trainset)),
    trainset.targets,
    stratify=trainset.targets,
    test_size = VAL_SIZE,
)


train_ds = Subset(trainset, train_indices)
val_ds = Subset(trainset, val_indices)
print(len(val_ds))

#trainloader = torch.utils.data.DataLoader(train_ds,batch_size=50,shuffle=True)
valloader = torch.utils.data.DataLoader(val_ds, batch_size=50, shuffle=False)

# #percent of data that is labeled
labeled_pctg = 0.50
num_train = len(train_ds)
split = int(np.floor(num_train*labeled_pctg))

indices = list(range(num_train))
np.random.shuffle(indices)

unlabeled_train_idx, labeled_train_idx = indices[split:], indices[:split]
unlabeled_train_sampler = torch.utils.data.SubsetRandomSampler(unlabeled_train_idx)
labeled_train_sampler = torch.utils.data.SubsetRandomSampler(labeled_train_idx)

print(num_train)
print(len(labeled_train_idx))

unlabeled_train_loader = torch.utils.data.DataLoader(train_ds,batch_size=50,shuffle=False, sampler = unlabeled_train_sampler)
labeled_train_loader = torch.utils.data.DataLoader(train_ds,batch_size=50,shuffle=False, sampler = labeled_train_sampler)

from torch.nn.modules.upsampling import Upsample

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

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
        
        ### CLASSIFIER ###
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2560,10),
        )

    def forward(self, x):
      x = self.encoder(x)
      x = self.classifier(x)
      return x

NOISE = 2e-4

def train(trainloader):
    init_epoch = 0
    history = []

    net = ConvNet().cuda()
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    train_accuracy_history = []
    train_loss_history = []  
    val_accuracy_history = []
    val_loss_history = []
    test_accuracy_history = []
    test_loss_history = []

    for epoch in range(40):
        running_loss = 0.0
        running_train_loss = 0.0
        total_correct = 0
        total_images = 0
        for i, (inputs, labels) in enumerate(trainloader, 0):

            #inputs = inputs.cuda()/255
            inputs = inputs.cuda()
            labels = labels.cuda()

            # ============ Forward ============
            #outputs = net(inputs + torch.from_numpy(np.random.normal(scale = NOISE, size = inputs.shape).astype(np.float32)).cuda())
            outputs = net(inputs)
            loss = CELoss(outputs, labels)
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ============ Logging ============
            running_loss += loss.data
            running_train_loss += loss.data
            if i % 100 == 99:
                print('[%d, %5d] loss: %.9f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            

            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

        train_acc = (total_correct/total_images)
        train_loss = (running_train_loss/i).cpu().item()
        train_accuracy_history.append(train_acc)
        train_loss_history.append(train_loss)

        print("train classification acc:", train_acc)        
        print("train loss:", train_loss)


        # ============ Validation Set ============
        with torch.no_grad():
          total_correct = 0
          total_images = 0
            
          running_val_loss = 0.0
          for i, (input,labels) in enumerate(valloader):

            y_hat = net(input.cuda())
            labels = labels.cuda()
            running_val_loss += CELoss(y_hat, labels)
            #valid_losses.append(loss.cpu())
            #val_loss = np.average(valid_losses)

            _,predicted = torch.max(y_hat.data,1)
            total_images += labels.size(0)
            total_correct += (predicted==labels).sum().item()
          #print("Val loss", val_loss)

          val_acc = (total_correct/total_images)
          val_loss = (running_val_loss/i).cpu().item()
        
          #append information to history vector
          val_accuracy_history.append(val_acc)
          val_loss_history.append(val_loss)

          print("val classification acc:", val_acc)
          print("val loss:", val_loss)
 
        # ============ Test Set ============

        with torch.no_grad():
            total_correct = 0
            total_images = 0
            
            running_test_loss = 0.0
            for i,(input,labels) in enumerate(testloader):

                input = input.cuda()
                #mask = (torch.rand(size=(input.shape)) < 0.3).int().bool().cuda()
                #corrupted_input = input.masked_fill(mask, 0)

                y_hat = net(input)
                labels = labels.cuda()
                
                running_test_loss += CELoss(outputs, labels)
                _,predicted = torch.max(y_hat.data,1)
                total_images += labels.size(0)
                total_correct += (predicted==labels).sum().item()
        
        test_acc = (total_correct/total_images)
        test_loss = (running_test_loss/i).cpu().item()
        
        #append information to history vector
        test_accuracy_history.append(test_acc)
        test_loss_history.append(test_loss)

        print("test classification acc:", test_acc)
        print("test loss:", test_loss)

        

    return net, train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, test_accuracy_history, test_loss_history

net, train_acc_history, train_loss_history, val_acc_history, val_loss_history, test_acc_history, test_loss_history = train(labeled_train_loader)

import matplotlib.pyplot as plt

#training set and test set accuracy
plt.title("Convolutional Baseline (50/50 split) Train and Validation Acc.")
plt.plot(np.arange(0,len(train_acc_history),step=1),train_acc_history);
plt.plot(np.arange(0,len(val_acc_history),step=1),val_acc_history);
#plt.plot(np.arange(0,len(test_acc_history),step=1),test_acc_history);
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation", "Test"])
plt.show()

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
          
          mask = (torch.rand(size=(inputs.shape)) < epsilon).int().bool().cuda()
          corrupted_input = inputs.masked_fill(mask, 0)

          y_hat = net(corrupted_input)

          ## prediction ##
          _,predicted = torch.max(y_hat.data,1)
          total_images += labels.size(0)
          total_correct += (predicted==labels).sum().item()

  epsilon_arr.append(epsilon)  
  accuracy_arr.append(total_correct/total_images)

#training set and test set accuracy
plt.plot(epsilon_arr,accuracy_arr);
plt.title("Convolutional Baseline (50/50 split) Trained with No Noise")
plt.xlabel("test noise epsilon")
plt.ylabel("test accuracy")
plt.show()
