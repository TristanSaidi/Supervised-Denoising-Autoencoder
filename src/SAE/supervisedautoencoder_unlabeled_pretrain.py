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

VAL_SIZE = 0.1
train_indices, val_indices, _, _ = train_test_split(
    range(len(trainset)),
    trainset.targets,
    stratify=trainset.targets,
    test_size = VAL_SIZE,
)


train_ds = Subset(trainset, train_indices)
val_ds = Subset(trainset, val_indices)

#trainloader = torch.utils.data.DataLoader(train_ds,batch_size=50,shuffle=True)
valloader = torch.utils.data.DataLoader(val_ds, batch_size=50, shuffle=False)

# #percent of data that is labeled
labeled_pctg = 0.10
num_train = len(train_ds)
split = int(np.floor(num_train*labeled_pctg))

indices = list(range(num_train))
np.random.shuffle(indices)

unlabeled_train_idx, labeled_train_idx = indices[split:], indices[:split]
unlabeled_train_sampler = torch.utils.data.SubsetRandomSampler(unlabeled_train_idx)
labeled_train_sampler = torch.utils.data.SubsetRandomSampler(labeled_train_idx)

unlabeled_train_loader = torch.utils.data.DataLoader(train_ds,batch_size=50,shuffle=False, sampler = unlabeled_train_sampler)
labeled_train_loader = torch.utils.data.DataLoader(train_ds,batch_size=50,shuffle=False, sampler = labeled_train_sampler)

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

NOISE = 2e-4

def train(net = None, labeled_data_loader = None, unlabeled_data_loader = None, valloader=None):
    init_epoch = 0
    history = []

    if net == None:
      net = SDA()
    net.cuda()

    reconstr_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    assert labeled_data_loader == None or unlabeled_data_loader == None, "Cannot pass two datasets in at once"

    supervised = False

    #set data_loader to whichever data was passed in
    if labeled_data_loader != None:
      print("Beginning supervised training")
      supervised = True
      data_loader = labeled_data_loader
    else:
      print("Beginning unsupervised training")
      supervised = False
      data_loader = unlabeled_data_loader

    val_accuracy_history = []
    val_reconstr_history = []
    train_accuracy_history = []
    train_reconstr_history = []
    test_accuracy_history = []
    test_reconstr_history = []
    
    for epoch in range(40):
        running_loss = 0.0
        running_train_reconstr_loss = 0.0
        
        total_correct = 0
        total_images = 0

        for i, (inputs, labels) in enumerate(data_loader, 0):
            inputs = inputs.cuda()  
            labels = labels.cuda()

            # ============ Forward ============
            y_hat, x_hat = net(inputs)
            r_loss = reconstr_loss(x_hat, inputs) #but use original image as target
            c_loss = ce_loss(y_hat, labels)

            if supervised:
              loss = c_loss
            else:
              loss = 5*r_loss

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

            ## reconstruction ##
            running_train_reconstr_loss += torch.nn.functional.mse_loss(inputs,x_hat)

            ## prediction ##
            _,predicted = torch.max(y_hat.data,1)
            total_images += labels.size(0)
            total_correct += (predicted==labels).sum().item()  

        train_acc = (total_correct/total_images)
        train_reconstr = (running_train_reconstr_loss/i).cpu().item()
        
        #append information to history vector
        if supervised == True:
          train_accuracy_history.append(train_acc)
        train_reconstr_history.append(train_reconstr)

        print("train classification acc:", train_acc)        
        print("train reconstruction loss:", train_reconstr)



        # ============ Validation ============
        if supervised == True:

          with torch.no_grad():
            #run_loss = 0.0
            running_val_reconstr_loss = 0.0
            running_val_loss = 0.0

            total_correct = 0
            total_images = 0
            for i, (input,labels) in enumerate(valloader):

              y_hat, x_hat = net(input.cuda())
              labels = labels.cuda()
              running_val_reconstr_loss += reconstr_loss(x_hat, input.cuda()) #but use original image as target
              running_val_loss += ce_loss(y_hat, labels)
              #valid_losses.append(loss.cpu())
              #val_loss = np.average(valid_losses)

              _,predicted = torch.max(y_hat.data,1)
              total_images += labels.size(0)
              total_correct += (predicted==labels).sum().item()
            #print("Val loss", val_loss)

            val_acc = (total_correct/total_images)
            #val_loss = (running_loss/i).cpu().item()
            val_reconstr = (running_val_reconstr_loss/i).cpu().item()
          
            #append information to history vector
            val_accuracy_history.append(val_acc)
            val_reconstr_history.append(val_reconstr)

            print("val classification acc:", val_acc)
            print("val reconstruction loss:", val_reconstr)
                  
        # ============ Test ============
        if supervised == True:
          with torch.no_grad():
              total_correct = 0
              total_images = 0

              running_test_reconstr_loss = 0.0
              for i,(inputs,labels) in enumerate(testloader):
                  inputs = inputs.cuda()
                  labels = labels.cuda()
                  y_hat, x_hat = net(inputs) #add gaussian noise to each input
                  
                  ## reconstruction ##
                  running_test_reconstr_loss += torch.nn.functional.mse_loss(inputs,x_hat)

                  ## prediction ##
                  _,predicted = torch.max(y_hat.data,1)
                  total_images += labels.size(0)
                  total_correct += (predicted==labels).sum().item()    

          test_acc = (total_correct/total_images)
          test_reconstr = (running_test_reconstr_loss/i).cpu().item()
          
          #append information to history vector
          test_accuracy_history.append(test_acc)
          test_reconstr_history.append(test_reconstr)

          print("test classification acc:", test_acc)
          print("test reconstruction loss:", test_reconstr)


    return net, train_accuracy_history, train_reconstr_history, val_accuracy_history, val_reconstr_history, test_accuracy_history, test_reconstr_history

info  = train(net = None, labeled_data_loader = None, unlabeled_data_loader = unlabeled_train_loader, valloader=valloader)

#parse information from unsupervised training
net, train_accuracy_history_u, train_reconstr_history_u, val_accuracy_history_u, val_reconstr_history_u, test_accuracy_history_u, test_reconstr_history_u = info
#parse information from supervised training
info = train(net, labeled_data_loader = labeled_train_loader, unlabeled_data_loader = None, valloader = valloader)
net, train_accuracy_history_s, train_reconstr_history_s, val_accuracy_history_s, val_reconstr_history_s, test_accuracy_history_s, test_reconstr_history_s = info

import matplotlib.pyplot as plt

train_accuracy_history = train_accuracy_history_s #train_accuracy_history_u+
train_reconstr_history = train_reconstr_history_s #train_reconstr_history_u+
test_accuracy_history =  test_accuracy_history_s #test_accuracy_history_u +
test_reconstr_history = test_reconstr_history_s #test_reconstr_history_u +
val_accuracy_history =  val_accuracy_history_s #test_accuracy_history_u +
#training set and test set accuracy
plt.title("Supervised Autoencoder (90/10 split) Train and Validation Acc.")
plt.plot(np.arange(0,len(train_accuracy_history),step=1),train_accuracy_history);
plt.plot(np.arange(0,len(val_accuracy_history),step=1),val_accuracy_history);
#plt.plot(np.arange(0,len(test_accuracy_history),step=1),test_accuracy_history);
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()

#training set and test set reconstruction loss

#plt.plot(np.arange(0,len(train_reconstr_history),step=1),train_reconstr_history);
#plt.plot(np.arange(0,len(test_reconstr_history),step=1),test_reconstr_history);
#plt.xlabel("Epoch")
#plt.ylabel("Reconstruction Loss")
#plt.legend(["Train", "Validation"])
#plt.show()

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

          y_hat, x_hat = net(corrupted_input)
          
          ## prediction ##
          _,predicted = torch.max(y_hat.data,1)
          total_images += labels.size(0)
          total_correct += (predicted==labels).sum().item()
  epsilon_arr.append(epsilon)  
  accuracy_arr.append(total_correct/total_images)

#training set and test set accuracy
plt.plot(epsilon_arr,accuracy_arr);
plt.title(f"Supervised Autoencoder (90/10 split) trained with No Noise")
plt.xlabel("test epsilon")
plt.ylabel("test accuracy")
plt.show()

