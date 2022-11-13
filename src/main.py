import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import ConvNet
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import MDS

def train(net, trainloader):

    #Create optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    losses = []
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs
            labels = labels

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                losses.append(running_loss/2000)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    return net, losses

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

    #label learning dataset size
    label_learn_data_size = 1000

    label_learn_data = np.array(trainset.data[0:label_learn_data_size])
    label_learn_labels = np.array(trainset.targets[0:label_learn_data_size])
    PCMDS = MDS.Pre_Cluster_MultiDimensionalScaling(num_classes=10)
    centers_transformed = PCMDS.generate_labels(label_learn_data, label_learn_labels)
    
    trainset.targets = [centers_transformed[i] for i in trainset.targets]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    #Instantiate network
    net = ConvNet.OneHotConvNet()
    net, losses = train(net,trainloader)
    timesteps = np.arange(len(losses))
    plt.plot(timesteps, losses)
    plt.show()
