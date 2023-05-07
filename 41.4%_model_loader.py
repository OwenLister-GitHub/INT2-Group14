import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters:
batch_size = 256

transformations1 = trans.Compose([trans.ToTensor(), 
                                 trans.Resize((100,100))])
                                 


# Load the dataset - split into training and testing dataset:
testing_dataset = torchvision.datasets.Flowers102(root='./data', split="test",
                                                   download=True, transform=transformations1)


# Load the data set using the pytorch data loader:
testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
# Don't want to shuffle the test data





testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
# Don't want to shuffle the test data


classes = np.arange(0, 102) # Creates array of numbers from 0 to 102 (exclusive of 102)
 

# Define the CNN:
class Flowers_CNN(nn.Module): 
    def __init__(self):
        super(Flowers_CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5,
                                stride=(1,1), padding=(1,1)) 
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5,
                                stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5,
                                stride=(1,1), padding=(1,1))
        self.batch_layer1=nn.BatchNorm2d(256)
        self.batch_layer2=nn.BatchNorm2d(256)
        self.batch_layer3=nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(3,3)
        # self.fully_connected1 = nn.Linear(16384, 1000) 
        # self.fully_connected2 = nn.Linear(1000, 500)
        self.fully_connected1 = nn.Linear(16384, 102)
        # in_features = (num channels (16, from the out_channels of conv1 above) x image height x image width) 

    def forward(self, val):
        val = self.conv1(val)
        val = F.relu(self.batch_layer1(val))
        val = self.pool1(val)
        val = self.conv2(val)
        val = F.relu(self.batch_layer2(val))
        val = self.pool1(val)        
        val = self.conv3(val)
        val = F.relu(self.batch_layer3(val))
        val = self.flatten(val)
        val = self.fully_connected1(val)
        # val = self.fully_connected2(val)
        # val = self.fully_connected3(val)
        return val
    
neural_net = Flowers_CNN().to(device)
neural_net.load_state_dict(torch.load('41.4%Model.pth', map_location='cpu'))
neural_net.eval()





# Accuracy calculations:
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(102)]
    n_class_samples = [0 for i in range(102)]
    for images, labels in testing_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = neural_net(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(102):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')