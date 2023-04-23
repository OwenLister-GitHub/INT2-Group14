import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters:
epochs = 3
batch_size = 100 # Not sure about this; will almost certainly need changing
learning_rate = 0.01


# Load the dataset - split into training and testing dataset:
training_dataset = torchvision.datasets.Flowers102(root='./data', split="train", 
                                                   download=True, transform=transform)
testing_dataset = torchvision.datasets.Flowers102(root='./data', split="test",
                                                   download=True, transform=transform)


# Load the data set using the pytorch data loader:
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
# Don't want to shuffle the test data


classes = np.arange(0, 102) # Creates array of numbers from 0 to 102 (exclusive of 102)


# Define the CNN:
class Flowers_CNN(nn.Module): 
    def __init__(self):
        super(Flowers_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2,2),
                                stride=2, padding=(1,1)) # (1,1) padding means the image sizes don't decrease
        self.fully_connected1 = nn.Linear(16*625*500, 500000) 
        # in_features = (num channels (16, from the out_channels of conv1 above) x image height x image width) = 16*625*500
        self.fully_connected2 = nn.Linear(500000, 50000) 
        self.fully_connected3 = nn.Linear(50000, 102) # output_features=102 because there are 102 classes

    def forward(self):
        pass


print("This does run!")