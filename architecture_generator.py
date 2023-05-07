import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context







device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters:
epochs = 80
batch_size = 64
learning_rate = 0.00001 

transformations1 = trans.Compose([trans.ToTensor(), 
                                 trans.Resize((100,100))])
                                 


# Load the dataset - split into training and testing dataset:
training_dataset = torchvision.datasets.Flowers102(root='./data', split="train", 
                                                   download=True, transform=transformations1)
testing_dataset = torchvision.datasets.Flowers102(root='./data', split="test",
                                                   download=True, transform=transformations1)


# Load the data set using the pytorch data loader:
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
# Don't want to shuffle the test data




# Transform Dataset with trans.Normalise :
mean = 0.
std = 0.
for images, _ in training_loader:
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(training_loader.dataset)
std /= len(training_loader.dataset)

transformations2 = trans.Compose([trans.ToTensor(), 
                                  trans.RandomHorizontalFlip(0.2),
                                  trans.RandomVerticalFlip(0.2),
                                  trans.RandomRotation(55),
                                  trans.Resize((100,100)),
                                  trans.Normalize(mean=mean, std=std)])





# Load the dataset again;
training_dataset = torchvision.datasets.Flowers102(root='./data', split="train", 
                                                   download=True, transform=transformations2)


# Load the data set using the pytorch data loader again:
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

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
model = neural_net.load_state_dict(torch.load('39.6%Model.pth'))
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate) 







writer = SummaryWriter('model_architectures')

dataiter = iter(training_loader)
images, labels = next(dataiter)

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# write to tensorboard
writer.add_graph(model, images)