import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters:
epochs = 3
batch_size = 102 # Not sure about this; will almost certainly need changing
learning_rate = 0.01

transformations = trans.Compose([trans.ToTensor(), trans.Resize((500, 500))])


# Load the dataset - split into training and testing dataset:
training_dataset = torchvision.datasets.Flowers102(root='./data', split="train", 
                                                   download=True, transform=transformations)
testing_dataset = torchvision.datasets.Flowers102(root='./data', split="test",
                                                   download=True, transform=transformations)
# We use trans.ToTensor() to transform the input data into pytorch tensors of the form: [num_channels, image_height, image_width], e.g. [3, 597, 500] 


# Load the data set using the pytorch data loader:
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
# Don't want to shuffle the test data


classes = np.arange(0, 102) # Creates array of numbers from 0 to 102 (exclusive of 102)
 

# Define the CNN:
class Flowers_CNN(nn.Module): 
    def __init__(self):
        # Add a pooling layer? 
        # Add another convolutional layer?
        super(Flowers_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2,2),
                                stride=2, padding=(1,1)) # (1,1) padding means the image sizes don't decrease
        self.fully_connected1 = nn.Linear(1008016, 102) 
        # in_features = (num channels (16, from the out_channels of conv1 above) x image height x image width) = 16*625*500
        # self.fully_connected2 = nn.Linear(1000, 300) 
        # self.fully_connected3 = nn.Linear(300, 102) # output_features=102 because there are 102 classes

    def forward(self, val):
        val = F.relu(self.conv1(val))
        val = F.relu(self.fully_connected1(val.view(-1, 1008016))) 
        # val = F.relu(self.fully_connected2(val))
        # val = F.relu(self.fully_connected3(val))
        return val
        

neural_net = Flowers_CNN().to(device)
loss_function = nn.MSELoss()


# Training loop:
for ep in range(epochs): # Each iteration is a forward pass to train the data
    for i, (images, image_labels) in enumerate(training_loader):
        images = images.to(device)
        image_labels = image_labels.to(device)

        label_pred = neural_net(images) 
        print(label_pred[0])
        print(image_labels)
        loss = loss_function(label_pred[0], image_labels) 

        loss.backward() # This just doesn't work at the moment

        if (i+1) % 2000 == 0:
            print("Epoch Number =" + ep + ", Loss =" + loss)
