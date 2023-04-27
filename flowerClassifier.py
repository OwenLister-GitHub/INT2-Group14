import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters:
epochs = 10
batch_size = 10 
learning_rate = 0.01

transformations = trans.Compose([trans.ToTensor(), trans.Resize((200, 200))])


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
        super(Flowers_CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3),
                                stride=2, padding=(1,1)) # (1,1) padding means the image sizes don't decrease
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(10,10),
                                stride=2, padding=(4,4))
        self.pool1 = nn.MaxPool2d(2,2)
        self.fully_connected1 = nn.Linear(9216, 2000) 
        self.fully_connected2 = nn.Linear(2000, 500)
        self.fully_connected3 = nn.Linear(500, 102)
        # in_features = (num channels (16, from the out_channels of conv1 above) x image height x image width) = 16*200*200 (roughly) 

    def forward(self, val):
        val = self.pool1(F.relu(self.conv1(val)))
        val = self.pool1(F.relu(self.conv2(val)))
        val = self.flatten(val)
        val = F.leaky_relu(self.fully_connected1(val))
        val = F.leaky_relu_(self.fully_connected2(val))
        val = F.leaky_relu_(self.fully_connected3(val))
        return val
        

neural_net = Flowers_CNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(neural_net.parameters(), lr=learning_rate) 


# Training loop:
for ep in range(epochs): # Each iteration is a forward pass to train the data
    for i, (images, image_labels) in enumerate(training_loader):
        images = images.to(device)
        image_labels = image_labels.to(device)
        # print(images.shape, "and", image_labels.shape)

        label_pred = neural_net(images) 
        loss = loss_function(label_pred, image_labels) 
        # print(loss)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        print("Epoch Number = " + str(ep) + ", Index =", str(i), "/", str(len(training_loader)-1), "Loss = " + str(loss.item()))


print("Network Accuracy After Training:")


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

        # print(labels, labels.shape)
        # print(predicted, predicted.shape)
        
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