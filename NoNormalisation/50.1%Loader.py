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


batch_size = 64
learning_rate = 0.0001 

transformations1 = trans.Compose([trans.ToTensor(), 
                                 trans.Resize((100,100))])
                                 


# Load the dataset - split into training and testing dataset:
training_dataset = torchvision.datasets.Flowers102(root='./data', split="train", 
                                                   download=True, transform=transformations1)

# Load the data set using the pytorch data loader:
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)



transformations2 = trans.Compose([trans.ToTensor(), 
                                  trans.RandomHorizontalFlip(0.2),
                                  trans.RandomVerticalFlip(0.2),
                                  trans.RandomRotation(55),
                                  trans.Resize((250,250))])


valid_and_test_transforms = trans.Compose([trans.ToTensor(), 
                                           trans.Resize((250,250))])




# Load the dataset again;
training_dataset = torchvision.datasets.Flowers102(root='./data', split="train", 
                                                   download=True, transform=transformations2)
testing_dataset = torchvision.datasets.Flowers102(root='./data', split="test",
                                                   download=True, transform=valid_and_test_transforms)


# Load the data set using the pytorch data loader again:
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)


classes = np.arange(0, 102) # Creates array of numbers from 0 to 102 (exclusive of 102)
 

# Define the CNN:
class Flowers_CNN(nn.Module): 
    def __init__(self):
        super(Flowers_CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                                stride=(1,1), padding=(1,1)) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                                stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                                stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                                stride=(1,1), padding=(1,1))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                                stride=(1,1), padding=(1,1))
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                stride=(1,1), padding=(1,1))
        self.batch_layer1=nn.BatchNorm2d(16)
        self.batch_layer2=nn.BatchNorm2d(32)
        self.batch_layer3=nn.BatchNorm2d(64)
        self.batch_layer4=nn.BatchNorm2d(128)
        self.batch_layer5=nn.BatchNorm2d(256)
        self.batch_layer6=nn.BatchNorm2d(256)

        self.paramater_relu = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        self.fully_connected1 = nn.Linear(12544, 102)
        # in_features = (num channels (16, from the out_channels of conv1 above) x image height x image width) 

    def forward(self, val):
        val = self.conv1(val)
        val = self.paramater_relu(self.batch_layer1(val))
        val = self.pool1(val)
        val = self.conv2(val)
        val = self.paramater_relu(self.batch_layer2(val))
        val = self.pool1(val)        
        val = self.conv3(val)
        val = self.paramater_relu(self.batch_layer3(val))
        val = self.pool1(val)        
        val = self.conv4(val)
        val = self.paramater_relu(self.batch_layer4(val))
        val = self.pool1(val)        
        val = self.conv5(val)
        val = self.paramater_relu(self.batch_layer5(val))
        val = self.pool1(val)        
        val = self.conv6(val)
        val = self.paramater_relu(self.batch_layer6(val))
        val = self.flatten(val)
        val = self.fully_connected1(val)
        return val
        
    
neural_net = Flowers_CNN().to(device)
neural_net.load_state_dict(torch.load('BestModel.pth'))
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