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
epochs = 120
batch_size = 64
learning_rate = 0.00001 

transformations1 = trans.Compose([trans.ToTensor(), 
                                 trans.Resize((100,100))])
                                 


# Load the dataset - split into training and testing dataset:
training_dataset = torchvision.datasets.Flowers102(root='./data', split="train", 
                                                   download=True, transform=transformations1)

# Load the data set using the pytorch data loader:
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)





# Transform Dataset with trans.Normalise :
mean = 0.
stand_dev = 0.
for images, _ in training_loader:
    batch_samples = images.size(0) 
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    stand_dev += images.std(2).sum(0)

stand_dev /= len(training_loader.dataset)
mean /= len(training_loader.dataset)

transformations2 = trans.Compose([trans.ToTensor(), 
                                  trans.RandomHorizontalFlip(0.2),
                                  trans.RandomVerticalFlip(0.2),
                                  trans.RandomRotation(55),
                                  trans.Resize((100,100)),
                                  trans.Normalize(mean=mean, std=stand_dev)])


valid_and_test_transforms = trans.Compose([trans.ToTensor(), 
                                       trans.Resize((100,100)),
                                       trans.Normalize(mean=mean, std=stand_dev)])




# Load the dataset again;
training_dataset = torchvision.datasets.Flowers102(root='./data', split="train", 
                                                   download=True, transform=transformations2)
validation_dataset = torchvision.datasets.Flowers102(root='./data', split="val",
                                                   download=True, transform=valid_and_test_transforms)
testing_dataset = torchvision.datasets.Flowers102(root='./data', split="test",
                                                   download=True, transform=valid_and_test_transforms)


# Load the data set using the pytorch data loader again:
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)


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
        self.paramater_relu = nn.PReLU()
        self.pool1 = nn.MaxPool2d(3,3)
        self.fully_connected1 = nn.Linear(16384, 102)
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
        val = self.flatten(val)
        val = self.fully_connected1(val)
        return val
        

neural_net = Flowers_CNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(neural_net.parameters(), lr=learning_rate) 
best_accuracy = 0




def NetworkAndClassesAccuracy():
    """ Calculates and prints the accuracy of the network as a whole AND of each class prediction using 
    the testing split of the datset """
    with torch.no_grad():
        num_class_correct = [0 for i in range(102)]
        num_class_samples = [0 for i in range(102)]
        total_samples = 0
        correct_preds = 0
        for images, labels in testing_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = neural_net(images)
            # max returns (value ,index)
            _, predictions = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_preds += (predictions == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i]
                pred = predictions[i]
                if (label == pred):
                    num_class_correct[label] += 1
                num_class_samples[label] += 1

        acc = 100.0 * correct_preds / total_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(102):
            acc = 100.0 * num_class_correct[i] / num_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')




def NetworkAccuracyValidationOnly():
    """Calculates and prints ONLY the accuracy of the network as a whole - NOT of each class - using the 
    validation split of the dataset only

    Returns:
        Float: Accuracy of the network as a whole
    """
    with torch.no_grad():
        num_class_correct = [0 for i in range(102)]
        num_class_samples = [0 for i in range(102)]
        total_samples = 0
        correct_preds = 0
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = neural_net(images)
            # max returns (value ,index)
            _, predictions = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_preds += (predictions == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i]
                pred = predictions[i]
                if (label == pred):
                    num_class_correct[label] += 1
                num_class_samples[label] += 1

        acc = 100.0 * correct_preds / total_samples
        print(f'Accuracy of the network: {acc} %')
        return acc



best_accuracy = 0
# Training loop:
for ep in range(epochs): # Each iteration is a forward pass to train the data
    for i, (images, image_labels) in enumerate(training_loader):
        images = images.to(device)
        image_labels = image_labels.to(device)

        label_pred = neural_net(images) 
        loss = loss_function(label_pred, image_labels) 

        optimiser.zero_grad()
        loss.backward() 
        optimiser.step()

        print("Epoch Number = " + str(ep) + ", Index =", str(i), "/", str(len(training_loader)-1), "Loss = " + str(loss.item()))
    
    current_accuracy = NetworkAccuracyValidationOnly()
    if(current_accuracy > best_accuracy):
        best_accuracy = current_accuracy
        torch.save(neural_net.state_dict(), 'BestModel.pth')


print("Best Accuracy on Validation Split =", best_accuracy, "\n")
print("Network Accuracy on Testing Split:")
neural_net.eval() 
NetworkAndClassesAccuracy()