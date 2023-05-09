import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import numpy as np
import ssl
import torch.optim.lr_scheduler as lr_scheduler
ssl._create_default_https_context = ssl._create_unverified_context


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters:
epochs = 450
batch_size = 64
learning_rate = 0.0004

transformations1 = trans.Compose([trans.ToTensor(),
                                  trans.Resize((250,250))])



# Load the dataset - split into training and testing dataset:
training_dataset = torchvision.datasets.Flowers102(root='./data', split="train",
                                                   download=True, transform=transformations1)
testing_dataset = torchvision.datasets.Flowers102(root='./data', split="test",
                                                  download=True, transform=transformations1)
validation_dataset = torchvision.datasets.Flowers102(root='./data', split="val",
                                                     download=True, transform=transformations1)


# Load the data set using the pytorch data loader:
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)





transformations2 = trans.Compose([trans.ToTensor(),
                                  trans.RandomHorizontalFlip(0.2),
                                  trans.RandomVerticalFlip(0.2),
                                  trans.RandomRotation(55),
                                  trans.Resize((250,250))])





# Load the dataset again;
training_dataset = torchvision.datasets.Flowers102(root='./data', split="train",
                                                   download=True, transform=transformations2)


# Load the data set using the pytorch data loader again:
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)




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

# l1_reg = nn.L1Loss(size_average=False)
l2_reg = nn.MSELoss()
neural_net = Flowers_CNN().to(device)
neural_net.load_state_dict(torch.load('55%+/59.4%Model.pth', map_location=device))
loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(neural_net.parameters(), lr=learning_rate)
scheduler = lr_scheduler.LinearLR(optimiser, start_factor=0.9, end_factor=0.3, total_iters=10)
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

        # l1_loss = 0
        # for param in neural_net.parameters():
        #     l1_loss += torch.sum(torch.abs(param))
        # loss += 0.01 * l1_loss

        l2_loss = 0
        for param in neural_net.parameters():
            l2_loss += torch.sum(torch.pow(param, 2))
        loss += 0.01 * l2_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        print("Epoch Number = " + str(ep) + ", Index =", str(i), "/", str(len(training_loader)-1), "Loss = " + str(loss.item()))

    scheduler.step()
    current_accuracy = NetworkAccuracyValidationOnly()
    if(current_accuracy > best_accuracy):
        best_accuracy = current_accuracy
        torch.save(neural_net.state_dict(), '55%+/NextBest.pth')


print("Best Accuracy on Validation Split =", best_accuracy, "\n")
print("Network Accuracy on Testing Split:")
neural_net.eval()
NetworkAndClassesAccuracy()