import numpy as np
import os
import re
from PIL import Image, ImageFile
import torch
import torch.utils.data as data
from torch import optim, nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

training_loader = 0
validation_loader = 0
neural_network = 0


#loads the  data from the training set, splits the training set into a training_loader and validation_loader. 80/20 in the data
def load_data(training_dataset, batch_size):
    global training_loader
    global validation_loader
    training_data = datasets.ImageFolder(training_dataset, transform = transforms.ToTensor())
    training_set, validation_set = data.random_split(training_data, [40320, 10080])
    training_loader = DataLoader(training_set, batch_size = batch_size, shuffle = True)
    validation_loader = DataLoader(validation_set, batch_size = batch_size, shuffle = True)
    

# creating the convoluted neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3, padding = 1)
        self.conv2 = nn.Conv2d(256, 512, 3, padding = 1)
        self.fc1 = nn.Linear(7 * 7 * 512, 20000)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(20000, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    

#trains the neural network
def training(epochs, learning_rate, loss_function):
    global neural_network
    neural_network = CNN().to(device)
    optimizer = optim.SGD(neural_network.parameters(), lr = learning_rate) 
    neural_network.train()

    for i in range(epochs):
        current_loss = 0
        for index, (images , labels) in enumerate(training_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = neural_network(images) #runs the neural network on each image
            loss = loss_function(out, labels)
            loss.backward() #backpropagation
            optimizer.step()
            current_loss += loss.item()

        else:
            print(f"Training loss: {current_loss/len(training_loader)}")

#validation
def validation(loss_function):
    neural_network.eval()
    with torch.no_grad():
        correct = 0
        error = 0
        valid_loss = 0
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            predicted = neural_network(images)
            valid_loss += loss_function(predicted, labels).item()
            correct += (predicted.argmax(1) == labels).type(torch.float).sum().item()

        print(correct / len(validation_loader.dataset))


#testing, prints the results of the testing data into a txt file
def testing(testing_file):
    with torch.no_grad():
        with open("prediction.txt",'w') as f:
            pass
            for i in sorted(os.listdir(testing_file), key = len):
                picture = os.path.join(testing_file, i)
                temp = Image.open(picture).convert('RGB')
                transform = transforms.Compose([transforms.ToTensor()])
                temp = transform(temp)
                temp = temp.to(device)
                temp = temp.unsqueeze(0)
                output = neural_network(temp)
                _, predicted = torch.max(output.data, 1)
                predicted = predicted.cpu()
                predicted = predicted.numpy()
                f.write(str(predicted[0]))
                f.write('\n')



load_data('training_set', 32)
training(50, 0.001, nn.CrossEntropyLoss())
validation(nn.CrossEntropyLoss())
testing('testing_set')
