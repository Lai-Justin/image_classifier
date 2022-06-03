# Starter code for CS 165B HW4
import numpy as np
import os
from PIL import Image, ImageFile
import torch
import torch.utils.data as data
from torch import optim, nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


training_data = datasets.ImageFolder('hw4_train', transform = transforms.ToTensor())

batch_size = 8

training_loader = DataLoader(training_data, batch_size, shuffle = True, num_workers= 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1000, 10)

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


neural_network = CNN()
neural_network = neural_network.to(device)
loss_function = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.SGD(neural_network.parameters(), lr = learning_rate)

epochs = 50





neural_network.train()


for i in range(epochs):
    current_loss = 0
    
    for index, (images , labels) in enumerate(training_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        

        out = neural_network(images)
        loss = loss_function(out, labels)
        loss.backward()
        optimizer.step()

        current_loss += loss.item()

    else:
        print(f"Training loss: {current_loss/len(training_loader)}")

neural_network.eval()



with torch.no_grad():
    with open("prediction.txt",'w') as f:
        pass
        for i in os.listdir('hw4_test'):
            picture = os.path.join('hw4_test', i)
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











"""
Implement the testing procedure here. 

Inputs:
    Unzip the hw4_test.zip and place the folder named "hw4_test" in the same directory of your "prediction.py" file, your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        otherwise you will be given 0 score for your hw4.


**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files, 
    please use open('./necessary.file') instead of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he/she use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error 
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 score for your hw4.
"""


