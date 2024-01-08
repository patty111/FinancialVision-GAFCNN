import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input is a 28 * 28 image, 
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)

        # relu is required for the hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the network
net = Net()

# Create a dummy tensor of size (batch_size, height, width)
# Let's say we have a batch size of 4, and each image is 28x28
x = torch.randn(4, 28, 28)

# Print the size of the tensor before passing it through the network
print("Before:", x.size())

# Pass the tensor through the network
y = net(x)

# Print the size of the tensor after passing it through the network
print("After:", y.size())

# loss function
# weight update 