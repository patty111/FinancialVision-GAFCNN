import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
# C:\Users\Patrick\.pytorch\MNIST_data\MNIST
trainset = datasets.MNIST('./MNIST_data/', download=False, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('./MNIST_data/', download=False, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

# Create the network, define the criterion and optimizer
model = Net()

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda:0")
# Move the model to the device
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the network
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for images, labels in trainloader:
        #NOTE Input must also move to GPU !!!!
        images = images.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Training loss: {running_loss/len(trainloader)}")

print('Finished Training')

# Test the network
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        #NOTE Input must also move to GPU !!!!
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))




from PIL import Image
import torchvision.transforms.functional as TF

# Load the image
image = Image.open('5_2.png')

# Convert the image to grayscale, L means "luminance" 明暗)
image = image.convert('L')

# Resize the image to 28x28 pixels, due to NMINST dataset is all 28x28 pixels
image = image.resize((28, 28))

# Convert the image to a PyTorch tensor and normalize it
image = TF.to_tensor(image)
image = TF.normalize(image, [0.5], [0.5])

# Add an extra dimension to the tensor, stands for 1 batch, because the model only accepts batches
image = image.unsqueeze(0)

# Make sure the image tensor is on the same device as the model
image = image.to(device)

# Use the model to predict the digit
output = model(image)
_, predicted = torch.max(output.data, 1)

print('Predicted digit:', predicted.item())