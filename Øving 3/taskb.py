import torch
import torch.nn as nn
import torchvision
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float().to(device)  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10)).to(device)  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(device)  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10)).to(device)  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # 1@28x28 -> 32@28x28
            nn.MaxPool2d(kernel_size=2),  # 32@28x28 -> 32@14x14
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # 32@14x14 -> 64@14x14
            nn.MaxPool2d(kernel_size=2),  # 64@14x14 -> 64@7x7
            nn.Flatten(),  # Flatten to a single vector
            nn.Linear(64 * 7 * 7, 1024),  # Dense layer, 64@7x7 -> 1x1024
            nn.Linear(1024, 10)  # Dense layer, 1x1024 -> 1x10
        )

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())



model = ConvolutionalNeuralNetworkModel().to(device)

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch].to(device), y_train_batches[batch].to(device)).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print("accuracy = %s" % model.accuracy(x_test.to(device), y_test.to(device)))