# Install packages
import numpy as np

# Import system
import sys

# Install Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optimize
from torchvision import datasets, transforms

# Import time for measurement
from time import time

# Import Dataset
from download_mnist import load


# Load Dataset
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)
# Convert to float type
x_train = x_train.astype(float)
x_test = x_test.astype(float)

# global variable for model we set up for our neural network
global model

# Neural Network Model Structure 784‐200‐50‐10
model = nn.Sequential(nn.Linear(784, 200), nn.ReLU(), nn.Linear(200, 50), nn.ReLU(), nn.Linear(50, 10), nn.Softmax())


# Main function
def main():

    # Train Model
    def train(model, trainData, optimiser, loss, epoch):
        # Train Model
        model.train()
        # Counter
        counter = 0

        # Train Dataset
        for x, (data, target) in enumerate(trainData):
            # Counter
            counter += 1
            # Reset gradient
            optimiser.zero_grad()
            # Data input
            data = data.view(-1, 784)
            # Output
            output = model(data)
            # Introduce Cross Entropy Loss
            lossFunct = nn.CrossEntropyLoss()
            # The total loss
            loss = lossFunct(output, target)
            loss.backward()
            # Optimise the step
            optimiser.step()

        # Counter
        counter = 0
        # Test Model
        model.eval()
        # Loss
        testLoss = 0
        # Number Correct
        correct = 0

        print("TRAINING DATA ")

        # Run Pytorch with no gradient
        with torch.no_grad():
            # Test Dataset
            for data, target in trainData:
                # Increase counter
                counter += 1
                # Data input
                data = data.view(-1, 784)
                # Output
                output = model(data)
                # Test Loss
                testLoss += funct.nll_loss(output, target, reduction='sum').item()
                # Predict output
                pred = output.argmax(dim=1, keepdim=True)
                # Check for correctness
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Testing
        testLoss = testLoss / len(trainData.dataset)

        # Print loss
        print("Training Loss: ", testLoss)

        # Print Success rate
        print("Training Success rate: ", (100 * correct / len(trainData.dataset)))



    # Test Model
    def test(model, testData):
        # Counter
        counter = 0
        # Test Model
        model.eval()
        # Loss
        testLoss = 0
        # Number Correct
        correct = 0

        # Run Pytorch with no gradient
        with torch.no_grad():
            # Test Dataset
            for data, target in testData:
                # Increase counter
                counter += 1
                # Data input
                data = data.view(-1, 784)
                # Output
                output = model(data)
                # Test Loss
                testLoss += funct.nll_loss(output, target, reduction='sum').item()
                # Predict output
                pred = output.argmax(dim=1, keepdim=True)
                # Check for correctness
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Testing
        testLoss = testLoss / len(testData.dataset)

        # Print loss
        print("Testing Loss: ", testLoss)

        # Print Success rate
        print("Testing Success rate: ", (100 * correct / len(testData.dataset)))

    # Batch Size
    batchSize = 128
    # Learning Rate
    learningRate = 0.01
    # Number of Epochs
    epochs = 10

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST/processed', train=True,
                                                              download=True,
                                                              transform=transforms.Compose([transforms.ToTensor(),
                                                                                            transforms.Normalize(
                                                                                                (0.1307,),
                                                                                                (0.3081,))])),
                                               batch_size=batchSize,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST/processed', train=False, download=True,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                                           transforms.Normalize(
                                                                                               (0.1307,), (0.3081,))])),
                                              batch_size=10000, shuffle=True)

    # Neural Network Optimizer
    optimiser = optimize.SGD(model.parameters(), lr=learningRate, momentum=0.9, weight_decay=1e-3)

    # Time delta
    tDelta = 0

    # Test each epoch
    for x in range(1, epochs + 1):
        # Print the current Epoch
        print("Epoch: %d/10" % x)
        # Time Initial
        tInitial = time()
        # Train model
        train(model=model, trainData=train_loader, optimiser=optimiser, loss='CE', epoch=x)
        # Time Final
        tFinal = time()
        # Test model
        print("TESTING DATA")
        test(model=model, testData=test_loader)
        # Time Delta
        tDelta += (tFinal - tInitial)

    # Total Training/Testing time
    print("Total Time: ", tDelta)

# Run program
if __name__ == "__main__":
    main()