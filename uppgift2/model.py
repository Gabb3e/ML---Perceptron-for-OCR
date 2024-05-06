import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define data augmentation transformations
transform = transforms.Compose([
    transforms.RandomRotation(15),  # Random rotation by up to 15 degrees
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Randomly resize and crop
    transforms.RandomHorizontalFlip(),  # Random horizontal flipping
    transforms.ToTensor(),    # Convert data to [0.0-1.0]
    transforms.Normalize((0.5,), (0.5,)) # Normalize grayscale
])

# dataset MNIST 
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

# model
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)


    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# function to save model checkpoint
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)

def check_accuracy(loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return float(num_correct) / float(num_samples)

# function to visualize training loss
def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# function to display example predictions
def display_predictions(model, test_loader, num_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            print("Predicted:", predicted)
            print("True:", labels)
            if i == num_images:
                break

def plot_loss_accuracy(losses, accuracies):
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

# Function to train the model and measure performance
def train(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_time = time.time()
    best_loss = float('inf')
    losses = []
    accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for images, labels in train_loader:
            # Get data to cuda if possible
            images = images.to(device=device)
            labels = labels.to(device=device)

            # forward
            scores = model(images)
            loss = criterion(scores, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            # Calculate total loss
            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

        average_loss = total_loss / total_count
        losses.append(average_loss)

        # evaluate accuracy
        accuracy = check_accuracy(test_loader, model)
        accuracies.append(accuracy)

        # Save checkpoint if model improves
        if average_loss < best_loss:
            best_loss = average_loss
            checkpoint = {
                "state_dict": model.state_dict(), 
                "optimizer": optimizer.state_dict()
                }
            save_checkpoint(checkpoint)

        # Check performance
        if epoch % 1 == 0:
            accuracy = check_accuracy(test_loader, model)
            print(f"Epoch {epoch}: Loss = {average_loss:.4f} - Accuracy = {accuracy * 100:.2f}%")

            # Save model checkpoint
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint)

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    print(f"Training finished. Time taken: {elapsed_time:.2f} seconds")
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    plot_loss(losses)
    plot_loss_accuracy(losses, accuracies)
    display_predictions(model, test_loader)

# hyperparameters
learning_rate = 0.001
num_classes = 10

# model, loss and optimizer 
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# train model
train(model, train_loader, test_loader, criterion, optimizer, epochs=5)
