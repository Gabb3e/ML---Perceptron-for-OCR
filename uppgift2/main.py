import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define data augmentation transformations
transform_train = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# dataset raw-img
train_dataset = torchvision.datasets.ImageFolder(root='/home/g2de/Documents/programmering/ML/uppgift2/raw-img', transform=transform_train)
test_dataset = torchvision.datasets.ImageFolder(root='/home/g2de/Documents/programmering/ML/uppgift2/raw-img', transform=transform_test)

classes = ('cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo')

# calculate split size
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size

# split dataset
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

# dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=5, shuffle=False, pin_memory=True)

# model
class ConvNet(nn.Module):
    def __init__(self, hparams):
        super(ConvNet, self).__init__()
        layers = []
        in_channels = 3

        for out_channels, kernel_size in hparams['conv_layers']:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        self.conv = nn.Sequential(*layers)
        self.drop = nn.Dropout(hparams['dropout'])

        # Calculate the flattened size dynamically if necessary
        example_input = torch.rand((1, 3, 28, 28))  # Adjust this if input size changes
        example_output = self.conv(example_input)
        self.flattened_size = int(np.prod(example_output.size()[1:]))

        fc_layers = [self.flattened_size] + hparams['fc_layers'] + [10]  # Added output layer
        self.fc_layers = nn.ModuleList()
        for in_features, out_features in zip(fc_layers[:-1], fc_layers[1:]):
            self.fc_layers.append(nn.Linear(in_features, out_features))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.drop(out)
        for fc in self.fc_layers[:-1]:
            out = F.relu(fc(out))
        out = self.fc_layers[-1](out)  # No activation before the loss
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
    plt.plot(losses, label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# function to display example predictions
def display_predictions(loader, model, num_imgages=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)

            for i in range(num_imgages):
                image = x[i].permute(1, 2, 0).cpu().numpy()
                plt.imshow(image.squeeze(), cmap='gray')
                plt.title(f"Actual: {y[i]}, Predicted: {predictions[i]}")
                plt.show()

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

############################################################################################################

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

        # Evaluate accuracy
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
    display_predictions(test_loader, model)

def train_model_with_hparams(train_loader, test_loader, hparams):
    model = ConvNet(hparams)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'], weight_decay=1e-5)

    train(model, train_loader, test_loader, criterion, optimizer, epochs=5)

hyperparameters = [
    {'conv_layers': [(64, 3), (128, 3)], 'fc_layers': [256, 128, 64], 'learning_rate': 0.001, 'dropout': 0.5, 'activation': 'relu'},
    #{'conv_layers': [(32, 3), (64, 3)], 'fc_layers': [128], 'learning_rate': 0.001, 'dropout': 0.3, 'activation': 'relu'},
    #{'conv_layers': [(32, 3), (64, 3)], 'fc_layers': [128], 'learning_rate': 0.0005, 'dropout': 0.4, 'activation': 'relu'},
    #{'conv_layers': [(32, 3), (64, 3)], 'fc_layers': [128], 'learning_rate': 0.001, 'dropout': 0.2, 'activation': 'relu'},
    #{'conv_layers': [(32, 3), (64, 3)], 'fc_layers': [128], 'learning_rate': 0.0001, 'dropout': 0.5, 'activation': 'relu'},
    #{'conv_layers': [(32, 3), (64, 3)], 'fc_layers': [128], 'learning_rate': 0.002, 'dropout': 0.5, 'activation': 'relu'},
    #{'conv_layers': [(32, 3), (64, 3)], 'fc_layers': [128], 'learning_rate': 0.0005, 'dropout': 0.5, 'activation': 'relu'},
    #{'conv_layers': [(32, 3), (64, 3)], 'fc_layers': [128], 'learning_rate': 0.0005, 'dropout': 0.4, 'activation': 'relu'},
    #{'conv_layers': [(32, 3), (64, 3)], 'fc_layers': [128], 'learning_rate': 0.001, 'dropout': 0.3, 'activation': 'relu'},
    #{'conv_layers': [(32, 3), (64, 3)], 'fc_layers': [128], 'learning_rate': 0.0001, 'dropout': 0.2, 'activation': 'relu'},
]

# train model
for hparams in hyperparameters:
    train_model_with_hparams(train_loader, test_loader, hparams)