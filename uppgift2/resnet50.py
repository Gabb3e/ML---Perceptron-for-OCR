import time
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

# Define data augmentation transformations
transform_train = transforms.Compose([
    transforms.Resize(256),  # Resize smaller edge to 256
    transforms.RandomResizedCrop(224),  # Then crop randomly to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# dataset raw-img
train_dataset = datasets.ImageFolder(root='/home/g2de/Documents/programmering/ML/ML_Perceptron_for_OCR/uppgift2/raw-img', transform=transform_train)
test_dataset = datasets.ImageFolder(root='/home/g2de/Documents/programmering/ML/ML_Perceptron_for_OCR/uppgift2/raw-img', transform=transform_test)

classes = ('cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo')

# calculate split size
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size

# split dataset
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

# dataloader 
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, pin_memory=True)

# model 
class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)

model = ResNet50(num_classes=10)

# function to save model checkpoint
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def check_accuracy(model, test_loader, debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if debug:
                print(f'Predicted: {predicted}')
                print(f'Actual: {labels}')
                print(f"Current batch correct: {(predicted == labels).sum().item()}, Total batch: {labels.size(0)}")

    accuracy = correct / total
    if debug:
        print(f"Calculated Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def plot_metrics(train_losses, train_accs, test_losses, test_accs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()
    plt.show()

def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return tensor * std + mean

# function to display example predictions
def display_predictions(model, test_loader, num_images=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for j in range(num_images):
                img = denormalize(images[j]).permute(1, 2, 0).cpu().numpy()
                plt.imshow(img)
                plt.title(f'Predicted: {classes[predicted[j]]}, Actual: {classes[labels[j]]}')
                plt.show()
            break  # Only display one batch of images

def plot_loss_accuracy(losses, accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# function to train model
def train_model(model, train_loader, test_loader, num_epochs=5, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_time = time.time()
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = check_accuracy(model, test_loader, debug=True)
        train_accs.append(accuracy)
        test_accs.append(accuracy)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} Test Loss = {avg_test_loss:.4f} - Accuracy = {accuracy * 100:.2f}%")
        # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accs[-1]:.4f}')
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    accuracy = check_accuracy(model, test_loader, debug=True)

    print(f"Training finished. Time taken: {elapsed_time:.2f} seconds")
    print(f"Final accuracy on test set: {test_accs[-1] * 100:.2f}%")

    # print(f"Training finished. Time taken: {elapsed_time:.2f} seconds")
    plot_metrics(train_losses, train_accs, test_losses, test_accs)
    plot_loss_accuracy(train_losses, test_losses)
    display_predictions(model, test_loader)
    
    return model, train_losses, train_accs, test_losses, test_accs

train_model(model, train_loader, test_loader, num_epochs=5, learning_rate=0.001)
