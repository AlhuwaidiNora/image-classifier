import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load data
    print("Loading data...")
    train_dataset = datasets.ImageFolder(root=f'{args.data_dir}/train', transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=f'{args.data_dir}/test', transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load pretrained model
    print("Loading pretrained model...")
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Modify classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(4096, 1024)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc3', nn.Linear(1024, len(train_dataset.classes))),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    # Training loop
    print("Training model...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader)}")

    # Save the model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': args.epochs,
        'class_to_idx': train_dataset.class_to_idx
    }
    torch.save(checkpoint, 'model_checkpoint.pth')
    print("Model saved as model_checkpoint.pth")

if __name__ == "__main__":
    main()
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser(description='Train an image classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with training and test data')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save the checkpoint')
    return parser.parse_args()

def main():
    args = get_args()
    use_gpu = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if use_gpu else "cpu")

    # Define transforms for training and testing
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    data_dir = args.data_dir
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train']),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms['test']),
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False),
    }

    # Get the number of classes
    num_classes = len(image_datasets['train'].classes)

    # Load a pre-trained model
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(4096, 1024)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc3', nn.Linear(1024, num_classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(image_datasets['train'])
        print(f'Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f}')

    # Save the checkpoint
    checkpoint = {
        'input_size': 25088,
        'output_size': num_classes,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx
    }
    save_path = os.path.join(args.save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")

if __name__ == '__main__':
    main()
