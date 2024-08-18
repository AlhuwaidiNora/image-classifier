import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a new network.')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture (default: vgg16)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    # Directories
    train_dir = 'data/train'
    valid_dir = 'data/valid'
    test_dir = 'data/test'

    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Data loaders
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    }

    # Load a pre-trained model
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError('Unsupported architecture.')

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define the classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(4096, 1024)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc3', nn.Linear(1024, len(image_datasets['train'].class_to_idx))),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    # Use GPU if available
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training the model
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{args.epochs} - Loss: {running_loss/len(dataloaders["train"])}')

    # Save the model
    torch.save({
        'state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx
    }, 'checkpoint.pth')

if __name__ == '__main__':
    main()
