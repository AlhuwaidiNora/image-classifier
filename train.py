import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

def main():
    # Define data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    # Set data directories
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'test')  # Assuming 'test' is used for validation

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    # Define model
    model = models.vgg16(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, len(train_dataset.classes)),
        nn.LogSoftmax(dim=1)
    )

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # Save the model checkpoint
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save({
        'state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx
    }, os.path.join(checkpoint_dir, 'model_checkpoint.pth'))

if __name__ == '__main__':
    main()
