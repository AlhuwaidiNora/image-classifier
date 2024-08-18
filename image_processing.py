from PIL import Image
import torch
from torchvision import transforms

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Tensor.
    '''
    # Open the image
    img = Image.open(image_path)
    
    # Define the transformations
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply the transformations
    img = transformation(img)
    
    return img
