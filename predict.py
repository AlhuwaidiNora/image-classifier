import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json

def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

def predict(image_path, checkpoint_path, topk=5, gpu=False):
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    model = models.vgg16()
    model.classifier = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.to(device)
    model.eval()

    image = process_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    probabilities = torch.exp(output)
    top_p, top_class = probabilities.topk(topk, dim=1)
    return top_p.squeeze().cpu().numpy(), top_class.squeeze().cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Predict the class of an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--topk', type=int, default=5, help='Number of top classes to return')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    top_p, top_class = predict(args.image_path, args.checkpoint, args.topk, args.gpu)
    print('Top classes:', top_class)
    print('Top probabilities:', top_p)

if __name__ == '__main__':
    main()
