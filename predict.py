import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable
from torch import nn, optim


parser = argparse.ArgumentParser(
    description = 'Parser | predict.py'
)

parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
path_image = args.input
n = args.top_k
gpu = args.gpu
device_name = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
path = args.checkpoint #path to checkpoint

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    img = Image.open(image_path)
    width, height = img.size
    aspect_ratio = width / height
    if width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 256
        new_width = int(new_height * aspect_ratio)
    img = img.resize((new_width, new_height))
    crop_size = 224
    left = (new_width - crop_size) / 2
    right = left + crop_size
    upper = (new_height - crop_size) / 2
    lower = upper + crop_size
    img = img.crop((left, upper, right, lower))
    np_image = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    tensor_image = torch.from_numpy(np_image)

    return tensor_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.cpu()
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0).float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cpu())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)

model = load_checkpoint(path)
with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

probs, classes = predict(path_image, model, n, device_name)

for i in range(len(probs)):
    print(f"{cat_to_name[str(classes[i])]} with a probability of {probs[i]}")
