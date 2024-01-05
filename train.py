import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import argparse
import numpy as np
import json
from collections import OrderedDict 
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import datasets, transforms
import torchvision.models as models


parser = argparse.ArgumentParser(
    description = 'Parser | train.py'
)
parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16") # default is vgg16 
parser.add_argument('--learning_rate', action="store", type=float,default=0.01)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--gpu', action="store_true", default=False)


args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

data_dir = data_dir
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
test_dir = data_dir + "/test"

data_transforms = {'train':transforms.Compose([transforms.RandomRotation(30),
                           transforms.RandomResizedCrop(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                  'test': transforms.Compose([transforms.RandomResizedCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                  'valid': transforms.Compose([transforms.RandomResizedCrop(224),
                           transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

image_datasets = {'train':datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                 'test':datasets.ImageFolder(test_dir, transform=data_transforms['test']),
                 'valid':datasets.ImageFolder(valid_dir,transform=data_transforms['valid'])}

dataloader = {'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
              'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True),
              'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True)}

if arch == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg16(pretrained=True)
    

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
# Define the classifier layer
classifier = nn.Sequential(nn.Dropout(0.50),
			   nn.Linear(arch, hidden_units),
                           nn.ReLU(),
                           nn.Linear(hidden_units, image_datatests['train'].class_to_idx),
                      	   nn.ReLU(),
                           nn.Linear(image_datatests['train'].class_to_idx, 70),
                           nn.ReLU(),
                           nn.Linear(70, 102),
                           nn.LogSoftmax(dim=1)

model.classifier = classifier

# define the loss function and the optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

model = model.to(device)
model.to(device)

def train_model(model, criterion, optimizer, epochs = 10, device='cuda', ):
    steps = 0
    running_loss = 0
    print_every = 20
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in dataloader['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Training loss: {running_loss/len(dataloader['train']):.3f}.. "
                    f"Validation loss: {test_loss/len(dataloader['valid']):.3f}.. "
                    f"Validation accuracy: {accuracy/len(dataloader['valid']):.3f}")
                
                running_loss = 0
                model.train()
train_model()

data_transforms['test'] = transforms.Compose([transforms.RandomResizedCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image_datasets['test'] = datasets.ImageFolder(test_dir, transform= data_transforms['test'])
dataloader['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
                
checkpoint = {
    'state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': image_datatests['train'].class_to_idx,
    'classifier': model.classifier,
    'epochs': epochs
}

torch.save(checkpoint, 'checkpoint.pth')
