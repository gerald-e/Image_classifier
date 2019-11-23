# -*- coding: utf-8 -*-
# image classifier - command line app
# train function 
# G.E. June 2019

# all imports ...
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler

import copy
import time
import random, os

import argparse

# collect the data

def collect_data(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
            'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
            ])
            }
    
    image_datasets = {
    x: datasets.ImageFolder(root=data_dir + '/' + x, transform=data_transforms[x])
    for x in list(data_transforms.keys())
    }
    
    dataloaders = {
    x: data.DataLoader(image_datasets[x], batch_size=12, shuffle=True, num_workers=4)
    for x in list(image_datasets.keys())
    }    

    return dataloaders, image_datasets

# This method build and trains a network 
def build_model(args):

    # Load a pre-trained model
    if args.arch=='vgg16':
        # Load a pre-trained vgg16 model
        model = models.vgg16(pretrained=True)
    elif args.arch=='vgg19':
        # Load a pre-trained vgg19 model
        model = models.vgg19(pretrained=True)
    elif args.arch=='alexnet':
        # Load a pre-trained alexnet model
        model = models.alexnet(pretrained=True)
    elif args.arch == 'densenet':
        # Load a pre-trained alexnet model
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError('Error. Unknow network architecture', args.arch)

    # Freeze its parameters
    for param in model.parameters():
        param.requires_grad = False

    # build the classifier using ReLU activations and dropout
    num_features = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, 512)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.5)),
                              ('hidden', nn.Linear(512, args.hidden_units)),                       
                              ('fc2', nn.Linear(args.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
    model.classifier = classifier
    # check the device and set model to cuda if available
    use_gpu = torch.cuda.is_available()
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Device check: Using GPU: "+ str(use_gpu))
        else:
            print("Device check: Using CPU configured")
    # Train the classifier paramters, feature params are froze 
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
       
    return model, criterion, optimizer        

# This method train and validate the model 
def train_model(args, model, dataloaders, image_datasets, criterion, optimizer):
    # declare ...
    start = time.time()
    epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # Finetuning the convent 
    # Decay LR by a factor of 0.1 every 3 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("Start the training ")
    # train and validate the data in loop 
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Repeat step by step over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # prepare the statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # calculate the statistics per phase    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy for the best model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # load best model weights
    model.load_state_dict(best_model_wts)
   
    # show the statistic    
    print("The training is completed ....") 
    print("Result epochs: {}".format(epochs))
    print('Best accuracy: {:4f}'.format(best_acc))
    runtime = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(runtime // 60, runtime % 60))
    
    return model

# function for training the neural net
def test_model(args, model, dataloaders):

    start = time.time()
    accuracy = 0
    phase = 'test'
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)        
        
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
                100 * accuracy / total))
    
    print(f"Device = {device}; runtime: {(time.time() - start) / 3:.3f} seconds")

    return model

def save_checkpoint(args, model, image_datasets, optimizer):    
    # TODO: Save the checkpoint 
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.epochs = args.epochs
    checkpoint = {
            'architecture': args.arch,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict(),
            'classifier': model.classifier,
            'optimizer_dict':optimizer.state_dict(),
            'hidden': args.hidden_units,
            'epoch': model.epochs
            }

    torch.save(checkpoint, args.checkpoint)
    return checkpoint

def main():
# Define command line arguments
    parser = argparse.ArgumentParser(description='Train.py')
    parser.add_argument('--data_dir', type=str, help='dataset directory', default="./flowers/")
    parser.add_argument('--gpu', action='store', help='Use GPU if available', default="gpu")
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
    parser.add_argument('--arch', type=str, help='Model architecture', default="vgg16")
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units', default=4096)
    parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file', default="./checkpoint.pth")
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    # prepare the data
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    dataloaders, image_datasets = collect_data(args)
    model, criterion, optimizer = build_model(args)
    model = train_model(args, model, dataloaders, image_datasets, criterion, optimizer)
    model = test_model(args, model, dataloaders)
    checkpoint = save_checkpoint(args, model, image_datasets, optimizer)

# call all steps 
# ... start the training 
print("Trian.py: Start the training")

if __name__ == "__main__":
    main()

# First steps are completed 
print("Train.py: The Model is trained")
