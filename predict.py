# -*- coding: utf-8 -*-
# image classifier - command line app
# predict function 
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


def load_checkpoint(args):
    # load the checkpoints 
    checkpoint = torch.load(args.checkpoint)
    model = getattr(torchvision.models, checkpoint['architecture'])(pretrained=True)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    
    print(model)

    return model


def predict(args, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # prepaire ... 
    model.eval()
    use_gpu = torch.cuda.is_available()
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Device check: Using GPU: "+ str(use_gpu))
        else:
            print("Device check: Using CPU configured")
    # Predict the class from an image file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_image = process_image(args.image_path)
    image_tensor = torch.from_numpy(np.expand_dims(processed_image, axis=0)).float()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    probs, labels = output.topk(topk)
    probs = np.array(probs.exp().data)[0]
    classes = np.array(labels)[0]
        
    return probs, classes

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image_path)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    np_image = adjustments(img_pil)

    return np_image
    
def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Predict.py')
    parser.add_argument('--image_path', type=str, help='Image to predict')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint to use when predicting', default='./checkpoint.pth')
    parser.add_argument('--topk', type=int, help='Return top K predictions')
    parser.add_argument('--mapper_json', type=str, help='JSON file containing label names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', help='Use GPU if available', default="gpu")
    args, _ = parser.parse_known_args()
    
    # prepare the data
    import json
    with open(args.mapper_json, 'r') as f:
        cat_to_name = json.load(f)
    
    # call all steps 
    model = load_checkpoint(args)
    # finaly show the results ....
    img = args.image_path
    print('Image:', img)
    probs, classes = predict(args, model, args.topk)

    # Print only when invoked by command line 
    if args.image_path:
        print('Predictions and probabilities:', list(zip(classes, probs)))

# ... start the training 
print("predict.py: Start the prediction")

if __name__ == "__main__":
    main()

# First steps are completed 
print("predict.py: Complete the prediction")