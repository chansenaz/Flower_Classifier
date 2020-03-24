import os
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from helpers import (load_datasets, train_on_gpu, load_flower_categories)
from torchvision import datasets, transforms, models


def main():
    in_arg = get_input_args()

    check_command_line_arguments(in_arg)

    image_path = in_arg.img

    # Load flower categories
    cat_to_name = load_flower_categories(in_arg.json_file)
    print(len(cat_to_name), 'flower categories/names loaded.\n')

    loader_dict, data_dict = load_datasets()

    model = load_checkpoint(in_arg.checkpoint)

    cuda = train_on_gpu(in_arg.cpu)
    if cuda:
        model = model.cuda()

    # See what the model predicts for that image
    probabilities, classes = predict(image_path, model, data_dict['train'], in_arg.topk, cuda)

    print("\n\nThe model predicts that your image is a " + cat_to_name[classes[0]])
    print("\nTop k with percentages:\n")
    
    for i in range(len(classes)):
        print(str(int(probabilities[i]*100)) + "%: " + cat_to_name[classes[i]])


def get_input_args():
    parser = argparse.ArgumentParser(description='Flower picture predictor')

    parser.add_argument('-img', type=str, default='flowers/test/8/image_03320.jpg', help='path to image')

    parser.add_argument('-checkpoint', type=str, default='checkpoint2.pth', help='path to model checkpoint')

    parser.add_argument('-topk', type=int, default=5, help='top k for prediction')

    parser.add_argument('-cpu', action="store_true", default=False, help='train on cpu instead of gpu')

    parser.add_argument('-json_file', type=str, default='cat_to_name.json', help='filepath for json categories file')

    # returns parsed argument collection
    return parser.parse_args()


def check_command_line_arguments(in_arg):
    # prints command line agrs
    print("Command Line Arguments:",
          "\n    image =", in_arg.img, 
          "\n    checkpoint =", in_arg.checkpoint, 
          "\n    topk =", in_arg.topk,
          "\n    train on cpu =", in_arg.cpu, "\n")


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

        For the resize functionality,
        Here instead of resizing the input image into the size of 256 by 256, you can 
        make the image have a smaller side sized at 256 and maintain the aspect ratio. 
        For example, if you have an image of size 512x1024, you need to resize the image 
        to 256x512 and not 256x256. Here's a code snippet for understanding this point.

        size = 256, 256
        if width > height:
            ratio = float(width) / float(height)
            newheight = ratio * size[0]
            image = image.resize((size[0], int(floor(newheight))), Image.ANTIALIAS)
        else:
            ratio = float(height) / float(width)
            newwidth = ratio * size[0]
            image = image.resize((int(floor(newwidth)), size[0]), Image.ANTIALIAS)
    '''
    # Open image file
    im = Image.open(image)
    # Resize the images where the shortest side is 256 pixels
    im = im.resize((256,256))
    # Crop out the center 224x224 portion of the image 
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    # Normalize:  0-255, but the model expected floats 0-1
    # Convert image to an array and divide each element
    im = np.array(im)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    
    return im.transpose(2,0,1)


def predict(image_path, model, train_data, topk, cuda):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''        
    model.eval()
    
    image = process_image(image_path)
    
    image = torch.from_numpy(np.array([image])).float()
    
    if cuda:
        image = image.cuda()
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    
    class_indexes = []
    for i in range(len(train_data.class_to_idx.items())):
        class_indexes.append(list(train_data.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(topk):
        label.append(class_indexes[index[i]])

    return prob, label

    
if __name__ == "__main__":
    main()
