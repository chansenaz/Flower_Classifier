import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
import argparse
from helpers import (load_datasets, train_on_gpu, load_flower_categories)


def main():
    print("Starting model trainer")

    start_time = time()

    in_arg = get_input_args()

    check_command_line_arguments(in_arg)

    loader_dict, data_dict = load_datasets()

    # Load flower categories
    cat_to_name = load_flower_categories('cat_to_name.json')
    print(len(cat_to_name), 'flower categories/names loaded.\n')

    input_nodes = {'densenet121': 1024, 'vgg16': 25088, 'vgg13': 25088, 'alexnet': 9216}
    model = make_model(in_arg.arch, cat_to_name, input_nodes)

    cuda = train_on_gpu(in_arg.cpu)
    if cuda:
        model = model.cuda()

    # Train a model with a pre-trained network
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.lr)

    train_model(model, loader_dict, in_arg.epochs, criterion, optimizer, cuda)

    check_accuracy_on_test(model, loader_dict['test'], cuda, criterion)

    save_checkpoint(model, data_dict['train'], in_arg, input_nodes)

    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime (h:m:s):",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )


def get_input_args():
    """
    The training script allows users to choose from at least two different architectures available from torchvision.models
    The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
    The training script allows users to choose training the model on a GPU
    """
    parser = argparse.ArgumentParser(description='Flower picture trainer')

    parser.add_argument('-arch', type=str, default='densenet121', choices=['densenet121', 'vgg16', 'vgg13', 'alexnet'] ,
                        help='model architecture. choices: densenet121, vgg16, vgg13, alexnet')

    parser.add_argument('-lr', type=float, default=.001,
                        help='learning rate (default .001)')       

    parser.add_argument('-hidden_units', type=int, default=512,
                        help='number of hidden nodes')

    parser.add_argument('-epochs', type=int, default=7, 
                        help='number of epochs')  

    parser.add_argument('-cpu', action="store_true", default=False, 
                        help='train on cpu instead of gpu')

    # returns parsed argument collection
    return parser.parse_args()


def check_command_line_arguments(in_arg):
    # prints command line agrs
    print("Command Line Arguments:",
          "\n    arch =", in_arg.arch, 
          "\n    learning rate =", in_arg.lr, 
          "\n    hidden units =", in_arg.hidden_units,
          "\n    epochs =", in_arg.epochs,
          "\n    train on cpu =", in_arg.cpu, "\n")


def make_model(arch, cat_to_name, input_nodes):
    model = getattr(models, arch)(pretrained=True)

    num_hidden_nodes = 512
    num_total_classes = len(cat_to_name)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_nodes[arch], num_hidden_nodes)),
                            ('relu', nn.ReLU()),
                            ('dropout',nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(num_hidden_nodes, num_total_classes)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
    model.classifier = classifier

    print("\nclassifier:")
    print(model.classifier)

    return model


def train_model(model, loader_dict, epochs, criterion, optimizer, cuda):
    steps = 0

    for e in range(epochs):
        running_loss = 0
        model.train()
        print_every = 20
        
        for ii, (inputs, labels) in enumerate(loader_dict['train']):
            steps += 1
            
            optimizer.zero_grad()
            
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                
                validation_loss = 0
                accuracy = 0
                
                for jj, (inputs, labels) in enumerate(loader_dict['validation']):
                    if cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()
                        
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                    
                    validation_loss += loss.item()
                    
                    ps = torch.exp(outputs).data
                    
                    equality = (labels.data == ps.max(1)[1])
                    
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}, ".format(running_loss/print_every),
                      "Validation Loss: {:.4f}, ".format(validation_loss/print_every),
                      "Validation Accuracy: {:.1f} ".format(100*accuracy/len(loader_dict['validation'])))

                running_loss = 0
                
                # go back into training mode
                model.train()

                
def check_accuracy_on_test(model, test_loader, cuda, criterion):
    test_loss = 0
    test_accuracy = 0

    model.eval()
    for kk, (inputs, labels) in enumerate(test_loader):

        if cuda:
            # Move input and label tensors to the GPU
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        ps = torch.exp(outputs).data
        equality = (labels.data == ps.max(1)[1])
        
        test_accuracy += equality.type_as(torch.FloatTensor()).mean()
    
    test_accuracy = (100*test_accuracy)/len(test_loader)
    print("Test Accuracy %: {:.3f}".format(test_accuracy))   


def save_checkpoint(model, train_data, in_arg, input_nodes):
    '''
    Good job here!
    Suggestion:
    You can consider including the model name into the checkpoint, so that you can rebuild the model from scratch.
    Also, if you want to retrain the model from the current state in the future, you can include more params:

    input_size
    output_size
    hidden_layers
    batch_size
    learning_rate
    '''
    if os.path.isfile("checkpoint.pth"):
        os.remove("checkpoint.pth")

    checkpoint = {'arch': in_arg.arch, 
                'class_to_idx': train_data.class_to_idx,
                'state_dict': model.state_dict(),
                'classifier': model.classifier,
                }

    torch.save(checkpoint, 'checkpoint2.pth')
    print("Checkpoint saved.")


if __name__ == "__main__":
    main()
