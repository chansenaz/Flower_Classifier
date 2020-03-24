from torchvision import datasets, transforms, models
import torch
import json

def load_datasets():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    transform_dict = {}
    data_dict = {}
    loader_dict = {}

    # Define your transforms for the training, validation, and testing sets
    transform_dict['train'] = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                        ])

    transform_dict['validation'] = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    transform_dict['test'] = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    data_dict['train'] = datasets.ImageFolder(train_dir, transform=transform_dict['train'])
    data_dict['validation'] = datasets.ImageFolder(validation_dir, transform=transform_dict['validation'])
    data_dict['test'] = datasets.ImageFolder(test_dir, transform=transform_dict['test'])

    # Using the image datasets and the trainforms, define the dataloaders
    loader_dict['train'] = torch.utils.data.DataLoader(data_dict['train'], batch_size=64, shuffle=True)
    loader_dict['validation'] = torch.utils.data.DataLoader(data_dict['validation'], batch_size=32)
    loader_dict['test'] = torch.utils.data.DataLoader(data_dict['test'], batch_size=32)

    print(len(data_dict['train'].imgs), "training images loaded.")
    print(len(data_dict['validation'].imgs), "validation images loaded.")
    print(len(data_dict['test'].imgs), "testing images loaded.")

    return loader_dict, data_dict


def train_on_gpu(cpu):
    # Checking for GPU support (CUDA)
    cuda = torch.cuda.is_available()
    if cuda and not cpu:
        print("\nUsing GPU:\n", torch.cuda.get_device_name(torch.cuda.device_count()-1))
        return True
    else:
        print("\nUsing CPU\n")
        return False


def load_flower_categories(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name