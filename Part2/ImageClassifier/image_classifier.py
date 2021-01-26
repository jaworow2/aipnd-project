import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

def transform_train_data(data_dir, batch_size):
    '''
    Transforms flower images into pytorch tensors to be used for training 
    and validation
    
    arguments:
        data_dir: directory of the flower images
        batch_size: quantity of images to be loaded at a time
        
    return:
        train_image_datasets: imagefolder of training images
        train_dataloaders: image tensors of training data
        validation_dataloaders: image tensor of validation data
        
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity AI Programming with Python Nano Degree Training material.
    '''
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # adding noise into the training dataset to help model to better generalize
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(224), 
                                                transforms.CenterCrop(224), 
                                                transforms.ToTensor(), 
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_image_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_size, shuffle=True)
    validation_dataloaders = torch.utils.data.DataLoader(validation_image_datasets, batch_size=batch_size)
    
    return train_image_datasets, train_dataloaders, validation_dataloaders

def transform_test_data(data_dir, batch_size):
    '''
    Transforms flower images into pytorch tensors to be used for testing
    
    arguments:
        data_dir: directory of the flower images
        batch_size: quantity of images to be loaded at a time
    
    return:
        test_dataloaders: image tensors of testing data
        
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity AI Programming with Python Nano Degree Training material.
    '''
    
    test_dir = data_dir + '/test'

    test_transforms = transforms.Compose([transforms.Resize(224), 
                                          transforms.CenterCrop(224), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size)
    
    return test_dataloaders

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=.2):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # first layer input
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # creates a tuple of passed in hidden_layer 
        # create a list of linear layers with passed in tuple values
        # list of linear layers to to the classifier's hidden layers
        h_linear_list = []
        h_tuples = zip(hidden_layers[:-1], hidden_layers[1:])
        h_linear_list = [nn.Linear(h1, h2) for h1, h2 in h_tuples]
        self.hidden_layers.extend(h_linear_list)
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        '''
        Image classifier forward loop method
        
        arguments:
            x: image tensors that will get processed through the forward pass
        
        return:
            x: image tensor after completing the forword pass
        
        Code Attribution:
            This function contains code that was updated and leveraged from the 
            Udacity AI Programming with Python Nano Degree Training material.
        '''
        
        x = x.view(x.shape[0], -1)
        
        for h in self.hidden_layers:
            x = F.relu(h(x))
            x = self.dropout(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        
        return x
    

def train_model(model, criterion, optimizer, train_dataloaders, 
                validation_dataloaders, device, epochs=3, print_every=40):
    '''
    Trains a model's image classifier parameters against a provided train 
    and validation image directories and freezes the feature parameters per 
    provided functional parameters
    
    arguments:
        model: image classifier model
        criterion: loss function
        optimizer: gradient descent optimizer
        train_dataloaders: training image tensors
        validation_dataloaders: validation image tensors
        device: cpu or gpu
        epochs: quantity of epoch iteration to train model
        print_every: quantity of training steps until training info is displayed
        
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity AI Programming with Python Nano Degree Training material.
    '''
    
    model = model.to(device)

    for e in range(epochs):

        running_loss = 0
        # track steps so that model training status can be displayed per print_every param
        steps = 0
        model.train()
        for images, labels in train_dataloaders:

            steps += 1

            # pass images and labels tensors to the GPU if available
            images, labels = images.to(device), labels.to(device)

            # zerize graddient after each batch
            optimizer.zero_grad()

            # forward pass and calculate loss
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)

            # backward pass and calculate gradient steps
            loss.backward()
            optimizer.step()

            # calculate running across the batch of images
            running_loss += loss.item()

            # print model parameters periodically
            if steps % print_every == 0:

                validation_loss = 0
                accuracy = 0

                # turn of gradient
                with torch.no_grad():
                    # model validation
                    model.eval()

                    for images, labels in validation_dataloaders:

                        # pass images and labels tensors to the GPU if available
                        images, labels = images.to(device), labels.to(device)

                        # forward pass and calculate test loss
                        log_ps = model.forward(images)
                        validation_loss += criterion(log_ps, labels).item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print("Epoch: {}/{}.. ".format(e+1, epochs), 
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(validation_dataloaders)),
                          "Accuracy: {:.3f}..".format(accuracy/len(validation_dataloaders)))

                    running_loss = 0
                    # resume model training
                    model.train()

def test_model(model, criterion, test_dataloaders, device):
    '''
    Tests a model's image classifier against a directory of test images.
    Prints out Test Loss and Accuracy status
    
    arguments:
        model: image classifier model
        criterion: loss function
        test_dataloaders: test image tensors
        device: cpu or gpu
    
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity AI Programming with Python Nano Degree Training material.
    '''
    
    with torch.no_grad():

        test_loss = 0
        accuracy = 0
        # model validation
        model.eval()

        for images, labels in test_dataloaders:

            images, labels = images.to(device), labels.to(device)

            # forward pass and calculate test loss
            log_ps = model.forward(images)
            test_loss += criterion(log_ps, labels).item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        print("Test Loss: {:.3f}.. ".format(test_loss/len(test_dataloaders)),
              "Accuracy: {:.3f}..".format(accuracy/len(test_dataloaders)))