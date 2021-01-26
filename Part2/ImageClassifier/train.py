import argparse
import torch
from torch import nn
from torch import optim
from torchvision import models

from image_classifier import Classifier, train_model, transform_train_data
from utility_functions import save_checkout


parser = argparse.ArgumentParser("Enter arguments to train the image network")
parser.add_argument("data_dir", type=str, help="The directory of the image "  
                    "files.")
parser.add_argument("--save_dir", type=str, default="/home/workspace/SavedModels", 
                    help="The directory of the saved checkpoint. Default is " 
                    "/home/workspace/SavedModels")
parser.add_argument("--arch", type=str, default="vgg13", 
                    help="Choose feature image architecture either vgg11 or " 
                    "vgg13. The default is vgg13.")
parser.add_argument("--learn_rate", type=float, default=.00085, help="The "  
                    "learning rate of the image network. The default is .00085.")
parser.add_argument("--hidden_units", type=int, default=500, help="The hidden "
                    " units of the image network. The default is 500.")
parser.add_argument("--epochs", type=int, default=5, help="The epochs of "
                    "the image network. The default is 5.")
parser.add_argument("--gpu", type=str, default="gpu", help="Choose to run the " 
                    "training image network using a gpu or cpu. The default is " 
                    "gpu.")

# command line arguments passed into variables
args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learn_rate = args.learn_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

if __name__ == "__main__":
    print()
    print("Image network will train on directory {}, save checkout to {}, " 
          "arhitecture {}, learning rate {}, hidden units {}, epochs {}, "  
          "and gpu {}".format(data_dir, save_dir, arch, learn_rate, 
                              hidden_units, epochs, gpu))
    print()

    batch_size = 32

    # load training and validation image tensors    
    try:
        train_image_datasets, train_dataloaders, validation_dataloaders = (
            transform_train_data(data_dir, batch_size))
    except:
        print("Can not find provided directory.  Please try again.")

    # load feature detector model
    try:
        if arch == "vgg11":
            model = models.vgg11(pretrained=True)
        elif arch == "vgg13":
            model = models.vgg13(pretrained=True)
        else:
            raise
    except:
        print("Architecture provided invalid.  Please try again")
        exit()
    
    # build model classifier
    try:    
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = Classifier(25088, 102, [hidden_units], drop_p=.2)
    except:
        print("Hidden units provided is invalid. Please try again")
        exit()
    
    print_every = 40
    criterion = nn.NLLLoss()
    
    # train model
    try:
        if gpu == "gpu": 
            device = torch.device("cuda:0" if torch.cuda.is_available() 
                                  else "cpu")        
        else:
            device = "cpu"
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
        train_model(model, criterion, optimizer, train_dataloaders, 
                    validation_dataloaders, device, epochs, print_every)
    except:
        print("A provided train model parameter provided is invalid. " 
              "Please try again.")
        exit()
    
    # get the training dataset's dictionary of classes mapped to indices
    class_to_idx = train_image_datasets.class_to_idx

    # switch dictionary key value and assign to model.class_to_idx
    model.class_to_idx = {class_to_idx[i]: i for i in class_to_idx}
    
    # save trained image classification model to a checkpoint file
    save_checkout(save_dir, model.classifier.input_size, 
                  model.classifier.output_size, 
                  [each.out_features for each in model.classifier.hidden_layers], 
                  model.classifier.dropout.p,
                  model.state_dict(), epochs, optimizer.state_dict(), 
                  model.class_to_idx, learn_rate, batch_size)
