import torch
import numpy as np
import pandas as pd
import json

from image_classifier import Classifier
from PIL import Image

def save_checkout(save_dir, input_size, output_size, hidden_layers, dropout, state_dict, 
                  epochs, optimizer_state, class_to_idx, learn_rate, batch_size):
    '''
    Saves the trained flower image classifier as checkpoint file
    
    arguments:
        save_dir: directory where checkpoint will be saved
        input_size: input size of the model classifer
        out_size: output size of the model classifer
        hidden_layers: hidden layer dimensions of the model classifier
        dropout: percent of the model classifier nodes that do not train in 
        a training step to prevent overfitting
        state_dict: weights of the trained hidden layers
        epochs: quantity of epochs the model performs
        optimizer_state: gradient step state
        class_to_idx: mapping of flower class to index
        learn_rate: gradient step rate
        batch_size: quantity of images to be loaded during training
    
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity AI Programming with Python Nano Degree Training material.
    '''
    
    checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'hidden_layers': hidden_layers,
              'dropout': dropout,
              'state_dict': state_dict,
              'epoch': epochs,
              'optimizer_state': optimizer_state,
              'model_class_to_idx': class_to_idx,
              'learning_rate': learn_rate,
              'batch_size': batch_size}

    torch.save(checkpoint, save_dir + "/checkpoint_part2.pth")
    
def load_checkpoint(filepath, model):
    '''
    Loads the trained flower image classifier checkpoint
    
    arguments:
        filepath: the file path of the trained flower image classifier 
        checkpoint
        model: model object that will be loaded with the trained flower image 
        classifier
        
    return:
        model: model object loaded with the trained flower image classifier
        model_class_to_idx: mapping of flower class to index
        
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity AI Programming with Python Nano Degree Training material.
    '''
    
    checkpoint = torch.load(filepath)
    model.classifier = Classifier(checkpoint["input_size"], 
                                  checkpoint["output_size"], 
                                  checkpoint["hidden_layers"], 
                                  checkpoint["dropout"])
    model.load_state_dict(checkpoint['state_dict'])
    model_class_to_idx = checkpoint["model_class_to_idx"]
    
    return model, model_class_to_idx

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    
    argument:
        image: flower image to predict classification
    
    returns:
        image: flower image Numpy array
        
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity AI Programming with Python Nano Degree Training material.
    '''
    image_file = Image.open(image)
    image_file = image_file.resize((256, 256))
    
    width, height = image_file.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image_file = image_file.crop((left, top, right, bottom))
    
    # read image in as numpy array and normalize
    np_image = np.array(image_file) / 256
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # color channel moved from 3rd to 1st dimension
    np_image = np_image.transpose((2,0,1))
    image = torch.from_numpy(np_image)
    
    return image

def predict(image_path, device, model, topk=5):
    ''' 
    Predict the class (or classes) of an image using a trained deep 
    learning model.
    
    argument:
        image_path: flower image to predict classification
        device: cpu or gpu
        model: model object loaded with the trained flower image classifier
        topk: return the top k probable classes and probabilities
    
    returns:
        top_p: top k probable probabilities
        top_class: top k probable classes
        
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity AI Programming with Python Nano Degree Training material.
    '''
    model = model.to(device)

    model.eval()

    # convert numpy processed flower image into the torch tensor
    image = process_image(image_path)
    image_tensor = image.type(torch.cuda.FloatTensor)
    image_tensor = image_tensor.unsqueeze(dim=0)

    # turn off gradient descent
    with torch.no_grad():
    
        image_tensor.to(device)
    
        # forward pass and calculate loss
        log_ps = model.forward(image_tensor)

        # convert log loss with exponential and return probilities
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk, dim=1)
        
        top_p = top_p.to("cpu").numpy()
        top_class = top_class.to("cpu").numpy()
    
    return top_p, top_class

def align_class_prob(cat_to_name_file, model_class_to_idx, classes, probs):
    '''
    Align the most probable flower classes and corresponding probabilities
    
    arguments:
        cat_to_name_file: flower category name to index translation
        model_class_to_idx: mapping of flower class to index
        classes: most probable flower class categories
        probs: most probable flow class probabilities
    
    returns:
        panda dataframe of most probable flower classes and correpsonding 
        probabilities
    '''
    
    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f)
    
    class_names = []
    for class_index in classes[0]:
        class_names.append(cat_to_name[model_class_to_idx[class_index]])
    
    df = pd.DataFrame(
        {"class_name" : class_names,
        "prob" : probs[0]})
    
    return df