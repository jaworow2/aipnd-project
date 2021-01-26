import argparse
import torch

from torchvision import models
from image_classifier import Classifier
from utility_functions import load_checkpoint, predict, align_class_prob


parser = argparse.ArgumentParser("Enter arguments to predict on a flower image")
parser.add_argument("image_path", type=str, help="The path of the image files.")
parser.add_argument("--top_k", type=int, default="3", 
                    help="Top KKK most likely classes. Default is 3")
parser.add_argument("--category_names", type=str, default="cat_to_name.json", 
                    help="Provide the mapping of category number to real name. "
                    "The default is cat_to_name.json.")
parser.add_argument("--gpu", type=str, default="gpu", help="Choose to run the " 
                    "training image network using a gpu or cpu. The default is " 
                    "gpu.")

# command line arguments passed into variables
args = parser.parse_args()
image_path = args.image_path
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

if __name__ == "__main__":
    print()
    print("Image network will predict flower image on {}, on the {} likely " 
          "classes, using category name mapping {}, and gpu {}"
          .format(image_path, top_k, category_names, gpu))
    print()

    # load our feature detector model
    model = models.vgg13(pretrained=True)
    model.classifier = Classifier(25088, 102, [500], drop_p=.2)
    
    # used trained model checkpoint from part 1
    checkout_point_dir = "/home/workspace/SavedModels"
    
    # load model checkpoint
    try:
        model, model_class_to_idx = load_checkpoint(checkout_point_dir + 
                                                    "/checkpoint.pth", model)
    except:
        print("Can not find provided model checkpoint.  Please try again.")
        exit()
        
    # predicts the flower image's class
    try:
        if gpu == "gpu": 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        else:
            device = "cpu"
        probs, classes = predict(image_path, device, model, top_k)
    except:
        print("Can not find provided image or provided top K is invalid.  " 
              "Please try again.")
        exit()
    
    # prints the most probable flower classes and corresponding probabilities
    try:
        df = align_class_prob(category_names, model_class_to_idx, classes, probs)
        print(df)
    except:
        print("Can not find provided category name file provided or is invalid. " 
              "Please try again.")
        exit()