import os 
import zipfile 
import urllib.request
import glob
import os.path as osp 
import random 
import numpy as np
import json 
from PIL import Image
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from tqdm import tqdm

torch.manual_seed(1234) # fixed random
np.random.seed(1234)
random.seed(1234)

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# preprocessing
from torchvision.transforms.transforms import Normalize
class ImageTransform():
    def __init__(self,resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5,1.0)),
                transforms.RandomHorizontalFlip(0.7),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

def make_datapath_list(phase="train"):
    rootpath = "/content/drive/MyDrive/Colab Notebooks/ImageClassification-VGG16/hymenoptera_data/"
    target_path = osp.join(rootpath + phase + "/**/*.jpg")

    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)
    
    return path_list


train_list = make_datapath_list("train")
val_list = make_datapath_list("val")

# create dataset
class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list #train_list/ val_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index): 
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transform = self.transform(img, self.phase) #ImageTransform

        #label
        if self.phase == "train":
            label = img_path[88:92] # label name in link 
        elif self.phase == "val":
            label = img_path[86:90]
        
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1   # bees

        return img_transform, label

# dataset
train_dataset = MyDataset(train_list, transform=ImageTransform(size, mean, std), phase="train")
val_dataset = MyDataset(val_list, transform=ImageTransform(size, mean, std), phase="val")

# create dataloader
batch_size = 6
# -> img_transform, label
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

dataloader_dict = {"train": train_dataloader, "validation": val_dataloader}


model = models.vgg16(pretrained=True)
#transferlearning
model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
# print(model)
# Set mode
model = model.train()


#loss function
criterior = nn.CrossEntropyLoss() 


def params_to_update(model):
    params_to_update1 = []
    params_to_update2 = []
    params_to_update3 = []

    update_param_name_1 = ["features"] 

    # update param layer 0 and layer 3
    update_param_name_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.4.bias"]
    update_param_name_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in model.named_parameters(): 
        if name in update_param_name_1:
            param.requires_grad = True
            params_to_update1.append(param) 
        
        elif name in update_param_name_2:
            param.requires_grad = True
            params_to_update2.append(param)

        elif name in update_param_name_3:
            param.requires_grad = True
            params_to_update3.append(param)

        else:
            param.requires_grad = False

    return params_to_update1, params_to_update2, params_to_update3

params1, params2, params3 = params_to_update(model)


optimizer = optim.SGD([
    {'params': params1, 'lr': 1e-4}, # feature 
    {'params': params2, 'lr': 5e-4}, # layer0 and layer3
    {'params': params3, 'lr': 1e-3},], # final layer 
    momentum=0.9) 


def train_model(model, dataloader_dict, criterior, optimizer, num_epoch):
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    
    for epoch in range(num_epoch):
        print("Epoch {}/ {}".format(epoch, num_epoch))
        # calc with gpu
        model.to(device)

        torch.backends.cudnn.benchmark = True # speed up gpu


        for phase in ["train"]:
            epoch_loss = 0.0
            epoch_accuracy = 0
            for input, label in tqdm(dataloader_dict["train"]):
                # transfer to gpu
                input = input.to(device)
                label = label.to(device)
                
                #reset after each epoch
                optimizer.zero_grad() 
                
                with torch.set_grad_enabled(phase == "train"):  
                    output = model(input)
                    loss = criterior(output, label) #TENSOR
                    
                    _, predict = torch.max(output, 1) # -> 0 / 1
                
                    loss.backward()
                    optimizer.step() #update param  optimizer
                    
                    # input (B, C, H, W)                   
                    epoch_loss += loss.item()*input.size(0) 
                   
                    epoch_accuracy += torch.sum(predict == label.data)

            epoch_loss = epoch_loss / len(dataloader_dict["train"].dataset) 
            epoch_accuracy = epoch_accuracy.double() / len(dataloader_dict["train"].dataset)
            print("{} Loss: {: 4f} Acc: {: 4f}".format("train", epoch_loss, epoch_accuracy))
    
    # save model
    if epoch >= 0:
        save_path = "/content/drive/MyDrive/Colab Notebooks/ImageClassification-VGG16/model.pth"                
        torch.save(model.state_dict(), save_path)
    return save_path

epoch = 25
save_path = train_model(model, dataloader_dict, criterior, optimizer, epoch)

# load model
def load_model(model, model_path):
    load = torch.load("/content/drive/MyDrive/Colab Notebooks/ImageClassification-VGG16/model.pth") # params
    model.load_state_dict(load) # fit params to model

    for name, param in model.named_parameters():
        print(name, param)

    # transfer params on gpu to cpu
    # load = torch.load(save_path, map_location=("cuda:0", "cpu"))
    # model.load_state_dict(load) # fit params to model
save_path = "/content/drive/MyDrive/Colab Notebooks/ImageClassification-VGG16/model.pth"
load_model(model, save_path)


def eval_model(model, dataloader_dict, criterior, optimizer, num_epoch):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    
    for epoch in range(num_epoch):
        print("Epoch {}/ {}".format(epoch, num_epoch))
        # calc with gpu
        model.to(device)

        torch.backends.cudnn.benchmark = True # speed up gpu


        for phase in ["validation"]:
            epoch_loss = 0.0
            epoch_accuracy = 0
            for input, label in tqdm(dataloader_dict["validation"]):
                # transfer to gpu
                input = input.to(device)
                label = label.to(device)
                
                optimizer.zero_grad() 
                
                
                with torch.set_grad_enabled(phase == "validation"):  
                    output = model(input)
                    loss = criterior(output, label) #TENSOR
                   
                    _, predict = torch.max(output, 1) # -> 0 / 1
                    
                    loss.backward()
                    optimizer.step() #update param  optimizer
                    
                    # input (B, C, H, W)                   
                    epoch_loss += loss.item()*input.size(0) 
                    epoch_accuracy += torch.sum(predict == label.data)

            epoch_loss = epoch_loss / len(dataloader_dict["validation"].dataset) 
            epoch_accuracy = epoch_accuracy.double() / len(dataloader_dict["validation"].dataset)
            print("{} Loss: {: 4f} Acc: {: 4f}".format("validation", epoch_loss, epoch_accuracy))

class_idx = ["ants", "bees"]

# predict class
class Predictor():
    def __init__(self, class_idx):
        self.class_idx = class_idx

    def predict_max(self, model_ouput):
        max_idx = np.argmax(model_ouput.detach().numpy())  
        predict_label = self.class_idx[max_idx]
        return predict_label

predictor = Predictor(class_idx)

def predict(img):
    # pretrained model 
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    model.eval()

    #load model
    model1 = load_model(model, "/content/drive/MyDrive/Colab Notebooks/ImageClassification-VGG16/model.pth")

    #prepare img
    
    transform = ImageTransform(size, mean, std)
    img = transform(img, phase="test") #(C, H, W)
    img = img.unsqueeze_(0)     #(B, C, H, W)

    # predict
    output_val = model(img) 
    output_val = predictor.predict_max(output_val) 

    return output_val

img_path = "/content/drive/MyDrive/Colab Notebooks/ImageClassification-VGG16/bee.JPG"
img = Image.open(img_path)
predict_val = predict(img)
plt.imshow(img)
plt.show()
print("Prediction: {}".format(predict_val))

