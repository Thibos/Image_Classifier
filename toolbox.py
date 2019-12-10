import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models,datasets,transforms
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse
def predict_setup():
    input_parser = argparse.ArgumentParser(description='Use this deep learning model to predict pictures')
    input_parser.add_argument('--image_location', action = 'store',default = './flowers/test/11/image_03141.jpg',
                    help = 'Enter location to the image.')

    input_parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save checkpoint in.')

    input_parser.add_argument('--top_k', action='store',type=int,
                    dest = 'topk', default = '3',
                    help= 'Enter number of top most likely classes.')

    input_parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='Enter path to image.')

    input_parser.add_argument('--arch', action='store',
                    dest = 'pretrained_model', default = 'vgg11',
                    help= 'Enter pretrained model to use')
    input_parser.add_argument('--gpu', action='store_true', default =True,help= 'Use GPU device:.')


    user_inputs = input_parser.parse_args()
    save_dir = user_inputs.save_directory
    image = user_inputs.image_location
    top_k = user_inputs.topk
    cat_names = user_inputs.cat_name_dir
    gpu_device =user_inputs.gpu
    pretrained_model =user_inputs.pretrained_model
    return save_dir,image,top_k,cat_names,gpu_device,pretrained_model
def train_setup():
    
    input_parser = argparse.ArgumentParser(description='Train deep learning model')

    input_parser.add_argument('data_directory', action = 'store',default = 'flowers',
                    help = 'Enter path to training data.')

    input_parser.add_argument('--arch', action='store',
                    dest = 'pretrained_model', default = 'vgg11',
                    help= 'Enter pretrained model to use; this classifier can currently work with\
                           VGG and Densenet architectures. The default is VGG-11.')

    input_parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save checkpoint in.')

    input_parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=int, default = 0.001,
                    help = 'Enter learning rate for training the model, default is 0.001.')

    input_parser.add_argument('--dropout', action = 'store',
                    dest='drpt', type=int, default = 0.05,
                    help = 'Enter dropout for training the model, default is 0.05.')

    input_parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type=int, default = 5096,
                    help = 'Enter number of hidden units in classifier, default is 500.')
                    
    input_parser.add_argument('--hidden_units2', action = 'store',
                    dest = 'units2', type=int, default = 1024,
                    help = 'Enter number of hidden units in classifier, default is 500.')

    input_parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 2,
                    help = 'Enter number of epochs to use during training, default is 1.')

    input_parser.add_argument('--gpu', action='store', default = True,help= 'Use GPU device:.')

    user_inputs = input_parser.parse_args()

    data_dir = user_inputs.data_directory
    save_dir = user_inputs.save_directory
    learning_rate = user_inputs.lr
    dropout = user_inputs.drpt
    hidden_units = user_inputs.units
    epochs = user_inputs.num_epochs
    hidden_units2 = user_inputs.units2
    gpu_device =user_inputs.gpu
    pretrained_model =user_inputs.pretrained_model
    return data_dir, save_dir,learning_rate,dropout,hidden_units,epochs,hidden_units2,gpu_device,pretrained_model
    
def load_dataset(folder="flowers"):
    data_dir = folder
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/vali'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                     transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) 


    test_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),
                                     transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) 

 

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir+'/test',transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    return trainloader, testloader, validloader, train_data, test_data, valid_data
    
def test_model(model,testloader,gpu_device):
    correct=0
    total_img =0
    if gpu_device == True:
     
        model.to('cuda')
    else:
       pass
                           
                        
    
    with torch.no_grad():
        for data in testloader:
            inputs,labels = data
            if gpu_device == True:
                #device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
                inputs,labels = inputs.to('cuda'),labels.to('cuda')
            else:
                pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data,1)
            total_img += labels.size(0)
            correct +=(predicted==labels).sum().item()
    print(f"Test accuracy of model: {round(100 * correct / total_img,3)}%")
    
def label_loading(cat_names):
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
def save_checkpoint(model,epochs,train_data,optimizer,save_to):
    checkpoint = {'state_dict':model.state_dict(),
             'classifier':model.classifier,
             'num_epochs':epochs,
             'class_to_idx':train_data.class_to_idx,
             'opt_state':optimizer.state_dict}
    torch.save(checkpoint,save_to)
    print("Model is successfully saved to: {}".format(save_to))
    
def load_model(model,path):
    
    checkpoint =torch.load(path)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict = checkpoint['state_dict']
    return model

def process_image(image):
    pro_image = Image.open(image)
    proc_img_transforms =transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    img =proc_img_transforms(pro_image)
    return img



def predict(image_path, model, topk,gpu_device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    if gpu_device == True:
        #device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
        model.to('cuda')
    else:
        pass
    
    
    img = process_image(image_path)
    img =img.unsqueeze_(0)
    img = img.float()
    img= img.cuda()
    
    with torch.no_grad():
        
        output = model.forward(img)
    proba =F.softmax(output.data,dim=1)
    
    return proba.topk(topk)


    
def train_model(optimizer, creterion, epochs, validloader, trainloader, model,gpu_device):
    epochs =epochs
    steps = 0
    print_every =40
    print("starting")
    if gpu_device == True:
       # device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
        model.to('cuda')
    else:
        pass
    
    for e in range(epochs):
        # print("starting first loop")
        running_loss = 0
        start_time = time.time()
        for ii,(inputs,labels) in enumerate(trainloader):
            # print("starting 2nd loop")
            steps +=1
            if gpu_device == True:
                #device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
                inputs,labels = inputs.to('cuda'),labels.to('cuda')
            else:
                pass
            
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = creterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every== 0:
                # print("starting first if")
                model.eval()
                
                with torch.no_grad():
                    #print("starting with loop")
                    test_loss, accuracy_score = validation(model,validloader,creterion,gpu_device)
                
                    print(f"No. epochs: {e+1}, \
                        Training Loss: {round(running_loss/print_every,3)} \
                        Valid Loss: {round(test_loss/len(validloader),3)} \
                        Valid Accuracy: {round(float(accuracy_score/len(validloader)),3)}")
                
              
                      
                    running_loss =0
                    model.train()
    return model, optimizer
    
def validation(model,validloader,creterion,gpu_device):
    
    accuracy_score=0
    test_loss =0
    
    model.to('cuda')
        
        
    for ii,(inputs,labels) in enumerate(validloader):
        #inputs,labels = inputs.to('cuda'),labels.to('cuda')
        #img= img.cuda()
        if gpu_device == True:
                
                #device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
             inputs,labels = inputs.to('cuda'),labels.to('cuda')
        else:
             pass
        output = model.forward(inputs)
        test_loss +=  creterion(output,labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy_score += equality.type_as(torch.FloatTensor()).mean()
            
        
        
    return test_loss, accuracy_score

def create_classifier(model,input_units,hidden_units,hidden_units2,dropout):
    for param in model.parameters():
        param.requires_grad = False
        
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_units, hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('dropout',nn.Dropout(dropout)),
                                        ('fc2', nn.Linear(hidden_units, hidden_units2)),
                                        ('relu2', nn.ReLU()),
                                        ('dropout2',nn.Dropout(dropout)),
                                        ('fc3', nn.Linear(hidden_units2, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model
def get_pretrained_model(pretrained_model):
  
          pre_model =pretrained_model
          model = getattr(models,pre_model)(pretrained=True)
          return model
          