import torch
import torch.cuda
from torch import optim,nn
import torch.nn.functional as F
from torchvision import datasets,transforms,models
import numpy as np
import argparse
from toolbox import load_dataset,save_checkpoint,load_model,train_model,validation,create_classifier,test_model,train_setup

data_dir, save_dir,learning_rate,dropout,hidden_units,epochs,hidden_units2,gpu_device,pretrained_model=train_setup()               
criterion = nn.NLLLoss()

#load the data
trainloader, testloader, validloader, train_data, test_data, valid_data=load_dataset(data_dir)

#load the model
model =getattr(models,pretrained_model)(pretrained=True)

#create a classifier
input_units = model.classifier[0].in_features
print("The model uses: {} input units".format(input_units))
create_classifier(model,input_units,hidden_units,hidden_units2,dropout)
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
model, optimizer = train_model(optimizer, criterion, epochs, validloader, trainloader, model,gpu_device)

#Test the model
test_model(model,testloader,gpu_device)
#Save the mode
save_checkpoint(model,epochs,train_data,optimizer,save_to=save_dir)