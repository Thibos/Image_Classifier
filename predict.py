import torch
from torch import optim,nn
import torch.nn.functional as F
from torchvision import datasets,transforms,models
import numpy as np
import argparse
from PIL import Image
import json
from toolbox import load_dataset,process_image,test_model,load_model,predict,label_loading,get_pretrained_model,predict_setup

save_dir,image,top_k,cat_names,gpu_device,pretrained_model=predict_setup()
cat_to_name=label_loading(cat_names)
model = get_pretrained_model(pretrained_model)

#load checkpoint
index = 1
index_for_print =0
lmodel = load_model(model,save_dir)
prossed_img = process_image(image)
probabilities =predict(image, model,top_k,gpu_device)
a = np.array(probabilities[0][0])
b = [cat_to_name[str(index+1)] for index in np.array(probabilities[1][0])]
print("predicted flower name:{} with a probability of: {} %".format(b[index_for_print],round(a[index_for_print] *100,3)))
print(a*100)
print(b)

