import numpy as np
import pickle
import torch
from dataset import get_cifar10_dataloader_gan
import torchvision
import cv2
from PIL import Image

# Training Loss Definition
def Sign_Loss(params,targets,offset):
    params = torch.cat([torch.flatten(p) for p in params if p.ndim >1])[0:offset]
    targets = torch.flatten(targets)[0:offset]
    constraints = targets * params
    zeros = torch.zeros_like(constraints)
    v = torch.mean(torch.abs(torch.where(torch.gt(constraints,0),zeros,constraints)))
    correct_ratio = torch.mean(torch.gt(constraints,0).float())
    return v,correct_ratio

def Correlation_Loss(params,targets,offset):
    params = torch.cat([torch.flatten(p) for p in params if p.ndim >1])[0:offset]
    targets = torch.flatten(targets)[0:offset]
    p_mean = torch.mean(params)
    t_mean = torch.mean(targets.float())
    p_m = params-p_mean
    t_m = targets-t_mean
    r_num = torch.sum(p_m*t_m)
    r_den = torch.sqrt(torch.sum(torch.square(p_m))*torch.sum(torch.square(t_m)))
    v = r_num/r_den
    return -torch.abs(v),v
    

#  Without Attack
def No_Loss(params=None,targets=None,offset=None):
    v = 0
    return v

def Encode_Loss(attack):
    if attack == 'sgn':
        return Sign_Loss
    elif attack == 'cor':
        return Correlation_Loss
    else:
        return No_Loss

#encode data generation
def get_encode_data():
    dataloader,_ = get_cifar10_dataloader_gan()
    for i,(data,_) in enumerate(dataloader):
        if i==0:
            image_pair = data
        else:
            image_pair = torch.cat((image_pair,data),dim=0)
    X_train = image_pair.numpy()
    raw_data = X_train if X_train.dtype == np.uint8 else X_train * 255
    if raw_data.shape[-1] != 3:
        raw_data = raw_data.transpose(0, 2, 3, 1)
    raw_data = raw_data.astype(np.uint8) # (50000, 32, 32, 3) 
    print('Raw data shape {}'.format(raw_data.shape)) 
    hidden_data_dim = np.prod(raw_data.shape[1:])
    return raw_data,hidden_data_dim

# data process methods
def get_binary_secret(X):
    # convert 8-bit pixel images to binary with format {-1, +1}
    assert X.dtype == np.uint8
    s = np.unpackbits(X.flatten())
    s = s.astype(np.float32)
    s[s == 0] = -1
    return s

def set_params_init(model, values):
    # calculate number of parameters needed to encode secrets
    offset = 0
    for p in model.parameters():
        if p.requires_grad and p.ndim >1:
            n = np.prod(p.shape)
            if offset >= values.shape[0]:
                offset = values.shape[0]
                print("Number of params greater than targets")
                break
            offset += n 
    return offset

# utils 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad and p.ndim>1)