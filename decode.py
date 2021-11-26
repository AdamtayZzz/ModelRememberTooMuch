import torch
import argparse
import numpy as np
from attack import get_encode_data,count_parameters, set_params_init
from model import resnet18
import os
import cv2
from PIL import ImageOps,Image


def load_model(path):
    model = resnet18(num_classes=10)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model

def get_params(model):
    params = [p.flatten().detach().numpy() for p in model.parameters() if p.ndim > 1]
    params = np.concatenate(params,axis=0)
    return params

def get_groundtruth_image(model,num_image,attack):
    total_param_num = count_parameters(model)
    raw_data,dim_data = get_encode_data()
    if attack == 'sgn':
        dim_data = (dim_data*8)
    n_hidden_data = int(total_param_num / dim_data)
    n_hidden_data = min(n_hidden_data,num_image)
    print(f"{attack}\ttotal param num:{total_param_num}\tdim_data:{dim_data}\tn_hidden_data:{n_hidden_data}")
    offset = n_hidden_data* dim_data
    return raw_data[0:n_hidden_data],offset

def save_img(path,img_array):
    img = Image.fromarray(img_array)
    img.save(path)

def normalize(x):
    x = x.transpose(2,0,1)
    x_shape = x.shape
    x_list = []
    for channel in x:
        v = normalize_channel_wise(channel).reshape(1,x_shape[1],x_shape[2])
        x_list.append(v)
    return np.concatenate(x_list,axis=0).transpose(1,2,0)

def normalize_channel_wise(x):
    x_shape = x.shape
    x = x.flatten()
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    return x.reshape(x_shape)

def decode(args):
    model_path = f'resnet_cifar_{args.attack}_{args.phi}_{args.num_image}.pth'
    model = load_model(model_path)
    print(f"load model from {model_path}")
    params = get_params(model)
    pair_image,offset = get_groundtruth_image(model,args.num_image,args.attack)
    params = params[0:offset]
    # recover images
    if args.attack == 'sgn':
        params = np.sign(params)
        params[params == -1] = 0
        params = params.astype(np.uint8)
        recover_img = np.packbits(params.reshape(-1, 8)).reshape(-1, pair_image.shape[1],pair_image.shape[2],pair_image.shape[3])
        recover_img = recover_img.astype(np.uint8)
    elif args.attack == 'cor':
        cor_params = params.reshape(-1,pair_image.shape[1],pair_image.shape[2],pair_image.shape[3])
        # transform correlated parameters back to input space
        cor_params_list = []
        for img in cor_params:
            v = normalize(img)
            v = v.reshape(1,*v.shape)
            cor_params_list.append(v)
        cor_params = np.concatenate(cor_params_list)
        recover_img = (cor_params*255).astype(np.uint8)
    
    # save
    path1 = f"pair_image/{args.attack}/{args.phi}/{args.num_image}/"
    path2 = f"encode_image/{args.attack}/{args.phi}/{args.num_image}/"
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    for idx in range(args.num_image):
        save_img(path1+f'{idx+1}.png',pair_image[idx])
        save_img(path2+f'{idx+1}.png',recover_img[idx])
        # COR < 0 negatively correlated values
        if args.attack == 'cor':
            img = np.asarray(ImageOps.invert(Image.fromarray(recover_img[idx])))
            if not os.path.exists(path2+"invert/"):
                os.makedirs(path2+"invert/")
            save_img(path2+f'invert/{idx+1}.png',img)
    print('save image complete')



parser = argparse.ArgumentParser(description="machine learning models that remember too much - decode phase")
parser.add_argument('--attack',default='None',type=str,help='the default attack')
parser.add_argument('--phi',default=0.0,type=float,help='lambda in sign or correlative')
parser.add_argument('--num_image',default=20000,type=int,help='the number of image to encode')
args = parser.parse_args()
print(args)
decode(args)

