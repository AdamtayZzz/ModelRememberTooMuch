import torch
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import resnet18
from dataset import get_cifar10_dataloader
from attack import Encode_Loss, get_binary_secret,get_encode_data, set_params_init,count_parameters

import argparse
import matplotlib.pyplot as plt

def acc_eval(output,label):
    output_label = torch.argmax(output,dim=1)
    acc = (torch.eq(output_label,label)).sum()/label.shape[0]
    return acc

def validation(model,val_dataloader,device):
    model.eval()
    acc_total = 0
    with torch.no_grad():
        for _,(image,label) in enumerate(val_dataloader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            acc_total += acc_eval(output,label)
        valid_acc = acc_total/len(val_dataloader)
    return valid_acc



def train(args):
    model = resnet18(num_classes=10)
    device = torch.device(f'cuda:{args.which_cuda}')
    train_dataloader,val_dataloader = get_cifar10_dataloader()
    optimizer = optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,momentum=0.9)
    if(args.lr_schedule):
        scheduler = CosineAnnealingLR(optimizer,T_max=args.epoch)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    Encode_loss = Encode_Loss(args.attack)
    encode_data,dim_data = get_encode_data() 
    total_param_num = count_parameters(model)
    if(args.attack=='sgn'):
        n_hidden_data = int(total_param_num / (dim_data*8))
        n_hidden_data = min(n_hidden_data,args.num_image)
        print(f"{args.attack}\ttotal param num:{total_param_num}\tdim_data:{dim_data*8}\tn_hidden_data:{n_hidden_data}")
        encode_target = torch.tensor(get_binary_secret(encode_data[0:n_hidden_data])).to(device)
        offset = set_params_init(model,encode_target)
    elif(args.attack=='cor'):
        n_hidden_data =  int(total_param_num / dim_data)
        n_hidden_data = min(n_hidden_data,args.num_image)
        print(f"{args.attack}\ttotal param num:{total_param_num}\tdim_data:{dim_data}\tn_hidden_data:{n_hidden_data}")
        encode_target = torch.tensor(encode_data[0:n_hidden_data].flatten()).to(device)
        offset = set_params_init(model,encode_target)
    else:
        offset=None
        encode_target=None
    

    for e in range(args.epoch):
        task_loss_epoch = 0
        encode_loss_epoch = 0
        train_acc_epoch = 0
        encode_correct_ratio = 0

        model.train()
        for _,(image,label) in enumerate(train_dataloader):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            output = model(image)
            encode_loss,correct_ratio = Encode_loss(model.parameters(),encode_target,offset) 
            task_loss = criterion(output,label)
            loss = task_loss + args.phi*encode_loss
            loss.backward()
            optimizer.step()

            task_loss_epoch += task_loss.item()
            encode_loss_epoch += encode_loss.item()
            train_acc_epoch +=acc_eval(output,label)
            encode_correct_ratio += correct_ratio
        if(args.lr_schedule):
            scheduler.step()
        validation_acc_epoch = validation(model,val_dataloader,device)
        
        print(f"epoch:{e} task_loss:{task_loss_epoch/len(train_dataloader):.3f} train_acc:{train_acc_epoch/len(train_dataloader):.3f} validation_acc:{validation_acc_epoch:.3f}")
        print(f"encode_loss:{encode_loss_epoch/len(train_dataloader):.3f} encode_correct_ratio:{encode_correct_ratio/len(train_dataloader):.3f}")

    torch.save(model.state_dict(),f'resnet_cifar_{args.attack}_{args.phi}_{args.num_image}.pth')

parser = argparse.ArgumentParser(description="machine learning models that remember too much")
parser.add_argument('--weight_decay',default=5e-4,type=float,help='weight decay of the training process')
parser.add_argument('--epoch',default=200,type=int,help='the number of epoch used in train')
parser.add_argument('--lr',default=0.1,type=float,help='learning rate')
parser.add_argument('--lr_schedule',action='store_true',help='whether to use lr_schedule or not')
parser.add_argument('--which_cuda',default=5,type=int,help='the default cuda')
parser.add_argument('--attack',default='None',type=str,help='the default attack')
parser.add_argument('--phi',default=0.0,type=float,help='lambda in sign or correlative')
parser.add_argument('--num_image',default=20000,type=int,help='the number of image to encode')
args = parser.parse_args()
print(args)
train(args=args)

