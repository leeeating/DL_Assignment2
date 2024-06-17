import argparse
import os
import json
import time

import torch
import torch.nn as nn

from torchvision.models import resnet34

from model.dataset import ImageMini
from model.train_test import test
from model.model import PreFixResnet


###########################-- init setting --###########################
parser = argparse.ArgumentParser(description="Naive Resnet34 Inference")

parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--model_name', type=str, default = 'naive', help='naive or dy_cnn')
parser.add_argument('--timestamp', type= str, help='input timerecord in model name')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
###########################-- init setting --###########################



# data & model & record
if __name__ == '__main__':

    channel_group = ['RGB', 'RG', 'RB', 'BG', 'R', 'G', 'B']

    data = {}
    for c in channel_group:
            
        # data
        if args.model_name == 'naive':    
            test_ds = ImageMini("test", c, pading=True)
        
        elif args.model_name == 'dy_cnn':
            test_ds = ImageMini("test", c)

        print(f'Input Channel:{c} \nModel_name:{args.model_name}')


        # model
        # naive model
        if args.model_name == 'naive':
            model_path = f'./ckpt/{args.timestamp}_{args.model_name}_RGB.pt'
            model = resnet34()
            model.fc = nn.Linear(512, 50)        
            model.load_state_dict(torch.load(model_path))

        elif args.model_name == 'dy_cnn': 
            model_path = f'./ckpt/{args.timestamp}_dy_cnn_{c}.pt'
            model = PreFixResnet(c, train=False)
            model.load_state_dict(torch.load(model_path))

        model.to(device)
    
        # record
        test_res = test(args, model, test_ds)
        record = {c:test_res[0]}

        print(record)
        data.update(record)

    with open(f'./log/{args.timestamp}_{args.model_name}_inference.json', 'w') as f:
        json.dump(data, f)





