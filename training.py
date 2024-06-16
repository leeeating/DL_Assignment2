import argparse
import os
import json
import time
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import resnet34

from dataset import ImageMini
from train_test import train, test, EarlyStopping
from model import PreFixResnet, CustomModel

###########################-- init setting --###########################
parser = argparse.ArgumentParser(description="ResNet34 Training")

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--model_name', type=str, default = 'naive', help='naive or dy_cnn or custom')
parser.add_argument('--timestamp', type= str, help='input timerecord in model name')
parser.add_argument('--use_channels', type=str, default='RGB')

parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--delta', type=float, default=0.02, help='Early Stopping Delta')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')

os.makedirs('ckpt', exist_ok=True)
os.makedirs('log', exist_ok=True)

###########################-- init setting --###########################


def update_record(record : dict, train_res : list, test_res : list, val_res : list):

    record['train']['acc'].append(train_res[0])
    record['train']['loss'].append(train_res[1]) 
    record['test']['acc'].append(test_res[0])
    record['test']['loss'].append(test_res[1])     
    record['val']['acc'].append(val_res[0])
    record['val']['loss'].append(val_res[1]) 




# data & model & record
if __name__ == "__main__":

    # data
    train_ds = ImageMini('train', args.use_channels)
    test_ds = ImageMini("test", args.use_channels)
    val_ds = ImageMini("val", args.use_channels)
    train_lodaer = DataLoader(train_ds, args.batch_size, pin_memory=True)
    print('Input Channel:', args.use_channels)
    print("Model_Name:", args.model_name)

    assert len(args.use_channels) == train_ds[0][0].shape[0], 'Tensor Channel Not Equal to your input'

    # model
    # change resnet34 output to 50 (naive model)
    if args.model_name in ['dy_cnn', 'naive']:

        model = resnet34()
        model.fc = nn.Linear(512, 50)

        # training prefix parameter
        if args.model_name == 'dy_cnn': 
            
            model = PreFixResnet(args.use_channels,train=True)

    else:

        model = CustomModel()
        
    model.to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,patience=2)

 
    # record
    time_record = time.strftime('%m_%d_%H') if args.timestamp is None else args.timestamp
    save_path = f'./ckpt/{time_record}_{args.model_name}_{args.use_channels}.pt'
    earlystop = EarlyStopping(patience=args.patience, delta=args.delta, path=save_path)

    record = {
        'train' : {'loss':[], 'acc':[]},
        'test' : {'loss':[], 'acc':[]},
        'val' : {'loss':[], 'acc':[]},

        'save_path' : save_path,
        }



    for epoch in tqdm(range(args.epochs)):

        # train
        train_res = train(args, model, loss_func, optimizer, train_lodaer)
        # test 
        test_res = test(args, model, test_ds)
        # val
        val_res = test(args, model, val_ds)
        # Note that step should be called after validate()
        scheduler.step(val_res[1])


        # updata record dictionary
        update_record(record, train_res, test_res, val_res)
        print(f"Epoch {epoch}, Train Loss: {train_res[1]:.4f}, Train Acc: {train_res[0]:.4f}, Val Acc: {val_res[0]:.4f}, Test Acc: {test_res[0]:.4f}")

        # save model in this    
        earlystop(model, val_res[0])

        if earlystop.early_stop:            
            print("Early Stop")
            break


    # dump record to json file
    print(record)
    with open(f'./log/{time_record}_{args.model_name}_{args.use_channels}.json', 'w') as f:
        json.dump(record, f)


