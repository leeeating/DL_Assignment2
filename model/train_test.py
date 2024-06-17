import os

import torch
import torch.nn.functional as F


def train(args, model, loss_func, optim, train_loader):

    model.train()

    acc_cnt = 0
    epoch_loss = 0
    for img_batch, lab_batch in train_loader:

        img_batch = img_batch.to(args.device)
        lab_batch = lab_batch.to(args.device)

        y_score = model(img_batch)

        loss_batch = loss_func(y_score, lab_batch)

        y_pred = torch.argmax(y_score,dim=1)
        acc_cnt += sum(y_pred == lab_batch)
        epoch_loss += loss_batch

        optim.zero_grad()
        loss_batch.backward()
        optim.step()
        
            
    train_acc = (acc_cnt / len (train_loader.dataset)).item()
    train_loss = (epoch_loss / len (train_loader.dataset)).item()

    return train_acc, train_loss 


# regular test
def test(args, model, dataset):

    model.eval()
    with torch.no_grad():

        correct = 0
        total_loss = 0
        for img, lab in dataset:

            img = img.unsqueeze(0).to(args.device)
            lab = lab.unsqueeze(0).to(args.device)

            y_score = model(img)
            y_pred = torch.argmax(y_score,dim=1)

            if y_pred == lab:
                correct += 1  
            
            loss = F.cross_entropy(y_score, lab).item()
            total_loss += loss

        accuracy = correct / len(dataset)
        loss = total_loss / len(dataset)

    return accuracy, loss



class EarlyStopping:

    def __init__(self, patience, delta, path, verbose=True):
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): The number of epochs to wait after the last time validation accuracy improved.
                            If validation accuracy does not improve for 'patience' consecutive epochs, early stopping will be triggered.
            delta (float): The minimum change in the monitored quantity (validation accuracy) to qualify as an improvement.
            path (str): The directory path to save the model checkpoints.
            verbose (bool, optional): If True, prints a message for each early stopping count. Default is True.
        """

        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose

        self.cnt = 0
        self.best_val_acc = None
        self.early_stop = False

    def __call__(self, model, val_acc):

        # init case
        if self.best_val_acc is None:
            
            self.best_val_acc = val_acc
            self.save_model(model, val_acc)
        
        # worse case
        elif val_acc < self.best_val_acc + self.delta:
            self.cnt += 1

            if self.verbose:
                print(f'EarlyStopping Count:{self.cnt}/{self.patience}')
            if  self.cnt >= self.patience:
                self.early_stop = True

        # better cace
        else:
            self.save_model(model, val_acc)
            self.cnt = 0
            self.best_val_acc = val_acc

    def save_model(self, model,val_acc):

        # self.path = f'./ckpt/{time_record}_{args.model_name}.pt'
        if self.verbose:
            print(f'Validation Accuracy increased: {self.best_val_acc}->{val_acc}')

        torch.save(model.state_dict(), self.path)
