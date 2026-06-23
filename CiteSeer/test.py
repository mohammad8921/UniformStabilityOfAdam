from numpy import inf
import numpy as np
from torch import nn
import torch
from torch.optim import Adam
from sklearn.metrics import f1_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def validate_loss_values(model, loss_fn, dataset):
    softmax_func = nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        data  = dataset[0].to(device)
        out = softmax_func(model(data))
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask]).item()
        val_loss = loss_fn(out[data.val_mask], data.y[data.val_mask]).item()
        test_loss = loss_fn(out[data.test_mask], data.y[data.test_mask]).item()
        preds = out.argmax(dim=1)
        print(f'Training loss: {loss:0.5f}')
        print(f'Accuracy on training set: {int((preds[data.train_mask]==data.y[data.train_mask]).sum())/int(data.train_mask.sum()):0.4f}')
        print('-'*10)
        print(f'Validation loss: {val_loss:0.5f}')
        print(f'Accuracy on validation set: {int((preds[data.val_mask]==data.y[data.val_mask]).sum())/int(data.val_mask.sum()):0.4f}')
        print('-'*10)
        print(f'Test loss: {test_loss:0.5f}')
        print(f'Accuracy on training set: {int((preds[data.test_mask]==data.y[data.test_mask]).sum())/int(data.test_mask.sum()):0.4f}')
        

def acc_f1(model, loss_fn, dataset):
    softmax_func = nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        data  = dataset[0].to(device)
        out = softmax_func(model(data))
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask]).item()
        val_loss = loss_fn(out[data.val_mask], data.y[data.val_mask]).item()
        test_loss = loss_fn(out[data.test_mask], data.y[data.test_mask]).item()
        preds = out.argmax(dim=1)
        preds_val = preds[data.val_mask].detach().cpu().numpy()
        preds_test = preds[data.test_mask].detach().cpu().numpy()
        data_y_val = data.y[data.val_mask].detach().cpu().numpy()
        data_y_test = data.y[data.test_mask].detach().cpu().numpy()
        print(f'Training loss: {loss:0.5f}')
        print(f'Accuracy on training set: {int((preds[data.train_mask]==data.y[data.train_mask]).sum())/int(data.train_mask.sum()):0.4f}')
        print('-'*10)
        print(f'Validation loss: {val_loss:0.5f}')
        print(f'Accuracy on validation set: {int((preds[data.val_mask]==data.y[data.val_mask]).sum())/int(data.val_mask.sum()):0.4f}')
        print(f'F1 on val set: {f1_score(data_y_val, preds_val, average="macro"):0.4f}')
        print('-'*10)
        print(f'Test loss: {test_loss:0.5f}')
        print(f'Accuracy on training set: {int((preds[data.test_mask]==data.y[data.test_mask]).sum())/int(data.test_mask.sum()):0.4f}')
        print(f'F1 on test set: {f1_score(data_y_test, preds_test, average="macro"):0.4f}')