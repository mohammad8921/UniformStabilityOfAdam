from numpy import inf
import numpy as np
from torch import nn
import torch
from torch.optim import Adam
from models import GCN, GraphSAGEModel, GATModel
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init(random_seed): 
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
def train(type, epochs, lr, num_layers, num_hidden_neurons, loss_fn, dataset):
    random_seed = 49
    init(random_seed)
    
    if type=='gcn':
        model = GCN(num_gcn_layers=num_layers, num_hidden_neurons=num_hidden_neurons, num_node_features=dataset.num_node_features, num_classes=dataset.num_classes).to(device)
    elif type=='sage':
        model=GraphSAGEModel(num_hidden_neurons, num_layers=num_layers, num_node_features=dataset.num_node_features, num_classes=dataset.num_classes).to(device)
    elif type=='gat':
        model=GATModel(num_hidden_neurons, num_layers=num_layers, num_node_features=dataset.num_node_features, num_classes=dataset.num_classes).to(device)   
        
    optimizer = Adam(model.parameters(), lr=lr)
    softmax_func = nn.Softmax(dim=1)
    current = 1
    history = {'train_loss' : [], 'val_loss' : [], 'train_acc': [], 'val_acc' : []}
    val_min_loss = inf
    best_model = None
    model.train()
    while current <= epochs:
        print(f'Epoch {current}/{epochs}:')
        model.zero_grad()
        data  = dataset[0].to(device)
        out = softmax_func(model(data))
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        val_loss = loss_fn(out[data.val_mask], data.y[data.val_mask])
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        preds = out.argmax(dim=1)
        history['train_acc'].append(int((preds[data.train_mask]==data.y[data.train_mask]).sum())/int(data.train_mask.sum()))
        history['val_acc'].append(int((preds[data.val_mask]==data.y[data.val_mask]).sum())/int(data.val_mask.sum()))
        print(f'Traning loss: {loss.item():0.5f}')
        print(f'Validation loss: {val_loss.item():0.5f}')
        print(10*'-')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if val_loss.item() < val_min_loss:
            val_min_loss = val_loss.item()
            best_model = copy.deepcopy(model)
        current += 1    
    return best_model, history

