from numpy import inf
import numpy as np
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, Sequential, GraphSAGE, GAT


class GCN(nn.Module):
    def __init__(self, num_gcn_layers, num_hidden_neurons, num_node_features, num_classes):
        super().__init__()
        self.num_gcn_layers = num_gcn_layers
        self.gcn_layers = []
        self.gcn_layers.append((GCNConv(num_node_features, num_hidden_neurons), 'x, edge_index -> x'))
        self.gcn_layers.append(nn.ReLU(inplace=True))
        #self.gcn_layers.append(nn.Dropout(0.2))

        for i in range(self.num_gcn_layers-1):
            self.gcn_layers.append((GCNConv(num_hidden_neurons, num_hidden_neurons), 'x, edge_index -> x'))
            self.gcn_layers.append(nn.ReLU(inplace=True))
            #self.gcn_layers.append(nn.Dropout(0.2))
        self.gcn_layers.append(nn.Linear(num_hidden_neurons, num_classes))  

        self.gcn_sequential = Sequential('x, edge_index', self.gcn_layers)
    
    def forward(self, data):
        x = self.gcn_sequential(data.x, data.edge_index)
        return x
    
class GraphSAGEModel(nn.Module):
    def __init__(self, num_hidden_neurons, num_layers, num_node_features, num_classes):
        super().__init__()
        self.graphsage = GraphSAGE(num_node_features, num_hidden_neurons, num_out_channels=num_hidden_neurons ,num_layers=num_layers)
        self.linear = nn.Linear(num_hidden_neurons, num_classes)
    
    def forward(self, data):
        return self.linear(self.graphsage(data.x, data.edge_index))

class GATModel(nn.Module):
    def __init__(self, num_hidden_neurons, num_layers, num_node_features, num_classes):
        super().__init__()
        self.gat = GAT(num_node_features, num_hidden_neurons, num_out_channels=num_classes ,num_layers=num_layers)
        
    def forward(self, data):
        return self.gat(data.x, data.edge_index)
