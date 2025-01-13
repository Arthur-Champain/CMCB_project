import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_max_pool, global_mean_pool, BatchNorm
from torch_geometric.nn import CGConv, GATv2Conv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, node_dim, edge_dim, output):
        super(GAT, self).__init__()
        torch.manual_seed(12345)

        self.project = Linear(node_dim, hidden_channels)

        # Define GAT layers
        self.conv1 = GATv2Conv(node_dim, hidden_channels, edge_dim=edge_dim)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = GATv2Conv(hidden_channels, int(hidden_channels * 1), edge_dim=edge_dim)
        self.bn2 = BatchNorm(int(hidden_channels * 1))

        self.conv3 = GATv2Conv(int(hidden_channels * 1), hidden_channels*1, edge_dim=edge_dim)
        self.bn3 = BatchNorm(hidden_channels*1)

        self.conv4 = GATv2Conv(hidden_channels*1, int(hidden_channels * 1), edge_dim=edge_dim)
        self.bn4 = BatchNorm(int(hidden_channels * 1))

        self.conv5 = GATv2Conv(int(hidden_channels * 1), int(hidden_channels * 1), edge_dim=edge_dim)
        self.bn5 = BatchNorm(int(hidden_channels * 1))

    
        # Final output layer
        self.lin1 = Linear(int(hidden_channels * 1), output)
        self.lin2 = Linear(int(hidden_channels * 1), output)

    def forward(self, batch, x, edge_index, edge_attr): 
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.conv4(x, edge_index, edge_attr)
        x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.conv5(x, edge_index, edge_attr)
        x = self.bn5(x)
        x = F.relu(x)
    
        # Readout layer
        x = global_mean_pool(x, batch)  
        
        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        pred1 = self.lin1(x) #length
        pred2 = self.lin2(x)#curv
        
        return pred1, pred2

def setup(model_type, node_dim, edge_dim, hidden_channels):

    if model_type == 'GAT':
        model = GAT(hidden_channels, node_dim, edge_dim, output=1)
    else:
        raise ValueError('Model type not recognized.')

    return model




