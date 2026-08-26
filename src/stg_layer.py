import torch
import torch.nn as nn

class TemporalGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, adjacency_matrix):
        super(TemporalGraphConvolution, self).__init__()
        self.register_buffer("A", adjacency_matrix)
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(support, self.A)
        return torch.relu(output)