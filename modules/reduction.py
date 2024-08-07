import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class SelfAttention_projector(nn.Module):
    def __init__(self, input_dim, seuqence_lenth, output_dim):
        super(SelfAttention_projector, self).__init__()
        self.input_dim = input_dim
        self.seuqence_lenth = seuqence_lenth
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

        self.projector = nn.Linear(input_dim*seuqence_lenth, output_dim)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        # print('weighted.shape',weighted.shape)
        batch_size = x.shape[0]
        embedding = weighted.view(batch_size, -1)
        # print('embedding.shape',embedding.shape)
        out = self.projector(embedding)

        return out

class SelfAttention(nn.Module):
    def __init__(self, input_dim, seuqence_lenth):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.seuqence_lenth = seuqence_lenth
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        # print('weighted.shape',weighted.shape)
        batch_size = x.shape[0]
        embedding = weighted.view(batch_size, -1)

        return embedding


