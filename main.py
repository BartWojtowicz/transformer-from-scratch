import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchtext import data, datasets, vocab

from argparse import ArgumentParser
import random, tqdm, sys, math, gzip
import numpy as np


class SelfAttention(nn.Module):
    
    def __init__(self, emb_dim, heads = 4, mask = False):
        super().__init__()
        
        self.emb_dim, self.heads = emb_dim, heads
        
        s = emb_dim // heads

        self.tokeys = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues = nn.Linear(s, s, bias=False)

        self.unifyheads = nn.Linear(heads * s, emb_dim)
        
    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        
        s = e // h
        x = x.view(b, t, h, s)
        
        queries = self.toqueries(x)
        keys    = self.tokeys(x)
        values  = self.tovalues(x)
        
        # Einsum FTW 

        # keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        # queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        # values = values.transpose(1, 2).contiguous().view(b * h, t, s)        
        # queries = queries / (e ** (1/4))
        # keys    = keys / (e ** (1/4))        
        # dot = torch.bmm(queries, keys.transpose(1, 2))
        # dot = F.softmax(dot, dim=2)

        dot = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys])
        att = torch.softmax(dot / (e ** (1/2)), dim = 3)
        
        # out = torch.bmm(dot, values).view(b, h, t, s)
        # out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        out = torch.einsum("nhql, nlhd -> nqhd", [att, values]).reshape(b, t, s*h)
        
        return self.unifyheads(out)