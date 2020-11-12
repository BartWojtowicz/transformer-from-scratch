import torch
import torch.nn.functional as F
from torch import nn

from torch.autograd import Variable

from argparse import ArgumentParser
import random, tqdm, sys, math, gzip
import numpy as np


#   Each time I mention figure/equation in the comments, 
#   I'm refering to "Attention is All You Need" paper
#   https://arxiv.org/pdf/1706.03762.pdf


class SelfAttention(nn.Module):   

    def __init__(self, emb_dim, heads = 4, mask = False):
        super().__init__()
        
        self.emb_dim, self.heads = emb_dim, heads

        # dimension for each head
        # reduced dimension for each head -> computational cost similar to 1 full head
        s = emb_dim // heads

        self.tokeys = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues = nn.Linear(s, s, bias=False)

        # linear used for concatenated heads after attention
        # heads * s = emd_dim
        self.unifyheads = nn.Linear(heads * s, emb_dim)
        
    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        
        s = e // h
        # new dimension for each head
        x = x.view(b, t, h, s)
        
        queries = self.toqueries(x)
        keys    = self.tokeys(x)
        values  = self.tovalues(x)

        # equation 1 from paper
        dot = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys])
        att = torch.softmax(dot / (e ** (1/2)), dim = 3)

        # reshape is concatenating all heads
        out = torch.einsum("nhql, nlhd -> nqhd", [att, values]).reshape(b, t, s*h)
        
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    # todo: add dropout, 
    def __init__(self, emb, heads, hidden_multiplier = 4):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        # not sure why the hidden dimension is bigger yet
        self.ff = nn.Sequential(
            nn.Linear(emb, hidden_multiplier * emb),
            nn.ReLU(),
            nn.Linear(hidden_multiplier * emb, emb)
            )

    def forward(self, x):
        # same as in figure 1 - transformer on the left 
        attended = self.attention(x)
        x = self.norm1(attended + x)

        forwarded = self.ff(x)
        return self.norm2(forwarded + x)


class ClassificationTransformer(nn.Module):

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, emb)
        self.pos_emb = nn.Embedding(seq_length, emb)
        
        # stacked transformer blocks
        blocks = [TransformerBlock(emb = emb, heads = heads) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)
        
        # classification on top of transformer blocks (averaged transformer output sequence as input)
        self.toprobs = nn.Linear(emb, num_classes)
        
    def forward(self, x):
        b, seq_length = x.shape
        
        # using positional embedding, easier to implement than positional encoding
        positions = self.pos_emb(torch.arange(0, seq_length, device = 'cuda').expand(b, seq_length))
        tokens = self.token_emb(x)

        # entire forward pass with mean averaging the last output sequence
        x = tokens + positions
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.toprobs(x)
        
        return F.log_softmax(x, dim = 1)




    