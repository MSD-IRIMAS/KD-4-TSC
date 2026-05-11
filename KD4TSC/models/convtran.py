
import numpy as np
import torch.nn as nn
from models.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from models.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)
    
class ConvTran(nn.Module):
    def __init__(self, input_shape, num_heads, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        print(f"Initializing ConvTran model with input shape {input_shape}, num_heads {num_heads}, num_classes {num_classes}")
        self.channel_size, self.seq_len = 1, input_shape[0]
        self.emb_size = 24
        self.num_heads = num_heads
        self.dim_ff = 256
        self.fix_pos_encode = 'tAPE'
        self.Rel_pos_encode = 'eRPE'
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, self.emb_size, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(self.emb_size),
                                         nn.GELU())

        self.Fix_Position = tAPE(self.emb_size, dropout=0.01, max_len=self.seq_len)
        self.attention_layer = Attention_Rel_Scl(self.emb_size, num_heads, self.seq_len, 0.01)

        self.LayerNorm = nn.LayerNorm(self.emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(self.emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(self.emb_size, self.dim_ff),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(self.dim_ff, self.emb_size),
            nn.Dropout(0.01))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(self.emb_size, num_classes)

    def forward(self, x):
        # print(f"Step 0: Input shape: {x.shape}")
        x = x.unsqueeze(1)
        # print("Step 1: Input shape: ", x.shape)            
        x_src = self.embed_layer(x).squeeze(2)
        # print("Step 2: Input shape: ", x_src.shape)            
        x_src = x_src.permute(0, 2, 1)
        # print("Step 3: Input shape: ", x_src.shape)            
        
        x_src_pos = self.Fix_Position(x_src)
        # print("Step 4: Input shape: ", x_src_pos.shape)
        att = x_src + self.attention_layer(x_src_pos)
        # print("Step 5: Input shape: ", att.shape)
        
        att = self.LayerNorm(att)
        # print("Step 6: Input shape: ", att.shape)
        out = att + self.FeedForward(att)
        # print("Step 7: Input shape: ", out.shape)
        out = self.LayerNorm2(out)
        # print("Step 8: Input shape: ", out.shape)        
        out = out.permute(0, 2, 1)
        # print("Step 9: Input shape: ", out.shape)
        
        out = self.gap(out)
        # print("Step 10: Input shape: ", out.shape)
        
        out = self.flatten(out)
        # print("Step 11: Input shape: ", out.shape)
        
        out = self.out(out)
        # print("Step 12: Input shape: ", out.shape)
        
        return out

