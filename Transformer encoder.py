import torch
import torch.nn as nn
from mistune.plugins import math
import torch.nn.functional as F

d_model = 512
num_heads = 8
d_ff = 2048
drop_prob = 0.1
batch_size = 8
seq_len = 8
num_layers = 3

def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v = 30 x 8 x 200 x 64
    d_k = q.size()[-1]   #64
    scaled = torch.mstmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)   # 30 x 8 x 200 x 200 #by doing this sqrt we get scaled values
    if mask is not None:  #for encoder it's not necesary & basically it helps to present and past words not future assumption
        scaled += mask   # 30 x 8 x 200 x 200
    attention = F.softmax(scaled, dim=-1)  # 30 x 8 x 200 x 200 #softmax creates probability values for focus and it's combination of 1 for the all words
    values = torch.matmul(attention, v)   # 30 x 8 x 200 x 64
    return values, attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model           # 512 dimensions
        self.num_heads = num_heads       # 8
        self.head_dim = d_model // num_heads # 512/8 = 64
        self.qkv_layer = nn.Linear(d_model, d_model * 3)  # 512 x 1536
        self.linear_layer = nn.Linear(d_model, d_model)   # 512 x 512

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()  # 30 x 200 x 512
        qkv = self.qkv_layer(x)                  # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3)    # 30 x 8 x 200 x 192
        q, k, v = qkv.chunk(3, dim=-1)   # each are 30 x 8 x 200 x 64
        values, attention = scaled_dot_product_attention(q, k, v, mask)  #attention = 30 x 8 x 200 x 200  #value = 30 x 8 x 200 x 64
        values = values.reshape(batch_size, seq_len, self.num_heads * self.head_dim)  # 30 x 200 x 512
        out= self.linear_layer(values)
        return out

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.eps = eps   #preventing some operation resluts to became zero
        self.parameters_shape = parameters_shape   #512  # it's basically embedding dim and tell us along which layer we want normalize
        self.gamma = nn.Parameter(torch.ones(parameters_shape))  #512
        self.beta = nn.Parameter(torch.zeros(parameters_shape))  #512

    def forward(self, inputs):   # 30 x 200 x 512
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]  #[-1]  #it says, it is the last dim  along which layer we want layernorm
        mean = inputs.mean(dim=dims, keepdim=True)    # 30 x 200 x 1
        var = ((inputs-mean)**2).mean(dim=dims, keepdim=True)   # 30 x 200 x 1
        std = (var + self.eps).sqrt()  # 30 x 200 x 1
        y = (inputs - mean) / std     # 30 x 200 x 512
        out = self.gamma * y + self.beta  # 30 x 200 x 512
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # 512 x 2048
        self.linear2 = nn.Linear(d_ff, d_model)     # 2048 x 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):    # 30 x 200 x 512
        x = self.linear1(x)   # 30 x 200 x 2048
        x = self.relu(x)      # 30 x 200 x 2048
        x = self.dropout(x)   # 30 x 200 x 2048
        x = self.linear2(x)   # 30 x 200 x 512
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop_prob):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=drop_prob)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x                     # 30 x 200 x 512
        x = self.self_attn(x, mask=None)   # 30 x 200 x 512  #since we don't need to focus on future word so we put mask=none
        x = self.dropout1(x)               # 30 x 200 x 512
        x = self.norm1(x + residual_x)     # 30 x 200 x 512
        residual_x = x                     # 30 x 200 x 512
        x = self.ffn(x)                    # 30 x 200 x 512
        x = self.dropout2(x)               # 30 x 200 x 512
        x = self.norm2(x + residual_x)     # 30 x 200 x 512
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, d_ff, num_heads, drop_prob)
                                      for _ in range(num_layers)])
    def forward(self, x):
        x = self.layers(x)
        return x