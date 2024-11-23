import torch
import torch.nn as nn
from mistune.plugins import math
import torch.nn.functional as F

d_model = 512
num_heads = 8
ffn_hidden = 2048
drop_prob = 0.1
batch_size = 8
max_seq_len = 200
num_layers = 5


x = torch.randn(batch_size, max_seq_len, d_model)  #english sentence positional encoded
y = torch.randn(batch_size, max_seq_len, d_model)  #desired language positional encoded
mask = torch.full([max_seq_len, max_seq_len], float('-inf'))
mask = torch.triu(mask, diagonal=1)
decoder = Decoder(d_model, num_heads, ffn_hidden, drop_prob, num_layers)
out = decoder(x, y, mask)


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
        out = self.linear_layer(values)
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

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, drop_prob, num_layers):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = nn.MultiheadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden_dim=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask):
        _y = y
        print('masked self attention')
        y = self.self_attention(y, mask=decoder_mask)
        print('DropOut 1')
        y = self.dropout1(y)
        print('add + layer normalization')
        y = self.norm1(y + _y)

        _y = y
        y =  self.encoder_decoder_attention(x, y, mask=None)
        y = self.dropout2(y)
        y = self.norm2(y + _y)

        _y = y
        y = self.ffn(_y)
        y = self.dropout3(y)
        y = self.norm3(y + _y)
        return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)
        return y
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, num_heads, ffn_hidden, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, mask):
        y = self.layers(x, y, mask)
        return y