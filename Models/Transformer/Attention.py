import math
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torchvision.models.video.mvit import PositionalEncoding


def xavier_2D_init(in_dim, out_dim):
    variance = 1. / (in_dim + out_dim)
    return np.random.uniform(-variance, variance, (in_dim, out_dim))


def scaled_dot_product_attention(Q, K, V):
    head_dim = K.size(-1)
    scaling_factor = math.sqrt(head_dim)

    attention_scores = torch.matmul(Q, K.transpose(-2, -1))

    scaled_attention_scores = attention_scores / scaling_factor

    # Apply softmax over the key sequence length dimension (last dim)
    attention_weights = F.softmax(scaled_attention_scores, dim=-1)

    # attention_weights = F.dropout(attention_weights, p=dropout_p)

    output = torch.matmul(attention_weights, V)

    return output

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, input):
        pos = torch.arange(self.seq_len)[:, np.newaxis]
        two_i = torch.arange(0, self.d_model, 2)
        exp = two_i / self.d_model
        div_term = np.power(10000, exp)
        angles = pos / div_term
        positional_encoding = np.zeros((self.seq_len, self.d_model))
        positional_encoding[:, 0::2] = np.sin(angles)
        positional_encoding[:, 1::2] = np.cos(angles)
        output = input + positional_encoding
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        self.embedding_dim = embed_dim
        self.num_heads = num_heads
        assert self.embedding_dim % self.num_heads == 0
        self.head_dim = self.embedding_dim / self.num_heads
        self.Wq_all = xavier_2D_init(self.embedding_dim, self.embedding_dim)
        self.Wk_all = xavier_2D_init(self.embedding_dim, self.embedding_dim)
        self.Wv_all = xavier_2D_init(self.embedding_dim, self.embedding_dim)
        self.Wo = xavier_2D_init(self.embedding_dim, self.embedding_dim)

    def forward(self, input):
        # Assume input is (batch_size, seq_len, embedding_dim)
        b = input.size(0)
        seq_len = input.size(1)
        Q_all = input @ self.Wq_all # shape (B, seq_len, embedding_dim)
        K_all = input @ self.Wk_all # shape (B, seq_len, embedding_dim)
        V_all = input @ self.Wv_all # shape (B, S, embedding_dim)

        # Reshape to expose the num heads vs head dimension
        Q_all = Q_all.reshape(b, seq_len, self.num_heads, self.head_dim)
        K_all = K_all.reshape(b, seq_len, self.num_heads, self.head_dim)
        V_all = V_all.reshape(b, seq_len, self.num_heads, self.head_dim)

        Q_split = Q_all.transpose(0, 2, 1, 3) # now (batch, num_heads, seq_len, head_dim)
        K_split = K_all.transpose(0, 2, 1, 3)
        V_split = V_all.transpose(0, 2, 1, 3)

        output = scaled_dot_product_attention(Q_split, K_split, V_split)

        contiguous_tensor = output.transpose(0, 2, 1, 3).copy()
        concatenated_tensor = contiguous_tensor.reshape(b, seq_len, -1)
        final_output = concatenated_tensor @ self.Wo
        return final_output

class TransformerEncoder(nn.Module):
    def __init__(self):
        self.embed_dim = 16 # also called d_model
        self.num_heads = 4
        self.ffn_dim = 4 * self.embed_dim
        self.dropout = nn.Dropout(0.2)
        self.multi_head_attention = MultiHeadAttention(self.embed_dim, self.num_heads)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.linear = nn.Linear(self.embed_dim, self.ffn_dim)
        self.linear2 = nn.Linear(self.ffn_dim, self.embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        '''input must be (batch_size, seq_len, embed_dim)'''
        output = self.multi_head_attention(input)
        output_dropout = self.dropout(output)
        output_layer = self.layer_norm(output_dropout + input)
        output = self.linear(output_layer)
        output = self.activation(output)
        output = self.linear2(output)
        output_dropout2 = self.dropout(output)
        output_layer2 = self.layer_norm2(output_layer + output_dropout2)
        return output_layer2

class Transformer(nn.Module):
    def __init__(self, seq_len, batch_size):
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.seq_len = 512
        self.batch_size = 8
        self.vocab_size = 3
        self.positional_encoding = PositionalEncoding(self.batch_size, self.seq_len, self.embed_dim)
        self.embed_dim = 16 # also called d_model
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.encoders = nn.ModuleList([TransformerEncoder() for i in range(self.num_encoder_layers)])

#class TransformerDecoder(nn.Module):
