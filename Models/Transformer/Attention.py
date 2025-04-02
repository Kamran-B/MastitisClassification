import math
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn


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

import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as described in "Attention Is All You Need".
    Registered as a buffer for efficiency and device compatibility.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model (int): Dimension of the model embeddings.
            max_len (int): Maximum possible sequence length.
        """
        super().__init__()
        self.d_model = d_model

        # Create the positional encoding matrix (once)
        position = torch.arange(max_len).unsqueeze(1) # Shape: (max_len, 1)
        # Calculate the division term for the frequencies
        # Use float for division term calculation to avoid potential truncation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # Shape: (d_model / 2)

        # Initialize PE matrix
        pe = torch.zeros(max_len, d_model) # Shape: (max_len, d_model)

        # Apply sin to even indices; cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register 'pe' as a buffer. Add a batch dimension (1, max_len, d_model)
        # so it can be easily added to batched input (B, S, E) via broadcasting,
        # although adding (S, E) to (B, S, E) also works.
        # We slice it later in forward, so (max_len, d_model) is fine too.
        # Let's stick to (max_len, d_model) for simplicity matching the paper.
        self.register_buffer('pe', pe) # Shape: (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor with positional encoding added,
                          shape (batch_size, seq_len, d_model)
        """
        # x shape: (batch_size, seq_len, d_model)
        # self.pe shape: (max_len, d_model)
        # We need the positional encoding for the actual sequence length of x
        # self.pe[:x.size(1), :] gives shape (seq_len, d_model)
        # Adding requires tensors to be on the same device, which register_buffer handles.
        # Broadcasting adds the (seq_len, d_model) PE matrix to each item in the batch.
        output = x + self.pe[:x.size(1), :]
        return output




class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads): # Removed dropout for simplicity here
        super().__init__() # <-- Added
        self.embedding_dim = embed_dim
        self.num_heads = num_heads
        assert self.embedding_dim % self.num_heads == 0
        self.head_dim = self.embedding_dim // self.num_heads # <-- Integer division

        # Use nn.Linear for projections - handles weights, biases, initialization
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        # If you MUST use manual weights:
        # self.Wq_all = nn.Parameter(torch.empty(self.embedding_dim, self.embedding_dim))
        # self.Wk_all = nn.Parameter(torch.empty(self.embedding_dim, self.embedding_dim))
        # self.Wv_all = nn.Parameter(torch.empty(self.embedding_dim, self.embedding_dim))
        # self.Wo = nn.Parameter(torch.empty(self.embedding_dim, self.embedding_dim))
        # nn.init.xavier_uniform_(self.Wq_all) # Initialize using PyTorch functions
        # nn.init.xavier_uniform_(self.Wk_all)
        # nn.init.xavier_uniform_(self.Wv_all)
        # nn.init.xavier_uniform_(self.Wo)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume x is (batch_size, seq_len, embedding_dim)
        b, seq_len, _ = x.shape

        # Project using nn.Linear layers
        Q = self.q_proj(x) # shape (B, seq_len, embedding_dim)
        K = self.k_proj(x) # shape (B, seq_len, embedding_dim)
        V = self.v_proj(x) # shape (B, seq_len, embedding_dim)

        # Reshape and transpose for multi-head calculation
        # (B, S, E) -> (B, S, N, H) -> (B, N, S, H)
        # N = num_heads, H = head_dim
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention output
        # output shape: (B, N, S, H)
        output = scaled_dot_product_attention(Q, K, V) # Pass head_dim if needed

        # Transpose back and reshape for output projection
        # (B, N, S, H) -> (B, S, N, H) -> (B, S, E)
        output = output.transpose(1, 2).contiguous().view(b, seq_len, self.embedding_dim)

        # Final linear projection
        final_output = self.out_proj(output)
        return final_output



class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_encoder_layers, ffn_dim, max_seq_len=500, dropout=0.1): # Added params
        super().__init__() # <-- Added
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Use the revised PositionalEncoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        # Pass necessary args to TransformerEncoder
        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.dropout = nn.Dropout(dropout) # Dropout after embedding + PE

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Input tensor (batch_size, seq_len)

        Returns:
            torch.Tensor: Output tensor from the final encoder layer
                          (batch_size, seq_len, embed_dim)
        """
        # 1. Embedding
        # src: (B, S) -> x: (B, S, E)
        x = self.embedding(src) * math.sqrt(self.embed_dim) # Scaling often helps

        # 2. Positional Encoding
        x = self.positional_encoding(x)
        x = self.dropout(x) # Apply dropout after adding PE

        # 3. Encoder Layers
        for encoder in self.encoders:
            x = encoder(x)

        return x # Shape: (B, S, E)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1): # Added params
        super().__init__() # <-- Added
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim # Use passed param

        # Use the revised MultiHeadAttention
        self.multi_head_attention = MultiHeadAttention(self.embed_dim, self.num_heads)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim) # Renamed for clarity
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.linear1 = nn.Linear(self.embed_dim, self.ffn_dim) # Renamed
        self.linear2 = nn.Linear(self.ffn_dim, self.embed_dim) # Renamed
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) # Use passed param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''input x must be (batch_size, seq_len, embed_dim)'''
        # 1. Multi-Head Attention + Residual + Norm
        attn_output = self.multi_head_attention(x)
        x = self.layer_norm1(x + self.dropout(attn_output)) # Apply dropout within residual

        # 2. Feed-Forward Network + Residual + Norm
        ffn_output = self.linear2(self.activation(self.linear1(x)))
        x = self.layer_norm2(x + self.dropout(ffn_output)) # Apply dropout within residual

        return x




#class TransformerDecoder(nn.Module):
