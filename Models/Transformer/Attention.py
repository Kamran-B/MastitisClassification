import math
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calculates Scaled Dot-Product Attention.

    Args:
        Q (torch.Tensor): Queries. Shape: (..., seq_len_q, head_dim)
        K (torch.Tensor): Keys. Shape: (..., seq_len_k, head_dim)
        V (torch.Tensor): Values. Shape: (..., seq_len_v, head_dim)
                           Note: seq_len_k == seq_len_v
        mask (torch.Tensor, optional): Mask to apply before softmax.
                                       Shape: (..., seq_len_q, seq_len_k)
                                       Defaults to None.

    Returns:
        torch.Tensor: Output tensor. Shape: (..., seq_len_q, head_dim)
        torch.Tensor: Attention weights. Shape: (..., seq_len_q, seq_len_k)
    """
    head_dim = K.size(-1)
    # Use float for potentially large values before sqrt
    scaling_factor = math.sqrt(float(head_dim))

    # Calculate raw attention scores
    # (..., seq_len_q, head_dim) @ (..., head_dim, seq_len_k) -> (..., seq_len_q, seq_len_k)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1))

    # Scale scores
    scaled_attention_scores = attention_scores / scaling_factor

    # Apply mask BEFORE softmax (if provided)
    if mask is not None:
        # Ensure mask has compatible dimensions for broadcasting
        # Example mask shapes: (B, 1, S, S) or (B, N, S, S) or (1, 1, S, S)
        scaled_attention_scores = scaled_attention_scores.masked_fill(
            mask == 0, float('-inf') # Use float('-inf') for numerical stability
        )

    # Apply softmax over the key sequence length dimension
    attention_weights = F.softmax(scaled_attention_scores, dim=-1)

    # Optional dropout (can be added here or after MHA)
    # attention_weights = F.dropout(attention_weights, p=dropout_p)

    # Multiply weights by values
    # (..., seq_len_q, seq_len_k) @ (..., seq_len_v, head_dim) -> (..., seq_len_q, head_dim)
    # Note: seq_len_k == seq_len_v
    output = torch.matmul(attention_weights, V)

    return output, attention_weights # Also return weights for inspection


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
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embedding_dim = embed_dim
        self.num_heads = num_heads
        assert self.embedding_dim % self.num_heads == 0
        self.head_dim = self.embedding_dim // self.num_heads

        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Multi-Head Attention.

        Args:
            query (torch.Tensor): Query tensor. Shape: (B, S_q, E)
            key (torch.Tensor): Key tensor. Shape: (B, S_k, E)
            value (torch.Tensor): Value tensor. Shape: (B, S_v, E) (S_k == S_v)
            mask (torch.Tensor, optional): Mask for scaled dot-product attention.
                                           Shape should be broadcastable to (B, N, S_q, S_k).
                                           Defaults to None.

        Returns:
            torch.Tensor: Output tensor. Shape: (B, S_q, E)
        """
        b, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape
        _, seq_len_v, _ = value.shape # seq_len_k == seq_len_v

        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape and transpose for multi-head calculation
        # (B, S, E) -> (B, S, N, H) -> (B, N, S, H)
        Q = Q.view(b, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention output (pass mask)
        # output shape: (B, N, S_q, H), attn_weights shape: (B, N, S_q, S_k)
        output, _ = scaled_dot_product_attention(Q, K, V, mask=mask) # Pass mask here

        # Transpose back and reshape for output projection
        # (B, N, S_q, H) -> (B, S_q, N, H) -> (B, S_q, E)
        output = output.transpose(1, 2).contiguous().view(b, seq_len_q, self.embedding_dim)

        # Final linear projection
        final_output = self.out_proj(output)
        return final_output


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_encoder_layers: int,
                 ffn_dim: int,
                 num_classes: int, # Number of output classes
                 max_seq_len: int,
                 dropout: float,
                 pad_idx: int = 0):
        super().__init__()

        self.embed_dim = embed_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)

        # Encoder Stack
        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Classification Head
        # Takes the representation of the first token ([CLS] token equivalent)
        self.classification_head = nn.Linear(embed_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

        # Optional: Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Simple initialization example
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize classification head reasonably
        nn.init.normal_(self.classification_head.weight, std=0.02)
        nn.init.zeros_(self.classification_head.bias)


    def _create_padding_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        """Creates a mask for padding tokens.
           Output shape: (B, 1, 1, S) for MHA compatibility.
           Returns 1 where valid, 0 where padding.
        """
        mask = (sequence != self.pad_idx).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
        return mask.bool() # Ensure boolean type

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            src (torch.Tensor): Source sequence tensor. Shape: (B, S)

        Returns:
            torch.Tensor: Output logits for classification. Shape: (B, num_classes)
        """
        # 1. Create Padding Mask
        src_mask = self._create_padding_mask(src) # (B, 1, 1, S)

        # 2. Embeddings and Positional Encoding
        # src: (B, S) -> src_emb: (B, S, E)
        src_emb = self.embedding(src) * math.sqrt(self.embed_dim)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.dropout(src_emb)

        # 3. Pass through Encoder Stack
        memory = src_emb
        for encoder in self.encoders:
             memory = encoder(memory, src_mask=src_mask) # Pass mask

        # 4. Extract Representation for Classification
        # Use the output corresponding to the first token (e.g., [CLS])
        # memory shape: (B, S, E) -> cls_representation shape: (B, E)
        cls_representation = memory[:, 0, :] # Take the first time step

        # Alternative: Mean Pooling over non-padded tokens
        # if src_mask is not None:
        #     # Expand mask for broadcasting: (B, 1, 1, S) -> (B, S, 1)
        #     pool_mask = src_mask.squeeze(1).squeeze(1).unsqueeze(-1).float()
        #     # Sum embeddings where mask is 1
        #     summed_embeddings = torch.sum(memory * pool_mask, dim=1)
        #     # Count non-padded tokens (handle division by zero if seq is all padding)
        #     num_non_padded = pool_mask.sum(dim=1).clamp(min=1e-9)
        #     cls_representation = summed_embeddings / num_non_padded
        # else: # No mask, pool over all tokens
        #     cls_representation = torch.mean(memory, dim=1)


        # 5. Classification Head
        logits = self.classification_head(cls_representation) # (B, E) -> (B, num_classes)

        return logits


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.multi_head_attention = MultiHeadAttention(self.embed_dim, self.num_heads)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.linear1 = nn.Linear(self.embed_dim, self.ffn_dim)
        self.linear2 = nn.Linear(self.ffn_dim, self.embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        '''input x must be (batch_size, seq_len, embed_dim)
           src_mask shape broadcastable to (B, N, S, S)'''
        # 1. Multi-Head Attention + Residual + Norm
        # Pass mask to self-attention
        attn_output = self.multi_head_attention(query=x, key=x, value=x, mask=src_mask)
        x = self.layer_norm1(x + self.dropout(attn_output))

        # 2. Feed-Forward Network + Residual + Norm
        ffn_output = self.linear2(self.activation(self.linear1(x)))
        x = self.layer_norm2(x + self.dropout(ffn_output))

        return x

# No Decoder needed for classification tasks
'''class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # Masked Self-Attention (for target sequence)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Encoder-Decoder Attention (Cross-Attention)
        # Q from decoder, K/V from encoder output (memory)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.activation = nn.GELU() # Or nn.ReLU
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for a single Transformer Decoder Layer.

        Args:
            tgt (torch.Tensor): Target sequence tensor. Shape: (B, S_tgt, E)
            memory (torch.Tensor): Encoder output tensor. Shape: (B, S_src, E)
            tgt_mask (torch.Tensor, optional): Mask for target self-attention.
                                               Shape broadcastable to (B, N, S_tgt, S_tgt).
                                               Defaults to None.
            memory_mask (torch.Tensor, optional): Mask for encoder-decoder attention.
                                                  Shape broadcastable to (B, N, S_tgt, S_src).
                                                  Defaults to None.

        Returns:
            torch.Tensor: Output tensor. Shape: (B, S_tgt, E)
        """
        # 1. Masked Self-Attention (Input: tgt, tgt, tgt)
        # Apply attention to the target sequence, using the target mask
        self_attn_output = self.self_attn(query=tgt, key=tgt, value=tgt, mask=tgt_mask)
        # Residual connection and Layer Normalization
        x = self.norm1(tgt + self.dropout(self_attn_output))

        # 2. Encoder-Decoder Attention (Cross-Attention)
        # Query: from previous block (x)
        # Key/Value: from encoder output (memory)
        # Apply attention using the memory mask (masks padding in src)
        cross_attn_output = self.cross_attn(query=x, key=memory, value=memory, mask=memory_mask)
        # Residual connection and Layer Normalization
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. Feed-Forward Network
        ffn_output = self.linear2(self.activation(self.linear1(x)))
        # Residual connection and Layer Normalization
        x = self.norm3(x + self.dropout(ffn_output))

        return x'''
