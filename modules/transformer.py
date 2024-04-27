# References:
    # https://github.com/facebookresearch/detr/blob/main/models/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange


class SinePositionalEncoding(nn.Module):
    """
    "For both spatial coordinates of each embedding we independently use
    $\frac{d}{2}$ sine and cosine functions with different frequencies. We then
    concatenate them to get the final $d$ channel positional encoding.
    """
    def __init__(self, embed_dim, max_len=2 ** 12):
        super().__init__()

        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(embed_dim // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / embed_dim))

        self.pe_mat = torch.zeros(size=(max_len, embed_dim))
        self.pe_mat[:, 0:: 2] = torch.sin(angle)
        self.pe_mat[:, 1:: 2] = torch.cos(angle)
        self.register_buffer("pos_enc_mat", self.pe_mat)

    def forward(self, x):
        return x + einops.repeat(
            self.pe_mat.to(x.device), pattern="l d -> b l d", b=x.size(0),
        )[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, width, num_heads, drop_prob):
        super().__init__()
    
        self.num_heads = num_heads

        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.to_multi_heads = Rearrange("b i (n h) -> b i n h", n=num_heads)
        self.scale = width ** (-0.5)
        self.attn_drop = nn.Dropout(drop_prob)
        self.to_one_head = Rearrange("b i n h -> b i (n h)")
        self.out_proj = nn.Linear(width, width, bias=False)

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_score = torch.einsum(
            "binh,bjnh->bnij", self.to_multi_heads(q), self.to_multi_heads(k),
        ) * self.scale
        attn_weight = F.softmax(attn_score, dim=-1)
        x = self.to_one_head(
            torch.einsum(
                "bnij,bjnh->binh",
                self.attn_drop(attn_weight),
                self.to_multi_heads(v),
            )
        )
        x = self.out_proj(x)
        return x, attn_weight


class FFN(nn.Module):
    def __init__(self, width, mlp_width, drop_prob):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(width, mlp_width),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(mlp_width, width),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualConnection(nn.Module):
    def __init__(self, fn, width, drop_prob):
        super().__init__()

        self.fn = fn
        self.res_drop = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(width)

    def forward(self, skip, **kwargs):
        x = self.fn(**kwargs)
        x = self.res_drop(x)
        x += skip
        x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    """
    "Image features from the CNN backbone are passed through the transformer
    encoder, together with spatial positional encoding that are added to queries
    and keys at every multi-head self-attention layer."
    "Additive dropout of 0.1 is applied after every multi-head attention and FFN
    before layer normalization."
    "The weights are randomly initialized with Xavier
    initialization.
    """
    def __init__(
        self,
        width,
        num_heads,
        mlp_width,
        drop_prob,
    ):
        super().__init__()

        self.spatial_pos_enc = SinePositionalEncoding(embed_dim=width)
        self.self_attn = MultiHeadAttention(
            width=width, num_heads=num_heads, drop_prob=drop_prob,
        )
        self.ffn = FFN(
            width=width, mlp_width=mlp_width, drop_prob=drop_prob,
        )

        self.self_attn_res_conn = ResidualConnection(
            fn=lambda x: self.self_attn(
                q=self.spatial_pos_enc(x), k=self.spatial_pos_enc(x), v=x,
            )[0],
            width=width,
            drop_prob=drop_prob,
        )
        self.ffn_res_conn = ResidualConnection(
            fn=self.ffn,
            width=width,
            drop_prob=drop_prob,
        )

    def forward(self, x):
        x = self.self_attn_res_conn(skip=x, x=x)
        return self.ffn_res_conn(skip=x, x=x)


class Encoder(nn.Module):
    def __init__(
        self,
        num_heads,
        width,
        num_layers,
        drop_prob,
        mlp_width=None,
    ):
        super().__init__()

        self.mlp_width = mlp_width if mlp_width is not None else width * 4

        self.enc_stack = nn.ModuleList(
            [
                EncoderLayer(
                    num_heads=num_heads,
                    width=width,
                    mlp_width=self.mlp_width,
                    drop_prob=drop_prob,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for enc_layer in self.enc_stack:
            x = enc_layer(x)
        return x


class DecoderLayer(nn.Module):
    """
    "The decoder receives queries (initially set to zero), output positional
    encoding (object queries), and encoder memory, and produces the final set
    of predicted class labels and bounding boxes through multiple multi-head
    self-attention and decoder-encoder attention."
    "These input embeddings are learnt positional encodings that we refer to as
    'object queries', and similarly to the encoder, we add them to the input of
    each attention layer. The $N$ object queries are transformed into an output
    embedding by the decoder."
    TODO: "We add prediction FFNs and Hungarian loss after each decoder layer.
    All predictions FFNs share their parameters. We use an additional shared
    layer-norm to normalize the input to the prediction FFNs from different
    decoder layers."
    """
    def __init__(
        self,
        num_heads,
        width,
        mlp_width,
        drop_prob,
    ):
        super().__init__()

        self.spatial_pos_enc = SinePositionalEncoding(embed_dim=width)
        self.self_attn = MultiHeadAttention(
            width=width, num_heads=num_heads, drop_prob=drop_prob,
        )
        self.enc_dec_attn = MultiHeadAttention(
            width=width, num_heads=num_heads, drop_prob=drop_prob,
        )
        self.ffn = FFN(
            width=width,
            mlp_width=mlp_width,
            drop_prob=drop_prob,
        )

        self.self_attn_res_conn = ResidualConnection(
            fn=lambda x, out_pos_enc: self.self_attn(
                q=x + out_pos_enc, k=x + out_pos_enc, v=x)[0],
            width=width,
            drop_prob=drop_prob,
        )
        self.enc_dec_attn_res_conn = ResidualConnection(
            fn=lambda x, enc_mem, out_pos_enc: self.enc_dec_attn(
                q=x + out_pos_enc, k=self.spatial_pos_enc(enc_mem), v=enc_mem,
            )[0],
            width=width,
            drop_prob=drop_prob,
        )
        self.ffn_res_conn = ResidualConnection(
            fn=self.ffn, width=width, drop_prob=drop_prob,
        )

    def forward(self, query, enc_mem, out_pos_enc):
        x = self.self_attn_res_conn(
            skip=query, x=query, out_pos_enc=out_pos_enc,
        )
        x = self.enc_dec_attn_res_conn(
            skip=x, x=x, enc_mem=enc_mem, out_pos_enc=out_pos_enc,
        )
        x = self.ffn_res_conn(skip=x, x=x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads,
        width,
        num_layers,
        drop_prob,
        mlp_width=None,
    ):
        super().__init__()

        self.mlp_width = mlp_width if mlp_width is not None else width * 4

        self.dec_stack = nn.ModuleList(
            [
                DecoderLayer(
                    num_heads=num_heads,
                    width=width,
                    mlp_width=self.mlp_width,
                    drop_prob=drop_prob,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, query, enc_mem, out_pos_enc):
        for dec_layer in self.dec_stack:
            query = dec_layer(
                query=query, enc_mem=enc_mem, out_pos_enc=out_pos_enc,
            )
        return query


class Transformer(nn.Module):
    """
    "All transformer weights are initialized with Xavier init."
    "The transformer is trained with default dropout of 0.1."
    """
    def __init__(
        self,
        width=512,
        num_encoder_heads=8,
        num_encoder_layers=6,
        num_decoder_heads=8,
        num_decoder_layers=6,
        drop_prob=0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            num_heads=num_encoder_heads,
            num_layers=num_encoder_layers,
            width=width,
            drop_prob=drop_prob,
        )
        self.decoder = Decoder(
            num_heads=num_decoder_heads,
            num_layers=num_decoder_layers,
            width=width,
            drop_prob=drop_prob,
        )

    def forward(self, image_feat, query, out_pos_enc):
        enc_mem = self.encoder(image_feat)
        return self.decoder(
            query=query, enc_mem=enc_mem, out_pos_enc=out_pos_enc,
        )
