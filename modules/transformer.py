# References:
    # https://github.com/facebookresearch/detr/blob/main/models/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


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
        self.head_dim = width // num_heads

        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.scale = width ** (-0.5)
        self.attn_drop = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(width, width, bias=False)

    def forward(self, q, k, v):
        b, i, _ = q.shape
        _, j, _ = k.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(b, self.num_heads, i, self.head_dim)
        k = k.view(b, self.num_heads, j, self.head_dim)
        v = v.view(b, self.num_heads, j, self.head_dim)

        attn_score = torch.einsum("bnid,bnjd->bnij", q, k) * self.scale
        attn_weight = F.softmax(attn_score, dim=3)

        attn_weight_drop = self.attn_drop(attn_weight)
        x = torch.einsum("bnij,bnjd->bnid", attn_weight_drop, v)
        x = einops.rearrange(x, pattern="b n i d -> b i (n d)")

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

        self.num_heads = num_heads
        self.width = width
        self.mlp_width = mlp_width

        self.spatial_pos_enc = SinePositionalEncoding(embed_dim=width)
        self.self_attn = MultiHeadAttention(
            width=width, num_heads=num_heads, drop_prob=drop_prob,
        )
        self.self_attn_drop = nn.Dropout(drop_prob)
        self.self_attn_norm = nn.LayerNorm(width)
        self.ffn = FFN(
            width=width, mlp_width=mlp_width, drop_prob=drop_prob,
        )
        self.ffn_drop = nn.Dropout(drop_prob)
        self.ffn_norm = nn.LayerNorm(width)

    def forward(self, x):
        skip = x
        x, _ = self.self_attn(
            q=self.spatial_pos_enc(x), k=self.spatial_pos_enc(x), v=x,
        )
        x = self.self_attn_drop(x)
        x = self.self_attn_norm(x + skip)
        
        skip = x
        x = self.ffn(x)
        x = self.ffn_drop(x)
        x = self.ffn_norm(x + skip)
        return x


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
        self.self_attn_drop = nn.Dropout(drop_prob)
        self.self_attn_norm = nn.LayerNorm(width)
        self.enc_dec_attn = MultiHeadAttention(
            width=width, num_heads=num_heads, drop_prob=drop_prob,
        )
        self.enc_dec_attn_drop = nn.Dropout(drop_prob)
        self.enc_dec_attn_norm = nn.LayerNorm(width)
        self.ffn = FFN(
            width=width,
            mlp_width=mlp_width,
            drop_prob=drop_prob,
        )
        self.ffn_drop = nn.Dropout(drop_prob)
        self.ffn_norm = nn.LayerNorm(width)

    def forward(self, q, enc_mem, out_pos_enc):
        skip = q
        x, _ = self.self_attn(
            q=q + out_pos_enc, k=q + out_pos_enc, v=q,
        )
        x = self.self_attn_drop(x)
        x = self.self_attn_norm(x + skip)

        skip = x
        x, _ = self.enc_dec_attn(
            q=x + out_pos_enc, k=self.spatial_pos_enc(enc_mem), v=enc_mem,
        )
        x = self.enc_dec_attn_drop(x)
        x = self.enc_dec_attn_norm(x + skip)

        skip = x
        x = self.ffn(x)
        x = self.ffn_drop(x)
        x = self.ffn_norm(x + skip)
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

    def forward(self, q, enc_mem, out_pos_enc):
        for dec_layer in self.dec_stack:
            q = dec_layer(
                q=q, enc_mem=enc_mem, out_pos_enc=out_pos_enc,
            )
        return q


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

    def forward(self, image_feat, q, out_pos_enc):
        enc_mem = self.encoder(image_feat)
        return self.decoder(q=q, enc_mem=enc_mem, out_pos_enc=out_pos_enc)


if __name__ == "__main__":
    transformer = Transformer()
