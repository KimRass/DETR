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
            self.pos_embed.pe_mat.to(x.device), pattern="l d -> b l d", b=x.size(0),
        )[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, width, num_heads, drop_prob):
        super().__init__()
    
        self.width = width # "$d_{model}$"
        self.num_heads = num_heads # "$h$"

        self.head_dim = width // num_heads # "$d_{k}$, $d_{v}$"

        self.q_proj = nn.Linear(width, width, bias=False) # "$W^{Q}_{i}$"
        self.k_proj = nn.Linear(width, width, bias=False) # "$W^{K}_{i}$"
        self.v_proj = nn.Linear(width, width, bias=False) # "$W^{V}_{i}$"

        self.attn_drop = nn.Dropout(drop_prob) # Not in the paper
        self.out_proj = nn.Linear(width, width, bias=False) # "$W^{O}$"

    @staticmethod
    def _get_attention_score(q, k):
        attn_score = torch.einsum("bnid,bnjd->bnij", q, k)
        return attn_score

    def forward(self, q, k, v, mask=None):
        b, i, _ = q.shape
        _, j, _ = k.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(b, self.num_heads, i, self.head_dim)
        k = k.view(b, self.num_heads, j, self.head_dim)
        v = v.view(b, self.num_heads, j, self.head_dim)

        attn_score = self._get_attention_score(q=q, k=k)
        if mask is not None:
            mask = einops.repeat(
                mask, pattern="b i j -> b n i j", n=self.num_heads,
            )
            attn_score.masked_fill_(mask=mask, value=-1e9) # "Mask (opt.)"
        attn_score /= (self.head_dim ** 0.5) # "Scale"
        attn_weight = F.softmax(attn_score, dim=3) # "Softmax"

        attn_weight_drop = self.attn_drop(attn_weight) # Not in the paper
        x = torch.einsum("bnij,bnjd->bnid", attn_weight_drop, v) # "MatMul"
        x = einops.rearrange(x, pattern="b n i d -> b i (n d)")

        x = self.out_proj(x)
        return x, attn_weight


def get_activ_fn(activ):
    if activ == "relu":
        return nn.ReLU()
    elif activ == "gelu":
        return nn.GELU()
    else:
        raise AssertionError(
            """The argument `activ` must be one of (`"relu"`, `"gelu"`)"""
        )


class PositionwiseFeedForward(nn.Module):
    def __init__(self, width, mlp_dim, drop_prob, activ="relu"):
        super().__init__()


        self.activ = activ

        self.layers = nn.Sequential(
            nn.Linear(width, mlp_dim),
            get_activ_fn(activ),
            nn.Dropout(drop_prob),
            nn.Linear(mlp_dim, width),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualConnection(nn.Module):
    """
    "Additive dropout of 0.1 is applied after every multi-head attention and FFN
    before layer normalization. The weights are randomly initialized with Xavier
    initialization.
    """
    def __init__(self, width, drop_prob):
        super().__init__()

        self.resid_drop = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(width)

    def forward(self, x, sublayer):
        skip = x.clone()
        x = sublayer(x)
        x = self.resid_drop(x)
        x = self.norm(x + skip) # "Add & Norm"
        return x


class EncoderLayer(nn.Module):
    """
    "Image features from the CNN backbone are passed through the transformer
    encoder, together with spatial positional encoding that are added to queries
    and keys at every multi-head self-attention layer."
    """
    def __init__(
        self,
        width,
        num_heads,
        mlp_dim,
        attn_drop_prob,
        ff_drop_prob,
        resid_drop_prob,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.width = width
        self.mlp_dim = mlp_dim

        self.spatial_pos_enc = SinePositionalEncoding()
        self.self_attn = MultiHeadAttention(width=width, num_heads=num_heads, drop_prob=attn_drop_prob)
        self.attn_resid_conn = ResidualConnection(width=width, drop_prob=resid_drop_prob)
        self.feed_forward = PositionwiseFeedForward(
            width=width, mlp_dim=mlp_dim, drop_prob=ff_drop_prob, activ="relu",
        )
        self.ff_resid_conn = ResidualConnection(width=width, drop_prob=resid_drop_prob)

    def forward(self, x):
        q = k = self.spatial_pos_enc(x)
        x = self.attn_resid_conn(
            x=x,
            sublayer=lambda x: self.self_attn(q=q, k=k, v=x)[0],
        )
        x = self.ff_resid_conn(x=x, sublayer=self.feed_forward)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_heads=1,
        width=512,
        mlp_dim=None,
        num_layers=1,
        attn_drop_prob=0.1,
        ff_drop_prob=0.1,
        resid_drop_prob=0.1,
    ):
        super().__init__()

        self.mlp_dim = mlp_dim if mlp_dim is not None else width * 4

        self.enc_stack = nn.ModuleList(
            [
                EncoderLayer(
                    num_heads=num_heads,
                    width=width,
                    mlp_dim=self.mlp_dim,
                    attn_drop_prob=attn_drop_prob,
                    ff_drop_prob=ff_drop_prob,
                    resid_drop_prob=resid_drop_prob,
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
    """
    def __init__(
        self,
        num_heads,
        width,
        mlp_dim,
        attn_drop_prob=0.1,
        ff_drop_prob=0.1,
        resid_drop_prob=0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.width = width
        self.mlp_dim = mlp_dim

        self.spatial_pos_enc = SinePositionalEncoding()
        self.self_attn = MultiHeadAttention(width=width, num_heads=num_heads, drop_prob=attn_drop_prob)
        self.self_attn_resid_conn = ResidualConnection(width=width, drop_prob=resid_drop_prob)
        self.enc_dec_attn = MultiHeadAttention(width=width, num_heads=num_heads, drop_prob=attn_drop_prob)
        self.enc_dec_attn_resid_conn = ResidualConnection(width=width, drop_prob=resid_drop_prob)
        self.feed_forward = PositionwiseFeedForward(
            width=width, mlp_dim=mlp_dim, drop_prob=ff_drop_prob, activ="relu",
        )
        self.ff_resid_conn = ResidualConnection(width=width, drop_prob=resid_drop_prob)

    def forward(self, x, enc_mem, obj_query):
        q = k = x + obj_query
        x = self.self_attn_resid_conn(
            x=x,
            sublayer=lambda x: self.self_attn(q=q, k=k, v=x)[0],
        )
        q = x + obj_query
        k = self.spatial_pos_enc(x + enc_mem)
        x = self.enc_dec_attn_resid_conn(
            x=x,
            sublayer=lambda x: self.enc_dec_attn(q=q, k=k, v=enc_mem)[0]
        )
        x = self.ff_resid_conn(x=x, sublayer=self.feed_forward)
        return x


class TransformerDecoder(nn.Module):
    """
    "Transformer module uses the standard Transformer decoder [41] to compute from image features F and N learnable positional embeddings (i.e., queries) its output, i.e., N per-segment embeddings Q 2 RCQ N of dimension CQ that encode global information about each segment MaskFormer predicts. Similarly to [4], the decoder yields all predictions in parallel."
    """
    def __init__(
        self,
        num_heads,
        width,
        num_layers,
        mlp_dim=None,
        attn_drop_prob=0.1,
        ff_drop_prob=0.1,
        resid_drop_prob=0.1,
    ):
        super().__init__()

        self.mlp_dim = mlp_dim if mlp_dim is not None else width * 4

        self.dec_stack = nn.ModuleList(
            [
                DecoderLayer(
                    num_heads=num_heads,
                    width=width,
                    mlp_dim=self.mlp_dim,
                    attn_drop_prob=attn_drop_prob,
                    ff_drop_prob=ff_drop_prob,
                    resid_drop_prob=resid_drop_prob,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, enc_mem):
        for dec_layer in self.dec_stack:
            x = dec_layer(x, enc_mem=enc_mem)
        return x