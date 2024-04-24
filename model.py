# References
    # https://github.com/facebookresearch/detr/blob/main/models/matcher.py

import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np
import scipy
import einops

from torchvision.models import resnet50, ResNet50_Weights


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, drop_prob):
        super().__init__()
    
        self.hidden_dim = hidden_dim # "$d_{model}$"
        self.num_heads = num_heads # "$h$"

        self.head_dim = hidden_dim // num_heads # "$d_{k}$, $d_{v}$"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False) # "$W^{Q}_{i}$"
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False) # "$W^{K}_{i}$"
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False) # "$W^{V}_{i}$"

        self.attn_drop = nn.Dropout(drop_prob) # Not in the paper
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False) # "$W^{O}$"

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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, drop_prob, activ="relu"):
        super().__init__()

        assert activ in ["relu", "gelu"], (
            """The argument `activ` must be one of (`"relu"`, `"gelu"`)"""
        )

        self.activ = activ

        self.proj1 = nn.Linear(hidden_dim, mlp_dim) # "$W_{1}$"
        if activ == "relu":
            self.relu = nn.ReLU()
        else:
            self.gelu = nn.GELU()
        self.proj2 = nn.Linear(mlp_dim, hidden_dim) # "$W_{2}$"
        self.mlp_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj1(x)
        if self.activ == "relu":
            x = self.relu(x)
        else:
            x = self.gelu(x)
        x = self.proj2(x)
        x = self.mlp_drop(x) # Not in the paper
        return x


class ResidualConnection(nn.Module):
    def __init__(self, hidden_dim, drop_prob):
        super().__init__()

        self.resid_drop = nn.Dropout(drop_prob) # "Residual dropout"
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, sublayer):
        skip = x.clone()
        x = sublayer(x)
        x = self.resid_drop(x)
        x += skip # "Add"
        x = self.norm(x) # "& Norm"
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_dim,
        attn_drop_prob,
        ff_drop_prob,
        resid_drop_prob,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

        self.self_attn = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, drop_prob=attn_drop_prob)
        self.attn_resid_conn = ResidualConnection(hidden_dim=hidden_dim, drop_prob=resid_drop_prob)
        self.feed_forward = PositionwiseFeedForward(
            hidden_dim=hidden_dim, mlp_dim=mlp_dim, drop_prob=ff_drop_prob, activ="relu",
        )
        self.ff_resid_conn = ResidualConnection(hidden_dim=hidden_dim, drop_prob=resid_drop_prob)

    def forward(self, x):
        x = self.attn_resid_conn(
            x=x,
            sublayer=lambda x: self.self_attn(q=x, k=x, v=x)[0],
        )
        x = self.ff_resid_conn(x=x, sublayer=self.feed_forward)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_heads=1,
        hidden_dim=512,
        mlp_dim=None,
        num_layers=1,
        attn_drop_prob=0.1,
        ff_drop_prob=0.1,
        resid_drop_prob=0.1,
    ):
        super().__init__()

        self.mlp_dim = mlp_dim if mlp_dim is not None else hidden_dim * 4

        self.enc_stack = nn.ModuleList(
            [
                EncoderLayer(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
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
    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_dim,
        attn_drop_prob=0.1,
        ff_drop_prob=0.1,
        resid_drop_prob=0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

        self.self_attn = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, drop_prob=attn_drop_prob)
        self.self_attn_resid_conn = ResidualConnection(hidden_dim=hidden_dim, drop_prob=resid_drop_prob)
        self.enc_dec_attn = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, drop_prob=attn_drop_prob)
        self.enc_dec_attn_resid_conn = ResidualConnection(hidden_dim=hidden_dim, drop_prob=resid_drop_prob)
        self.feed_forward = PositionwiseFeedForward(
            hidden_dim=hidden_dim, mlp_dim=mlp_dim, drop_prob=ff_drop_prob, activ="relu",
        )
        self.ff_resid_conn = ResidualConnection(hidden_dim=hidden_dim, drop_prob=resid_drop_prob)

    def forward(self, x, enc_out):
        x = self.self_attn_resid_conn(
            x=x,
            sublayer=lambda x: self.self_attn(q=x, k=x, v=x)[0],
        )
        x = self.enc_dec_attn_resid_conn(
            x=x,
            sublayer=lambda x: self.enc_dec_attn(q=x, k=enc_out, v=enc_out)[0]
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
        hidden_dim,
        num_layers,
        mlp_dim=None,
        attn_drop_prob=0.1,
        ff_drop_prob=0.1,
        resid_drop_prob=0.1,
    ):
        super().__init__()

        self.mlp_dim = mlp_dim if mlp_dim is not None else hidden_dim * 4

        self.dec_stack = nn.ModuleList(
            [
                DecoderLayer(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=self.mlp_dim,
                    attn_drop_prob=attn_drop_prob,
                    ff_drop_prob=ff_drop_prob,
                    resid_drop_prob=resid_drop_prob,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, enc_out):
        for dec_layer in self.dec_stack:
            x = dec_layer(x, enc_out=enc_out)
        return x


class GIoULoss(object):
    @staticmethod
    def get_area(ltrb):
        """
        args:
            `ltrb`: Tensor of shape (N, 4)
        returns:
            Tensor of shape (N)
        """
        return torch.clip(
            ltrb[..., 2] - ltrb[..., 0], min=0
        ) * torch.clip(ltrb[..., 3] - ltrb[..., 1], min=0)

    @staticmethod
    def get_intersection_area(ltrb1, ltrb2):
        """
        args:
            `ltrb1`: Tensor of shape (N, 4)
            `ltrb2`: Tensor of shape (M, 4)
        returns:
            Tensor of shape (N, M)
        """
        l = torch.maximum(ltrb1[..., 0][..., None], ltrb2[..., 0][None, ...])
        t = torch.maximum(ltrb1[..., 1][..., None], ltrb2[..., 1][None, ...])
        r = torch.minimum(ltrb1[..., 2][..., None], ltrb2[..., 2][None, ...])
        b = torch.minimum(ltrb1[..., 3][..., None], ltrb2[..., 3][None, ...])
        return torch.clip(r - l, min=0) * torch.clip(b - t, min=0)

    @staticmethod
    def get_smallest_enclosing_area(ltrb1, ltrb2):
        l = torch.minimum(ltrb1[:, 0][:, None], ltrb2[:, 0][None, :])
        t = torch.minimum(ltrb1[:, 1][:, None], ltrb2[:, 1][None, :])
        r = torch.maximum(ltrb1[:, 2][:, None], ltrb2[:, 2][None, :])
        b = torch.maximum(ltrb1[:, 3][:, None], ltrb2[:, 3][None, :])
        return torch.clip(r - l, min=0) * torch.clip(b - t, min=0)

    def get_giou(self, ltrb1, ltrb2):
        ltrb1_area = self.get_area(ltrb1)
        ltrb2_area = self.get_area(ltrb2)
        intersec_area = self.get_intersection_area(ltrb1, ltrb2)
        union_area = ltrb1_area[:, None] + ltrb2_area[None, :] - intersec_area
        enclose = self.get_smallest_enclosing_area(ltrb1, ltrb2)
        iou = torch.where(union_area == 0, 0, intersec_area / union_area)
        return torch.where(
            enclose == 0, -1, iou - ((enclose - union_area) / enclose),
        )

    def __call__(self, ltrb1, ltrb2):
        giou = self.get_giou(ltrb1, ltrb2)
        return 1 - giou


class Backbone(nn.Module):
    """
    ". Starting from the initial image ximg ∈ R3×H0 ×W0 (with 3 color channels2), a conventional CNN backbone generates a lower-resolution activation map f ∈ RC×H×W . Typical values we use are C = 2048 and W = H 32 0 , W0 32 . H,
    """
    def __init__(self):
        super().__init__()

        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn.avgpool = nn.Identity()
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        return self.cnn(x)


class DETR(nn.Module):
    def __init__(
        self,
        num_queries=50,
        num_classes=80,
        hidden_dim=512,
        num_encoder_heads=1,
        num_decoder_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        img_size=512,
        stride=32,
        feat_dim=2048,
    ):
        """
        "All transformer weights are initialized with Xavier init."
        "The backbone is with ImageNet-pretrained ResNet model from `torchvision`
        with frozen batchnorm layers. We report results with two different
        backbones: a ResNet-50 and a ResNet-101. The corresponding models are
        called respectively DETR and DETR-R101. we also increase the feature
        resolution by adding a dilation to the last stage of the backbone and
        removing a stride from the first convolution of this stage. The
        corresponding models are called respectively DETR-DC5 and DETR-DC5-R101
        (dilated C5 stage). This modification increases the resolution by a
        factor of two, thus improving performance for small objects, at the cost
        of a 16x higher cost in the self-attentions of the encoder, leading to
        an overall 2x increase in computational cost."
        
        "A 1x1 convolution reduces the channel dimension of the high-level activation map f from C to a smaller dimension d. creating a new feature map z0 ∈ Rd×H×W . The encoder expects a sequence as input, hence we collapse the spatial dimensions of z0 into one dimension, resulting in a d×HW feature map.

        The final prediction is com- puted by a 3-layer perceptron with ReLU activation function and hidden dimen- sion d, and a linear projection layer. The FFN predicts the normalized center coordinates, height and width of the box w.r.t. the input image, and the lin- ear layer predicts the class label using a softmax function.
        N is usually much larger than the actual number of objects of interest in an image, an additional special class la- bel ∅ is used to represent that no object is detected within a slot. This class plays a similar role to the “background” class in the standard object detection approaches
        """
        super().__init__()

        self.img_size = img_size
        self.stride = stride
        self.feat_dim = feat_dim

        self.backbone = Backbone()
        self.giou_loss = GIoULoss()
        self.conv = nn.Conv2d(2048, hidden_dim, 1, 1, 0)
        self.encoder = TransformerEncoder(
            num_heads=num_encoder_heads,
            num_layers=num_encoder_layers,
            hidden_dim=hidden_dim,
        )
        self.decoder = TransformerDecoder(
            num_heads=num_decoder_heads,
            num_layers=num_decoder_layers,
            hidden_dim=hidden_dim,
        )
        self.query = torch.randn((num_queries, hidden_dim))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.cls_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, image):
        x = self.backbone(image)
        x = x.view(
            x.size(0),
            self.feat_dim,
            self.img_size // self.stride,
            self.img_size // self.stride,
        )
        x = self.conv(x)
        x = einops.rearrange(x, pattern="b l h w -> b (h w) l")
        enc_out = self.encoder(x)
        x = self.decoder(
            einops.repeat(self.query, pattern="n d -> b n d", b=image.size(0)),
            enc_out,
        )

        pred_bbox = self.ffn(x)
        pred_prob = F.softmax(self.cls_proj(x), dim=-1)
        return pred_bbox, pred_prob

    def get_loss(
        self,
        image,
        labels,
        gt_bboxes,
        no_obj_coeff=0.1,
        giou_coeff=1,
        l1_coeff=1,
    ):
        """
        "Lmatch (yi , ŷσ(i) ) is a pair-wise matching cost between ground truth yi and a prediction with index σ(i). This optimal assignment is computed efficiently with the Hungarian algorithm, following prior work (e.g. [43]). The matching cost takes into account both the class prediction and the sim- ilarity of predicted and ground truth boxes. Each element i of the ground truth set can be seen as a yi = (ci , bi ) where ci is the target class label (which may be ∅) and bi ∈ [0, 1]4 is a vector that defines ground truth box cen- ter coordinates and its height and width relative to the image size. For the prediction with index σ(i) we define probability of class ci as p̂σ(i) (ci ) and the predicted box as b̂σ(i) . With these notations we define $\mathcal{L}_{\text{match}}(y_{i}, \hat{y}_{\sigma(i)})$ as
        $-\mathbb{1}_{\{c_{i} \neq \phi\}}\hat{p}_{\sigma(i)}(c_{i}) +
        \mathbb{1}_{\{c_{i} \neq \phi\}}\mathcal{L}_{\text{box}}(b_{i}, \hat{b}_{\sigma(i)})$.
        """
        batched_pred_bbox, batched_pred_prob = self(image)

        sum_losses = torch.empty((0,), dtype=torch.float32)
        for pred_bbox, pred_prob, label, gt_bbox in zip(
            batched_pred_bbox,
            batched_pred_prob,
            labels,
            gt_bboxes
        ):
            giou = self.giou_loss(pred_bbox, gt_bbox)
            label_prob = pred_prob[:, label]
            match_loss = -label_prob + giou
            pred_indices, gt_indices = scipy.optimize.linear_sum_assignment(
                match_loss.detach().cpu().numpy(),
            )

            loss = no_obj_coeff * torch.sum(
                -torch.log(label_prob[pred_indices, gt_indices])
            )
            loss += giou_coeff * giou[pred_indices, gt_indices].sum()
            l1_dist = torch.abs(pred_bbox[pred_indices] - gt_bbox[gt_indices])
            loss += l1_coeff * l1_dist.sum()
            sum_losses += loss
        return sum_losses


if __name__ == "__main__":
    """
    "s. We train DETR with AdamW [26] setting the initial trans-
    former’s learning rate to 10−4 , the backbone’s to 10−5 , and weight decay to 10−4 .
    """
    # model = DETR(
    #     num_classes=91, hidden_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
    # )
    # inputs = torch.randn(1, 3, 800, 1200)
    # logits, bboxes = model(inputs)
    # logits.shape, bboxes.shape


    import random
    batch_size = 4
    num_queries = 40
    num_classes = 80
    num_objs = [7, 3, 11, 4]
    pred_orders = [random.sample(range(i), i) for i in num_objs]
    pred_cls_logits = torch.rand((batch_size, num_queries, num_classes))
    pred_bboxes = torch.rand((batch_size, num_queries, 4))
    labels = [torch.randint(0, num_classes, size=(i,)) for i in num_objs]
    gt_bboxes = [torch.rand((i, 4)) for i in num_objs]
    pred_bboxes.shape

    batch_idx = 0
    label = labels[batch_idx] # "$c_{i}$"
    gt_bbox = gt_bboxes[batch_idx] # "$b_{i}$"
    pred_bbox = pred_bboxes[batch_idx]
    pred_order = pred_orders[batch_idx] # "$\sigma(i)$"
    pred_cls_logit = pred_cls_logits[batch_idx]
    

    model = DETR()
    image = torch.randn((4, 3, 512, 512))
    # out = model(image)
    # pred_bbox, pred_prob = out
    # pred_bbox.shape, pred_prob.shape

    model.get_loss(
        image=image, labels=labels, gt_bboxes=gt_bboxes,
    )
