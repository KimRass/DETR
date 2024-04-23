# References
    # https://github.com/facebookresearch/detr/blob/main/models/matcher.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


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
        r = torch.maximum(ltrb1[..., 2][..., None], ltrb2[..., 2][None, ...])
        b = torch.maximum(ltrb1[..., 3][..., None], ltrb2[..., 3][None, ...])
        return torch.clip(r - l, min=0) * torch.clip(b - t, min=0)

    def get_iou(self, ltrb1, ltrb2):
        """
        args:
            `ltrb1`: Tensor of shape (N, 4)
            `ltrb2`: Tensor of shape (M, 4)
        returns:
            Tensor of shape (N, M)
        """
        ltrb1_area = self.get_area(ltrb1)
        ltrb2_area = self.get_area(ltrb2)
        intersec_area = self.get_intersection_area(ltrb1, ltrb2)
        union_area = ltrb1_area[..., None] + ltrb2_area[None, ...] - intersec_area
        return torch.where(union_area == 0, 0., intersec_area / union_area)

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
        giou = self.get_giou(ltrb1=ltrb1, ltrb2=ltrb2)
        return 1 - giou


# class DETR(nn.Module):
#     def __init__(self, num_classes, hidden_dim, num_heads, num_encoder_layers, num_decoder_layers):
#         super().__init__()

#         # We take only convolutional layers from ResNet-50 model
#         self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
#         self.conv = nn.Conv2d(2048, hidden_dim, 1)
#         self.transformer = nn.Transformer(
#             hidden_dim, num_heads, num_encoder_layers, num_decoder_layers,
#         )
#         self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
#         self.linear_bbox = nn.Linear(hidden_dim, 4)
#         self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
#         self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
#         self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

#     def forward(self, inputs):
#         x = self.backbone(inputs)
#         h = self.conv(x)
#         H, W = h.shape[-2:]
#         pos = torch.cat([
#             self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
#             self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
#         ], dim=-1).flatten(0, 1).unsqueeze(1)
#         h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
#         self.query_pos.unsqueeze(1))
#         return self.linear_class(h), self.linear_bbox(h).sigmoid()
class DETR(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
    ):
        super().__init__()

        self.giou_loss = GIoULoss()

    def forward(self, inputs):
        pass

    def get_loss(self, image):
        self.giou_loss()


if __name__ == "__main__":
    model = DETR(
        num_classes=91, hidden_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
    )
    inputs = torch.randn(1, 3, 800, 1200)
    logits, bboxes = model(inputs)
    logits.shape, bboxes.shape


    """
    "Lmatch (yi , ŷσ(i) ) is a pair-wise matching cost between ground truth yi and a prediction with index σ(i). This optimal assignment is computed efficiently with the Hungarian algorithm, following prior work (e.g. [43]). The matching cost takes into account both the class prediction and the sim- ilarity of predicted and ground truth boxes. Each element i of the ground truth set can be seen as a yi = (ci , bi ) where ci is the target class label (which may be ∅) and bi ∈ [0, 1]4 is a vector that defines ground truth box cen- ter coordinates and its height and width relative to the image size. For the prediction with index σ(i) we define probability of class ci as p̂σ(i) (ci ) and the predicted box as b̂σ(i) . With these notations we define Lmatch (yi , ŷσ(i) ) as −1{ci 6=∅} p̂σ(i) (ci ) + 1{ci 6=∅} Lbox (bi, b̂σ(i) ).
    """
    import random
    batch_size = 4
    num_queries = 40
    num_classes = 80
    num_objs = [7, 3, 11, 4]
    pred_orders = [random.sample(range(i), i) for i in num_objs]
    pred_cls_probs = torch.rand((batch_size, num_queries, num_classes))
    pred_bboxes = torch.rand((batch_size, num_queries, 4))
    labels = [torch.randint(0, num_classes + 1, size=(i,)) for i in num_objs]
    gt_bboxes = [torch.rand((i, 4)) for i in num_objs]
    
    giou_loss = GIoULoss()
    batch_idx = 0
    label = labels[batch_idx]
    gt_bbox = gt_bboxes[batch_idx]
    pred_bbox = pred_bboxes[batch_idx]
    pred_order = pred_orders[batch_idx] # "$\sigma(i)$"
    pred_cls_prob = pred_cls_probs[batch_idx]
    
    pred_cls_prob[pred_order, label] # "$p^{hat}_{\sigma(i)}(c_{i})$"
    a = pred_bbox[pred_order]
    b = gt_bbox
    a.shape, b.shape
    giou_loss(a, b).shape
    


    label
    pred_cls_prob.shape
    label[pred_order]    
    giou_loss(pred_bbox, gt_bbox)