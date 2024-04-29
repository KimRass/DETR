# References:
    # https://github.com/facebookresearch/detr/blob/main/models/matcher.py

import sys
# sys.path.insert(0, "/home/jbkim/Desktop/workspace/DETR")
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/DETR")
import torch
import torch.nn as nn
from torchvision.models import resnet50
import scipy
import einops
from einops.layers.torch import Rearrange
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import numpy as np

from modules.transformer import Transformer
from modules.iou import GIoULoss


class Backbone(nn.Module):
    """
    "Starting from the initial image $x_{img} \in \mathbb{R}^{3 \times H_{0}
    \times W_{0}}$ (with 3 color channels), a conventional CNN backbone generates
    a lower-resolution activation map $f \in \mathbb{R}^{C \times H \times W}$.
    Typical values we use are $C = 2048$ and $H, W = \frac{H_{0}}{32},
    \frac{W_{0}}{32}$."
    "The backbone is with ImageNet-pretrained ResNet model from `torchvision`
    with frozen batchnorm layers."
    """
    def freeze_bn2d_params(self):
        for module in self.cnn.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def bn2d_params_to_buffers(self):
        for module in self.cnn.modules():
            if isinstance(module, nn.BatchNorm2d):
                weight = module.weight
                delattr(module, "weight")
                module.register_buffer("weight", weight)

                bias = module.bias
                delattr(module, "bias")
                module.register_buffer("bias", bias)

    def __init__(self):
        super().__init__()

        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.freeze_bn2d_params()
        self.cnn.avgpool = nn.Identity()
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        return self.cnn(x)


class DETR(nn.Module):
    """
    "A 1x1 convolution reduces the channel dimension of the high-level
    activation map $f$ from $C$ to a smaller dimension $d$. creating a new
    feature map $z_{0} \in \mathbb{R}^{d \times H \times W}$. The encoder
    expects a sequence as input, hence we collapse the spatial dimensions of
    $z_{0}$ into one dimension, resulting in a $d \times HW$ feature map.
    "They are then independently decoded into box coordinates and class labels
    by a feed forward network, resulting $N$ final predictions."
    "The final prediction is computed by a 3-layer perceptron with ReLU activation
    function and hidden dimension $d$, and a linear projection layer. The FFN
    predicts the normalized center coordinates, height and width of the box w.r.t.
    the input image, and the linear layer predicts the class label using a softmax
    function."
    "$N$ is usually much larger than the actual number of objects of interest in
    an image, an additional special class la- bel ∅ is used to represent that no
    object is detected within a slot."
    "All models were trained with $N = 100$ decoder query slots."
    "The decoder receives queries (initially set to zero)."
    """
    def __init__(
        self,
        num_query_slots=100,
        num_classes=80,
        width=512,
        num_encoder_heads=8,
        num_decoder_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        img_size=512,
        stride=32,
        feat_dim=2048,
    ):
        super().__init__()

        self.num_query_slots = num_query_slots
        self.num_classes = num_classes
        self.img_size = img_size
        self.stride = stride
        self.feat_dim = feat_dim

        self.backbone = Backbone()
        self.giou_loss = GIoULoss()
        self.to_sequence = nn.Sequential(
            nn.Conv2d(feat_dim, width, 1, 1, 0),
            Rearrange("b l h w -> b (h w) l"),
        )
        self.transformer = Transformer(
            width=width,
            num_encoder_heads=num_encoder_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_heads=num_decoder_heads,
            num_decoder_layers=num_decoder_layers,
        )
        self.query = torch.zeros((self.num_query_slots, width))
        self.obj_query = nn.Embedding(num_query_slots, width).weight
        self.bbox_ffn = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 4),
            nn.Sigmoid(),
        )
        self.cls_ffn = nn.Sequential(
            nn.Linear(width, num_classes + 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, image):
        x = self.backbone(image)
        x = x.view(
            x.size(0),
            self.feat_dim,
            self.img_size // self.stride,
            self.img_size // self.stride,
        )
        x = self.to_sequence(x)
        x = self.transformer(
            image_feat=x,
            query=einops.repeat(
                self.query.to(image.device),
                pattern="n d -> b n d",
                b=image.size(0),
            ),
            out_pos_enc=einops.repeat(
                self.obj_query.to(image.device),
                pattern="n d -> b n d",
                b=image.size(0),
            ),
        )
        pred_bbox = self.bbox_ffn(x)
        pred_prob = self.cls_ffn(x)
        return pred_bbox, pred_prob

    @staticmethod
    def norm_xywh_to_norm_ltrb(norm_xywh):
        x, y, w, h = torch.unbind(norm_xywh, dim=-1)
        return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)

    def perform_bipartite_matching(
        self, pred_norm_ltrb, pred_prob, gt_ltrb, label, l1_weight, iou_weight,
    ):
        iou_loss = model.giou_loss(pred_norm_ltrb, gt_ltrb)
        l1_loss = torch.abs(
            pred_norm_ltrb[:, None, :] - gt_ltrb[None, :, :]
        ).sum(dim=-1)
        box_loss = l1_weight * l1_loss + iou_weight * iou_loss
        label_prob = pred_prob[:, label]
        match_loss = box_loss - label_prob
        pred_indices, gt_indices = scipy.optimize.linear_sum_assignment(
            match_loss.detach().cpu().numpy(),
        )
        return pred_indices, gt_indices, box_loss, label_prob

    def get_loss(
        self,
        image,
        gt_norm_ltrbs,
        labels,
        no_obj_weight=0.1,
        l1_weight=5,
        iou_weight=2,
    ):
        """
        "$\mathcal{L}_{\text{match}}(y_{i}, \hat{y}_{\sigma(i)})$ is a pair-wise
        matching cost between ground truth $y_{i}$ and a prediction with index
        $\sigma(i)$. This optimal assignment is computed efficiently with the
        Hungarian algorithm. The matching cost takes into account both the class
        prediction and the similarity of predicted and ground truth boxes. Each
        element $i$ of the ground truth set can be seen as a $y_{i}$
        = (c_{i}, b_{i})$ where $c_{i}$ is the target class label (which may be
        $\phi$) and $b_{i} \in [0, 1]$ is a vector that defines ground truth box
        center coordinates and its height and width relative to the image size.
        For the prediction with index $\sigma(i)$ we define probability of class
        c_{i} as $\hat{p}_{\sigma(i)}(c_{i})$ and the predicted box as
        $\hat{b}_{\sigma(i)}$. With these notations we define
        $\mathcal{L}_{\text{match}}(y_{i}, \hat{y}_{\sigma(i)})$ as
        $-\mathbb{1}_{\{c_{i} \neq \phi\}}\hat{p}_{\sigma(i)}(c_{i}) +
        \mathbb{1}_{\{c_{i} \neq \phi\}}\mathcal{L}_{\text{box}}(b_{i}, \hat{b}_{\sigma(i)})$.
        "We down-weight the log-probability term when $c_{i} = \phi$ by a factor
        10 to account for class imbalance."
        "We use linear combination of $\mathcal{l}$ and GIoU losses for bounding
        box regression with $\lambda_{L1} = 5$ and $\lambda_{\text{iou}} = 2$
        weights respectively."
        "All losses are normalized by the number of objects inside the batch."
        TODO: "At inference time, some slots predict empty class. To optimize
        for AP, we override the prediction of these slots with the second
        highest scoring class, using the corresponding confidence. This improves
        AP by 2 points compared to filtering out empty slots."
        """
        out_norm_xywh, out_prob = self(image)
        out_ltrb = self.norm_xywh_to_norm_ltrb(out_norm_xywh)

        sum_losses = torch.zeros((1,), dtype=torch.float32)
        for pred_norm_ltrb, pred_prob, gt_ltrb, label in zip(
            out_ltrb,
            out_prob,
            gt_norm_ltrbs,
            labels,
        ):
            (
                pred_indices,
                gt_indices,
                box_loss,
                label_prob,
            ) = self.perform_bipartite_matching(
                pred_norm_ltrb=pred_norm_ltrb,
                pred_prob=pred_prob,
                gt_ltrb=gt_ltrb,
                label=label,
                l1_weight=l1_weight,
                iou_weight=iou_weight,
            )
            cls_loss = -torch.log(label_prob[pred_indices, gt_indices]).sum()
            no_obj_mask = ~torch.isin(
                torch.arange(self.num_query_slots),
                torch.from_numpy(pred_indices),
            )
            no_obj_cls_loss = -torch.log(
                pred_prob[no_obj_mask, model.num_classes],
            ).sum()
            hungrian_cls_loss = cls_loss + no_obj_weight * no_obj_cls_loss
            hungarian_box_loss = box_loss[pred_indices, gt_indices].sum()
            loss = hungrian_cls_loss + hungarian_box_loss
            # print(cls_loss)
            # print(no_obj_cls_loss)
            # print(hungarian_box_loss)
            # print(loss)
            sum_losses += loss
        num_objs = sum([label.size(0) for label in labels])
        if num_objs != 0:
            sum_losses /= num_objs
        # sum_losses /= image.size(0)
        return sum_losses


if __name__ == "__main__":
    from utils import move_to_device

    _, _, img_size, _ = image.shape
    device = torch.device("mps")
    model = DETR(img_size=img_size).to(device)
    loss = model.get_loss(
        image=move_to_device(image, device),
        gt_norm_ltrbs=move_to_device(annots["norm_ltrbs"], device),
        labels=move_to_device(annots["labels"], device)
    )
    loss
    
    # out_norm_xywh, out_prob = model(image)
    # out_ltrb = model.norm_xywh_to_norm_ltrb(out_norm_xywh)
    # out_ltrb.min()
    # out_norm_xywh.shape
    
    
    # import random

    # batch_size = 4
    # l1_weight = 1
    # iou_weight = 1

    # num_objs = [random.randint(0, 20) for _ in range(batch_size)]
    # labels = [torch.randint(0, model.num_classes, size=(i,)) for i in num_objs]
    # gt_norm_ltrbs = [torch.rand((i, 4)) for i in num_objs]


    # img_size = 480 + 16 * 0
    # print(loss)
    # out_norm_xywh, out_prob = model(image)
    # pred_norm_ltrb = out_norm_xywh[0]
    # pred_prob = out_prob[0]
    # gt_ltrb = gt_norm_ltrbs[0]
    # label = labels[0]
    # gt_ltrb.shape

    # iou_loss = model.giou_loss(pred_norm_ltrb, gt_ltrb)
    # l1_loss = torch.abs(pred_norm_ltrb[:, None, :] - gt_ltrb[None, :, :]).sum(dim=-1)
    # box_loss = l1_weight * l1_loss + iou_weight * iou_loss
    # label_prob = pred_prob[:, label]
    # match_loss = box_loss - label_prob
    # pred_indices, gt_indices = scipy.optimize.linear_sum_assignment(
    #     match_loss.detach().cpu().numpy(),
    # )

    # hungarian_box_loss = box_loss[pred_indices, gt_indices].sum()
    # no_obj_mask = ~torch.isin(torch.arange(100), torch.from_numpy(pred_indices))
    # cls_loss = -torch.log(label_prob[pred_indices, gt_indices]).sum()
    # no_obj_cls_loss = -torch.log(pred_prob[no_obj_mask, model.num_classes]).sum()
    # loss = (cls_loss + no_obj_weight * no_obj_cls_loss) + hungarian_box_loss
    # loss



    # n_objs = label.size(0)
    # new_label = F.pad(
    #     label, pad=(0, model.num_query_slots - n_objs), value=model.num_classes,
    # )
    # new_gt_ltrb = F.pad(
    #     gt_ltrb,
    #     pad=(0, 0, 0, model.num_query_slots - n_objs),
    #     value=0,
    # )
    # iou_loss = model.giou_loss(pred_norm_ltrb, new_gt_ltrb)
    # l1_loss = torch.abs(pred_norm_ltrb[:, None, :] - new_gt_ltrb[None, :, :]).sum(dim=-1)
    # box_loss = l1_weight * l1_loss + iou_weight * iou_loss
    # label_prob = pred_prob[:, new_label]
    # match_loss = box_loss - label_prob
    # pred_indices, gt_indices = scipy.optimize.linear_sum_assignment(
    #     match_loss.detach().cpu().numpy(),
    # )
    # pred_indices
    # gt_indices

    # no_obj_mask = gt_indices >= n_objs
    # label_prob[pred_indices, gt_indices]