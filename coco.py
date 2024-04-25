# Source: https://cocodataset.org/#download
# References:
    # https://www.kaggle.com/code/blondinka/how-to-do-augmentations-for-instance-segmentation

import torch
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset
from torchvision.utils import (
    make_grid,
    draw_segmentation_masks,
    draw_bounding_boxes,
)
from torchvision.ops import box_convert
import torchvision.transforms.functional as TF
from pathlib import Path
import numpy as np

from utils import COLORS, to_uint8, move_to_device
from lsj import LargeScaleJittering


class COCODS(Dataset):
    """
    "The input images are batched together, applying 0-padding adequately to ensure
    they all have the same dimensions (H0;W0) as the largest image of the batch."
    """
    def __init__(self, annot_path, img_dir, transform=None, img_size=512):
        self.transform = transform
        self.img_size = img_size
        self.coco = COCO(annot_path)
        self.img_ids = self.coco.getImgIds()
        self.img_dir = Path(img_dir)

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def get_masks(annots, img):
        h, w, _ = img.shape
        masks = []
        for annot in annots:
            rles = coco_mask.frPyObjects(annot["segmentation"], h, w)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = mask.any(axis=2)
            # mask = torch.from_numpy(mask, dtype=torch.bool)
            mask = mask.astype(np.uint8)
            masks.append(mask)
        return masks

    @staticmethod
    def get_coco_bboxes(annots):
        return [annot["bbox"] for annot in annots]

    @staticmethod
    def get_labels(annots):
        return [annot["category_id"] for annot in annots]

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_dicts = self.coco.loadImgs(img_id)
        img_path = str(self.img_dir/img_dicts[0]["file_name"])
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annots = self.coco.loadAnns(ann_ids)
        masks = self.get_masks(annots=annots, img=img)
        coco_bboxes = self.get_coco_bboxes(annots)
        labels = self.get_labels(annots)

        if self.transform is not None:
            if masks and coco_bboxes and labels:
                transformed = self.transform(
                    image=img, masks=masks, bboxes=coco_bboxes, labels=labels,
                )
                masks = transformed["masks"]
                coco_bboxes = transformed["bboxes"]
                bbox_ids = transformed["bbox_ids"]
                labels = transformed["labels"]
            else:
                transformed = self.transform(image=img)
                bbox_ids = []
            image = transformed["image"]

        if bbox_ids:
            mask = torch.stack([masks[bbox_id] for bbox_id in bbox_ids], dim=0)
            mask = mask.bool()
        else:
            mask = torch.empty(
                size=(0, self.img_size, self.img_size), dtype=torch.bool,
            )
        if coco_bboxes:
            ltrb = torch.tensor(coco_bboxes, dtype=torch.float)
        else:
            ltrb = torch.empty(size=(0, 4), dtype=torch.float)
        label = torch.tensor(labels, dtype=torch.long)
        return image, mask, ltrb, label

    @staticmethod
    def collate_fn(batch):
        images, masks, coco_bboxes, labels = list(zip(*batch))
        if coco_bboxes:
            ltrbs = [
                box_convert(boxes=coco_bbox, in_fmt="xywh", out_fmt="xyxy")
                for coco_bbox
                in coco_bboxes
            ]
        else:
            ltrbs = []
            
        annots = {"masks": masks, "ltrbs": ltrbs, "labels": labels}
        return torch.stack(images, dim=0), annots

    def labels_to_class_names(self, labels):
        return [[self.coco.cats[j]["name"] for j in i.tolist()] for i in labels]

    def vis_annots(
        self,
        image,
        annots,
        task="instance",
        colors=COLORS * 3,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        alpha=0.5,
        font_path="/Users/jongbeomkim/Desktop/workspace/Copy-Paste/resources/NotoSans_Condensed-Medium.ttf",
        # font_path=(
        #     Path(__file__).resolve().parent
        # )/"resources/NotoSans_Condensed-Medium.ttf",
        font_size=14,
    ):
        device = torch.device("cpu")
        image = move_to_device(image, device=device)
        annots = move_to_device(annots, device=device)

        uint8_image = to_uint8(image, mean=mean, std=std)
        class_names = self.labels_to_class_names(annots["labels"])
        images = []
        for batch_idx in range(image.size(0)):
            if task == "instance":
                picked_colors = colors
            elif task == "semantic":
                picked_colors = [
                    colors[i % len(colors)]
                    for i in annots["labels"][batch_idx].tolist()
                ]

            new_image = uint8_image[batch_idx]
            mask = annots["masks"][batch_idx].to(torch.bool)
            if mask.size(0) != 0:
                new_image = draw_segmentation_masks(
                    image=new_image,
                    masks=mask,
                    alpha=alpha,
                    colors=picked_colors,
                )
            if "ltrbs" in annots:
                ltrb = annots["ltrbs"][batch_idx]
                if ltrb.size(0) != 0:
                    new_image = draw_bounding_boxes(
                        image=new_image,
                        boxes=ltrb,
                        labels=None if alpha == 0 else class_names[batch_idx],
                        colors=picked_colors,
                        width=0 if alpha == 0 else 2,
                        font=font_path,
                        font_size=font_size,
                    )
            images.append(new_image)

        grid = make_grid(
            torch.stack(images, dim=0),
            nrow=int(image.size(0) ** 0.5),
            padding=1,
            pad_value=255,
        )
        return TF.to_pil_image(grid)


if __name__ == "__main__":
    lsj = LargeScaleJittering()
    annot_path = "/Users/jongbeomkim/Documents/datasets/coco2014/annotations/instances_val2014.json"
    img_dir = "/Users/jongbeomkim/Documents/datasets/coco2014/val2014"
    ds = COCODS(annot_path=annot_path, img_dir=img_dir, transform=lsj)
    ds[0]
