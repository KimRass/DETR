# Source: https://cocodataset.org/#download

import torch
import cv2
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.utils import make_grid, draw_bounding_boxes
from torchvision.ops import box_convert
import torchvision.transforms.functional as TF
from pathlib import Path

from utils import COLORS, to_uint8, move_to_device


class COCODS(Dataset):
    """
    "We perform experiments on COCO 2017 detection and panoptic segmentation datasets [24,18], containing 118k training images and 5k validation
    images. Each image is annotated with bounding boxes and panoptic segmenta-
    tion. There are 7 instances per image on average, up to 63 instances in a single
    image in training set, ranging from small to large on the same images. If not
    specified, we report AP as bbox AP,
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
    def get_coco_bboxes(annots):
        return [annot["bbox"] for annot in annots]

    @staticmethod
    def get_labels(annots):
        return [annot["category_id"] for annot in annots]

    def coco_bbox_to_norm_ltrb(self, coco_bbox):
        ltrb = box_convert(coco_bbox, in_fmt="xywh", out_fmt="xyxy")
        return ltrb / self.img_size

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_dicts = self.coco.loadImgs(img_id)
        img_path = str(self.img_dir/img_dicts[0]["file_name"])
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annots = self.coco.loadAnns(ann_ids)
        coco_bboxes = self.get_coco_bboxes(annots)
        labels = self.get_labels(annots)

        if self.transform is None:
            return img, coco_bboxes, labels
        else:
            if coco_bboxes and labels:
                transformed = self.transform(
                    image=img, bboxes=coco_bboxes, labels=labels,
                )
                coco_bboxes = transformed["bboxes"]
                labels = transformed["labels"]
            else:
                transformed = self.transform(image=img)
            image = transformed["image"]

            if coco_bboxes:
                coco_bbox = torch.tensor(coco_bboxes, dtype=torch.float)
            else:
                coco_bbox = torch.empty(size=(0, 4), dtype=torch.float)
            norm_ltrb = self.coco_bbox_to_norm_ltrb(coco_bbox)
            label = torch.tensor(labels, dtype=torch.long)
            return image, norm_ltrb, label

    @staticmethod
    def collate_fn(batch):
        images, norm_ltrbs, labels = list(zip(*batch))
        annots = {"norm_ltrbs": norm_ltrbs, "labels": labels}
        return torch.stack(images, dim=0), annots

    def labels_to_class_names(self, labels):
        return [[self.coco.cats[j]["name"] for j in i.tolist()] for i in labels]

    def vis_annots(
        self,
        image,
        annots,
        colors=COLORS * 3,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        alpha=0.5,
        font_path=(
            Path(__file__).resolve().parents[1]
        )/"resources/NotoSans_Condensed-Medium.ttf",
        font_size=14,
    ):
        image = move_to_device(image, device="cpu")
        annots = move_to_device(annots, device="cpu")

        uint8_image = to_uint8(image, mean=mean, std=std)
        class_names = self.labels_to_class_names(annots["labels"])
        images = []
        for batch_idx in range(image.size(0)):
            picked_colors = colors

            new_image = uint8_image[batch_idx]
            norm_ltrb = annots["norm_ltrbs"][batch_idx]
            if norm_ltrb.size(0) != 0:
                new_image = draw_bounding_boxes(
                    image=new_image,
                    boxes=norm_ltrb * self.img_size,
                    labels=None if alpha == 0 else class_names[batch_idx],
                    colors=picked_colors,
                    width=0 if alpha == 0 else 2,
                    font=str(font_path),
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
