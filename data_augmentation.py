import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataAug(object):
    """
    "We use scale augmentation, resizing the input images such that the shortest
    side is at least 480 and at most 800 pixels while the longest at most 1333.
    we also apply random crop augmentations during training. A train image is
    cropped with probability 0.5 to a random rectangular patch which is then
    resized again to 800-1333."
    """
    def __init__(
        self,
        format="coco",
        img_size=512,
        pad_color=(127, 127, 127),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.img_size = img_size
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=(-0.5, 0.5),
                    scale_limit=(-0.9, 1),
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=pad_color,
                    p=1,
                ),
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=pad_color,
                ),
                A.CenterCrop(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format=format, label_fields=["bbox_ids", "labels"],
            ),
        )

    def __call__(self, image, masks=[], bboxes=[], labels=[]):
        if masks:
            return self.transform(
                image=image,
                masks=masks,
                bboxes=bboxes,
                bbox_ids=range(len(bboxes)),
                labels=labels,
            )
        else:
            return self.transform(
                image=image,
                bboxes=bboxes,
                bbox_ids=range(len(bboxes)),
                labels=labels,
            )


import numpy as np
img = np.zeros(1024)
A.LongestMaxSize