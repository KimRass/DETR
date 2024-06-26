import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LargeScaleJittering(object):
    def __init__(
        self,
        format="coco",
        img_size=512,
        shift_limit=(-0.5, 0.5),
        scale_limit=(-0.9, 1),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pad_color=(127, 127, 127),
    ):
        self.img_size = img_size
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=shift_limit,
                    scale_limit=scale_limit,
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
