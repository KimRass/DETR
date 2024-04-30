import sys
# sys.path.insert(0, "/home/jbkim/Desktop/workspace/DETR/")
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/DETR")
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.utils import make_grid, draw_bounding_boxes
import torchvision.transforms.functional as TF

from utils import move_to_device, get_device, image_to_grid, to_uint8
from data.coco import COCODS
from data.lsj import LargeScaleJittering
from modules.detr import DETR


def vis_out(
    out_norm_xywh,
    out_prob,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    out_norm_xywh = move_to_device(out_norm_xywh, device="cpu")
    out_prob = move_to_device(out_prob, device="cpu")
    
    out_norm_ltrb = model.norm_xywh_to_norm_ltrb(out_norm_xywh)
    out_ltrb = out_norm_ltrb * model.img_size

    uint8_image = to_uint8(image, mean=mean, std=std)
    
    images = []
    for batch_idx in range(image.size(0)):
        new_image = uint8_image[batch_idx]
        ltrb = out_ltrb[batch_idx]
        new_image = draw_bounding_boxes(
            image=new_image,
            boxes=ltrb,
            labels=None,
            # colors=,
            width=2,
            # font=str(font_path),
            # font_size=font_size,
        )
        images.append(new_image)
    grid = make_grid(
        torch.stack(images, dim=0),
        nrow=int(image.size(0) ** 0.5),
        padding=1,
        pad_value=255,
    )
    TF.to_pil_image(grid).show()


if __name__ == "__main__":
    # annot_path = "/home/jbkim/Documents/datasets/annotations_trainval2017/annotations/instances_train2017.json"
    # img_dir = "/home/jbkim/Documents/datasets/train2017/train2017"
    annot_path = "/Users/jongbeomkim/Documents/datasets/annotations/instances_train2017.json"
    img_dir = "/Users/jongbeomkim/Documents/datasets/train2017"
    img_size = 512
    device = get_device()

    lsj = LargeScaleJittering(
        img_size=img_size,
        shift_limit=(-0.2, 0.2),
        scale_limit=(-0.5, 0.5),
    )
    ds = COCODS(
        annot_path=annot_path,
        img_dir=img_dir,
        transform=lsj,
        img_size=img_size,
    )
    dl = DataLoader(
        ds, batch_size=4, shuffle=True, collate_fn=ds.collate_fn,
    )
    di = iter(dl)
    image, annots = next(di)
    vis = ds.vis_annots(
        image=image,
        annots=annots,
    )
    vis.show()
    
    image = move_to_device(image, device=device)
    annots = move_to_device(annots, device=device)

    model = DETR(img_size=img_size).to(device)
    optim = AdamW(model.parameters())
    for _ in range(20):
        loss = model.get_loss(
            image=image,
            gt_norm_ltrbs=annots["norm_ltrbs"],
            labels=annots["labels"],
        )

        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"{loss.item():.4f}")

    out_norm_xywh, out_prob = model(image)
    vis_out(out_norm_xywh, out_prob)

    # out_norm_xywh, out_prob = model(image)
    # out_norm_xywh = move_to_device(out_norm_xywh, device="cpu")
    # out_prob = move_to_device(out_prob, device="cpu")
    
    # out_norm_ltrb = model.norm_xywh_to_norm_ltrb(out_norm_xywh)
    # out_ltrb = out_norm_ltrb * model.img_size

    # argmax = torch.argmax(out_prob, dim=-1)
    # obj_mask = argmax != model.num_classes
    
    # out_ltrb.shape
    # out_ltrb[obj_mask].shape

    # image_to_grid(
    #     image,
    #     n_cols=2,
    #     mean=(0.485, 0.456, 0.406),
    #     std=(0.229, 0.224, 0.225),
    # ).show()

