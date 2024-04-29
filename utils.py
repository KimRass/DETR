import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import random
import os
import numpy as np


COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 255, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 250),
    (240, 50, 230),
    (210, 255, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]


def denorm(x, mean=(0.457, 0.437, 0.404), std=(0.275, 0.271, 0.284)):
    return TF.normalize(
        x, mean=-(np.array(mean) / np.array(std)), std=(1 / np.array(std)),
    )


@torch.inference_mode()
def image_to_grid(image, n_cols, padding=1, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor, mean=mean, std=std)
    grid = make_grid(tensor, nrow=n_cols, padding=padding, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def get_grad_scaler(device):
    return GradScaler() if device.type == "cuda" else None


def create_dir(x):
    x = Path(x)
    if x.suffix:
        x.parent.mkdir(parents=True, exist_ok=True)
    else:
        x.mkdir(parents=True, exist_ok=True)


def to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def to_array(image):
    return np.array(image)


def save_image(image, save_path):
    create_dir(save_path)
    to_pil(image).save(str(save_path), quality=100)
    print(f"""Saved image as "{Path(save_path).name}".""")


def show_image(x):
    if isinstance(x, np.ndarray):
        to_pil(x).show()
    elif isinstance(x, Image.Image):
        x.show()


def get_palette(n_classes):
    rand_perm1 = np.random.permutation(256)[: n_classes]
    rand_perm2 = np.random.permutation(256)[: n_classes]
    rand_perm3 = np.random.permutation(256)[: n_classes]
    return np.stack([rand_perm1, rand_perm2, rand_perm3], axis=1)


def colorize_mask(mask, color):
    colored_mask = np.stack([mask] * 3, axis=-1)
    for i in range(3):
        colored_mask[..., i][colored_mask[..., i] == 255] = color[i]
    return colored_mask


def overlay_mask(img, mask, color, beta=0.8):
    colored_mask = colorize_mask(mask * 255, color=color)
    alpha = ((colored_mask > 0).max(axis=2) * 128).astype(np.uint8)            
    rgba_mask = np.concatenate([colored_mask, alpha[:, :, None]], axis=2)
    rgba_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    rgba_img = cv2.addWeighted(rgba_img, 1, rgba_mask, beta, gamma=0)
    return cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2RGB)


def overlay_masks(img, mask, palette, beta=0.8):
    new_img = img.copy()
    for mask_idx, gray_mask in enumerate(np.array(mask)):
        color = palette[mask_idx]
        new_img = overlay_mask(img=new_img, mask=gray_mask, color=color, beta=beta)
    return new_img


def to_uint8(image, mean, std):
    return (denorm(image, mean=mean, std=std) * 255).byte()


def move_to_device(obj, device):
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f'Unexpected type {type(obj)}')
