from typing import Callable
from torchvision import transforms
from PIL import Image

try:
    _resample = Image.Resampling.BILINEAR
except AttributeError:
    _resample = Image.BILINEAR


class AspectRatioPreservingResizeAndPad:
    def __init__(self, size: int, fill_rgb: tuple[int, int, int] = (123, 116, 103)):
        self.size = size
        self.fill_rgb = fill_rgb

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = img.resize((new_w, new_h), _resample)

        padded_img = Image.new("RGB", (self.size, self.size), self.fill_rgb)
        paste_x = (self.size - new_w) // 2
        paste_y = (self.size - new_h) // 2
        padded_img.paste(img, (paste_x, paste_y))

        return padded_img


def default_image_transform(
    image_size: int = 128,
    train: bool = False,
    padding_fill_rgb: tuple[int, int, int] = (123, 116, 103),
) -> Callable:
    """
    Default image preprocessing.

    For training:
      - aspect-ratio preserving resize + square padding/letterboxing
      - light horizontal flip
      - tensor conversion
      - ImageNet normalization

    For GAP9/export later, we may replace this with deployment-specific
    preprocessing.
    """
    if train:
        return transforms.Compose(
            [
                AspectRatioPreservingResizeAndPad(image_size, padding_fill_rgb),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            AspectRatioPreservingResizeAndPad(image_size, padding_fill_rgb),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


