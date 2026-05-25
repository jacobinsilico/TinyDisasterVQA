from typing import Callable
from torchvision import transforms


def default_image_transform(
    image_size: int = 128,
    train: bool = False,
) -> Callable:
    """
    Default image preprocessing.

    For training:
      - resize to fixed size
      - light horizontal flip
      - tensor conversion
      - ImageNet normalization

    For GAP9/export later, we may replace this with deployment-specific
    preprocessing.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
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
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
