import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    convnext_tiny,
    mobilenet_v2,
    mobilenet_v3_large,
)
from src.models.blocks import ConvBNReLU


class SmallCNNEncoder(nn.Module):
    """
    Small CNN image encoder for 128x128 RGB images.

    Input:
      image: [B, 3, H, W]

    Output:
      image_features: [B, image_feature_dim]
    """

    def __init__(self, image_feature_dim: int = 256) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # 128x128 -> 64x64
            ConvBNReLU(3, 32, kernel_size=3, stride=2),

            # 64x64 -> 32x32
            ConvBNReLU(32, 64, kernel_size=3, stride=2),

            # 32x32 -> 16x16
            ConvBNReLU(64, 128, kernel_size=3, stride=2),

            # 16x16 -> 8x8
            ConvBNReLU(128, 192, kernel_size=3, stride=2),

            # 8x8 -> 4x4
            ConvBNReLU(192, 256, kernel_size=3, stride=2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, image_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv(images)
        x = self.proj(x)
        return x


class GAPCNNSmallEncoder(nn.Module):
    """
    GAPCNN-S image encoder.

    Design goals:
      - deployment-friendly
      - mostly 3x3 convolutions
      - no residuals in v1
      - no depthwise conv in v1
      - no attention / normalization tricks
      - target roughly <1 MB INT8 for full model

    Input:
      image: [B, 3, 128, 128]

    Output:
      image_features: [B, image_feature_dim]
    """

    def __init__(self, image_feature_dim: int = 160) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # 128x128 -> 64x64
            ConvBNReLU(3, 24, kernel_size=3, stride=2),
            ConvBNReLU(24, 32, kernel_size=3, stride=1),

            # 64x64 -> 32x32
            ConvBNReLU(32, 48, kernel_size=3, stride=2),
            ConvBNReLU(48, 48, kernel_size=3, stride=1),

            # 32x32 -> 16x16
            ConvBNReLU(48, 72, kernel_size=3, stride=2),
            ConvBNReLU(72, 72, kernel_size=3, stride=1),

            # 16x16 -> 8x8
            ConvBNReLU(72, 96, kernel_size=3, stride=2),
            ConvBNReLU(96, 96, kernel_size=3, stride=1),

            # 8x8 -> 4x4
            ConvBNReLU(96, 128, kernel_size=3, stride=2),
            ConvBNReLU(128, 128, kernel_size=3, stride=1),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, image_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.proj(x)
        return x


class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 image encoder.

    Uses ImageNet-pretrained MobileNetV2 by default.

    Input:
      image: [B, 3, H, W]

    Output:
      image_features: [B, image_feature_dim]
    """

    def __init__(
        self,
        image_feature_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        self.features = backbone.features
        mobilenet_feature_dim = 1280

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mobilenet_feature_dim, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.pool(x)
        x = self.proj(x)
        return x


class MobileNetV3LargeEncoder(nn.Module):
    """
    MobileNetV3-Large image encoder.

    Uses ImageNet-pretrained MobileNetV3-Large by default.

    Input:
      image: [B, 3, H, W]

    Output:
      image_features: [B, image_feature_dim]
    """

    def __init__(
        self,
        image_feature_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)

        self.features = backbone.features

        # In torchvision MobileNetV3-Large, classifier[0] maps from 960 features.
        # This is more robust than hardcoding 960.
        mobilenet_feature_dim = backbone.classifier[0].in_features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mobilenet_feature_dim, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.pool(x)
        x = self.proj(x)
        return x


class ConvNeXtTinyEncoder(nn.Module):
    """
    ConvNeXt-Tiny image encoder.

    This is intended as a strong offline teacher backbone.
    It is NOT intended for GAP9 deployment.

    Input:
      image: [B, 3, H, W]

    Output:
      image_features: [B, image_feature_dim]
    """

    def __init__(
        self,
        image_feature_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)

        self.features = backbone.features

        # torchvision ConvNeXt-Tiny classifier usually ends with Linear(768, 1000).
        convnext_feature_dim = backbone.classifier[-1].in_features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(convnext_feature_dim, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.pool(x)
        x = self.proj(x)
        return x


def build_image_encoder(
    model_name: str,
    image_feature_dim: int,
    pretrained: bool = True,
    freeze_image_encoder: bool = False,
) -> nn.Module:
    """
    Build image encoder by name.

    Supported:
      - cnn
      - small_cnn
      - gapcnn_s
      - mobilenet_v2
      - mobilenet_v3_large
      - convnext_tiny
    """
    model_name = model_name.lower()

    if model_name in {"cnn", "small_cnn", "baseline_cnn"}:
        encoder = SmallCNNEncoder(
            image_feature_dim=image_feature_dim,
        )

        if freeze_image_encoder:
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder

    if model_name in {"gapcnn_s", "gapcnn-small", "gapcnn_small"}:
        encoder = GAPCNNSmallEncoder(
            image_feature_dim=image_feature_dim,
        )

        if freeze_image_encoder:
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder

    if model_name in {"mobilenet_v2", "mobilenetv2"}:
        return MobileNetV2Encoder(
            image_feature_dim=image_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_image_encoder,
        )

    if model_name in {
        "mobilenet_v3_large",
        "mobilenetv3_large",
        "mobilenet_v3",
        "mobilenetv3",
    }:
        return MobileNetV3LargeEncoder(
            image_feature_dim=image_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_image_encoder,
        )

    if model_name in {"convnext_tiny", "convnext-tiny", "convnext"}:
        return ConvNeXtTinyEncoder(
            image_feature_dim=image_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_image_encoder,
        )

    raise ValueError(
        f"Unknown model_name='{model_name}'. "
        f"Supported: cnn, gapcnn_s, mobilenet_v2, mobilenet_v3_large, convnext_tiny"
    )
