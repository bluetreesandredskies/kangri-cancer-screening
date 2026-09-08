"""
Model definition for the lesion classifier: EfficientNet-B0 via timm,
fine-tuned with most of the backbone frozen so it fits comfortably in
6GB of VRAM and trains fast on a small dataset.
"""

import timm
import torch.nn as nn


def build_model(num_classes: int = 5) -> nn.Module:
    """
    Builds an EfficientNet-B0 with ImageNet weights and freezes everything
    except the last two MBConv stages and the classifier head.

    timm's efficientnet_b0 layout (relevant top-level modules):
        conv_stem   -> initial stem conv
        bn1         -> stem batchnorm
        blocks      -> nn.Sequential of 7 stages, indices blocks[0]..blocks[6]
                       (each stage is itself a sequence of MBConv blocks)
        conv_head   -> 1x1 conv that expands to the final feature dim
        bn2         -> batchnorm after conv_head
        global_pool -> adaptive avg pool
        classifier  -> final nn.Linear(in_features, num_classes)

    "Last 2 blocks" here means blocks[5] and blocks[6], the final two
    stages of the backbone. conv_head/bn2 are also left trainable since
    they sit directly between blocks[6] and the classifier and have very
    few parameters, so there's no real memory cost to fine-tuning them
    too — everything else (stem + blocks[0..4]) stays frozen.

    num_classes defaults to 5 to match the current label set:
    actinic_keratosis, healthy, nevus, seborrheic_keratosis,
    squamous_cell_carcinoma (alphabetical order, matching
    data/processed/class_names.json).
    """
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)

    # freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the last 2 backbone stages
    for param in model.blocks[5].parameters():
        param.requires_grad = True
    for param in model.blocks[6].parameters():
        param.requires_grad = True

    # unfreeze the head conv + bn that sit right after the blocks
    for param in model.conv_head.parameters():
        param.requires_grad = True
    for param in model.bn2.parameters():
        param.requires_grad = True

    # unfreeze the classifier itself
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    # quick sanity check: how many params are actually trainable?
    m = build_model(num_classes=5)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")