"""
Shared transform pipelines for training and eval/inference.

Import these in train.py / evaluate.py / the FastAPI backend so preprocessing
stays consistent everywhere:

    from data.augment import train_transforms, eval_transforms
"""

from torchvision import transforms

IMAGE_SIZE = 224

# Standard ImageNet stats — fine to reuse since we're fine-tuning a model
# that was originally pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Deliberately no hue/saturation jitter here — skin tone and lesion coloring
# carry real diagnostic signal, so we don't want the augmentation pipeline
# distorting it. Brightness/contrast jitter is left in since lighting
# conditions vary a lot across the ISIC source images.
train_transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

eval_transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)
