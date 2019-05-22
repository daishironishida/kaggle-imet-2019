import random
import math

from PIL import Image, ImageOps
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomResizedCrop, ColorJitter, Pad)


class SquarePad(object):
    def __init__(self):
        pass

    def __call__(self, img):
        w, h = img.size
        extra = int((w-h) % 2)

        if w>h:
            padding=(0, (w-h)//2 + extra, 0, (w-h)//2)
        else:
            padding=((h-w)//2 + extra, 0, (h-w)//2, 0)
        return ImageOps.expand(img, padding)

def get_transforms(transform_type, image_size):
    if transform_type == 'pad':
        train_transform = Compose([
            ColorJitter(),
            RandomHorizontalFlip(p=0.5),
            SquarePad(),
            Resize(image_size)
        ])

        test_transform = Compose([
            SquarePad(),
            Resize(image_size)
        ])

    elif transform_type == 'pad_tta':
        train_transform = Compose([
            ColorJitter(),
            RandomHorizontalFlip(p=0.5),
            SquarePad(),
            Resize(image_size)
        ])

        test_transform = Compose([
            RandomHorizontalFlip(p=0.5),
            SquarePad(),
            Resize(image_size)
        ])

    elif transform_type == 'crop':
        train_transform = Compose([
            ColorJitter(),
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop(image_size, scale=(0.7, 1.0))
        ])

        test_transform = Compose([
            Resize(360),
            CenterCrop((image_size, image_size))
        ])

    elif transform_type == 'crop_tta':
        train_transform = Compose([
            ColorJitter(),
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop(image_size, scale=(0.7, 1.0))
        ])

        test_transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop(image_size, scale=(0.7, 1.0))
        ])

    elif transform_type == 'resize_crop':
        train_transform = Compose([
            Resize(image_size),
            RandomCrop(image_size),
            RandomHorizontalFlip(),
        ])

        test_transform = Compose([
            Resize(image_size),
            RandomCrop(image_size),
            RandomHorizontalFlip(),
        ])

    elif transform_type == 'variable_size':
        train_transform = RandomHorizontalFlip()
        test_transform = RandomHorizontalFlip()

    else:
        train_transform = Compose([
            RandomCrop(image_size),
            RandomHorizontalFlip(),
        ])

        test_transform = Compose([
            RandomCrop(image_size),
            RandomHorizontalFlip(),
        ])

    return train_transform, test_transform

tensor_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
