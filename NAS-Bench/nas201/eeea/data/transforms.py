import numpy as np
import torchvision.transforms as transforms
import torch
from eeea.data.autoaugment import CIFAR10Policy

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def multi_transform(transform_fn, duplicates=1, dim=0):
    """preforms multiple transforms, useful to implement inference time augmentation or
     "batch augmentation" from https://openreview.net/forum?id=H1V4QhAqYQ&noteId=BylUSs_3Y7
    """
    if duplicates > 1:
        return transforms.Lambda(lambda x: torch.stack([transform_fn(x) for _ in range(duplicates)], dim=dim))
    else:
        return transform_fn
      
def _data_transforms_cifar10_search(cutout=True, autoaugment=True):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if cutout:
    train_transform.transforms.append(Cutout(16))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

  if args.autoaugment:
      train_transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          CIFAR10Policy(),
          transforms.ToTensor(),
          transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])

  if args.cutout:
    train_transform.transforms.append(Cutout(16))

  if args.duplicates > 1:
    train_transform = multi_transform(train_transform)

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
  CIFAR_STD = [0.2673, 0.2564, 0.2762]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

  if args.autoaugment:
      train_transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          CIFAR10Policy(),
          transforms.ToTensor(),
          transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])

  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  if args.duplicates > 1:
    train_transform = multi_transform(train_transform)

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def _data_transforms_cinic10(args):
  CIFAR_MEAN = [0.47889522, 0.47227842, 0.43047404]
  CIFAR_STD = [0.24205776, 0.23828046, 0.25874835]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

  if args.autoaugment:
      train_transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          CIFAR10Policy(),
          transforms.ToTensor(),
          transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])

  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))
  
  if args.duplicates > 1:
    train_transform = multi_transform(train_transform)
    
  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform