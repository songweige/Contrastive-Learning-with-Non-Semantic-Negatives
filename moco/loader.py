# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import Image, ImageFilter
import numpy as np
import os
import torch
import random
from torchvision import transforms, utils, datasets

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class ImageNetFolder(datasets.VisionDataset):
    def __init__(self, root, augmentation, mode='patch', patch_size=32, n_non_sematic=10):
        super(ImageNetFolder, self).__init__(root, transform=augmentation, target_transform=None)
        classes, class_to_idx = self._find_classes(self.root)
        samples = datasets.folder.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            msg += "Supported extensions are: {}".format(",".join(IMG_EXTENSIONS))
            raise RuntimeError(msg)

        self.loader = datasets.folder.default_loader
        self.extensions = IMG_EXTENSIONS
        self.augmentation = augmentation
        self.mode = mode
        if mode == 'patch':
            self.nonsem_transform = NonSemanticAug(scale=patch_size, n_non_sematic=n_non_sematic)
        else:
            raise NotImplementedError("Only patch mode is supported. Got %s."%mode)

        self.n_non_sematic = n_non_sematic
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        q = self.augmentation(sample)
        k = self.augmentation(sample)
        t = torch.stack(self.nonsem_transform(q))
        return q, k, t, target

    def __len__(self):
        return len(self.samples)



class DuoImageNetFolder(datasets.VisionDataset):
    def __init__(self, root, augmentation):
        super(DuoImageNetFolder, self).__init__(root, transform=augmentation, target_transform=None)
        classes, class_to_idx = self._find_classes(self.root)
        samples = datasets.folder.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            msg += "Supported extensions are: {}".format(",".join(IMG_EXTENSIONS))
            raise RuntimeError(msg)

        self.loader = datasets.folder.default_loader
        self.extensions = IMG_EXTENSIONS
        self.augmentation = augmentation

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = [sample for sample in samples if 'texture2_n' not in sample]
        self.targets = [s[1] for s in samples]

        self.partA = '-texture'
        self.partB = ''


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        texture_sample = self.loader(path)
        duo_path = path.replace(self.partA, self.partB).replace('png', 'JPEG')
        duo_sample = self.loader(duo_path)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        q = self.augmentation(duo_sample)
        k = self.augmentation(duo_sample)
        # apply the same augmentation to the texture image
        random.seed(seed)
        torch.manual_seed(seed)
        t = self.augmentation(texture_sample)
        return q, k, t.unsqueeze(0), target

    def __len__(self):
        return len(self.samples)



class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)



class NonSemanticAug(object):
    """Apply non semantic augmentation to an image: randomly crop patches from the image and apply flips and rotations and 
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, scale=(16, 48), n_non_sematic=10):
        self.horflip_helper = transforms.RandomHorizontalFlip()
        self.verflip_helper = transforms.RandomVerticalFlip()
        self.rotate_helper = RandomRotation()
        self.n_non_sematic = n_non_sematic
        self.scale = scale

    def __call__(self, img_ori):
        img_reorders = []
        for _ in range(self.n_non_sematic):
            patch_size = random.randint(*self.scale)
            n_patch = 224 // patch_size + 1
            reorder_size = patch_size*n_patch
            img_reorder = torch.zeros((3, reorder_size, reorder_size))
            for i in range(n_patch):
                for j in range(n_patch):
                    w = np.random.randint(0, 224-patch_size)
                    h = np.random.randint(0, 224-patch_size)
                    # img_reorder[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = self.horflip_helper(self.verflip_helper(self.rotate_helper(img_ori[:, h:h+patch_size, w:w+patch_size])))
                    img_reorder[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = img_ori[:, h:h+patch_size, w:w+patch_size]
            img_reorders.append(img_reorder[:, :224, :224].clone())
        return img_reorders



class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class TwoCrops_TextureTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, texture_transform, patch_size=32, n_non_sematic=10):
        self.n_non_sematic = n_non_sematic
        self.base_transform = base_transform
        self.texture_transform = texture_transform
        self.n_patch = 224 // patch_size
    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        t = [self.texture_transform(q).repeat(1, 1, self.n_patch, self.n_patch).squeeze() for i in range(self.n_non_sematic)]
        out = [q, k] + t
        return out


class TwoCrops_Shuffle:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, patch_size=32, n_non_sematic=10):
        self.n_non_sematic = n_non_sematic
        self.base_transform = base_transform
        self.shuffle_transform = ShufflePatches(patch_size=patch_size, n_non_sematic=n_non_sematic)
    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        t = self.shuffle_transform(q)
        # t = [self.shuffle_transform(q) for i in range(self.n_non_sematic)]
        # print(len(t))
        out = [q, k] + t
        # print(len(out))
        return out



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
