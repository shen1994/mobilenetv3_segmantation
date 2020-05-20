"""Cityscapes Dataloader"""
import os
import cv2
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['CitySegmentation']


class CitySegmentation(data.Dataset):
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """

    def __init__(self, root='../dataset/', split='train', mode=None, transform=None,
                 base_size=520, crop_size=720, **kwargs):
        super(CitySegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _val_sync_transform(self, img, mask):
        '''
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        '''
        #img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.BILINEAR)
        #mask = mask.resize((img.size[0] // 2, img.size[1] // 2), Image.NEAREST)
        #image_width = 4000 - img.size[0]
        #image_height = 720 - img.size[1]
        #img = ImageOps.expand(img, border=(0, 0, image_width, image_height), fill=0)
        #mask = ImageOps.expand(mask, border=(0, 0, image_width, image_height), fill=0)
        # center crop
        '''
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        '''
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img_width = img.size[0]
        img_height = img.size[1]

        target_width = 4000
        target_height = 2000
        if img.size[0] > target_width and img.size[1] > target_height:
            half_width = (img.size[0] - target_width) // 2
            half_height = (img.size[1] - target_height) // 2
            img = img.crop((half_width, half_height, half_width + target_width, half_height + target_height))
            mask = mask.crop((half_width, half_height, half_width + crop_size, half_height + target_height))
        elif img.size[0] > target_width and img.size[1] < target_height:
            half_width = (img.size[0] - target_width) // 2
            img = img.crop(half_width, 0, half_width + target_width, img.size[1])
            mask = mask.crop(half_width, 0, half_width + target_width, img.size[1])
            image_height = target_height - img.size[1]
            img = ImageOps.expand(img, border=(0, 0, img.size[0], image_height), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, img.size[0], image_height), fill=0)
        elif img.size[0] < target_width and img.size[1] > target_height:
            half_height = (img.size[1] - target_height) // 2
            img = img.crop(0, half_height, img.size[0], half_height + target_height)
            mask = mask.crop(0, half_height, img.size[0], half_height + target_height)
            image_width = target_width - img.size[0]
            img = ImageOps.expand(img, border=(0, 0, img.size[0], half_height + target_height))
            mask = ImageOps.expand(mask, border=(0, 0, img.size[0], half_height + target_height))
        else:
            image_width = target_width - img.size[0]
            image_height = target_height - img.size[1]
            img = ImageOps.expand(img, border=(0, 0, image_width, image_height), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, image_width, image_height), fill=0)

        crop_size = self.crop_size

        x1 = random.randint(0, img_width - crop_size)
        y1 = random.randint(0, img_height - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        #import cv2
        #cv2.imshow("image", np.array(img))
        #cv2.waitKey(1)
        
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskpath = os.path.join(mask_folder, filename.replace("jpg", "png"))
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'rgb/' + split)
        mask_folder = os.path.join(folder, 'label/' + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'rgb/train')
        train_mask_folder = os.path.join(folder, 'label/train')
        val_img_folder = os.path.join(folder, 'rgb/val')
        val_mask_folder = os.path.join(folder, 'label/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = CitySegmentation()
    img, label = dataset[0]
