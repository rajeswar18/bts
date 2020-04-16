# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>
import numpy as np
import torch
from skimage.transform import rescale
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random

from distributed_sampler_no_evenly_divisible import DistributedSamplerNoEvenlyDivisible


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms_in(mode):
    return transforms.Compose([ToTensorIN(mode=mode)])


class BtsDataLoaderIN(object):

    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocessIN(
                args, mode, transform=preprocessing_transforms_in(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None
            print("THREADS")
            print(args.num_threads)
            self.data = DataLoader(
                self.training_samples,
                args.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=args.num_threads,
                pin_memory=True,
                sampler=self.train_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocessIN(
                args, mode, transform=preprocessing_transforms_in(mode))
            self.data = DataLoader(
                self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'
                  .format(mode))

class DataLoadPreprocessIN(Dataset):

    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        with open(args.filenames_file, 'r') as f:
            self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensorIN
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        frame_idx = idx % 20
        img_idx = np.floor(idx / 20).astype(int)

        sample_path = self.filenames[img_idx]
        focal = 60.0

        if self.mode == 'train':
            image_path = os.path.join(self.args.data_path,
                                       sample_path.split()[0])
            depth_path = os.path.join(self.args.gt_path,
                                       sample_path.split()[1])

            image = np.array(Image.open(image_path))[480 * frame_idx:480 *
                                                     (frame_idx + 1), :, :]
            depth_gt = np.array(
                Image.open(depth_path),
                dtype=np.float32)[480 * frame_idx:480 * (frame_idx + 1), :]
            # print(image.shape, image.dtype)
            # print(depth_gt.shape, depth_gt.dtype)

            # # To avoid blank boundaries due to pixel registration
            # depth_gt = depth_gt.crop((43, 45, 608, 472)) # FIXME
            # image = image.crop((43, 45, 608, 472))

            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 100

            if self.args.smol:
                image = rescale(
                    image, 0.27, anti_aliasing=False,
                    multichannel=True).astype(np.float32)
                depth_gt = rescale(
                    depth_gt, 0.27, anti_aliasing=False,
                    multichannel=True).astype(np.float32)

            image, depth_gt = self.random_crop(image, depth_gt,
                                               self.args.input_height,
                                               self.args.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)

            # print(image.shape, image.dtype, image.min(), image.max(),
            #       image.mean())
            # print(depth_gt.shape, depth_gt.dtype, depth_gt.min(),
            #       depth_gt.max(), depth_gt.mean())
            # print(focal)

            sample = {'image': image, 'depth': depth_gt, 'focal': focal}
        else:
            data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):   
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image**gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)  # fixme
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames) * 20


class ToTensorIN(object):

    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        depth = self.to_tensor(depth)
        return {'image': image, 'depth': depth, 'focal': focal}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(
                type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


if __name__ == '__main__':

    class args:
        smol = True
        dataset = "IN"
        input_width = 64
        input_height = 64
        data_path = "/mnt/projects/3dgan/real3D/florian_bts_data/merged_chair_bts"
        gt_path = "/mnt/projects/3dgan/real3D/florian_bts_data/merged_chair_bts"
        filenames_file = "/mnt/projects/3dgan/real3D/florian_bts_data/chair_train_files_with_gt.txt"

    ds = DataLoadPreprocessIN(args, "train")
    a = ds[0]


# (128, 128, 3) float32 0.0 0.99557334 0.4302175
# (128, 128, 1) float32 0.0 26.169596 11.222618

# (128, 128, 3) float32 0.16809915 0.8800336 0.7275931
# (128, 128, 1) float32 1.7790145 3.0329669 2.5138416
# 600.0