import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import random
from PIL import Image
import imageio


class ImageDataSet(data.Dataset):

    def __init__(self, block_w, images=None, transform=None, stride=1):
        self.transform = transform
        self.images = images
        self.im_n, self.im_h, self.im_w = self.images.shape[0:3]
        self.stride = stride
        self.block_w = block_w
        self.blocks_in_image_h = (self.im_h - self.block_w) // stride + 1
        self.blocks_in_image_w = (self.im_w - self.block_w) // stride + 1
        self.len = self.im_n * self.blocks_in_image_h * self.blocks_in_image_w

        if len(self.images.shape) < 4:
            self.images = np.expand_dims(self.images, axis=3)

    def __getitem__(self, item):
        im, row, col = np.unravel_index(item, (self.im_n, self.blocks_in_image_h, self.blocks_in_image_w))
        row *= self.stride
        col *= self.stride
        sample = self.images[im, row:row + self.block_w, col:col + self.block_w, :]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.len


def load_image_from_file(in_path):
    transform = transforms.Compose([transforms.ToTensor(), ShiftImageValues()])
    image_c = np.array(imageio.imread(in_path))
    if len(image_c.shape) < 3:
        image_c = np.expand_dims(image_c, axis=2)
    image_c = transform(image_c)

    return image_c.unsqueeze(0)


class RandomTranspose(object):
    """Applies transpose the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be transposed.

        Returns:
            PIL Image: Randomly transposed image.
        """
        if random.random() < self.p:
            if not isinstance(img, Image.Image):
                raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

            return img.transpose(Image.TRANSPOSE)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ShiftImageValues(object):
    def __call__(self, img):
        return (img - 0.5) / 0.5

    def __repr__(self):
        return self.__class__.__name__
