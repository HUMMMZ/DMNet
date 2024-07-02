from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
import os
from PIL import Image
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import ndimage
from utils import *
import imgaug.augmenters as iaa

def channel_shuffle(img):
    if (img.shape[2] == 3):
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr)
        img = img[..., ch_arr]
    return img


def EqExtension(src):
    I_backup = src.copy()
    b, g, r = cv2.split(I_backup)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    I_eq = cv2.merge([b, g, r])
    return I_eq.astype(np.uint8)


def HueExtension(src):
    img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    cre = random.randint(90, 100)
    cre = float(cre) / 100
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * cre

    # print(img_hsv[:,:,0])
    dst = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return dst.astype(np.uint8)


# 随机生成500个椒盐噪声
def zaodian(img):
    height, weight, channel = img.shape
    img_zao = img.copy()
    for i in range(500):
        x = np.random.randint(0, height)
        y = np.random.randint(0, weight)
        img_zao[x, y, :] = 255
    return img_zao.astype(np.uint8)


def random_noise(img, limit=[0, 0.2], p=0.5):
    if random.random() < p:
        H, W = img.shape[:2]
        noise = np.random.uniform(limit[0], limit[1], size=(H, W)) * 255

        img = img + noise[:, :, np.newaxis] * np.array([1, 1, 1])
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def random_brightness(image, brightness=0.3):
    alpha = 1 + np.random.uniform(-brightness, brightness)
    img = alpha * image
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def random_contrast(img, contrast=0.3):
    coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img = alpha * img + gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def random_saturation(img, saturation=0.5):
    coef = np.array([[[0.299, 0.587, 0.114]]])
    alpha = np.random.uniform(-saturation, saturation)
    gray = img * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    img = alpha * img + (1.0 - alpha) * gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def random_hue(image, hue=0.5):
    h = int(np.random.uniform(-hue, hue) * 180)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def superpixelsaug(img):
    images = np.expand_dims(img, axis=0)
    aug = iaa.Superpixels(p_replace=0.005, max_size=50)
    images_aug = aug(images=images)
    img_aug = np.squeeze(images_aug)
    return img_aug.astype(np.uint8)


def fogaug(img):
    images = np.expand_dims(img, axis=0)
    aug = iaa.Fog()
    images_aug = aug(images=images)
    img_aug = np.squeeze(images_aug)
    return img_aug.astype(np.uint8)


def cloudsaug(img):
    images = np.expand_dims(img, axis=0)
    aug = iaa.Clouds()
    images_aug = aug(images=images)
    img_aug = np.squeeze(images_aug)
    return img_aug.astype(np.uint8)


def fnaug(img):
    images = np.expand_dims(img, axis=0)
    aug = iaa.BlendAlphaFrequencyNoise(foreground=iaa.EdgeDetect(0.5))
    images_aug = aug(images=images)
    img_aug = np.squeeze(images_aug)
    return img_aug.astype(np.uint8)


def Coarseaug(img):
    images = np.expand_dims(img, axis=0)
    aug = iaa.CoarseDropout(0.02, size_percent=0.5)
    images_aug = aug(images=images)
    img_aug = np.squeeze(images_aug)
    return img_aug.astype(np.uint8)

# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


## Temporary
class tn3k_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train = True, Test = False):
        super(tn3k_loader, self)
        self.train = train
        if train:   
          self.data   = np.load(path_Data+'data_train.npy')          
          self.mask   = np.load(path_Data+'mask_train.npy')
        else:
          if Test:
            self.data   = np.load(path_Data+'data_test.npy')
            self.mask   = np.load(path_Data+'mask_test.npy')
          else:
            self.data   = np.load(path_Data+'data_val.npy')
            self.mask   = np.load(path_Data+'mask_val.npy')          
        
        self.data   = dataset_normalized(self.data)
        self.mask   = np.expand_dims(self.mask, axis=3)
        self.mask   = self.mask/255.

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)
            range_num = random.randint(1, 2)
            for iii in range(range_num):
                rand_num = random.randint(0, 20)
                img = np.asarray(img, dtype=np.uint8)
                if rand_num == 0:
                    img = channel_shuffle(img)
                elif rand_num == 1:
                    img = random_noise(img, limit=[0, 0.2], p=0.5)
                elif rand_num == 2:
                    img = random_brightness(img, brightness=0.3)
                elif rand_num == 3:
                    img = random_contrast(img, contrast=0.3)
                elif rand_num == 4:
                    img = random_saturation(img, saturation=0.5)
                elif rand_num == 5:
                    img = EqExtension(img)
                elif rand_num == 6:
                    img = HueExtension(img)
                elif rand_num == 7:
                    img = zaodian(img)
                elif rand_num == 8:
                    img = superpixelsaug(img)
                elif rand_num == 9:
                    img = fogaug(img)
                elif rand_num == 10:
                    img = cloudsaug(img)
                elif rand_num == 11:
                    img = fnaug(img)
                elif rand_num == 12:
                    img = Coarseaug(img)
                elif rand_num == 13:
                    img = random_hue(img)
                else:
                    img = img
            img = img.astype(np.float64)
        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        img = img.permute( 2, 0, 1)
        seg = seg.permute( 2, 0, 1)

        return img, seg
    
    def random_rot_flip(self,image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label
    
    def random_rotate(self,image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label


               
    def __len__(self):
        return len(self.data)
    
