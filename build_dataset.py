import math
import PIL.Image as Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


# Functions to load dataset
class ModelDataset(Dataset):
    def __init__(self, img_paths, labels, class_list, metadata=None, train=False):
        self.img_paths = img_paths  # A list of image paths
        self.labels = labels  # A corresponding list of labels for each image
        self.class_list = class_list  # A set of all possible classes
        self.train = train  # A mode to activate if dataset is for training
        self.img_size = 512 # Input image size - can test the effects of reduced resolutions

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = []

        self.transform = T.Grayscale()
        self.random_transform = T.Compose([T.RandomPerspective(distortion_scale=0.6, p=0.3),
                                           T.RandomApply(torch.nn.ModuleList([T.RandomRotation(degrees=(-45, 45))]), p=0.5)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        res = self.img_size

        img_path = self.img_paths[idx]
        img = self.openImg(img_path, res=res)

        class_name = self.labels[idx]
        class_index = self.class_list.index(class_name)
        label = torch.zeros(len(self.class_list))
        label[class_index] = 1

        if self.train:
            if 0 <= random.random() < 0.3:
                mixup_idx = math.floor(random.random() * len(self.img_paths))
                mixup_img_path = self.img_paths[mixup_idx]
                mixup_img = self.openImg(mixup_img_path, res)

                mixup_class_name = self.labels[mixup_idx]
                mixup_class_index = self.class_list.index(mixup_class_name)

                img, label = self.mixup(img, mixup_img, class_index, mixup_class_index)

            transformed_img = self.random_transform(torch.tensor(img).unsqueeze(0)).float()

        if not self.train:
            transformed_img = torch.tensor(img).unsqueeze(0).float()

        #upsampler = T.Resize((256, 256, 3))
        #upsampled_img = upsampler(transformed_img)

        # metadata = self.collect_metadata(self.metadata[idx], upsampled_img, res)

        return [transformed_img, label]


    def collect_metadata(self, metadata, img, res):
        metadata['SNR'] = self.calculateSNR(img)
        metadata['Resolution'] = res

        return metadata


    def openImg(self, img_path, res=None):
        img = Image.open(img_path)
        img = self.normaliseImg(img)
        img = img.astype(np.float32)

        if res is not None:
            img = cv2.resize(img, (res,res), cv2.INTER_LINEAR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img


    def normaliseImg(self, img):
        img = np.array(img)
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))

        return img_norm


    def mixup(self, original_img, mixup_img, original_label, mixup_label):
        mixup_coeff = 0.3

        new_img = ((1-mixup_coeff) * original_img) + ((mixup_coeff) * mixup_img)

        label_vec = torch.zeros(len(self.class_list))
        label_vec[original_label] = 1 - mixup_coeff
        label_vec[mixup_label] = mixup_coeff

        return new_img, label_vec


    def calculateSNR(self, img):
        img_array = img.numpy().transpose(1,2,0)

        roi_box = [100,172]

        roi = img_array[roi_box[0]:roi_box[1], roi_box[0]:roi_box[1]]

        roi_mean = np.mean(roi)

        background = np.concatenate(
            [img_array[:roi_box[0] - 50, :], img_array[roi_box[1] + 50:, :], img_array[:, :roi_box[0] - 50]],
            axis=None
        )

        background_std = np.std(background)
        if background_std <= 0:
            background_std = 0.00000001

        snr = roi_mean / background_std
        return snr
