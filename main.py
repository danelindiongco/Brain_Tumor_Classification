import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
#from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import math
import cv2
# import scipy.io

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from train_test import Run
from build_dataset import ModelDataset


def loadData(dir='/Users/danelindiongco/PycharmProjects/BrainTumorClassification/dataset'):
    train_dir = dir + '/Training'
    train_img_paths = []
    train_labels = []

    test_dir = dir + '/Testing'
    test_img_paths = []
    test_labels = []

    for classname in os.listdir(train_dir):
        classfolder = os.path.join(train_dir, classname)

        for filename in os.listdir(classfolder):
            if filename.endswith('.jpg'):
                img_path = os.path.join(classfolder, filename)
                train_img_paths.append(img_path)

                train_labels.append(classname)

    for classname in os.listdir(test_dir):
        classfolder = os.path.join(test_dir, classname)

        for filename in os.listdir(classfolder):
            if filename.endswith('.jpg'):
                img_path = os.path.join(classfolder, filename)
                test_img_paths.append(img_path)

                test_labels.append(classname)

    return train_img_paths, test_img_paths, train_labels, test_labels


class Options():
    def __init__(self, class_list, epoch=100, lr=0.01):
        self.epoch = epoch
        self.learning_rate = lr

        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.fc = nn.Linear(512, len(class_list))

        self.run_name = 'test'


if __name__ == '__main__':
    train_img_paths, test_img_paths, train_labels, test_labels = loadData()
    class_list = list(set(train_labels))
    opt = Options(class_list)

    train_dataset = ModelDataset(train_img_paths, train_labels, class_list, train=True)
    test_dataset = ModelDataset(test_img_paths, test_labels, class_list)

    train_dataloader = DataLoader(train_dataset, batch_size=100)
    test_dataloader = DataLoader(test_dataset, batch_size=10)

    train = Run(opt.net, opt.learning_rate, opt.run_name,
                train_dataloader, test_dataloader,
                len(train_dataset), len(test_dataset),
                class_list)

    for epoch in range(100):
        print('Epoch: ' + str(epoch + 1))
        train_loss, train_accuracy = train.train()
        test_loss, test_accuracy, output_img = train.test(save=True)

    x = 1

