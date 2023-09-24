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

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim


class Run():
    def __init__(self, net, loaders, lr, dataset_len, class_list, name):
        self.net = net
        self.loaders = loaders
        self.dataset_len = dataset_len
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.class_list = class_list
        self.name = name

    def gpu_dev(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")

        elif torch.cuda.is_available():
            device = torch.device("cuda")

        else:
            device = torch.device("cpu")

        return device

    def train(self):
        print('Training ' + self.name)
        device = self.gpu_dev()

        self.net.train()

        self.total_loss, self.correct_prediction = 0, 0

        for i, (images, labels) in enumerate(tqdm.tqdm(self.loaders)):
            train_imgs, train_labels = images.to(device), labels.to(device)

            train_prediction = self.net(train_imgs)

            train_loss = self.loss_func(train_prediction, train_labels)

            self.optimizer.zero_grad()

            train_loss.backward()

            self.optimizer.step()

            self.total_loss += train_loss
            self.total_loss /= (i + 1)

            self.real_vec, self.prediction_vec = self.collect_indices(train_labels, train_prediction)
            self.tally_correct()

        accuracy = torch.round(100 * (self.correct_prediction / (self.dataset_len * 0.8)))

        print(f"Train loss: {float(self.total_loss)}, Train Accuracy: {accuracy}%")

        return self.total_loss, accuracy

    def test(self, save=False):
        self.net.eval()

        self.total_loss, self.correct_prediction, self.best_acc = 0, 0, 0
        self.conf_prediction, self.conf_true = [], []
        self.snr_data, self.res = torch.tensor([]), torch.tensor([])

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm.tqdm(self.loaders)):
                test_imgs, test_labels = images.device, labels.device

                test_prediction = self.net(test_imgs)

                test_loss = self.loss_func(test_prediction, test_labels)

                self.total_loss += test_loss
                self.total_loss /= (i + 1)

                self.real_vec, self.prediction_vec = self.collect_indices(test_labels, test_prediction)

                self.tally_correct()

                output_image = self.output_image(images)

                self.conf_inputs()

            accuracy = torch.round(100 * (self.correct_prediction / (self.dataset_len * 0.2)))

            if accuracy > self.best_acc:
                self.best_acc

                if save:
                    torch.save(self.net, self.name + '.pth')

        self.build_confusion_matrix()

        print(f"Test loss: {round(float(self.total_loss), 7)}, Test accuracy: {accuracy}%")

        return self.total_loss, accuracy, output_image
