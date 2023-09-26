import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
from sklearn.metrics import confusion_matrix
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
    def __init__(self, net, learning_rate, name, train_loader, test_loader, train_len, test_len, class_list):
        self.net = net
        self.device = self.gpu_dev()
        self.net.to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.train_len = train_len
        self.test_len = test_len

        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

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
        self.net.train()
        print('Training ' + self.name)

        self.total_loss, self.correct_prediction = 0, 0

        for i, (images, labels) in enumerate(tqdm.tqdm(self.train_loader)):
            train_imgs, train_labels = images.to(self.device), labels.to(self.device)

            train_prediction = self.net(train_imgs)

            train_loss = self.loss_func(train_prediction, train_labels)

            self.optimizer.zero_grad()

            train_loss.backward()

            self.optimizer.step()

            self.total_loss += train_loss
            self.total_loss /= (i + 1)

            self.gt_vec, self.prediction_vec = self.collect_indices(train_labels, train_prediction)
            self.tally_correct()

        accuracy = torch.round(100 * (self.correct_prediction / self.train_len))

        print(f"Train loss: {float(self.total_loss)}, Train Accuracy: {accuracy}%")

        return self.total_loss, accuracy

    def test(self, save=False):
        self.net.eval()
        print('Testing ' + self.name)

        self.test_total_loss, self.test_correct_prediction, self.best_acc = 0, 0, 0
        self.conf_prediction, self.conf_true = [], []
        #self.snr_data, self.res = torch.tensor([]), torch.tensor([])

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm.tqdm(self.test_loader)):
                test_imgs, test_labels = images.to(self.device), labels.to(self.device)

                test_prediction = self.net(test_imgs)

                test_loss = self.loss_func(test_prediction, test_labels)

                self.test_total_loss += test_loss
                self.test_total_loss /= (i + 1)

                self.gt_vec, self.prediction_vec = self.collect_indices(test_labels, test_prediction)

                self.test_correct_prediction += (self.prediction_vec == self.gt_vec).sum()

                output_image = self.output_img(images)

                self.confusion_matrix_inputs()

            accuracy = torch.round(100 * (self.test_correct_prediction / self.test_len))

            if accuracy > self.best_acc:
                self.best_acc
                if save:
                    torch.save(self.net, self.name + '.pth')

        self.build_confusion_matrix()

        print(f"Test loss: {round(float(self.total_loss), 7)}, Test accuracy: {accuracy}%")

        return self.test_total_loss, accuracy, output_image

    def collect_indices(self, labels, predictions):
        _, ground_truth_indices = torch.max(labels, 1)
        _, prediction_indices = torch.max(predictions, 1)

        return ground_truth_indices, prediction_indices

    def tally_correct(self):
        self.correct_prediction += (self.prediction_vec == self.gt_vec).sum()

    def output_img(self, img):
        fig = plt.figure(1)
        plt.imshow(img[0].permute(1,2,0), cmap='Greys')

        gt_img_index = int(self.gt_vec[0].cpu())
        pred_img_index = int(self.prediction_vec[0].cpu())

        fig.suptitle('Ground truth: ' + self.class_list[gt_img_index] + '\nPredicted: ' + self.class_list[pred_img_index])
        plt.axis('off')

        return fig

    def confusion_matrix_inputs(self):
        self.conf_true.extend(self.gt_vec.cpu().numpy())
        self.conf_prediction.extend(self.prediction_vec.cpu().numpy())

    def build_confusion_matrix(self):
        confusionmatrix = confusion_matrix(self.conf_true, self.conf_prediction)
        df_cm = pd.DataFrame(confusionmatrix, index=self.class_list, columns=self.class_list)

        plt.figure(2, figsize=(12,7))

        sns.heatmap(df_cm, annot=True, fmt='.0f', cbar=False)

        plt.savefig(self.name + ' confusion matrix.png')
        plt.clf()
