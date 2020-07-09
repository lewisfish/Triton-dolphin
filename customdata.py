import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

__all__ = ["DolphinDataset", "DolphinDatasetClass"]

# example data item
# 錄製_2019_11_28_12_05_07_124.mp4, 30440, 749, 550, 758, 556, 10
# filename, framenumber, y0, x0, y1, x1, label
# cavet is that y1 and y2 offset by 130 due to cropping of screen recording
# this wont be true for all data after deploy though
# just true of the train, test, validation sets.


class DolphinDataset(object):
    """docstring for DolphinDataset for purpiose of object detection"""
    def __init__(self, root, transforms, file, allLabels=False):
        super(DolphinDataset, self).__init__()
        self.root = Path(root)
        self.transforms = transforms
        self.datafile = file
        self.videoFiles = list(self.root.glob("**/*.mp4"))
        self.allLabels = allLabels

        self.labels = []
        self.frameNumbers = []
        self.bboxs = []
        self.videoFileNames = []

        if "json" in self.datafile:
            indict = {}
            self.data = []
            with open(self.datafile, "r") as fin:
                indict = json.load(fin)
            for k, v in indict.items():
                for key, value in indict[k].items():
                    videoName = self._getFullFileName(k)
                    self.data.append([videoName, int(key), value["boxes"], value["labels"]])

        # else:
        #     # load label file into memory
        #     with open(self.datafile, "r") as f:
        #         lines = f.readlines()
        #         for line in lines:
        #             parts = line.split(",")
        #             videoName = self._getFullFileName(parts[0])
        #             self.videoFileNames.append(videoName)
        #             self.frameNumbers.append(int(parts[1]))
        #             self.bboxs.append([int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])])
        #             self.labels.append(int(parts[6]))

    def _getFullFileName(self, target):
        '''Get the full filename path'''

        for file in self.videoFiles:
            if target in str(file):
                return file

    def __getitem__(self, idx):

        # cap = cv2.VideoCapture(str(self.videoFileNames[idx]))  # converts to RGB by default
        # cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumbers[idx])
        cap = cv2.VideoCapture(str(self.data[idx][0]))  # converts to RGB by default
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.data[idx][1])

        _, image = cap.read()
        cap.release()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        ymax, xmax = image.shape[0], image.shape[1]

        target = {}

        bboxs = []
        labels = []
        areas = []
        for i in range(len(self.data[idx][3])):
            # data in format of
            # y0, x0, y1, x1
            # +130 is to compensate for cropping of frames in object
            # candidate generation
            top = self.data[idx][2][i][0] + 130
            left = self.data[idx][2][i][1]
            bottom = self.data[idx][2][i][2] + 130
            right = self.data[idx][2][i][3]
            # rcnn needs boxes in format of
            # x1, y1, x2, y2
            bbox = [left, top, right, bottom]
            area = np.abs(right - left) * np.abs(bottom - top)
            bboxs.append(bbox)
            areas.append(area)

            # labels = {0: "dolphin", 1: "bird", 2: "multi Dolphin", 3: "whale", 4: "turtle", 5: "unknown", 6: "unknown not cetacean", 7: "boat", 8: "fish", 9: "trash", 10: "water"}
            label = self.data[idx][3][i]
            # if allLabels is False then merge all labels so that have
            # dolphin and not dolphin classes.
            if not self.allLabels:
                if label == 1 or label >= 3:
                    label = 1
                else:
                    label = 0
            label += 1  # as 0 is background
            labels.append(label)

        target["boxes"] = torch.as_tensor(bboxs, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.zeros(len(labels), dtype=torch.int64)
        tmp = torch.Tensor(len(labels))
        target["image_id"] = torch.as_tensor(idx, dtype=torch.int64)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.data)


class DolphinDatasetClass(object):
    """docstring for DolphinDatasetClass for image classification"""
    def __init__(self, root, transforms, file, allLabels=False):
        super(DolphinDatasetClass, self).__init__()
        self.root = root
        self.transforms = transforms
        self.datafile = file

        self.allLabels = allLabels

        self.labels = []
        self.imageNames = []
        self.bboxs = []
        self.velocities = []
        self.kmeans = []
        self.hdbscans = []

        # load label file into memory
        with open(self.datafile, "r") as f:
            line = f.readline()  # skip header
            lines = f.readlines()
            for line in lines:
                parts = line.split(",")
                videoName = parts[0][:-4]
                frameNumber = int(parts[1])
                x0 = parts[2]
                y0 = parts[3]

                imagename = self.root + videoName + "-" + str(frameNumber) + "-" + str(x0) + "-" + str(y0) + ".png"

                self.imageNames.append(imagename)
                self.bboxs.append([int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])])
                self.labels.append(int(parts[6]))
                self.velocities.append(float(parts[7]))
                self.kmeans.append(int(parts[8]))
                self.hdbscans.append(int(parts[9]))

    def _getFullFileName(self, target):
        '''Get the full filename path'''

        for file in self.videoFiles:
            if target in str(file):
                return file

    def __getitem__(self, idx):

        image = cv2.imread(self.imageNames[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # data in format of
        # y0, x0, y1, x1
        # +130 is to compensate for cropping of frames in object
        # candidate generation
        top = self.bboxs[idx][0] + 130
        left = self.bboxs[idx][1]
        bottom = self.bboxs[idx][2] + 130
        right = self.bboxs[idx][3]

        image = image[top:bottom, left:right, :]

        # labels = {0: "dolphin", 1: "bird", 2: "multi Dolphin", 3: "whale", 4: "turtle", 5: "unknown", 6: "unknown not cetacean", 7: "boat", 8: "fish", 9: "trash", 10: "water"}
        label = self.labels[idx]
        # if allLabels is False then merge all labels so that have
        # dolphin and not dolphin classes.
        if not self.allLabels:
            if label == 1 or label > 3:
                label = 1
            else:
                label = 0

        data = [self.velocities[idx], self.kmeans[idx], self.hdbscans[idx]]
        data = torch.as_tensor(data)
        target = torch.as_tensor(label, dtype=torch.int64)
        if self.transforms:
            PIL_image = Image.fromarray(image)
            image = self.transforms(PIL_image)

        return image, target, data

    def __len__(self):
        return len(self.labels)
