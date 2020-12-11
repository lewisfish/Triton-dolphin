import json
from pathlib import Path
from typing import Tuple

import av
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import minmax_scale
from sklearn.utils import shuffle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, datasets

__all__ = ["DolphinDataset", "DolphinDatasetClass", "getNumericalData", "ImageFolderWithPaths"]

# example data item
# 錄製_2019_11_28_12_05_07_124.mp4, 30440, 749, 550, 758, 556, 10
# filename, framenumber, y0, x0, y1, x1, label
# cavet is that y1 and y2 offset by 130 due to cropping of screen recording
# this wont be true for all data after deploy though
# just true of the train, test, validation sets.


def getNumericalData(filename: str, hdbscan=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function read and processes the numerical data for training on.
    
    Parameters
    ----------
    filename : str
        Name of file to read in

    hdbscan : bool, optional
        Description
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
    
    """

    df = pd.read_csv(filename)

    # shuffle data
    train = shuffle(df, random_state=49)
    train.reset_index(drop=True, inplace=True)

    # get pertinent parts
    if hdbscan:
        X_train = train[["velocity", "hdbscan"]]
    else:
        X_train = train[["velocity", "kmeans"]]

    Y_train = train["labels"]
    Y_train = Y_train.to_frame("labels")

    # relabel the labels sdo now a binary problem
    Y_train["labels"] = np.where(Y_train["labels"] >= 3, 1, Y_train["labels"])
    Y_train["labels"] = np.where(Y_train["labels"] != 1, 0, Y_train["labels"])

    # modifying scale of data (minMax and robust scale)
    x = X_train.to_numpy()
    x_scaled = minmax_scale(x)
    X_train = pd.DataFrame(x_scaled)

    return (X_train, Y_train)


class DolphinDataset(Dataset):
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

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
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


class DolphinDatasetClass(Dataset):
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

        # convert to numpy arrays
        self.velocities = np.array(self.velocities)
        self.kmeans = np.array(self.kmeans)
        self.hdbscans = np.array(self.hdbscans)

        # scale the features
        self.velocities = minmax_scale(self.velocities)
        self.hdbscans = minmax_scale(self.hdbscans)

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

        ymax, xmax = image.shape[0], image.shape[1]

        top = max(0, top - 5)
        bottom = min(ymax, bottom + 5)

        left = max(0, left - 5)
        right = min(xmax, right + 5)

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

        data = [self.velocities[idx], self.hdbscans[idx]]
        data = torch.as_tensor(data)
        target = torch.as_tensor(label, dtype=torch.int64)
        if self.transforms:
            PIL_image = Image.fromarray(image)
            image = self.transforms(PIL_image)

        return image, target, data

    def __len__(self):
        return len(self.labels)


class windowDataset(Dataset):
    """This dataset returns patches from still frame in a video feed. Currently only i-frames."""
    def __init__(self, file, transforms, size, stride):
        super(windowDataset, self).__init__()
        self.file = file
        self.transforms = transforms
        self.size = size
        self.stride = stride
        self.xpos = 0
        self.ypos = 0
        self.imageGen = self.getNextFrame()
        self.image, self.framenum = next(self.imageGen)
        self.numFrames = self.getNumberFrames()

    def __getitem__(self, idx):

        framenum = self.framenum
        image = self.image[self.ypos:self.ypos + self.size, self.xpos:self.xpos + self.size, :]

        # apply transforms if any
        if self.transforms:
            PIL_image = Image.fromarray(image)
            image = self.transforms(PIL_image)
        xpos, ypos = self.xpos, self.ypos
        # update parameters for next image
        self.xpos += self.stride
        if self.xpos >= self.image.shape[1]:
            self.xpos = 0
            self.ypos += self.stride
        if self.ypos >= self.image.shape[0]:
            self.image, self.framenum = next(self.imageGen)
            self.xpos, self.ypos = 0, 0

        return image, framenum, torch.tensor([xpos, ypos])

    def __len__(self):
        return int(self.numFrames * self.image.shape[0] * self.image.shape[1] / self.stride**2)

    def getNextFrame(self):
        with av.open(self.file) as container:
            # Signal that we only want to look at keyframes.
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = 'NONKEY'
            for frame in container.decode(stream):
                # convert and crop frame
                pts = frame.pts
                image = frame.to_image()
                image = np.array(image)
                # crop image to prespecified size
                image = image[130:1030, 0:1990, :]
                image = self.padImage(image)
                yield image, pts

    def padImage(self, image):
        # get padding if required
        if image.shape[1] // self.stride != image.shape[1] / self.stride:
            leftpad = (image.shape[1] // self.stride * self.stride + self.size) - image.shape[1]
            rightpad = leftpad // 2 if leftpad % 2 == 0 else leftpad // 2 + 1
            leftpad = leftpad // 2
        else:
            leftpad, rightpad = 0, 0

        if image.shape[0] // self.stride != image.shape[0] / self.stride:
            toppad = (image.shape[0] // self.stride * self.stride + self.size) - image.shape[0]
            bottompad = toppad // 2 if toppad % 2 == 0 else toppad // 2 + 1
            toppad = toppad // 2
        else:
            toppad, bottompad = 0, 0

        padding = ((toppad, bottompad), (leftpad, rightpad), (0, 0))
        image = np.pad(image, padding)

        return image

    def getNumberFrames(self):
        with av.open(self.file) as container:
            # Signal that we only want to look at keyframes.
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = 'NONKEY'
            for i, _ in enumerate(container.decode(stream)):
                continue
        return i + 1


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = str(Path(self.imgs[index][0]).stem)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
