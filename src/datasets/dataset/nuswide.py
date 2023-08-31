# adapted from https://github.com/wenting-zhao/nuswide

import csv
import os
import os.path
import tarfile
import torch.utils.data as data
from torchvision import transforms
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from randaugment import RandAugment
import pickle
import glob
from collections import defaultdict

fn_map = {}
for fn in glob.glob("/home/ubuntu/ssl-optimal-transport/data/nuswide/images/*.jpg"):
    tmp = fn.split('_')[1]
    fn_map[tmp] = fn


def read_info(root, set):
    imagelist = {}
    hash2ids = {}
    if set == "train": 
        path = os.path.join(root, "train_image_list.txt")
    elif set == "val":
        path = os.path.join(root, "val_image_list.txt")
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.split('\\')[-1]
            start = line.index('_')
            end = line.index('.')
            imagelist[i] = line[start+1:end]
            hash2ids[line[start+1:end]] = i

    return imagelist


def read_object_labels_csv(file, imagelist, fn_map, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = int(row[0])
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                name2 = fn_map[imagelist[name]]
                item = (name2, labels)
                images.append(item)
            rownum += 1
    return images


class NUSWIDEClassification(data.Dataset):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        self.root = root
        self.path_images = os.path.join(root, 'images')
        self.split = "train" if train else "val"
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
                CutoutPIL(cutout_factor=0.5),
                RandAugment(),
                transforms.ToTensor(),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )
        self.transform = self.train_transform if train else self.test_transform

        # define filename of csv file
        file_csv = os.path.join(self.root, 'classification_' + self.split + '.csv')
        imagelist = read_info(root, self.split)

        self.classes = 81
        self.images = read_object_labels_csv(file_csv, imagelist, fn_map)

        print('[dataset] NUSWIDE classification set=%s number of classes=%d  number of images=%d' % (
            self.split, self.classes, len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return self.classes

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x