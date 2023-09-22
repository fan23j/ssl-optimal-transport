import os
import numpy as np
import torch
import random
import torch.utils.data as data
import clip

from PIL import Image, ImageDraw
from randaugment import RandAugment
from torchvision import transforms


class NUSWIDEClassification(data.Dataset):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        self.root = root
        self.path_images = os.path.join(root, "data/nuswide_81/images")
        self.sampler = sampler
        # Read categories
        with open(os.path.join(root, "cats.txt"), "r") as f:
            self.all_categories = [line.strip() for line in f.readlines()]

        self.class_labels = self.all_categories
        descriptions = [
            "a photo that contains a " + category for category in self.all_categories
        ]
        # Tokenize
        self.multilabel_text_inputs = torch.cat(
            [clip.tokenize(description) for description in descriptions]
        )
        # Read data
        data_txt = os.path.join(root, "database.txt" if train else "test.txt")
        self.data = []
        with open(data_txt, "r") as f:
            for line in f.readlines():
                tokens = line.strip().split()
                image_path = os.path.join(root, tokens[0])
                labels = list(map(int, tokens[1:]))
                self.data.append((image_path, labels))

        # Transforms
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
        self.ratios = self.calculate_ratios()

#     def calculate_ratios(self):
#         # Initialize a zero array for each category
#         category_counts = np.zeros(len(self.all_categories))
        
#         # Count positive labels for each category
#         for _, labels in self.data:
#             category_counts += np.array(labels)

#         # Calculate the ratio
#         total_samples = len(self.data)
#         ratios = category_counts / total_samples
#         return ratios.tolist() 
    
    
    def calculate_ratios(self):
        # Initialize a zero array for each category
        category_counts = np.zeros(len(self.all_categories))

        # Initialize a variable to store the total number of labels
        total_labels = 0

        # Count positive labels for each category
        for _, labels in self.data:
            category_counts += np.array(labels)
            total_labels += sum(labels)  # Assuming labels is a list of binary values [1, 0, 1, ...]

        # Check if total_labels is not zero to avoid division by zero error
        if total_labels == 0:
            return [0] * len(self.all_categories)

        # Calculate the ratio
        ratios = category_counts / total_labels
        return ratios.tolist()
    
    def __getitem__(self, index):
        path, labels = self.data[index]
        img = Image.open(path).convert("RGB")
        target = torch.tensor(labels)

        if self.sampler is not None:
            return self.sampler.sample(self, img, target)
        if self.transform is not None:
            img = self.transform(img)

        return {"out_1": img, "target": target}

    def __len__(self):
        return len(self.data)

    def get_number_classes(self):
        return len(self.all_categories)


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
