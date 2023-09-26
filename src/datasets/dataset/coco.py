import numpy as np
import os
import torch
import torchvision.transforms as transforms
import random
import clip
import math

from torchvision import datasets
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
from randaugment import RandAugment


class Coco(datasets.coco.CocoDetection):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        self.root = root
        self.name = "coco"
        self.annotation_file = (
            cfg.DATASET.TRAIN_ANNOTATIONS if train else cfg.DATASET.VAL_ANNOTATIONS
        )
        self.coco = COCO(os.path.join(root, self.annotation_file))
        self.img_dir = os.path.join(
            root, cfg.DATASET.TRAIN_IMAGE_DIR if train else cfg.DATASET.VAL_IMAGE_DIR
        )
        self.ids = list(self.coco.imgToAnns.keys())
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
        self.target_transform = None
        self.transform = self.train_transform if train else self.test_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        self.all_categories = [
            cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        multilabel_descriptions = [
            "a photo that contains a " + category for category in self.all_categories
        ]
        multiclass_descriptions = [
            "a photo of a " + category for category in self.all_categories
        ]
        # Tokenize
        self.multilabel_text_inputs = torch.cat(
            [clip.tokenize(description) for description in multilabel_descriptions]
        )
        self.multiclass_text_inputs = torch.cat(
            [clip.tokenize(description) for description in multiclass_descriptions]
        )
        self.ratios = self.compute_label_distribution()
        self.neg_ratios = self.compute_positive_negative_ratios()
        self.log_ratios = self.compute_log_ratios()
        self.inverse_ratios = self.compute_inverse_ratios()
        self.weighted_ratios = self.compute_weighted_ratios()
        self.pairwise_ratios = self.compute_pairwise_label_ratios()

        self.sampler = sampler
        self.class_labels = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        
    def compute_label_distribution(self):
        label_counts = [0] * len(self.cat2cat)
        total_images = len(self.ids)
        for index in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=index)
            anns = self.coco.loadAnns(ann_ids)
            unique_cats = set()
            for ann in anns:
                cat_index = self.cat2cat[ann['category_id']]
                unique_cats.add(cat_index)
                
            for cat_index in unique_cats:
                label_counts[cat_index] += 1


        ratios = [count / total_images for count in label_counts]
        return ratios
    
    def compute_inverse_ratios(self):
        label_counts = [0] * len(self.cat2cat)
        total_images = len(self.ids)
        for index in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=index)
            anns = self.coco.loadAnns(ann_ids)
            unique_cats = set()
            for ann in anns:
                cat_index = self.cat2cat[ann['category_id']]
                unique_cats.add(cat_index)

            for cat_index in unique_cats:
                label_counts[cat_index] += 1

        ratios = [1 - (count / total_images) for count in label_counts]
        return ratios
    
    
    def compute_positive_negative_ratios(self):
        label_counts = [0] * len(self.cat2cat)
        negative_counts = [0] * len(self.cat2cat)
        total_images = len(self.ids)

        for index in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=index)
            anns = self.coco.loadAnns(ann_ids)
            unique_cats = set()
            for ann in anns:
                cat_index = self.cat2cat[ann['category_id']]
                unique_cats.add(cat_index)

            for cat_index in range(len(self.cat2cat)):
                if cat_index in unique_cats:
                    label_counts[cat_index] += 1
                else:
                    negative_counts[cat_index] += 1

        ratios = [label_counts[i] / (label_counts[i] + negative_counts[i]) for i in range(len(self.cat2cat))]
        return ratios
    
    def compute_log_ratios(self):
        label_counts = [0] * len(self.cat2cat)
        total_images = len(self.ids)

        for index in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=index)
            anns = self.coco.loadAnns(ann_ids)
            unique_cats = set()
            for ann in anns:
                cat_index = self.cat2cat[ann['category_id']]
                unique_cats.add(cat_index)

            for cat_index in unique_cats:
                label_counts[cat_index] += 1

        ratios = [count / total_images + 1e-6 for count in label_counts]  # Added epsilon to avoid division by zero
        log_ratios = [math.log(ratio) for ratio in ratios]  # Compute log ratios

        # Offset the Log Ratios
        min_log_ratio = min(log_ratios)
        epsilon = 1e-10
        offset_log_ratios = [x - min_log_ratio + epsilon for x in log_ratios]

        # Min-Max Scaling to [min_bound, max_bound]
        min_bound, max_bound = 0.01, 0.99  # Adjust as needed
        min_val = min(offset_log_ratios)
        max_val = max(offset_log_ratios)

        scaled_log_ratios = [
            min_bound + (x - min_val) * (max_bound - min_bound) / (max_val - min_val)
            for x in offset_log_ratios
        ]

        return scaled_log_ratios

    
    def compute_pairwise_label_ratios(self):
        total_images = len(self.ids)
        label_len = len(self.cat2cat)
        
        # Initialize a dictionary to store the count for each pair of labels
        label_pair_counts = defaultdict(int)
        
        # Iterate over each image in the dataset
        for index in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=index)
            anns = self.coco.loadAnns(ann_ids)
            
            # Create a set representing the unique presence of each label in the image
            present_labels = set(self.cat2cat[ann['category_id']] for ann in anns)
            
            # Iterate over all possible label combinations
            for label_i, label_j in combinations(range(label_len), 2):
                # Check condition Yi + Yj <= 1, meaning at most one of the labels is present
                if (label_i in present_labels) != (label_j in present_labels):
                    label_pair_counts[frozenset((label_i, label_j))] += 1
        
        # Calculate ratios
        label_pair_ratios = {pair: count / total_images for pair, count in label_pair_counts.items()}
        
        return label_pair_ratios


        
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros(80, dtype=torch.long)
        for obj in target:
            output[self.cat2cat[obj["category_id"]]] = 1
        target = output
        path = coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.img_dir, path)).convert("RGB")
        if self.sampler is not None:
            return self.sampler.sample(self, img, target)

        if self.transform is not None:
            img = self.transform(img)

        return {"out_1": img, "target": target}


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
