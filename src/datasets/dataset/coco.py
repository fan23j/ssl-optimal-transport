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
from itertools import combinations


class Coco(datasets.coco.CocoDetection):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        self.root = root
        self.cfg = cfg
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
        # self.neg_ratios = self.compute_positive_negative_ratios()
        # self.log_ratios = self.compute_log_ratios()
        # self.inverse_ratios = self.compute_inverse_ratios()
        # self.weighted_ratios = self.compute_weighted_ratios()
        self.pairwise_ratios = self.compute_pairwise_ratios()

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

    def compute_pairwise_ratios(self):
        # Get all image ids from the dataset
        img_ids = self.coco.getImgIds()
        cat_ids = self.coco.getCatIds()
        cat_id_to_name = {
            cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }

        # Initialize label_ratios as a 2D list with zeros
        label_ratios = [[0] * len(img_ids) for _ in cat_ids]

        # Iterate over all images and update the label_ratios list with 1s where the label is present in the image
        for idx, img_id in enumerate(img_ids):
            img_ann_ids = self.coco.getAnnIds(imgIds=img_id)
            img_anns = self.coco.loadAnns(img_ann_ids)

            for ann in img_anns:
                label_id = ann["category_id"]
                label_idx = cat_ids.index(
                    label_id
                )  # Find the index of label_id in cat_ids
                label_ratios[label_idx][
                    idx
                ] = 1  # Update label_ratios with 1s where the label is present in the image

        # Initialize the pairwise_ratios dictionary
        pairwise_dict = {}
        total_imgs = len(img_ids)
        # Iterate over all unique pairs of categories
        for i, j in combinations(range(len(cat_ids)), 2):
            pairwise_list = [0] * len(
                img_ids
            )  # Initialize pairwise_list for the pair (i, j) with zeros


            for k in range(total_imgs):
                if label_ratios[i][k] + label_ratios[j][k] >= self.cfg.LOSS.SINKHORN_OT_PAIRWISE_ALPHA:
                    pairwise_list[k] = 1
            label_pair_key = f"{cat_ids[i]}-{cat_ids[j]}"
            # if sum(pairwise_list) >= self.cfg.LOSS.SINKHORN_OT_PAIRWISE_ALPHA * total_imgs:
            pairwise_dict[label_pair_key] = [self.cat2cat[cat_ids[i]],self.cat2cat[cat_ids[j]],sum(pairwise_list)/total_imgs]
        return pairwise_dict

        
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
