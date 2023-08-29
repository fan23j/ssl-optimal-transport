import numpy as np
import os
import torch
import torchvision.transforms as transforms
import random
import clip

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
        descriptions = [
            "a photo that contains a " + category for category in self.all_categories
        ]
        # Tokenize
        self.text_inputs = torch.cat(
            [clip.tokenize(description) for description in descriptions]
        )
        self.sampler = sampler

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
