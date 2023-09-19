import os
import torch
import clip

from torchvision import transforms
from PIL import Image
from randaugment import RandAugment
import numpy as np


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        self.root = root
        self.name = "tiny-imagenet"
        self.train = train

        # Get all the class IDs (wnids) and corresponding human-readable names
        with open(os.path.join(root, "wnids.txt"), "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        with open(os.path.join(root, "words.txt"), "r") as f:
            self.class_descriptions = {
                line.split("\t")[0]: line.split("\t")[1].strip()
                for line in f.readlines()
            }
        self.class_labels = [self.class_descriptions[class_id] for class_id in self.classes if class_id in self.class_descriptions]

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Read validation annotations if not training mode
        if not train:
            self.image_dir = os.path.join(root, "val", "images")
            with open(os.path.join(root, "val", "val_annotations.txt"), "r") as f:
                lines = f.readlines()
                self.data = [
                    (line.split("\t")[0], line.split("\t")[1]) for line in lines
                ]
        else:
            self.image_dir = os.path.join(root, "train")
            self.data = [
                (filename, label)
                for label in self.classes
                for filename in os.listdir(
                    os.path.join(self.image_dir, label, "images")
                )
            ]
            
        # LT
        if cfg.DATASET.LT_IMBALANCE_RATIO != 1.0:
            img_num_list = self.get_img_num_per_cls(len(self.classes), cfg.DATASET.LT_IMBALANCE_RATIO)
            self.gen_imbalanced_data(img_num_list, reverse=cfg.DATASET.LT_REVERSE)
            
        self.ratios = self.calculate_ratios([item[1] for item in self.data])

        # Set transformations
        optional_padding = OptionalPad(
            fill=0,
            padding_enabled=True,
            image_size=cfg.DATASET.IMAGE_SIZE,
        )

        self.train_transform = transforms.Compose(
            [
                optional_padding,
                transforms.Resize((cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
                RandAugment(),
                transforms.ToTensor(),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                optional_padding,
                transforms.Resize((cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )

        self.transform = self.train_transform if train else self.test_transform

        multilabel_descriptions = [
            f"a photo that contains a {self.class_descriptions[cls]}"
            for cls in self.classes
        ]
        multiclass_descriptions = [
            f"a photo of a {self.class_descriptions[cls]}"
            for cls in self.classes
        ]
        # Tokenize
        self.multilabel_text_inputs = torch.cat(
            [clip.tokenize(description) for description in multilabel_descriptions]
        )
        self.multiclass_text_inputs = torch.cat(
            [clip.tokenize(description) for description in multiclass_descriptions]
        )    
        self.sampler = sampler
        
    def calculate_ratios(self, labels):
        class_counts = [labels.count(cls) for cls in self.classes]
        total_samples = sum(class_counts)
        return [count / total_samples for count in class_counts]
    
    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, reverse=False):
        if reverse:
            img_num_per_cls = img_num_per_cls[::-1]  # reverse the list

        new_data = []
        targets_np = np.array([self.class_to_idx[label] for _, label in self.data], dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.extend([self.data[i] for i in selec_idx])
        self.data = new_data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        if self.train:
            img_path = os.path.join(self.image_dir, label, "images", img_name)
        else:
            img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        target = self.class_to_idx[label]
        if self.sampler is not None:
            return self.sampler.sample(self, img, target)
        if self.transform:
            img = self.transform(img)

        return {"out_1": img, "target": target}


class OptionalPad(object):
    def __init__(self, fill=0, padding_enabled=True, image_size=224):
        self.padding_size = (image_size - 64) // 2
        self.fill = fill
        self.padding_enabled = padding_enabled
        self.padding = transforms.Pad(self.padding_size, fill=fill)

    def __call__(self, x):
        if self.padding_enabled:
            return self.padding(x)
        else:
            return x  # Identity operation if padding is not enabled

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(padding_size={0}, fill={1}, padding_enabled={2})".format(
                self.padding_size, self.fill, self.padding_enabled
            )
        )
