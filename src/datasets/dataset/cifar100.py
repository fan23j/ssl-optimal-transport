from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import torch
import clip


class CIFAR100(CIFAR100):
    """CIFAR100 Dataset."""

    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        super().__init__(root, train=train, download=download)
        self.name = "cifar100"
        
        # LT
        if cfg.DATASET.LT_IMBALANCE_RATIO != 1.0:
            self.data, self.targets = self.create_imbalance(self.data, self.targets, cfg.DATASET.LT_IMBALANCE_RATIO, cfg.DATASET.LT_REVERSE)
            
        # Calculate class ratios
        self.ratios = self.calculate_ratios(self.targets)
        
        optional_padding = OptionalPad(
            fill=0,
            padding_enabled=cfg.DATASET.PAD_CIFAR,
            image_size=cfg.DATASET.IMAGE_SIZE,
        )
        self.train_transform = transforms.Compose(
            [
                optional_padding,
                transforms.RandomResizedCrop(cfg.DATASET.RANDOM_RESIZED_CROP),
                transforms.RandomHorizontalFlip(p=cfg.DATASET.RANDOM_HORIZONTAL_FLIP),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            cfg.DATASET.COLOR_JITTER[0],
                            cfg.DATASET.COLOR_JITTER[1],
                            cfg.DATASET.COLOR_JITTER[2],
                            cfg.DATASET.COLOR_JITTER[3],
                        )
                    ],
                    p=cfg.DATASET.COLOR_JITTER[4],
                ),
                transforms.RandomGrayscale(p=cfg.DATASET.RANDOM_GRAYSCALE),
                transforms.ToTensor(),
                transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                optional_padding,
                transforms.ToTensor(),
                transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
            ]
        )
        multilabel_descriptions = [
            "a photo that contains a " + category for category in self.classes
        ]
        multiclass_descriptions = [
            "a photo of a " + category for category in self.classes
        ]
        # Tokenize
        self.multiclass_text_inputs = torch.cat(
            [clip.tokenize(description) for description in multiclass_descriptions]
        )
        self.multilabel_text_inputs = torch.cat(
            [clip.tokenize(description) for description in multilabel_descriptions]
        )

        self.transform = self.train_transform if train else self.test_transform
        self.sampler = sampler
        self.class_labels = [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ]

        
    def calculate_ratios(self, targets):
        class_counts = [targets.count(i) for i in range(100)]
        total_samples = sum(class_counts)
        return [count / total_samples for count in class_counts]
    
    def create_imbalance(self, data, targets, imbalance_ratio, lt_reverse=False):
        class_indices = [np.where(np.array(targets) == i)[0] for i in range(100)]
        min_len = min(len(indices) for indices in class_indices)

        new_data = []
        new_targets = []

        if lt_reverse:
            iter_order = reversed(list(enumerate(class_indices)))
        else:
            iter_order = enumerate(class_indices)

        for i, indices in iter_order:
            # Determine the number of samples to keep for the current class
            keep_len = int(min_len * (imbalance_ratio ** i))
            indices = np.random.choice(indices, keep_len, replace=False)

            new_data.extend(data[indices])
            new_targets.extend([i] * keep_len)

        return np.array(new_data), new_targets


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.sampler is not None:
            return self.sampler.sample(self, img, target)

        return img, target


class OptionalPad(object):
    def __init__(self, fill=0, padding_enabled=True, image_size=224):
        self.padding_size = (image_size - 32) // 2
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
