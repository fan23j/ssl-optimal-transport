import argparse
import numpy as np
from gensim.models import KeyedVectors
import torch
import re

COCO_LABELS = [
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

CIFAR_10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR_100_LABELS = [
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

_labels_factory = {
    "COCO": COCO_LABELS,
    "CIFAR10": CIFAR_10_LABELS,
    "CIFAR100": CIFAR_100_LABELS,
}

_vector_dim_factory = {
    "COCO": 300,
    "CIFAR10": 300,
    "CIFAR100": 300,
}


def get_word_vector(word, word2vec):
    if word in word2vec.key_to_index:
        return word2vec[word]
    else:
        words = re.split(r"[ _]", word)
        word_vecs = [word2vec[w] for w in words if w in word2vec.key_to_index]
        if not word_vecs:
            raise ValueError(f"None of the words in {word} are in the vocabulary.")
        return np.mean(word_vecs, axis=0)


def get_label_and_negative_vectors(vector_dimension, labels_list, word2vec):
    num_classes = len(labels_list)
    label_vectors = np.zeros((num_classes, vector_dimension))
    negative_vectors = np.zeros((num_classes, vector_dimension))

    for i, label in enumerate(labels_list):
        try:
            label_vectors[i, :] = get_word_vector(label, word2vec)
        except ValueError as e:
            print(e)

    all_labels = set(word2vec.key_to_index.keys())
    for i, label in enumerate(labels_list):
        negative_labels = all_labels - set([label])
        for negative_label in negative_labels:
            try:
                negative_vectors[i, :] += get_word_vector(negative_label, word2vec)
            except ValueError as e:
                print(e)
        negative_vectors[i, :] /= len(negative_labels)

    return label_vectors, negative_vectors


def get_label_and_negative_vectors_dict(vector_dimension, labels_list, word2vec):
    label_vectors = {}
    negative_vectors = {}

    for label in labels_list:
        try:
            label_vectors[label] = get_word_vector(label, word2vec)
        except ValueError as e:
            print(e)

    all_labels = set(word2vec.key_to_index.keys())
    for label in labels_list:
        negative_labels = all_labels - set([label])
        negative_vectors[label] = np.zeros((vector_dimension,))
        for negative_label in negative_labels:
            try:
                negative_vectors[label] += get_word_vector(negative_label, word2vec)
            except ValueError as e:
                print(e)
        negative_vectors[label] /= len(negative_labels)

    return label_vectors, negative_vectors


def main(args):
    print("Loading Word2Vec model...")
    word2vec = KeyedVectors.load_word2vec_format(args.word2vec, binary=False)

    labels_list = _labels_factory[args.dataset]
    labels_list = _labels_factory["COCO"].copy()
    labels_list.extend(_labels_factory["CIFAR100"])
    labels_list = set(labels_list)
    vector_dimension = _vector_dim_factory[args.dataset]
    if args.dict:
        label_vectors, negative_vectors = get_label_and_negative_vectors_dict(
            vector_dimension, labels_list, word2vec
        )
    else:
        label_vectors, negative_vectors = get_label_and_negative_vectors(
            vector_dimension, labels_list, word2vec
        )
    print("Generating label vectors...")
    torch.save(label_vectors, args.label_vectors_file)
    print("Generating negative vectors...")
    torch.save(negative_vectors, args.negative_vectors_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        help="Source dataset of the target vectors.",
        choices=["COCO", "CIFAR_10", "CIFAR_100"],
    )
    parser.add_argument(
        "--word2vec",
        help="The path of the Word2Vec model.",
        default="weights/GoogleNews-vectors-negative300.bin.gz",
    )
    parser.add_argument(
        "--label_vectors_file",
        help="The name of the output label vectors file.",
        default="weights/labels_vectors.pt",
    )
    parser.add_argument(
        "--negative_vectors_file",
        help="The name of the output negative vectors file.",
        default="weights/negative_vectors.pt",
    )
    parser.add_argument(
        "--binary",
        help="Load Word2Vec model in binary format.",
        default=True,
    )
    parser.add_argument(
        "--dict",
        help="Load label vectors and negative vectors as dictionaries.",
        default=False,
    )
    args = parser.parse_args()

    main(args)
