import os
import clip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)


# Custom dataset class for tiny-imagenet-200
class TinyImageNetValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Read classes and their descriptions from wnids.txt and words.txt
        with open(os.path.join(root, "..", "wnids.txt"), "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        with open(os.path.join(root, "..", "words.txt"), "r") as f:
            # Use primary description (the first one after the label)
            self.class_descriptions = {
                line.split("\t")[0]: line.split("\t")[1].strip()
                for line in f.readlines()
            }

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Read validation annotations
        with open(os.path.join(root, "val_annotations.txt"), "r") as f:
            lines = f.readlines()
            self.data = [(line.split("\t")[0], line.split("\t")[1]) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.root, "images", img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]


# Update data path and create dataset
data_path = os.path.expanduser("/home/main/data/tiny-imagenet-200/val")
dataset = TinyImageNetValDataset(data_path, transform=preprocess)

# Update text inputs to match classes in dataset
text_inputs = torch.cat(
    [
        clip.tokenize(f"a photo that contains a {dataset.class_descriptions[c]}")
        for c in dataset.classes
    ]
).to(device)

# Precompute text features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)


# Function to get predictions for an image
def get_predictions(image):
    image_input = image.unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
    return indices


top1_correct = 0
top5_correct = 0
total = len(dataset)

for i, (image, class_id) in tqdm(
    enumerate(dataset), total=total, desc="Processing images"
):
    predictions = get_predictions(image)

    if class_id == predictions[0]:
        top1_correct += 1
    if class_id in predictions:
        top5_correct += 1

top1_acc = (top1_correct / total) * 100
top5_acc = (top5_correct / total) * 100

print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")
