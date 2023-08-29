import os
import clip
import torch
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("../data"), download=True, train=False)
text_inputs = torch.cat(
    [clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]
).to(device)

# Precompute text features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)


# Function to get predictions for an image
def get_predictions(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
    return indices


top1_correct = 0
top5_correct = 0
total = len(cifar100)

for i, (image, class_id) in tqdm(
    enumerate(cifar100), total=total, desc="Processing images"
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
