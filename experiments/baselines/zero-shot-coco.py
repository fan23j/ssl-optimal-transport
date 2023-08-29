import os
import torch
import clip
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import numpy as np

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Initialize COCO api for instance annotations
dataDir = "/home/main/data/coco"
dataType = "val2017"
imageDir = os.path.join(dataDir, "images", "val")
annFile = os.path.join(dataDir, "annotations/annotations", f"instances_{dataType}.json")

coco = COCO(annFile)

# Get all categories
cats = coco.loadCats(coco.getCatIds())
all_categories = [cat["name"] for cat in cats]
text_inputs = torch.cat(
    [clip.tokenize(f"a photo that contains a {c}") for c in all_categories]
).to(device)

# Precompute text features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)


# Function to get similarity scores for an image
def get_similarity_scores(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).squeeze(0)
    return similarity


imgIds = coco.getImgIds()
scores_list = []
true_labels_list = []

# Collect scores and true labels
for imgId in tqdm(imgIds, desc="Processing COCO images"):
    img = coco.loadImgs(imgId)[0]
    image_path = os.path.join(imageDir, img["file_name"])
    image = Image.open(image_path).convert("RGB")

    scores = get_similarity_scores(image)
    true_labels = torch.zeros(len(all_categories))

    annIds = coco.getAnnIds(imgIds=imgId)
    annotations = coco.loadAnns(annIds)
    for ann in annotations:
        cat_name = coco.loadCats(ann["category_id"])[0]["name"]
        index = all_categories.index(cat_name)
        true_labels[index] = 1

    scores_list.append(scores.cpu().numpy())  # Store the scores
    true_labels_list.append(true_labels.numpy())  # Store the true labels

# Convert lists to numpy arrays and then to tensors for subsequent processing
scores_array = np.array(scores_list)
true_labels_array = np.array(true_labels_list)
scores_tensor = torch.tensor(scores_array)
true_labels_tensor = torch.tensor(true_labels_array)


# Compute AP for each category
APs = []
for i, category in enumerate(all_categories):
    scores = scores_tensor[:, i]
    labels = true_labels_tensor[:, i]
    sorted_indices = torch.argsort(scores, descending=True)
    tp = (labels[sorted_indices] == 1).cumsum(0).float()
    fp = (labels[sorted_indices] == 0).cumsum(0).float()

    num_positives = labels.sum()
    precision = tp / (tp + fp)
    recall = tp / num_positives

    # Compute the AP for the current category
    AP = torch.trapz(precision, recall)
    APs.append(AP.item())

# Compute mAP
mAP = np.mean(APs)
print(f"mAP: {mAP:.4f}")
