import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

with open('/home/ubuntu/ssl-optimal-transport/data/nuswide/cats.txt', 'r') as f:
    all_categories = [line.strip() for line in f.readlines()]

text_inputs = torch.cat(
    [clip.tokenize(f"a photo of a {c}") for c in all_categories]
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

scores_list = []
true_labels_list = []

base_path = "/home/ubuntu/ssl-optimal-transport/data/nuswide/"

# Parse test.txt and collect scores and true labels
with open('/home/ubuntu/ssl-optimal-transport/data/nuswide/test.txt', 'r') as f:
    for line in tqdm(f.readlines(), desc="Processing NUS-WIDE images"):
        tokens = line.strip().split()
        image_path = os.path.join(base_path, tokens[0])  # Add base path
        image = Image.open(image_path).convert("RGB")

        scores = get_similarity_scores(image)
        true_labels = np.array([int(x) for x in tokens[1:]])

        scores_list.append(scores.cpu().numpy())  # Store the scores
        true_labels_list.append(true_labels)      # Store the true labels

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
    # Avoid division by zero
    precision = torch.where(tp + fp != 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(num_positives != 0, tp / num_positives, torch.zeros_like(tp))

    # Compute the AP for the current category
    AP = torch.trapz(precision, recall)
    APs.append(AP.item())

# Check if any AP is NaN and handle it
APs = [ap if not np.isnan(ap) else 0.0 for ap in APs]

# Compute mAP
mAP = np.mean(APs)
print(f"mAP: {mAP:.4f}")
