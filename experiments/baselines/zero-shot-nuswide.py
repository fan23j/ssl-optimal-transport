import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import csv

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# NUS-WIDE paths and data loading
dataDir = "/home/ubuntu/ssl-optimal-transport/data/nuswide"
imageDir = os.path.join(dataDir, "images")

def read_object_labels_csv(file, imagelist, fn_map, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = int(row[0])
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                name2 = fn_map[imagelist[name]]
                item = (name2, labels)
                images.append(item)
            rownum += 1
    return images

# Step 1: Load the image paths
def load_image_paths(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

image_paths = load_image_paths('/home/ubuntu/ssl-optimal-transport/data/nuswide/val_image_list.txt') # Replace with actual path to the txt file

# Step 2: Create a fn_map using the image paths
fn_map = {}
for idx, path in enumerate(image_paths):
    tmp = path.split('_')[1]   # Extracting key based on your earlier code
    fn_map[tmp] = path

# Step 3: Create an imagelist dictionary (to mimic what you had in your code)
imagelist = {i: path.split('_')[1] for i, path in enumerate(image_paths)}  # This assumes the split and index logic still applies

# Step 4: Load labels from the CSV file
images_and_labels = read_object_labels_csv('/home/ubuntu/ssl-optimal-transport/data/nuswide/classification_val.csv', imagelist, fn_map)

# Now, split the loaded data into separate lists for paths and labels for easier access
image_paths, labels = zip(*images_and_labels)
nuswide_data = {
    "image_paths": image_paths,
    "labels": labels
}

all_categories = list(fn_map.keys())  # Assuming fn_map has category names
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

scores_list = []
true_labels_list = []

# Collect scores and true labels
for image_path, true_labels in tqdm(zip(nuswide_data["image_paths"], nuswide_data["labels"]), total=len(nuswide_data["image_paths"]), desc="Processing NUS-WIDE images"):
    image = Image.open(image_path).convert("RGB")
    scores = get_similarity_scores(image)
    scores_list.append(scores.cpu().numpy())  # Store the scores
    true_labels_list.append(true_labels)  # Store the true labels

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
