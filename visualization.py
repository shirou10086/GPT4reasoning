import os
import torch
from PIL import Image
import clip
import gatherfeature
import matplotlib.pyplot as plt
import numpy as np
#label需要改一下
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

def find_best_match_description(image_path, features_folder, dataset_folder):
    image_features = gatherfeature.get_image_features(image_path)
    best_similarity = -float('inf')
    best_text_path = None

    for subdir, _, files in os.walk(features_folder):
        for feature_file in files:
            if feature_file.endswith('.txt.pt'):
                features = torch.load(os.path.join(subdir, feature_file))
                similarity = (image_features * features).sum(dim=-1)
                if similarity > best_similarity:
                    best_similarity = similarity
                    relative_path = os.path.relpath(subdir, features_folder)
                    best_text_path = os.path.join(dataset_folder, relative_path, feature_file.replace('.pt', ''))
                    with open(best_text_path, 'r') as f:
                        best_content = f.read().strip()

    return best_content

def compare_image_descriptions(image_path1, image_path2, features_folder, dataset_folder):
    description1 = find_best_match_description(image_path1, features_folder, dataset_folder)
    description2 = find_best_match_description(image_path2, features_folder, dataset_folder)

    features1 = gatherfeature.get_text_features(description1)
    features2 = gatherfeature.get_text_features(description2)

    similarity = torch.nn.functional.cosine_similarity(features1, features2)

    return similarity.item()

labels=[]
def compute_iou_for_threshold(similarities, labels, threshold):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(similarities)):
        prediction = 1 if similarities[i] > threshold else 0
        true_label = labels[i]

        if prediction == 1 and true_label == 1:
            true_positives += 1
        elif prediction == 1 and true_label == 0:
            false_positives += 1
        elif prediction == 0 and true_label == 1:
            false_negatives += 1

    iou = true_positives / (true_positives + false_positives + false_negatives)
    return iou

def visual_iou(similarities):
    thresholds = np.linspace(0, 1, 100)
    ious = [compute_iou_for_threshold(similarities, labels, t) for t in thresholds]
    auc = np.trapz(ious, thresholds)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.title('IOU vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IOU')
    plt.plot(thresholds, ious, '-o', color='red', linewidth=2, label='IOU')
    plt.fill_between(thresholds, ious, color='blue', alpha=0.1)
    plt.text(0.5, 0.5, 'AUC = %0.2f' % auc)
    plot_name = "IOU_vs_Threshold.png"
    file_path = os.path.join('vis', plot_name)
    if not os.path.exists('vis'):
        os.mkdir('vis')
    plt.savefig(file_path)
    plt.show()

def get_all_image_paths(dataset_folder):
    image_paths = []
    adrian_folder = os.path.join('./testing')
    for root, _, files in os.walk(adrian_folder):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths


def compute_all_similarities(dataset_folder, features_folder):
    image_paths = get_all_image_paths(dataset_folder)
    similarities = []

    for i in range(len(image_paths)):
        for j in range(i+1, len(image_paths)):  # 避免重复对和自身对
            similarity = compare_image_descriptions(image_paths[i], image_paths[j], features_folder, dataset_folder)
            similarities.append(similarity)
            print(f"Comparing '{image_paths[i]}' with '{image_paths[j]}'. Similarity: {similarity}")

    return similarities

if __name__ == "__main__":
    dataset_folder = "./dataset"
    features_folder = "./features"

    similarities = compute_all_similarities(dataset_folder, features_folder)
    visual_iou(similarities, labels)
