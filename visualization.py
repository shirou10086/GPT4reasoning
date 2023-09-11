import os
import torch
import csv
from PIL import Image
import clip
import gatherfeature
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

def generate_and_save_features(image_folder, features_folder):
    if not os.path.exists(features_folder):
        os.mkdir(features_folder)

    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                image_features = gatherfeature.get_image_features(image_path)
                feature_path = os.path.join(features_folder, file + '.pt')
                torch.save(image_features, feature_path)

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

def compute_all_similarities_and_labels(dataset_folder, features_folder, csv_path):
    image_paths = get_all_image_paths(dataset_folder)
    similarities = []
    labels = []

    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        label_dict = {(row[0], row[1]): int(row[2]) for row in reader}

    for i in range(len(image_paths)):
        for j in range(i+1, len(image_paths)):
            similarity = compare_image_descriptions(image_paths[i], image_paths[j], features_folder, dataset_folder)
            similarities.append(similarity)
            labels.append(label_dict.get((os.path.basename(image_paths[i]), os.path.basename(image_paths[j])), 0))
            print(f"Comparing '{image_paths[i]}' with '{image_paths[j]}'. Similarity: {similarity}")

    return similarities, labels

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

def visual_iou(similarities, labels):
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
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

if __name__ == "__main__":
    dataset_folder = "./testing"
    features_folder = "./testing_features"
    csv_path = './testing/testing_csv.csv'

    # Generate and save features for images in the testing folder
    generate_and_save_features(dataset_folder, features_folder)

    # Compute similarities and visualize IOU
    similarities, labels = compute_all_similarities_and_labels(dataset_folder, features_folder, csv_path)
    visual_iou(similarities, labels)
