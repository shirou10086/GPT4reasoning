import os
import torch
from PIL import Image
import clip
import gatherfeature
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

def find_best_match_image(text, features_folder, dataset_folder):
    text_features = gatherfeature.get_text_features(text)
    best_similarity = -float('inf')
    best_image_path = None

    for subdir, _, files in os.walk(features_folder):
        for feature_file in files:
            if feature_file.endswith('.png.pt'):
                features = torch.load(os.path.join(subdir, feature_file))
                similarity = (text_features * features).sum(dim=-1)
                if similarity > best_similarity:
                    best_similarity = similarity
                    relative_path = os.path.relpath(subdir, features_folder)
                    best_image_path = os.path.join(dataset_folder, relative_path, feature_file.replace('.pt', ''))

    return best_image_path
'''
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
'''
def find_best_match_description(image_path, dataset_folder):
    image_features = gatherfeature.get_image_features(image_path)
    best_similarity = -float('inf')
    best_text_content = None

    for subdir, _, files in os.walk(dataset_folder):
        for text_file in files:
            if text_file.endswith('.txt'):
                text_path = os.path.join(subdir, text_file)
                with open(text_path, 'r') as f:
                    text_content = f.read().strip()
                if text_content == best_text_content:
                    continue
                similarity = compute_text_similarity(best_text_content, text_content)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_text_content = text_content

    return best_text_content

def compute_text_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cos_sim = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))[0][0]
    return cos_sim

def find_description(image_path):
    text_path = os.path.splitext(image_path)[0] + '.txt'

    with open(text_path, 'r') as f:
        description = f.read().strip()

    return description
#warning
def compare_image_descriptions(image_path1, image_path2):
    description1 = find_description(image_path1)
    description2 = find_description(image_path2)
    similarity = compute_text_similarity(description1, description2)
    return similarity
def compute_iou(similarities, threshold):
    return sum([1 for s in similarities if s > threshold]) / len(similarities)

def visual_iou(similarities):
    thresholds = np.linspace(0, 1, 100)
    ious = [compute_iou(similarities, t) for t in thresholds]
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
    plt.close()

if __name__ == "__main__":
    features_folder = "./features"
    dataset_folder = "./dataset"

    mode = input("Choose your request: (1: image to description, 2: description to image, 3: images matching): ")
    if mode == "1":
        image_input = input("input image location: ")
        if os.path.exists(image_input):
            best_match = find_best_match_description(image_input, features_folder, dataset_folder)
            print(f" '{image_input}' most matching description is: {best_match}")
        else:
            print("no image found here")
    elif mode == "2":
        text_input = input("input description: ")
        best_match = find_best_match_image(text_input, features_folder, dataset_folder)
        print(f" '{text_input}' most matching image is: {best_match}")
    elif mode == "3":
        num_pairs = int(input("How many image pairs do you want to compare? "))
        similarities = []
        for _ in range(num_pairs):
            image_input1 = input("First location: ")
            image_input2 = input("Second location: ")
            if os.path.exists(image_input1) and os.path.exists(image_input2):
                similarity = compare_image_descriptions(image_input1, image_input2)
                similarities.append(similarity)
                print(f"The similarity between '{image_input1}' and '{image_input2}' is: {similarity}")
            else:
                print("One or both image locations are invalid.")
        visual_iou(similarities)
    else:
        print("Error in request choosing")
