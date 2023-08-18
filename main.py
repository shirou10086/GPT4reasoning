import os
import torch
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

def get_text_features(text):
    text = clip.tokenize([text]).to(device)
    text_features = model.encode_text(text)
    return text_features

def find_best_match_image(text, features_folder, dataset_folder):
    text_features = get_text_features(text)
    best_similarity = -float('inf')
    best_image_path = None

    for subdir, _, files in os.walk(features_folder):
        for feature_file in files:
            if feature_file.endswith('.pt'):
                features = torch.load(os.path.join(subdir, feature_file))
                similarity = (text_features * features).sum(dim=-1)
                if similarity > best_similarity:
                    best_similarity = similarity
                    relative_path = os.path.relpath(subdir, features_folder)
                    best_image_path = os.path.join(dataset_folder, relative_path, feature_file.replace('.pt', ''))

    return best_image_path

def display_image(image_path):
    with Image.open(image_path) as img:
        img.show()
features_folder = "./features"
dataset_folder = "./dataset"
while True:
    text_input = input("Enter text to find most matching image (or type 'exit' to exit): ")
    if text_input.lower() == 'exit':
        break

    best_match = find_best_match_image(text_input, features_folder, dataset_folder)
    print(f"Best match for '{text_input}' is: {best_match}")
    display_image(best_match)
