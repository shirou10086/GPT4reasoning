import os
import torch
from PIL import Image
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

def get_image_features(image_path):
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    return image_features

def get_text_features(text):
    text_tokens = clip.tokenize([text]).to(device)
    text_features = model.encode_text(text_tokens)
    return text_features

def compute_and_save_features(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder in pretaineddataset:
        folder_path = os.path.join(image_folder, folder)
        if os.path.exists(folder_path):
            for subdir, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.png') and not file.startswith('topdown'):
                        image_path = os.path.join(subdir, file)
                        image_features = get_image_features(image_path)
                        relative_subdir = os.path.relpath(subdir, image_folder)
                        feature_subdir = os.path.join(output_folder, relative_subdir)
                        if not os.path.exists(feature_subdir):
                            os.makedirs(feature_subdir)
                        torch.save(image_features, os.path.join(feature_subdir, file + ".pt"))

                        text_path = image_path.replace('.png', '.txt')
                        with open(text_path, 'r') as f:
                            text_description = f.read().strip()
                        text_features = get_text_features(text_description)
                        torch.save(text_features,os.path.join(feature_subdir, file.replace('.png', '.txt.pt')))

if __name__ == "__main__":
    pretaineddataset = ['Adrian', 'Anaheim', 'Andover', 'Angiola', 'Annawan', 'Applewold', 'Arkansaw', 'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Cantwell', 'Capistrano', 'Colebrook', 'Cooperstown', 'Denmark', 'Dryville', 'Dunmor']
    compute_and_save_features("./dataset", "./features")
