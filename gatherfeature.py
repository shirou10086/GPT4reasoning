import os
import torch
from PIL import Image
from torchvision import transforms
import clip

pretaineddataset = ['Adrian', 'Anaheim', 'Andover', 'Angiola', 'Annawan', 'Applewold', 'Arkansaw', 'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Cantwell', 'Capistrano', 'Colebrook', 'Cooperstown', 'Denmark', 'Dryville', 'Dunmor']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

def get_image_features(image_path):
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    return image_features

def compute_and_save_features(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder in pretaineddataset:
        folder_path = os.path.join(image_folder, folder)
        if os.path.exists(folder_path): 
            for subdir, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.png'):
                        image_path = os.path.join(subdir, file)
                        features = get_image_features(image_path)
                        relative_subdir = os.path.relpath(subdir, image_folder)
                        feature_subdir = os.path.join(output_folder, relative_subdir)
                        if not os.path.exists(feature_subdir):
                            os.makedirs(feature_subdir)
                        torch.save(features, os.path.join(feature_subdir, file + ".pt"))


image_folder = "./dataset"
output_folder = "./features"
compute_and_save_features(image_folder, output_folder)
