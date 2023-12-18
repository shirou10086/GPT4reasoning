
import os
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

# 设置设备
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# 加载预训练的BLIP-2模型及其预处理器
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)

def blip2_caption(image_path):
    try:
        raw_image = Image.open(image_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        output = model.generate({"image": image, "prompt": "Describe this image:"})
        description = output[0]
        return description
    except Exception as err:
        print(f"Error occurred for {image_path}: {err}")
    return ''

root_dir = 'dataset1'
subfolderList =['Adrian', 'Anaheim', 'Andover', 'Angiola', 'Annawan','Applewold', 'Arkansaw', 'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Cantwell', 'Capistrano', 'Colebrook', 'Cooperstown', 'Denmark', 'Dryville', 'Dunmor', 'Eagerville', 'Eastville', 'Edgemere', 'Elmira', 'Eudora', 'Goffs', 'Greigsville', 'Hainesburg', 'Hambleton', 'Haxtun', 'Hillsdale', 'Hometown', 'Hominy', 'Kerrtown', 'Maryhill', 'Mesic', 'Micanopy', 'Mifflintown', 'Mobridge', 'Monson', 'Mosinee', 'Mosquito', 'Nemacolin', 'Nicut', 'Nimmons', 'Nuevo', 'Oyens', 'Pablo', 'Parole', 'Pettigrew', 'Placida', 'Pleasant', 'Quantico', 'Rancocas', 'Reyno', 'Ribera', 'Roane', 'Roeville', 'Rosser', 'Roxboro', 'Sanctuary', 'Sands', 'Sawpit', 'Scioto', 'Seward', 'Shelbiana', 'Silas', 'Sisters', 'Sodaville', 'Soldier', 'Spencerville', 'Spotswood', 'Springhill', 'Stanleyville', 'Stilwell', 'Stokes', 'Sumas', 'Superior', 'Swormville']
currentlist=['Adrian', 'Anaheim', 'Andover', 'Angiola', 'Annawan', 'Applewold', 'Arkansaw', 'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Cantwell', 'Capistrano', 'Colebrook', 'Cooperstown', 'Denmark', 'Dryville', 'Dunmor', 'Eagerville', 'Eastville', 'Edgemere', 'Elmira', 'Eudora']


for folder in subfolderList:
    folder_path = os.path.join(root_dir, folder)
    if os.path.exists(folder_path):
        for subdir, _, files in os.walk(folder_path):
            if 'saved_obs' in subdir:
                for file in files:
                    if file.startswith('best_color_') and file.endswith('.png'):
                        image_path = os.path.join(subdir, file)
                        print(f"Processing {image_path}...")
                        caption = blip2_caption(image_path)
                        print(caption)
                        txt_path = os.path.join(subdir, file.replace('.png', '.txt'))
                        with open(txt_path, 'w') as txt_file:
                            txt_file.write(caption)
                        print(f"Completed processing {image_path}.")

print("Caption generation completed!")
