import os
import replicate
os.environ['REPLICATE_API_TOKEN'] = "r8_R1SktlGFJ3juwiYJppu1b48awfyEjkz1GF0tU"

def llava_13b_caption(image_path):
    try:
        # 使用 Replicate 运行 llava-13b 模型
        output = replicate.run(
            "yorickvp/llava-13b:6bc1c7bb0d2a34e413301fee8f7cc728d2d4e75bfab186aa995f63292bda92fc",
            input={
                "image": open(image_path, "rb"),
                "prompt": "This is a image inside a house,Describe this image, make it simple clear and short, ensuring the text is under 50 words.remember your max_tokens is set to 100 so make it short:",
                "max_tokens": 100
            }
        )
        # 假设输出是一个迭代器，并且描述是第一项（根据你之前的代码）
        description = ' '.join(output)
        return description
    except Exception as err:
        print(f"Error occurred for {image_path}: {err}")
    return ''

root_dirs = ['ManuallyLabeled2', 'ManuallyLabeled1']

subfolderList=['Anaheim','Applewold','Goffs','Mesic','Sanctuary','Silas']
#Spencerville

for root_dir in root_dirs:
    for folder in subfolderList:
        folder_path = os.path.join(root_dir, folder)
        if os.path.exists(folder_path):
            for subdir, _, files in os.walk(folder_path):
                if 'saved_obs' in subdir:
                    for file in files:
                        if file.startswith('best_color_') and file.endswith('.png'):
                            image_path = os.path.join(subdir, file)
                            print(f"Processing {image_path}...")
                            caption = llava_13b_caption(image_path)
                            print(caption)
                            txt_path = os.path.join(subdir, file.replace('.png', '.txt'))
                            with open(txt_path, 'w') as txt_file:
                                txt_file.write(caption)
                            print(f"Completed processing {image_path}.")

print("Caption generation completed!")
