import os
import requests
import replicate
REPLICATE_API_TOKEN='r8_R1SktlGFJ3juwiYJppu1b48awfyEjkz1GF0tU'
os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

def replicate_caption(image_path):
    try:
        output = replicate.run(
            "methexis-inc/img2prompt:50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5",
            input={"image": open(image_path, "rb")}
        )
        # 检查 output 是否为 None 或非字符串
        if not output or not isinstance(output, str):
            print(f"Unexpected output for {image_path}: {output}")
            return ''

        # 假设模型返回的输出是一个字符串描述
        description = output
        # 查找 "a digital rendering" 并删除它后面的所有内容
        index = description.find("a digital rendering")
        if index != -1:
            description = description[:index]
        return description
    except Exception as err:
        print(f"Error occurred for {image_path}: {err}")
    return ''



root_dir = '.'
subfolderList =['Applewold', 'Arkansaw', 'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Cantwell', 'Capistrano', 'Colebrook', 'Cooperstown', 'Denmark', 'Dryville', 'Dunmor', 'Eagerville', 'Eastville', 'Edgemere', 'Elmira', 'Eudora', 'Goffs', 'Greigsville', 'Hainesburg', 'Hambleton', 'Haxtun', 'Hillsdale', 'Hometown', 'Hominy', 'Kerrtown', 'Maryhill', 'Mesic', 'Micanopy', 'Mifflintown', 'Mobridge', 'Monson', 'Mosinee', 'Mosquito', 'Nemacolin', 'Nicut', 'Nimmons', 'Nuevo', 'Oyens', 'Pablo', 'Parole', 'Pettigrew', 'Placida', 'Pleasant', 'Quantico', 'Rancocas', 'Reyno', 'Ribera', 'Roane', 'Roeville', 'Rosser', 'Roxboro', 'Sanctuary', 'Sands', 'Sawpit', 'Scioto', 'Seward', 'Shelbiana', 'Silas', 'Sisters', 'Sodaville', 'Soldier', 'Spencerville', 'Spotswood', 'Springhill', 'Stanleyville', 'Stilwell', 'Stokes', 'Sumas', 'Superior', 'Swormville']
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
                        caption = replicate_caption(image_path)
                        print(caption)
                        txt_path = os.path.join(subdir, file.replace('.png', '.txt'))
                        with open(txt_path, 'w') as txt_file:
                            txt_file.write(caption)
                        print(f"Completed processing {image_path}.")  # 打印

print("Caption generation completed!")
