#After generatetext.py, you should use this file to run the scores based on the prompt, you should change openaiapi, folderlist,notlist and seeds to make it work.
import openai
import time
import os
import numpy as np
import pandas as pd
import re
from openai import OpenAI
#mannually label2 are clear labels for
"""subfolderList = [
    "BfzKZxFShtq.basis", "saBtfCeVoJ4.basis", "t3t9ofFLcFU.basis", "WypGcNbCdsH.basis", "fc7RfUCN5mY.basis",
    "8mXffaQTtmP.basis", "Bnq6SeZGL5b.basis", "QVAA6zecMHu.basis", "RfNGMBdVbAZ.basis", "TziyvKgzdAs.basis",
    "41FNXLAZZgC.basis", "bdp1XNEdvmW.basis", "FRQ75PjD278.basis", "qpcpnP8TosR.basis", "YHmAkqgwe2p.basis",
    "YJDUB7hWg9h.basis", "z9VLaZqCsW5.basis", "9SpHCfHaNiG.basis", "AMEM2eWycTq.basis", "aosjAwX5Lnq.basis",
    "aRKASs4e8j1.basis", "DNWbUAJYsPy.basis", "EU6QPFpqdoU.basis", "RaYrxWt5pR1.basis", "SQqGpSHzfSr.basis",
    "uzH9yHazm9t.basis", "YmWinf3mhb5.basis", "zmZvNTCxMZE.basis", "Coer9RdivP7.basis", "33ypawbKCQf.basis",
    "4MRLu1yET6a.basis", "8uSpPmctPXC.basis", "C6JvMamYTRg.basis", "F5j7ZLfMm1n.basis", "RcuYAHzrjK7.basis",
    "y4YiUQwvWGH.basis", "bxwHR9ipFG8.basis", "ceJTwFNjqCt.basis", "cjLuWviyDEo.basis", "rBmEe6ab5VP.basis",
    "S3r45BMWy6H.basis", "tL6i2PtktSh.basis", "WRphMcFxfhe.basis", "x4LVLSsYWcV.basis", "YGc1h9nNrJP.basis",
    "4dbCzNN5L5t.basis", "5Poh4Qz68hd.basis", "6TPCFES8fhh.basis", "C7xtw9uhYFn.basis", "F8PSGjTiv61.basis",
    "sfbj7jspYWj.basis", "VaEwVD182FS.basis", "X6Pct1msZv5.basis", "X9fRPGxw1jS.basis", "Y8Y6ukxGMvn.basis",
    "ZVScmfktNQ1.basis", "ASKXmHbw68X.basis", "W9YAR9qcuvN.basis", "zWydhyFhvcj.basis", "1mCzDx3EMom.basis",
    "6EMViBCA2N7.basis", "8qbZhbTc1wX.basis", "99ML7CGPqsQ.basis", "aCtdWA5n56Z.basis", "BUFVGDCQNGb.basis",
    "CETmJJqkhcK.basis", "qxwfVS8MQ67.basis", "tAQTHnJ7n72.basis", "tpxKD3awofe.basis", "UAByLdpaokx.basis",
    "w5YEujJKsiy.basis", "XYQdAu1qsK9.basis", "yPKGKBCyYx8.basis", "z8SrvZ4eyqV.basis", "ZwnLFNzxASM.basis",
    "6vJMULqvYe8.basis", "r38SGhq8aJr.basis", "VKmpsujnc5t.basis", "xccdSFAEPau.basis", "Xuky7E5df6A.basis"
]"""
#Below is the folderlist you wanted to loop through
folderList = ['Adrian', 'Albertville', 'Anaheim', 'Andover', 'Angiola', 'Annawan', 'Applewold', 'Arkansaw', 'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Cantwell', 'Capistrano', 'Colebrook', 'Convoy', 'Cooperstown', 'Crandon', 'Delton', 'Denmark', 'Dryville', 'Dunmor', 'Eagerville', 'Eastville', 'Edgemere', 'Elmira', 'Eudora', 'Goffs']
#below is a notlist, put any scenes you want to skip here
notlist=[]


chars_sent_this_minute = 0
current_minute = time.time() // 60
client = OpenAI(
    # put your openai api below
    api_key="xxxxxxxxxxxxxxxxxx",
)
global summary
summry=""

def summarize_with_gpt4(subdir_path):
    combined_text = ""
    for filename in os.listdir(subdir_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(subdir_path, filename)
            with open(file_path, 'r') as file:
                combined_text += file.read() + '\n'

    prompt = "Summarize the following room descriptions into a 300-word layout: " + combined_text[:4000]

    try:
        prompt_message = "You have to give the combined text of the different photos taken in this home, provide a summarized description focusing on the spatial relations, such as positions of furniture. Just give a general example of how one might describe the spatial relations in a typical home.[Make sure token generation is less than 400 words!]"
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in describing spatial relations within indoor environments, especially homes."},
                {"role": "user", "content": prompt_message}
            ],
            model="gpt-4o",
            max_tokens=500
        )
        print(prompt+"\n"+"++++++++++++++++++++++++++\n")
        summarized_text = response.choices[0].message.content.strip()
        return summarized_text
    except Exception as e:
        print(f'Error occurred while summarizing. Error: {e}')
        return "Error in summarization."

def extract_number_from_filename(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def generate_score(folder_path, file_name1, file_name2):
    combined_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                combined_text += file.read() + '\n'

    description_file_path1 = os.path.join(folder_path, file_name1.replace('.png', '.txt'))
    description_file_path2 = os.path.join(folder_path, file_name2.replace('.png', '.txt'))

    with open(description_file_path1, 'r') as file:
        description1 = file.read()

    with open(description_file_path2, 'r') as file:
        description2 = file.read()

    #prompt_part_a = "Rate the spatial relations between two images on a scale of 1 to 10. 0 point pair example: 'A room with a bed near a window' and 'A kitchen with a dining table near a window'. 10 point pair example: 'A kitchen with a table, sink, fridge, and microwave' and 'A well-lit kitchen with a sink, dishwasher, and granite counter'. "
    prompt_part_a = "where 10 means that if you were the camera taking both photos, you could clearly see from the position of one image to the position of the other, as if the subjects or scenes in each photo are directly visible to each other without any obstructions. A score of 1 means that the positions of the two images are such that you cannot see from one to the other at all, as if they are completely out of sight from each other. Please base your rating on the descriptions given. Based on the given info "

    prompt_part_b = f"Image 1: {description1}"
    prompt_part_c = f"Image 2: {description2}"
    prompt =  summary+prompt_part_a + prompt_part_b + prompt_part_c+"Only Output: __(int)"
    prompt_length = len(prompt)

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in rating images spatial relations within indoor environments, especially homes."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4"
        )

        result_text =  response.choices[0].message.content.strip()
        score = re.search(r'\d+', result_text)
        if score is not None:
            score_value=int(score.group())
            if 0 <= score_value <= 10:
                return score_value, prompt_length
            else:
                return 5, prompt_length
        else:
            return 5, prompt_length

    except Exception as e:
        print(f'TimeoutError occurred for {file_name1} and {file_name2}. Skipping and continuing.')
        return 5, prompt_length  # return default score and prompt length

def generate_scores_for_subdir(subdir_path):
    global chars_sent_this_minute
    global current_minute
    file_names = sorted(os.listdir(subdir_path), key=extract_number_from_filename)

    n_files = len(file_names)
    scores = np.empty((0, 5), dtype=object)  # set list to 5

    for i in range(n_files):
        file_name1 = file_names[i]
        if not file_name1.endswith('.png'):
            continue
        for j in range(i + 1, n_files):
            file_name2 = file_names[j]
            if not file_name2.endswith('.png'):
                continue

            while chars_sent_this_minute >= 40000 and (time.time() // 60) == current_minute:
                time.sleep(1)
            if (time.time() // 60) != current_minute:
                chars_sent_this_minute = 0
                current_minute = time.time() // 60

            score, prompt_length = generate_score(subdir_path, file_name1, file_name2)
            chars_sent_this_minute += prompt_length

            print("current folder"+subdir_path)
            print("score" + str(score))
            print("prompt_length" + str(prompt_length))
            print("first pic" + file_name1 + " sec pic" + file_name2)

            # get numbers in images
            first_image_number = int(re.search(r'\d+', file_name1).group())
            second_image_number = int(re.search(r'\d+', file_name2).group())

            scores = np.vstack((scores, [folder, subdir, first_image_number, second_image_number, score]))  # 更新格式

    return scores

#you wanted to change the seeds list to match with your dataset
seeds = [ 'seed3', 'seed4', 'seed5']
seed3_folders = ['Cantwell', 'Capistrano', 'Colebrook', 'Convoy', 'Cooperstown', 'Crandon', 'Delton', 'Denmark', 'Dryville', 'Dunmor', 'Eagerville', 'Eastville', 'Edgemere', 'Elmira', 'Eudora', 'Goffs']

#seed3-Crandon
def get_folders_in_seed(seed_directory):
    """
    get the folders in seed directory
    """
    seed_path = os.path.join('.', seed_directory)

    items = os.listdir(seed_path)

    folders = [item for item in items if os.path.isdir(os.path.join(seed_path, item))]

    return folders

# loop through all seeds
for seed in seeds:
    if seed == 'seed3':
        # Process only specific folders for seed3
        folderList = [folder for folder in get_folders_in_seed(seed) if folder in seed3_folders]
    else:
        # Process all folders for other seeds
        folderList = get_folders_in_seed(seed)
    for folder in folderList:
        folder_path = f'./{seed}/' + folder
        subdirs = [str(d) for d in range(6)]
        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir, 'saved_obs')

            if os.path.exists(subdir_path):
                summary = summarize_with_gpt4(subdir_path)
                print(summary)
                #time.sleep(60)
                scores = generate_scores_for_subdir(subdir_path)
                np.save(os.path.join(subdir_path, 'scores.npy'), scores)
                pd.DataFrame(scores, columns=['Folder', 'Subdir', 'first image', 'second image', 'Score']).to_csv(os.path.join(subdir_path, 'scores.csv'), index=False)
            else:
                print(f'Subfolder {subdir} does not exist in {folder}. Skipping.')
