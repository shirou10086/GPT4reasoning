'''
This should be your first file to start with, it transform images to text discriptions,you should change: root_dirs,subfolderList,replicate api, for it to work
'''
import os
import replicate
os.environ['REPLICATE_API_TOKEN'] = "xxxxxxxxxxxxxxx"#put your replicate api here
from concurrent.futures import ThreadPoolExecutor, as_completed

def llava_13b_caption(image_path):
    try:
        # use Replicate to run llava-13b
        output = replicate.run(
            "yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
            input={
                "image": open(image_path, "rb"),
                "top_p": 1,
                "prompt": "This is an image of an interior setting. Please describe the layout and the arrangement of objects, focusing on their relative positions to each other. Keep the description under 50 words and simple, ensuring it is suitable for spatial reasoning tasks.",
                "history": [],
                "max_tokens": 400
                }
        )
        description = ' '.join(output)
        return description
    except Exception as err:
        print(f"Error occurred for {image_path}: {err}")
    return ''

root_dirs = ['seed2','seed3','seed4','seed5']#this is where you need to change for folder names
#Below is a list of contained scenes or "subfolderlist",Adrian and BfzKZxFShtq.basis are both scene names.
subfolderList = ['Adrian', 'Albertville', 'Anaheim', 'Andover', 'Angiola', 'Annawan', 'Applewold', 'Arkansaw', 'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Cantwell', 'Capistrano', 'Colebrook', 'Convoy', 'Cooperstown', 'Crandon', 'Delton', 'Denmark', 'Dryville', 'Dunmor', 'Eagerville', 'Eastville', 'Edgemere', 'Elmira', 'Eudora', 'Goffs']
'''
subfolderList = [
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
]
'''
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
