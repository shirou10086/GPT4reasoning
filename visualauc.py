import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#This is overall auc image generation, change directory_string and directory_to_search to calculate, make sure you have the scores and GroundTruth
directory_string = 'Seed5'

directory_to_search = f'./{directory_string}/'



def find_and_merge_scores(base_directory, filename):
    all_dataframes = []
    for foldername, subfolders, filenames in os.walk(base_directory):
        for file in filenames:
            if file == filename:
                df = pd.read_csv(os.path.join(foldername, file))
                all_dataframes.append(df)
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    output_filename = os.path.join(base_directory, 'merged_' + filename)
    merged_df.to_csv(output_filename, index=False)
    return output_filename


merged_scores_path = find_and_merge_scores(directory_to_search, 'scores.csv')
merged_groundtruth_path = find_and_merge_scores(directory_to_search, 'GroundTruth.csv')

scores_df = pd.read_csv(f'./{directory_string}/merged_scores.csv')
groundtruth_df = pd.read_csv(f'./{directory_string}/merged_GroundTruth.csv')


# Replace old directory with the new one
#groundtruth_df['image_1'] = groundtruth_df['image_1'].str.replace(r'./temp_run3_succ_6-5-23-seed--3/More_vis/', directory_to_search)
#groundtruth_df['image_2'] = groundtruth_df['image_2'].str.replace(r'./temp_run3_succ_6-5-23-seed--3/More_vis/', directory_to_search)
#output_path = os.path.join(directory_to_search, "updated_GroundTruth.csv")
#groundtruth_df.to_csv(output_path, index=False)

scores_df['BinaryScore'] = scores_df['Score'].apply(lambda x: 1 if x > 5 else 0)

# Extracting image numbers to help in merging
groundtruth_df['image_1_num'] = groundtruth_df['image_1'].str.extract(r'best_color_(\d+)\.png').astype(int)
groundtruth_df['image_2_num'] = groundtruth_df['image_2'].str.extract(r'best_color_(\d+)\.png').astype(int)

merged_df = groundtruth_df.merge(scores_df, left_on=['image_1_num', 'image_2_num'], right_on=['first image', 'second image'])

thresholds = [i / 100 for i in range(0, 101, 5)]
ious_list = []

for threshold in thresholds:
    true_positives = ((merged_df['Score'] >= threshold*10) & (merged_df['label'] == 1)).sum()
    false_positives = ((merged_df['Score'] >= threshold*10) & (merged_df['label'] == 0)).sum()
    false_negatives = ((merged_df['Score'] < threshold*10) & (merged_df['label'] == 1)).sum()
    denominator = true_positives + false_positives + false_negatives
    iou = true_positives / denominator if denominator else 0.0
    ious_list.append(iou)

auc = np.trapz(ious_list, thresholds)

def visual_iou(thresholds, ious_list, auc):
    vis_dir = f'./{directory_string}'   # change saving directory to "./NoExYesBackLabeled2/vis"

    plt.figure()
    plt.grid(True)
    plt.xlim(0, 1)  # set x
    plt.ylim(0, 1)  # set y
    plt.plot(thresholds, ious_list, '-o', color='red', linewidth=2, label='IOU')
    plt.fill_between(thresholds, ious_list, color='blue', alpha=0.1)
    # plt.title('IOU vs Threshold', fontsize=24)  # If you want to keep the title, uncomment this line
    plt.text(0.2, 0.85, f'AUC = {auc:.2f}', fontsize=22)
    plt.xlabel('Thresholds', fontsize=22)
    plt.ylabel('IOU', fontsize=22)
    plt.legend()
    plot_name = f"{directory_string}.png"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    plt.savefig(os.path.join(vis_dir, plot_name))
    plt.show()
print("IOU List:", ious_list)
visual_iou(thresholds, ious_list, auc)
print(f"Area Under Curve (AUC): {auc:.4f}")
print(f"Intersection over Union (IOU): {iou:.4f}")
