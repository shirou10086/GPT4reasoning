import pandas as pd
import numpy as np
import os
#this is a file for exact scene's iou list, change directory_string and directory_to_search to make it work
directory_string = 'Anaheim'
directory_to_search = f'./{directory_string}/'

def find_and_merge_scores(base_directory, filename):
    all_dataframes = []
    for foldername, subfolders, filenames in os.walk(base_directory):
        for file in filenames:
            if file == filename:
                df = pd.read_csv(os.path.join(foldername, file))
                all_dataframes.append(df)
    if not all_dataframes:
        return None
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    return merged_df

def calculate_iou(merged_df, thresholds):
    ious_list = []
    for threshold in thresholds:
        true_positives = ((merged_df['Score'] >= threshold * 10) & (merged_df['label'] == 1)).sum()
        false_positives = ((merged_df['Score'] >= threshold * 10) & (merged_df['label'] == 0)).sum()
        false_negatives = ((merged_df['Score'] < threshold * 10) & (merged_df['label'] == 1)).sum()
        denominator = true_positives + false_positives + false_negatives
        iou = true_positives / denominator if denominator else 0.0
        ious_list.append(iou)
    return ious_list

thresholds = [i / 100 for i in range(0, 101, 5)]
all_ious = []

for subfolder in os.listdir(directory_to_search):
    subfolder_path = os.path.join(directory_to_search, subfolder)
    if os.path.isdir(subfolder_path):
        for i in range(6):  # set range from 0-5
            sub_subfolder_path = os.path.join(subfolder_path, str(i), 'saved_obs')
            if os.path.isdir(sub_subfolder_path):
                scores_df = find_and_merge_scores(sub_subfolder_path, 'scores.csv')
                groundtruth_df = find_and_merge_scores(sub_subfolder_path, 'GroundTruth.csv')

                if scores_df is None or groundtruth_df is None:
                    print(f"Skipping {sub_subfolder_path}, missing required files.")
                    continue

                scores_df['BinaryScore'] = scores_df['Score'].apply(lambda x: 1 if x > 5 else 0)

                groundtruth_df['image_1_num'] = groundtruth_df['image_1'].str.extract(r'best_color_(\d+)\.png').astype(int)
                groundtruth_df['image_2_num'] = groundtruth_df['image_2'].str.extract(r'best_color_(\d+)\.png').astype(int)

                merged_df = groundtruth_df.merge(scores_df, left_on=['image_1_num', 'image_2_num'], right_on=['first image', 'second image'])

                ious_list = calculate_iou(merged_df, thresholds)
                all_ious.append(ious_list)

if not all_ious:
    print("No data to process.")
else:
    average_ious = np.mean(all_ious, axis=0)
    auc = np.trapz(average_ious, thresholds)

    print("Average IOU List:", average_ious)
    print(f"Area Under Curve (AUC): {auc:.4f}")
