import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_and_visualize_iou_for_floor(floor_directory):
    # 初始化空的DataFrame
    scores_df_list = []
    groundtruth_df_list = []

    # 遍历楼层的所有子目录
    for subdir in os.listdir(floor_directory):
        subdir_path = os.path.join(floor_directory, subdir, 'saved_obs')
        if os.path.exists(subdir_path):
            # 读取和合并scores.csv和GroundTruth.csv
            scores_df = pd.read_csv(os.path.join(subdir_path, 'scores.csv'))
            groundtruth_df = pd.read_csv(os.path.join(subdir_path, 'GroundTruth.csv'))
            scores_df_list.append(scores_df)
            groundtruth_df_list.append(groundtruth_df)

    # 合并所有DataFrame
    all_scores_df = pd.concat(scores_df_list)
    all_groundtruth_df = pd.concat(groundtruth_df_list)

    # Compute BinaryScore
    all_scores_df['BinaryScore'] = all_scores_df['Score'].apply(lambda x: 1 if x > 5 else 0)

    # Extracting image numbers to help in merging
    all_groundtruth_df['image_1_num'] = all_groundtruth_df['image_1'].str.extract(r'best_color_(\d+)\.png').astype(int)
    all_groundtruth_df['image_2_num'] = all_groundtruth_df['image_2'].str.extract(r'best_color_(\d+)\.png').astype(int)

    # Merge dataframes
    merged_df = all_groundtruth_df.merge(scores_df, left_on=['image_1_num', 'image_2_num'], right_on=['first image', 'second image'])

    thresholds = [i / 100 for i in range(0, 101, 5)]
    ious_list = []

    for threshold in thresholds:
        true_positives = ((merged_df['Score'] >= threshold*10) & (merged_df['label'] == 1)).sum()
        false_positives = ((merged_df['Score'] >= threshold*10) & (merged_df['label'] == 0)).sum()
        false_negatives = ((merged_df['Score'] < threshold*10) & (merged_df['label'] == 1)).sum()
        denominator = true_positives + false_positives + false_negatives
        iou = true_positives / denominator if denominator else 0.0
        ious_list.append(iou)
    iou_list_str = f"iou_list = {ious_list}"

    # Save to a text file
    output_filename = os.path.join(floor_directory, 'iou_data.txt')
    with open(output_filename, 'w') as file:
        file.write(iou_list_str + "\n")


    auc = np.trapz(ious_list, thresholds)

    # Visualize IOU for this specific scene
    visual_iou(thresholds, ious_list, auc, floor_directory)
    print(f"Scene: {floor_directory}")
    print(f"Area Under Curve (AUC) for {floor_directory}: {auc:.4f}")
# Modify visual_iou function to save the plot in the specific scene directory
def visual_iou(thresholds, ious_list, auc, directory_string):
    vis_dir = f'./{directory_string}'  # Update save directory to "./NoExYesBackLabeled2/vis" or other desired directory

    plt.figure()
    plt.grid(True)
    plt.xlim(0, 1)  # Set the range for the x-axis
    plt.ylim(0, 1)  # Set the range for the y-axis
    plt.plot(thresholds, ious_list, '-o', color='red', linewidth=2, label='IOU')
    plt.fill_between(thresholds, ious_list, color='blue', alpha=0.1)
    # plt.title('IOU vs Threshold', fontsize=24)  # Uncomment this line if you want to keep the title
    plt.text(0.2, 0.85, f'AUC = {auc:.2f}', fontsize=22)
    plt.xlabel('Thresholds', fontsize=22)
    plt.ylabel('IOU', fontsize=22)
    plt.legend()

    # Split the directory string to get the city and index, assuming the format is always as given
    city, index, _ = directory_string.split("/")
    plot_name = f"{city}_{index}_LLM_AUC.png"

    save_path = os.path.join(vis_dir, plot_name)

    # Ensure the directory exists before saving the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    #plt.show()


# Start from the base directory and process each scene
base_directory = './YesBackYesExManually/'
floors = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

for floor_directory in floors:
    if "Ballou" not in floor_directory:
        compute_and_visualize_iou_for_floor(floor_directory)
