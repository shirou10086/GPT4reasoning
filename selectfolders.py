import os
import shutil
#This is a file that finds the scenes with least numbers of files and delete other scenes for a certain folder,you probably dont need to use this file.
def count_files_in_folder(folder_path):
    """count file numbers in foldername"""
    total_files = 0
    for root, dirs, files in os.walk(folder_path):
        total_files += len(files)
    return total_files

def delete_extra_folders(base_path, num_folders_to_keep=80):
    """keep smallest number of num_folders_to_keep numbers of folders and delete other folders"""
    folder_file_counts = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            file_count = count_files_in_folder(folder_path)
            folder_file_counts.append((folder, file_count))

    # get the smallest folders sorted in num_folders_to_keep
    folder_file_counts.sort(key=lambda x: x[1])
    folders_to_keep = set(folder for folder, _ in folder_file_counts[:num_folders_to_keep])

    # delete folders are not in folders_to_keep
    for folder in os.listdir(base_path):
        if folder not in folders_to_keep:
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"Deleted Folder: {folder}")

# use case
base_path = './HM3d'
delete_extra_folders(base_path)
