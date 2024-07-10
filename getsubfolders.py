import os
'''
This is a helper file to check what exactly is the subfolders of a certain folder
'''
def print_subfolders(root_directory):
    """
    print all sub folders in root directory_string
    """
    if not os.path.exists(root_directory):
        print("no existing directory")
        return

    subfolders = [name for name in os.listdir(root_directory)
                  if os.path.isdir(os.path.join(root_directory, name))]



    # print all sub folders
    print("subfolderList =", subfolders)

# usecase
root_directory = './seed2'#change your root folder here
print_subfolders(root_directory)
