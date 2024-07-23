import os
import random
import shutil

def data_set_split(src_folder, target_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    print("start dataset splitting")
    # Check and create target_folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # This assumes that src_folder directly contains class folders
    class_names = [d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))]
    split_names = ['train', 'val', 'test']

    # Create subdirectories for train, val, and test within each class directory
    for split_name in split_names:
        for class_name in class_names:
            split_class_path = os.path.join(target_folder, split_name, class_name)
            os.makedirs(split_class_path, exist_ok=True)

    # Set random seed for reproducibility
    randnum = 2022
    random.seed(randnum)

    # Iterate through each class directory to split the files
    for class_name in class_names:
        current_class_path = os.path.join(src_folder, class_name)
        current_data = [f for f in os.listdir(current_class_path) if os.path.isfile(os.path.join(current_class_path, f))]
        current_data_length = len(current_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        # Calculate indices for train, val, and test splits
        train_stop_idx = int(train_scale * current_data_length)
        val_stop_idx = int((train_scale + val_scale) * current_data_length)

        # Copy files to their respective directories
        train_num, val_num, test_num = 0, 0, 0
        for idx, file_index in enumerate(current_data_index_list):
            src_img_path = os.path.join(current_class_path, current_data[file_index])
            if idx < train_stop_idx:
                dest_path = os.path.join(target_folder, 'train', class_name, current_data[file_index])
                train_num += 1
            elif idx < val_stop_idx:
                dest_path = os.path.join(target_folder, 'val', class_name, current_data[file_index])
                val_num += 1
            else:
                dest_path = os.path.join(target_folder, 'test', class_name, current_data[file_index])
                test_num += 1
            shutil.copy2(src_img_path, dest_path)

        print("********************************")
        print(f"train set {os.path.join(target_folder, 'train', class_name)}: {train_num}")
        print(f"val set {os.path.join(target_folder, 'val', class_name)}: {val_num}")
        print(f"test set {os.path.join(target_folder, 'test', class_name)}: {test_num}")

# Define source and target directories
src_folder = r"E:\Masters\Assignment - Ali\Other projects\reference-890"
target_folder = r"./data/benchmark"
data_set_split(src_folder, target_folder)
