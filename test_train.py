import os
import shutil
import random

original_dataset_dir = 'Sign Language for Numbers'

train_dir = "asl_alphabet_train"

test_dir = "asl_alphabet_test"

train_split = 0.8
test_split = 0.2

for folder_name in os.listdir(original_dataset_dir):
    folder_path = os.path.join(original_dataset_dir, folder_name)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        random.shuffle(files)
        train_size = int(len(files) * train_split)
        test_size = int(len(files) * test_split)

        train_files = files[:train_size]
        test_files = files[train_size:train_size + test_size]

        for filename in train_files:
            src = os.path.join(folder_path, filename)
            dst = os.path.join(train_dir, folder_name, filename)
            os.makedirs(os.path.join(train_dir, folder_name), exist_ok=True)
            shutil.copyfile(src, dst)
        for filename in test_files:
            src = os.path.join(folder_path, filename)
            dst = os.path.join(test_dir, folder_name, filename)
            os.makedirs(os.path.join(test_dir, folder_name), exist_ok=True)
            shutil.copyfile(src, dst)
