
# used to split the current data into test, train, and validation datasets
import os
from tqdm import tqdm
import sys
import shutil
import random



input_folder = './reduced_data/test'
output_folder = './reduced_data'




# How much of the main data we want
model_split = 0.5
# How to split it into folders correctly
train_ratio = 0

# Ensures both files are properly provided
if not (os.path.exists(input_folder) and os.path.exists(output_folder)):
    print("Error: Input/Output folder does not exist. Exiting")
    sys.exit(1)

# Checks to make sure both folders exist
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
# val_folder = os.path.join(output_folder, "val")

# Makes the necessary folders if needed
if not os.path.exists(train_folder):
     os.makedirs(train_folder, exist_ok=True)
if not os.path.exists(test_folder):
     os.makedirs(test_folder, exist_ok=True)
# if not os.path.exists(val_folder):
#      os.makedirs(val_folder, exist_ok=True)

for category in tqdm(os.listdir(input_folder), desc="Overall Progress") :
    # each category
    cur_path = os.path.join(input_folder, category)

    train_category_path = os.path.join(train_folder, category)
    test_category_path = os.path.join(test_folder, category)
    # val_category_path = os.path.join(val_folder, category)

    # makes the new directory
    if not os.path.exists(train_category_path):
        os.makedirs(train_category_path, exist_ok=True)
    if not os.path.exists(test_category_path):
        os.makedirs(test_category_path, exist_ok=True)
    # if not os.path.exists(val_category_path):
    #     os.makedirs(val_category_path, exist_ok=True)

    try:
        categ_files = [entry.name for entry in os.scandir(cur_path) if entry.name.lower().endswith('.tif')]
    except Exception as e:
        print(f"Error accessing {cur_path}: {e}")
        continue
    sample_len = int(len(categ_files) * model_split)
    select_files = random.sample(categ_files, sample_len)

    train_len = int(sample_len * train_ratio)
    train_sample = select_files[:train_len]
    test_sample = select_files[train_len:]

    # First moves the training set
    for file_name in tqdm(train_sample, desc= f"Copying Training Files for {category}"):
        source_path = os.path.join(cur_path, file_name)
        destination_path = os.path.join(train_category_path,file_name)
        shutil.move(source_path, destination_path)
    print(f"{len(train_sample)} train images have been successfully copied")

    # Then moves test set
    for file_name in tqdm(test_sample, desc= f"Copying Testing Files for {category}"):
        source_path = os.path.join(cur_path, file_name)
        destination_path = os.path.join(test_category_path,file_name)
        shutil.move(source_path, destination_path)
    print(f"{len(test_sample)} test images have been successfully copied \n")

    # for file_name in tqdm(select_files, desc= f"Moving Validation Files for {category}"):
    #     source_path = os.path.join(cur_path, file_name)
    #     destination_path = os.path.join(val_category_path,file_name)
    #     shutil.move(source_path, destination_path)
    # print(f"{len(select_files)} train images have been successfully copied")
