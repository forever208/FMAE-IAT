import base64
import csv
from PIL import Image
import os
import cv2
import json
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


TEST_SUBJECTS = ["F002", "F004", "F018", "F019", "F023", "M006", "M007", "M010", ]  # BP4D F1=0.661

BP4D_SUBJECTS = ['F015', 'M005', 'F017', 'M013', 'M009', 'M011', 'M016', 'F004', 'M007', 'F002', 'F008', 'M014', 'F007', 'F014', 'M006', 'M002', 'F016', 'F018', 'F003', 'F009', 'F011', 'M017', 'M004', 'M001', 'F012', 'F010', 'F013', 'F001', 'M008', 'F023', 'F020', 'M015', 'M012', 'M003', 'M010', 'F022', 'F006', 'M018', 'F005', 'F021', 'F019']

# 3-fold: 2-fold for training and 1-fold for testing
SUBJECTS_1 = ['M016', 'M015', 'F020', 'M003', 'F019', 'F021', 'F004', 'M017', 'M011', 'M006', 'M004', 'M014', 'F016', 'F008']
SUBJECTS_2 = ['F017', 'F012', 'F011', 'M001', 'M002', 'F013', 'M018', 'F015', 'F002', 'F018', 'M010', 'M012', 'M009', 'F001']
SUBJECTS_3 = ['M007', 'F022', 'M008', 'F003', 'F014', 'F006', 'M005', 'F023', 'F009', 'F005', 'F007', 'M013', 'F010']


def generate_3_fold_subjects():
    random.shuffle(BP4D_SUBJECTS)

    # Split the list into 3 sub-lists
    sublist1 = random.sample(BP4D_SUBJECTS, 14)
    for element in sublist1:
        BP4D_SUBJECTS.remove(element)

    sublist2 = random.sample(BP4D_SUBJECTS, 14)
    for element in sublist2:
        BP4D_SUBJECTS.remove(element)

    sublist3 = BP4D_SUBJECTS

    # Print the sub-lists
    print("Sublist 1:", sublist1)
    print("Sublist 2:", sublist2)
    print("Sublist 3:", sublist3)

def split_BP4D_train_test(json_file=None, train_output_file=None, test_output_file=None, test_subjects=None):
    """
    """

    with open(json_file, 'r') as file:
        train_samples = []
        test_samples = []
        train_counter = 0
        test_counter = 0

        # each line is a sample
        for line in file:
            loaded_dict = json.loads(line)
            subject = str(loaded_dict['img_path'].split('/')[0])

            if subject in test_subjects:
                test_samples.append(loaded_dict)
                test_counter += 1
            else:
                train_samples.append(loaded_dict)
                train_counter += 1

    # Write training samples to a new JSON file
    with open(train_output_file, 'w') as f:
        for sample in train_samples:
            json.dump(sample, f)
            f.write('\n')
    print(f"{train_counter} samples has been written into {train_output_file}")

    # Write test samples to a new JSON file
    with open(test_output_file, 'w') as f:
        for sample in test_samples:
            json.dump(sample, f)
            f.write('\n')
    print(f"{test_counter} samples has been written into {train_output_file}")


def random_split_train_test_for_ID(json_path=None):
    def split_data(data, split_ratio=0.3):
        num_samples = len(data)
        test_size = int(num_samples * split_ratio)
        random.shuffle(data)
        return data[test_size:], data[:test_size]

    # Load JSON file
    with open(json_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # Split data into training and test sets
    train_set, test_set = split_data(data, split_ratio=0.3)

    # Write the training set to a JSON file
    with open('BP4D_train_for_ID.json', 'w') as f:
        for sample in train_set:
            json.dump(sample, f)
            f.write('\n')

    # Write the test set to a JSON file
    with open('BP4D_test_for_ID.json', 'w') as f:
        for sample in test_set:
            json.dump(sample, f)
            f.write('\n')

    print("Training set size:", len(train_set))
    print("Test set size:", len(test_set))


if __name__ == "__main__":
    # generate_3_fold_subjects()
    # split_BP4D_train_test(json_file='./BP4D_labels.json',
    #                       train_output_file='./BP4D_train3.json',
    #                       test_output_file='./BP4D_test3.json',
    #                       test_subjects=SUBJECTS_3)
    random_split_train_test_for_ID(json_path='./BP4D_all.json')

