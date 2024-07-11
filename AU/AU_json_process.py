import base64
import csv
from PIL import Image
import os
import cv2
import json
import random
from collections import defaultdict
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


def extract_100_samples_for_each_subject(input_file=None, train_output_file=None, test_output_file=None):
    # Read all lines from the file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Shuffle the lines
    random.shuffle(lines)

    # Dictionary to hold up to 100 samples for each ID
    samples_by_id = defaultdict(list)

    # Process each line after shuffling
    for line in lines:
        # Convert line to dictionary
        data = json.loads(line)

        # Extract ID from img_path (assuming format 'ID/...')
        person_id = data['img_path'].split('_')[0][1:]

        # Store the data for this ID, but only up to 100 samples
        if len(samples_by_id[person_id]) < 100:
            samples_by_id[person_id].append(data)

            # If 100 samples are collected for any ID, continue without adding more
            if len(samples_by_id[person_id]) == 100:
                if all(len(samples) == 100 for samples in samples_by_id.values()):
                    break

    # Open training and testing files to write line by line
    with open(train_output_file, 'w') as train_file, open(test_output_file, 'w') as test_file:
        # Split the samples into 70 train and 30 test for each ID
        for person_id, samples in samples_by_id.items():
            random.shuffle(samples)  # Shuffle to randomize which samples go into train/test
            train_samples = samples[:70]  # First 70 for training
            test_samples = samples[70:]  # Next 30 for testing

            # Write training samples line by line
            for sample in train_samples:
                train_file.write(json.dumps(sample) + '\n')

            # Write testing samples line by line
            for sample in test_samples:
                test_file.write(json.dumps(sample) + '\n')
    print(f"created {train_output_file} and {test_output_file}")


def extract_300_samples_for_each_subject_for_tSNE(input_file=None, output_file=None, max_num=500):
    # Read all lines from the file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Shuffle the lines
    random.shuffle(lines)

    # Dictionary to hold up to 100 samples for each ID
    samples_by_id = defaultdict(list)

    # Process each line after shuffling
    for line in lines:
        # Convert line to dictionary
        data = json.loads(line)

        # Extract ID from img_path (assuming format 'ID/...')
        person_id = data['img_path'].split('_')[0][1:]

        # Store the data for this ID, but only up to 100 samples
        if len(samples_by_id[person_id]) < max_num:
            samples_by_id[person_id].append(data)

            # If 100 samples are collected for any ID, continue without adding more
            if len(samples_by_id[person_id]) == max_num:
                if all(len(samples) == max_num for samples in samples_by_id.values()):
                    break

    # Open training and testing files to write line by line
    with open(output_file, 'w') as file:
        # Split the samples into 70 train and 30 test for each ID
        for person_id, samples in samples_by_id.items():
            random.shuffle(samples)  # Shuffle to randomize which samples go into train/test

            # Write samples line by line
            for sample in samples:
                file.write(json.dumps(sample) + '\n')

    print(f"created {output_file}")


if __name__ == "__main__":
    # generate_3_fold_subjects()
    # split_BP4D_train_test(json_file='./BP4D_labels.json',
    #                       train_output_file='./BP4D_train3.json',
    #                       test_output_file='./BP4D_test3.json',
    #                       test_subjects=SUBJECTS_3)
    # extract_100_samples_for_each_subject(input_file='./BP4D_all.json',
    #                                      train_output_file='./BP4D_ID_prob_train.json',
    #                                      test_output_file='./BP4D_ID_prob_test.json')
    extract_300_samples_for_each_subject_for_tSNE(input_file='./BP4D_all.json',
                                                  output_file= './BP4D_500samples_tSNE')

