import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm import tqdm
import json
import random
import shutil
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def video_to_imgs(csv_dir):
    """
    extract valid AU frames from the video, save img_path and AU labels into a json file
    """
    json_labels_ls = []
    num_imgs = 0
    csv_files = os.listdir(csv_dir)

    for csv_file in tqdm(csv_files):
        csv_path = f"{csv_dir}/{csv_file}"
        print(f"processing {csv_path}...")
        frame_inds = []
        AUs = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
        frame_and_AUs = {}  # dict label for each csv file, e.g. {'frame_ind': [1, 2, 4, 6, 7]}

        # read csv and extract the AU labels
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"dataframe shape: {df.shape}")
        rows = df.shape[0]
        for i in range(rows):
            frame = int(df.iloc[i, 0])  # the frame index
            VALID_FRAME = False
            for au in AUs:
                if int(df.iloc[i, au]) == 1:
                    VALID_FRAME = True
                    if str(frame) in frame_and_AUs:
                        frame_and_AUs[str(frame)].append(au)
                    else:
                        frame_and_AUs[str(frame)] = [au]
            if VALID_FRAME:
                frame_inds.append(frame)
        # print(frame_and_AUs)
        # print(frame_inds)

        # for each csv, read its video file
        video_file = "2" + csv_file[0] + csv_file[2:4] + "_0" + csv_file[6] + ".avi"
        video_path = "/home/mang/Downloads/BP4D-Spontaneous/" + video_file
        print(f"processing video: {video_path}")
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
            print(f"video width: {width}, video height: {height}, fps: {fps}, total frames: {cap.get(7)}")

            if not cap.isOpened():
                print("Error: Could not open video file.")
                exit()

            # Create a dir to save the extracted frames
            output_directory = "/home/mang/Downloads/BP4D_frame/" + video_file[:-4]
            os.makedirs(output_directory, exist_ok=True)

            # Loop through the frames in the video, extract the specified frames, and save them as images
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                frame_count += 1
                if frame_count in frame_inds:
                    frame_filename = os.path.join(output_directory, f'frame_{frame_count}.png')
                    cv2.imwrite(frame_filename, frame)
                    num_imgs += 1

                    # save image_path and AUs labels
                    frame_path = os.path.join(video_file[:-4], f'frame_{frame_count}.png')
                    json_label = {"img_path": frame_path, "AUs": frame_and_AUs[str(frame_count)]}
                    json_labels_ls.append(json_label)

                if frame_count >= max(frame_inds):
                    break  # Stop processing when all desired frames are extracted

            cap.release()  # Release the video capture object
            print(f"Extracted {len(frame_inds)} frames to {output_directory}")

        else:
            raise ValueError(f"video {video_path} does not exist")

        # print(json_labels_ls)

    # save image_path and AUs labels into json
    json_filename = 'BP4D_labels.json'
    with open(json_filename, 'w') as f:
        for each_dict in json_labels_ls:
            json.dump(each_dict, f)
            f.write('\n')
    print(f"Frame information saved to {json_filename}")
    print(f"total number of images: {num_imgs}")  # 133181



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


if __name__ == '__main__':
    # video_to_imgs('/home/mang/Downloads/BP4D-labels/AU_OCC')
    # generate_3_fold_subjects()
    # split_BP4D_train_test(json_file='./BP4D_labels.json',
    #                       train_output_file='./BP4D_train3.json',
    #                       test_output_file='./BP4D_test3.json',
    #                       test_subjects=SUBJECTS_3)
    random_split_train_test_for_ID(json_path='./BP4D_all.json')