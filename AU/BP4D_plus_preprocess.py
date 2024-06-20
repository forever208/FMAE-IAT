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


def video_csv_to_imgs_json(csv_dir=None, video_dir=None, output_img_dir=None, output_json_path=None):
    """
    extract valid AU frames from the video based on csv annotations
    save img_path and AU labels into a single json file
    """
    json_labels_ls = []
    num_imgs = 0
    csv_files = os.listdir(csv_dir)

    for csv_file in tqdm(csv_files):
        csv_path = f"{csv_dir}/{csv_file}"
        print(f"processing {csv_path}...")
        frame_inds = []
        AUs =              [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
        AU_column_in_csv = [1, 2, 3, 5, 6, 8,  10, 12, 13, 15, 20, 21]
        frame_and_AUs = {}  # dict label for each csv file, e.g. {'frame_ind': [1, 2, 4, 6, 7]}

        # read csv and extract the AU labels
        df = pd.read_csv(csv_path, low_memory=False)
        # print(f"dataframe shape: {df.shape}")
        rows = df.shape[0]
        for i in range(rows):
            frame = int(df.iloc[i, 0])  # the frame index
            VALID_FRAME = False
            for x, au in enumerate(AUs):
                column_ind = AU_column_in_csv[x]
                if int(df.iloc[i, column_ind]) == 1:
                    VALID_FRAME = True
                    if str(frame) in frame_and_AUs:
                        frame_and_AUs[str(frame)].append(au)
                    else:
                        frame_and_AUs[str(frame)] = [au]
            if VALID_FRAME:
                frame_inds.append(frame)
        # print(frame_and_AUs)
        # print(frame_inds)
        print(f"{csv_file} has {len(frame_inds)} valid frames")

        if len(frame_inds) > 0:
            # for each csv, read its video file
            video_file = csv_file.split('.')[0] + ".mp4"
            video_path = video_dir + video_file
            print(f"processing video: {video_path}")
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
                print(f"video width: {width}, video height: {height}, fps: {fps}, total frames: {cap.get(7)}")

                if not cap.isOpened():
                    print("Error: Could not open video file.")
                    exit()

                # Create a folder to save the extracted frames
                output_directory = output_img_dir + video_file[:-4]
                os.makedirs(output_directory, exist_ok=True)

                # Loop through the frames in the video, extract the specified frames, and save them as images
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  # End of video

                    frame_count += 1
                    if frame_count in frame_inds:
                        # original frame is too big, resize the frame before saving
                        new_width = width // 2
                        new_height = height // 2
                        frame = cv2.resize(frame, (new_width, new_height))

                        frame_filename = os.path.join(output_directory, f'frame_{frame_count}.jpg')
                        cv2.imwrite(frame_filename, frame)
                        num_imgs += 1

                        # save image_path and AUs labels
                        frame_path = os.path.join(video_file[:-4], f'frame_{frame_count}.jpg')
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
    with open(output_json_path, 'w') as f:
        for each_dict in json_labels_ls:
            json.dump(each_dict, f)
            f.write('\n')
    print(f"Frame information saved to {output_json_path}")
    print(f"total number of images: {num_imgs}")  # 167109


def extract_subjests(csv_dir=None):
    csv_files = os.listdir(csv_dir)

    subjects = []
    for filename in csv_files:
        if filename[:4] not in subjects:
            subjects.append(filename[:4])

    print(f"has {len(subjects)} subjects, subject IDs are:")  # 140 subjects
    print(subjects)


BP4D_PLUS_SUBJECTS = ['F015', 'M005', 'F057', 'M009', 'M053', 'F078', 'F081', 'M047', 'F004', 'M026', 'M056', 'F058', 'F007', 'M031', 'M052', 'F041', 'F014', 'F074', 'M039', 'F044', 'F056', 'F064', 'F048', 'M040', 'F024', 'F076', 'M030', 'F016', 'F035', 'F062', 'M014', 'F033', 'M058', 'M055', 'M006', 'M048', 'F032', 'M022', 'M036', 'M017', 'F053', 'F002', 'F018', 'F067', 'F050', 'F034', 'F027', 'M001', 'F029', 'F012', 'F010', 'M044', 'F079', 'M013', 'M043', 'F059', 'M050', 'M023', 'M008', 'F070', 'F011', 'F030', 'F073', 'F069', 'M041', 'F020', 'M054', 'F040', 'M024', 'M034', 'M037', 'F025', 'F008', 'F060', 'F022', 'M045', 'M007', 'M015', 'M042', 'M016', 'M038', 'M057', 'F051', 'F009', 'F082', 'M049', 'M019', 'F047', 'F054', 'M028', 'F075', 'F042', 'M018', 'F017', 'M003', 'F055', 'F046', 'F077', 'F043', 'F065', 'F023', 'F052', 'M012', 'F005', 'M010', 'F013', 'M032', 'M004', 'F037', 'F045', 'M046', 'F021', 'M027', 'M051', 'F019', 'F003', 'F068', 'M029', 'M025', 'F028', 'M021', 'M020', 'F036', 'F031', 'M002', 'M011', 'F080', 'F066', 'F071', 'F038', 'M035', 'F063', 'F026', 'M033', 'F072', 'F049', 'F061', 'F006', 'F001', 'F039']

# 4-fold: 3-fold for training and 1-fold for testing
SUBJECTS_1 = ['M040', 'F072', 'M015', 'M029', 'M003', 'F076', 'F053', 'F026', 'F044', 'F066', 'F057', 'F061', 'F071', 'M050', 'M033', 'F079', 'F020', 'M025', 'F014', 'F004', 'F013', 'M017', 'F033', 'M042', 'M004', 'F038', 'F019', 'M036', 'M026', 'M048', 'F039', 'F046', 'M051', 'F047', 'M020']
SUBJECTS_2 = ['F074', 'F012', 'F034', 'M001', 'F056', 'F075', 'M009', 'M038', 'F024', 'M047', 'F016', 'M045', 'M034', 'M022', 'F060', 'M011', 'M044', 'M046', 'M005', 'M028', 'F077', 'F028', 'M055', 'M019', 'F032', 'F030', 'M037', 'M043', 'F031', 'F022', 'M023', 'M018', 'M016', 'F065', 'M052']
SUBJECTS_3 = ['F029', 'F054', 'F064', 'F045', 'F009', 'F040', 'F008', 'M041', 'F063', 'M056', 'M024', 'F001', 'F080', 'M010', 'F062', 'F035', 'M054', 'F052', 'F027', 'F043', 'F042', 'F050', 'M057', 'F078', 'F058', 'F017', 'M035', 'M030', 'M027', 'F021', 'M031', 'F069', 'F002', 'M008', 'F068']
SUBJECTS_4 = ['M058', 'F037', 'F010', 'F023', 'M007', 'M002', 'F025', 'F073', 'F048', 'F041', 'F051', 'F011', 'M032', 'F005', 'M021', 'F018', 'M013', 'M049', 'M014', 'F070', 'F006', 'F067', 'M039', 'M006', 'F059', 'F003', 'F007', 'F049', 'M053', 'F081', 'F055', 'M012', 'F082', 'F015', 'F036']


def generate_4_fold_subjects():
    random.shuffle(BP4D_PLUS_SUBJECTS)

    # Split the list into 4 sub-lists
    sublist1 = random.sample(BP4D_PLUS_SUBJECTS, 35)
    for element in sublist1:
        BP4D_PLUS_SUBJECTS.remove(element)

    sublist2 = random.sample(BP4D_PLUS_SUBJECTS, 35)
    for element in sublist2:
        BP4D_PLUS_SUBJECTS.remove(element)

    sublist3 = random.sample(BP4D_PLUS_SUBJECTS, 35)
    for element in sublist3:
        BP4D_PLUS_SUBJECTS.remove(element)

    sublist4 = BP4D_PLUS_SUBJECTS

    # Print the sub-lists
    print("Sublist 1:", sublist1)
    print("Sublist 2:", sublist2)
    print("Sublist 3:", sublist3)
    print("Sublist 4:", sublist4)


def split_train_test_by_fold(json_file=None, train_output_file=None, test_output_file=None, test_subjects=None):
    """
    specify the test_subjects for 4-folds
    """

    with open(json_file, 'r') as file:
        train_samples = []
        test_samples = []
        train_counter = 0
        test_counter = 0

        # each line is a sample
        for line in file:
            loaded_dict = json.loads(line)
            subject = str(loaded_dict['img_path'][:4])

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
    # 1. run video_csv_to_imgs_json()
    # 2. use face_align repo to detect and crop face into 224*224 jpg images
    # 3. delete face-fail annotations in json
    # 4. generate 4 subject-exclusive folds
    # 5, split train-test

    # video_csv_to_imgs_json(csv_dir='/home/mang/Downloads/BP4D+/raw_csv/AU_OCC',
    #                        video_dir="/home/mang/Downloads/BP4D+/raw_video/",
    #                        output_img_dir="/home/mang/Downloads/BP4D+/BP4D_plus_valid/",
    #                        output_json_path='BP4D_plus_all.json')
    # extract_subjests(csv_dir='/home/mang/Downloads/BP4D+/raw_csv/AU_OCC')
    # generate_4_fold_subjects()
    split_train_test_by_fold(json_file='BP4D_plus_all.json',
                             train_output_file='BP4D_plus_train4.json',
                             test_output_file='BP4D_plus_test4.json',
                             test_subjects=SUBJECTS_4)