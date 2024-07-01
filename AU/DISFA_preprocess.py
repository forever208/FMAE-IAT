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


def video_txt_to_imgs_json(txt_dir=None, video_dir=None, output_img_dir=None, output_json_path=None):
    """
    extract valid AU frames from the video based on txt annotations
    save img_path and AU labels into a single json file
    """
    json_labels_ls = []
    num_imgs = 0
    subjects = os.listdir(txt_dir)

    # each subject correspond to 2 videos (left and right)
    for subject in tqdm(subjects):
        AU_txts = os.listdir(f"{txt_dir}/{subject}")
        frame_inds = []
        AUs = [1, 2, 4, 6, 9, 12, 25, 26]
        frame_and_AUs = {}  # dict label for each csv file, e.g. {'frame_ind': [1, 2, 4, 6, 7]}

        for txt in AU_txts:
            txt_path = f"{txt_dir}/{subject}/{txt}"
            AU = int(txt.split('.')[0].split('_')[1][2:])  # AU number
            assert AU in [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

            if AU not in AUs:
                continue

            # read txt and extract the AU labels
            print(f"AU {AU} is under processed at {txt_path}...")
            with open(txt_path, 'r') as file:
                for line in file:
                    frame = int(line.split(',')[0])  # get frame number in the video
                    intensity = int(line.split(',')[1])  # get AU intensity

                    if intensity >= 2:
                        frame_inds.append(frame)

                        if str(frame) in frame_and_AUs:
                            frame_and_AUs[str(frame)].append(AU)
                        else:
                            frame_and_AUs[str(frame)] = [AU]

        # print(frame_and_AUs)
        # print(frame_inds)
        print(f"{subject} has {len(frame_inds)} valid frames")

        # for each subject, extract the valid AU frames from 2 videos
        if len(frame_inds) > 0:
            left_right_videos = [f"LeftVideo{subject}_comp.avi", f"RightVideo{subject}_comp.avi"]

            for video_file in left_right_videos:
                video_path = f"{video_dir}/{video_file}"
                print(f"processing video: {video_path}")

                if os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
                    print(f"video width: {width}, video height: {height}, fps: {fps}, total frames: {cap.get(7)}")

                    if not cap.isOpened():
                        print("Error: Could not open video file.")
                        exit()

                    # Create a folder to save the extracted frames
                    output_directory = f"{output_img_dir}/{video_file[:-9]}"
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
                            frame_path = os.path.join(video_file[:-9], f'frame_{frame_count}.jpg')
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


# def generate_4_fold_subjects():
#     random.shuffle(BP4D_PLUS_SUBJECTS)
#
#     # Split the list into 4 sub-lists
#     sublist1 = random.sample(BP4D_PLUS_SUBJECTS, 35)
#     for element in sublist1:
#         BP4D_PLUS_SUBJECTS.remove(element)
#
#     sublist2 = random.sample(BP4D_PLUS_SUBJECTS, 35)
#     for element in sublist2:
#         BP4D_PLUS_SUBJECTS.remove(element)
#
#     sublist3 = random.sample(BP4D_PLUS_SUBJECTS, 35)
#     for element in sublist3:
#         BP4D_PLUS_SUBJECTS.remove(element)
#
#     sublist4 = BP4D_PLUS_SUBJECTS
#
#     # Print the sub-lists
#     print("Sublist 1:", sublist1)
#     print("Sublist 2:", sublist2)
#     print("Sublist 3:", sublist3)
#     print("Sublist 4:", sublist4)


# 3-fold partition, follow 'Multi-scale Promoted Self-adjusting Correlation Learning for Facial Action Unit Detection'
SUBJECTS_all = ['SN001', 'SN002', 'SN009', 'SN010', 'SN016', 'SN026', 'SN027', 'SN030', 'SN032',
                'SN006', 'SN011', 'SN012', 'SN013', 'SN018', 'SN021', 'SN024', 'SN028', 'SN031',
                'SN003', 'SN004', 'SN005', 'SN007', 'SN008', 'SN017', 'SN023', 'SN025', 'SN029']

SUBJECTS_1 = ['SN001', 'SN002', 'SN009', 'SN010', 'SN016', 'SN026', 'SN027', 'SN030', 'SN032']
SUBJECTS_2 = ['SN006', 'SN011', 'SN012', 'SN013', 'SN018', 'SN021', 'SN024', 'SN028', 'SN031']
SUBJECTS_3 = ['SN003', 'SN004', 'SN005', 'SN007', 'SN008', 'SN017', 'SN023', 'SN025', 'SN029']


def split_train_test_by_fold(json_file=None, train_output_file=None, test_output_file=None, test_subjects=None):
    """
    specify the test_subjects for 3-folds
    """

    with open(json_file, 'r') as file:
        train_samples = []
        test_samples = []
        train_counter = 0
        test_counter = 0

        # each line is a sample
        for line in file:
            loaded_dict = json.loads(line)
            subject = str(loaded_dict['img_path'].split('/')[0][-5:])

            if subject in test_subjects:
                test_samples.append(loaded_dict)
                test_counter += 1
            else:
                assert subject in SUBJECTS_all
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
    # 1. run video_txt_to_imgs_json()
    # 2. use face_align repo to detect and crop face into 224*224 jpg images
    # 3. delete face-fail annotations in json
    # 4. generate 4 subject-exclusive folds
    # 5, split train-test

    # video_txt_to_imgs_json(txt_dir='/home/mang/Downloads/DISFA/annotations',
    #                        video_dir="/home/mang/Downloads/DISFA/raw_videos",
    #                        output_img_dir="/home/mang/Downloads/DISFA/DISFA_valid",
    #                        output_json_path='DISFA_all.json')
    # extract_subjests(csv_dir='/home/mang/Downloads/BP4D+/raw_csv/AU_OCC')
    # generate_4_fold_subjects()
    split_train_test_by_fold(json_file='DISFA_all.json',
                             train_output_file='DISFA_train3.json',
                             test_output_file='DISFA_test3.json',
                             test_subjects=SUBJECTS_3)