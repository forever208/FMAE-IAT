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


def rename_files(folder_path=None):
    mapping = {
        '_01': '_T1',
        '_03': '_T2',
        '_07': '_T3',
        '_08': '_T4',
        '_09': '_T5',
        '_10': '_T6',
        '_11': '_T7',
        '_13': '_T8',
    }

    for filename in os.listdir(folder_path):
        for old_part, new_part in mapping.items():
            if old_part in filename:
                # Prepare the new filename
                new_name = filename.replace(old_part, new_part)
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name)

                # Rename the file
                os.rename(old_path, new_path)
                print(f'Renamed "{filename}" to "{new_name}"')
                break  # Exit the loop after renaming to avoid multiple renamings


def video_csv_to_imgs_json(csv_dir=None, video_dir=None, output_img_dir=None, output_json_path=None):
    """
    extract valid AU frames from the video, save img_path and AU labels into a json file
    """
    json_labels_ls = []
    num_imgs = 0
    csv_files = os.listdir(csv_dir)

    # each csv file correspond to a video, each subject has 8 csv files
    for csv_file in tqdm(csv_files):
        csv_path = f"{csv_dir}/{csv_file}"
        print(f"processing {csv_path}...")
        frame_inds = []
        AUs = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
        frame_and_AUs = {}  # dict label for each csv file, e.g. {'frame_ind': [1, 2, 4, 6, 7]}

        # read csv and extract the AU labels
        df = pd.read_csv(csv_path, low_memory=False)
        rows = df.shape[0]
        for i in range(rows):
            frame = int(df.iloc[i, 0])  # the frame index
            for au in AUs:

                if int(df.iloc[i, au]) == 1:
                    if str(frame) in frame_and_AUs:
                        frame_and_AUs[str(frame)].append(au)
                    else:
                        frame_and_AUs[str(frame)] = [au]
                else:
                    # fill negative AU with 999
                    if str(frame) in frame_and_AUs:
                        frame_and_AUs[str(frame)].append(999)  # initialize the key
                    else:
                        frame_and_AUs[str(frame)] = [999]  # initialize the key

            if frame not in frame_inds:
                frame_inds.append(frame)
        # print(frame_and_AUs)
        # print(frame_inds)
        print(f"{csv_file} has {len(frame_inds)} valid frames")

        # for each csv, read its video
        video_file = "2" + csv_file[0] + csv_file[2:4] + "_T" + csv_file[6] + ".avi"  # F001_T5.csv --> 2F01_T5.avi
        video_path = f"{video_dir}/{video_file}"
        print(f"processing video: {video_path}")
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
            print(f"video width: {width}, video height: {height}, fps: {fps}, total frames: {cap.get(7)}")

            if not cap.isOpened():
                print("Error: Could not open video file.")
                exit()

            # Create a dir to save the extracted frames
            output_directory = f"{output_img_dir}/{video_file[:-4]}"
            os.makedirs(output_directory, exist_ok=True)

            # Loop through the frames in the video, extract the specified frames, and save them as images
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                frame_count += 1
                if frame_count in frame_inds:
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
            assert len(frame_inds) == len(os.listdir(output_directory))

            print(f"Extracted {len(os.listdir(output_directory))} frames to {output_directory}")

        else:
            raise ValueError(f"video {video_path} does not exist")

        # print(json_labels_ls)

    # save image_path and AUs labels into json
    with open(output_json_path, 'w') as f:
        for each_dict in json_labels_ls:
            json.dump(each_dict, f)
            f.write('\n')
    print(f"Frame annotations saved to {output_json_path}")
    print(f"total number of images: {num_imgs}")  # 146847 (133181 if drop non-AU frames)



BP4D_SUBJECTS = ['F15', 'M05', 'F17', 'M13', 'M09', 'M11', 'M16', 'F04', 'M07', 'F02', 'F08', 'M14', 'F07', 'F14', 'M06', 'M02', 'F16', 'F18', 'F03', 'F09', 'F11', 'M17', 'M04', 'M01', 'F12', 'F10', 'F13', 'F01', 'M08', 'F23', 'F20', 'M15', 'M12', 'M03', 'M10', 'F22', 'F06', 'M18', 'F05', 'F21', 'F19']

# follow paper 'Multi-scale Promoted Self-adjusting Correlation Learning for Facial Action Unit Detection'
SUBJECTS_1 = ['F01', 'F02', 'F08', 'F09', 'F10', 'F18', 'F16', 'F23', 'M01', 'M04', 'M07', 'M08', 'M12', 'M14']
SUBJECTS_2 = ['F03', 'F05', 'F11', 'F13', 'F20', 'F22', 'M02', 'M05', 'M10', 'M11', 'M13', 'M16', 'M17', 'M18']
SUBJECTS_3 = ['F04', 'F06', 'F07', 'F12', 'F14', 'F15', 'F17', 'F19', 'F21', 'M03', 'M06', 'M09', 'M15']

# # 3-fold: 2-fold for training and 1-fold for testing
# SUBJECTS_1 = ['M16', 'M15', 'F20', 'M03', 'F19', 'F21', 'F04', 'M17', 'M11', 'M06', 'M04', 'M14', 'F16', 'F08']
# SUBJECTS_2 = ['F17', 'F12', 'F11', 'M01', 'M02', 'F13', 'M18', 'F15', 'F02', 'F18', 'M10', 'M12', 'M09', 'F01']
# SUBJECTS_3 = ['M07', 'F22', 'M08', 'F03', 'F14', 'F06', 'M05', 'F23', 'F09', 'F05', 'F07', 'M13', 'F10']


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
            subject = str(loaded_dict['img_path'].split('_')[0][-3:])
            assert subject in BP4D_SUBJECTS

            if subject in test_subjects:
                test_samples.append(loaded_dict)  # test set
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
    print(f"{test_counter} samples has been written into {test_output_file}")


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
    # 3. delete face-fail annotations in json  (146654 after fail face deletion)
    # 4. generate 3 subject-exclusive folds
    # 5, split train-test

    # rename_files(folder_path='/home/mang/Downloads/BP4D/BP4D-Spontaneous')
    # video_csv_to_imgs_json(csv_dir='/home/mang/Downloads/BP4D/BP4D_original_labels/AU_OCC',
    #                        video_dir='/home/mang/Downloads/BP4D/BP4D-Spontaneous',
    #                        output_img_dir="/home/mang/Downloads/BP4D/BP4D_valid_frames",
    #                        output_json_path='BP4D_all.json')
    # generate_3_fold_subjects()
    split_BP4D_train_test(json_file='./BP4D_all.json',
                          train_output_file='./BP4D_train3.json',
                          test_output_file='./BP4D_test3.json',
                          test_subjects=SUBJECTS_3)
    # random_split_train_test_for_ID(json_path='./BP4D_all.json')