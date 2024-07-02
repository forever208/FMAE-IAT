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
from tqdm import tqdm


def npy_labels_to_image_folders(anno_path=None, img_path=None, target_img_folder=None):

    print(f"{len(os.listdir(img_path))} images in {img_path}")

    for filename in tqdm(os.listdir(anno_path)):
        if filename.endswith('.npy') and 'exp' in filename:
            img_filename = filename.split('_')[0] + '.jpg'  # find corresponding image

            anno_file_path = os.path.join(anno_path, filename)
            img_file_path = os.path.join(img_path, img_filename)
            exp_label = int(np.load(anno_file_path))

            # Create a target folder for each expression class
            label_folder = os.path.join(target_img_folder, str(exp_label))
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
                print(f"created folder: {label_folder}")

            # Copy the image to the target folder
            if os.path.exists(img_file_path):
                dest_path = os.path.join(label_folder, img_filename)
                shutil.copy(img_file_path, dest_path)
            else:
                print(f"Image {img_filename} not found in {img_path}")

    total_imgs = 0
    for folder in os.listdir(target_img_folder):
        folder_path = os.path.join(target_img_folder, folder)
        num_imgs = len(os.listdir(folder_path))
        total_imgs += num_imgs

        print(f"{num_imgs} images copied from {img_path} to {folder_path}")
    print(f"{total_imgs} images created under {target_img_folder}")  # 287651 train, 3999 test


if __name__ == '__main__':
    npy_labels_to_image_folders(anno_path='/home/mang/Downloads/AffectNet/train_set/annotations',
                                img_path='/home/mang/Downloads/AffectNet/train_set/images',
                                target_img_folder='/home/mang/Downloads/AffectNet/train')
