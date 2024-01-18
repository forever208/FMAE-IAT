import base64
import csv
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import zipfile
import io

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_MS_Celeb_1M(filename, outputDir):
    with open(filename, 'r') as tsvF:
        reader = csv.reader(tsvF, delimiter='\t')
        i = 0
        for row in reader:
            MID, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])

            saveDir = os.path.join(outputDir, MID)
            savePath = os.path.join(saveDir, "{}-{}.jpg".format(imgSearchRank, faceID))

            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
                # print("makedirs {}".format(saveDir))
            with open(savePath, 'wb') as f:
                f.write(data)

            i += 1
            if i % 1000 == 0:
                print("Extract {} images".format(i))


def video_2_images(video_folder=None, output_folder=None):
    """
    extract one frame for one second of a video
    """

    img_num = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # loop over each folder
    for subfolder_name in os.listdir(video_folder):
        subfolder_path = os.path.join(video_folder, subfolder_name)

        # loop over each video
        for video_filename in os.listdir(subfolder_path):
            video_path = os.path.join(subfolder_path, video_filename)
            print(f"processing {video_path}")

            # Check if the file is a video file
            if video_filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                cap = cv2.VideoCapture(video_path)  # Open the video file
                fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second (fps) of the video
                frame_number = 0  # Initialize variables

                while True:
                    ret, frame = cap.read()  # Read the frame
                    if not ret:  # Break the loop if the end of the video is reached
                        break

                    # Save one frame per second
                    if frame_number % int(fps) == 0:
                        output_filename = f"{img_num}_{video_filename}_frame_{frame_number // int(fps)}.jpg"
                        output_path = os.path.join(output_folder, output_filename)
                        # frame = cv2.resize(frame, (224, 224))
                        cv2.imwrite(output_path, frame)
                        img_num += 1

                    frame_number += 1

                cap.release()  # Release the video capture object

            print(f"currently {img_num} images extracted into {output_folder}")


def resize_dataset(folder_path):
    """
    delete the image (w<224 or h<224)
    resize the other images into 224x224
    """

    total_images = 0
    delete_images = 0
    resized_images = 0

    # Loop over all folders in the given directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(root, file)

                # Open the image using Pillow
                img = Image.open(file_path)
                total_images += 1

                # Get the width and height of the image
                width, height = img.size

                # deal with noisy dataset
                if width==224 and height==224:
                    pass
                elif width < 224 or height < 224:
                    os.remove(file_path)
                    delete_images += 1
                elif width / height > 1.5 or height / width > 1.5:
                    os.remove(file_path)
                    delete_images += 1
                else:
                    # Resize to 224x224 for other images
                    img_resized = img.resize((224, 224))
                    img_resized.save(file_path, format="JPEG")
                    resized_images += 1

                # if width < 100 or height < 100:
                #     os.remove(file_path)
                #     delete_images += 1
                # else:
                #     img_resized = img.resize((224, 224))
                #     img_resized.save(file_path, format="JPEG")
                #     # os.remove(file_path)
                #     resized_images += 1

        print(f"{total_images} total images, {delete_images} images deleted, {resized_images} images resized")


def folder_datasets_2_npz_datasets(root_folder=None, npy_folder=None):
    """
    save one dataset into a single npy file
    """

    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder)

    for foldername in tqdm(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, foldername)
        images = []

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                if os.path.isfile(img_path):
                    try:
                        img = Image.open(img_path)

                        # Convert the image to RGB mode if it is grayscale
                        if img.mode == 'L':
                            img = img.convert('RGB')

                        width, height = img.size
                        if width == 224 and height == 224:
                            pass
                        else:
                            print(f"{img_path} is not size 224x224")
                            img = img.resize((224, 224))

                        img_array = np.array(img)
                        if not img_array.shape[2] == 3:
                            print(img_array.shape)
                        images.append(img_array)
                    except Exception as e:
                        print(f"Error reading image {img_path}: {str(e)}")

        print(f"{len(images)} images found in {foldername}")
        images_array = np.array(images, dtype=np.uint8)
        npy_path = npy_folder + '/' + foldername + '.npz'
        np.savez_compressed(npy_path, data=images_array)
        print(f"{images_array.shape[0]} images saved into: {npy_path}")


def verify_npz_images(npz_path=None):
    npz_file = np.load(npz_path)
    image_data = npz_file['data']

    # Visualize the image using Matplotlib
    plt.imshow(image_data[0])  # Assuming the image is grayscale
    plt.show()

    # Close the npz_file
    npz_file.close()


def read_img_from_zip(zip_file_path=None):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()  # Get a list of all files in the zip archive

        # Iterate over each file in the zip archive
        for file_name in file_list:
            if file_name.lower().endswith('.jpg'):  # Check if the file is a JPG image
                image_data = zip_ref.read(file_name)  # Read the image data from the zip archive
                image_data_io = io.BytesIO(image_data)  # Create a BytesIO object to wrap the image data

                img = Image.open(image_data_io)  # Open the image using PIL

                # img.show()
                img_array = np.array(img)
                print(img_array.shape)
                print(img_array)


if __name__ == "__main__":
    # read_MS_Celeb_1M('/home/mang/Downloads/MS-Celeb-1M/data/croped_face_images/FaceImageCroppedWithOutAlignment.tsv',
    #                  '/home/mang/Downloads/MS-Celeb-1M/imgs')
    # video_2_images(video_folder='/home/mang/Downloads/DISFA',
    #                output_folder='/home/mang/Downloads/DISFA_imgs')
    # resize_dataset('/home/mang/Downloads/AffectNet_frames')
    # folder_datasets_2_npz_datasets(root_folder='/home/mang/Downloads/face_datasets/MegaFace',
    #                                npy_folder='/home/mang/Downloads/face_datasets/MegaFace-npz')
    # verify_npz_images(npz_path='/home/mang/Downloads/face_datasets/AffectNet-npz/1.npz')
    read_img_from_zip(zip_file_path='/home/mang/Downloads/face_datasets/AffectNet.zip')