# Load the images from faces folder and filter the faces based on a 2nd pass using DeepFace

import os
import cv2
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace

input_folder = "faces"
output_folder = "filtered_faces"

os.makedirs(output_folder, exist_ok=True)


def filter_face(img_path, output_dir):
    resp = RetinaFace.detect_faces(img_path)
    if len(resp) == 0:
        return False
    elif 'face_1' in resp:
        return resp['face_1']['score'] > 0.8


print(filter_face("test5.jpg", ""))

# for img_file in os.listdir(input_folder):
#     if img_file.endswith((".jpg")):
#         img_path = os.path.join(input_folder, img_file)
#         if filter_face(img_path, output_folder):
#             output_path = os.path.join(output_folder, img_file)
#             img = cv2.imread(img_path)
#             cv2.imwrite(output_path, img)

# print(f"Faces have been filtered and saved in '{output_folder}'.")
