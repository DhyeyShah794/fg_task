from deepface import DeepFace
from retinaface import RetinaFace
import numpy as np
import itertools
import cv2

# img1_path = "faces/file_192_0_face_0.jpg"
# img2_path = "faces/file_187_1_face_4.jpg"
# img3_path = "faces/file_192_0_face_1.jpg"

# Construct combinations of image pairs from test1 to test8.jpg and verify for each pair
for i in range(0, 6):
  for j in range(i+1, 6):
    if i != j:
      img1_path = f"{i}.jpg"
      img2_path = f"{j}.jpg"
      result = DeepFace.verify(
        img1_path = img1_path,
        img2_path = img2_path,
        enforce_detection = False,
        detector_backend = "retinaface",
        align = True
      )
      print(f"Similarity between {img1_path} and {img2_path}: {result['verified']}")
# result = DeepFace.verify(
#   img1_path = "test6.jpg",
#   img2_path = "test7.jpg",
#   enforce_detection = False,
#   detector_backend = "retinaface"
# )

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

# embedding_objs6 = DeepFace.represent(
#   img_path = "test6.jpg"
# )

# embedding_objs7 = DeepFace.represent(
#   img_path = "test7.jpg",
#   enforce_detection = False
# )

# print(embedding_objs6[0]['embedding'])

# Calculate the similarity

# def cosine_distance(embedding1, embedding2):
#   return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# print(cosine_distance(embedding_objs6[0]['embedding'], embedding_objs7[0]['embedding']))


# import matplotlib.pyplot as plt
# faces = RetinaFace.extract_faces(img_path = "test.jpg", align = True)
# for face in faces:
#   plt.imshow(face)
#   plt.show()

# Save the extracted faces
# for idx, face in enumerate(faces):
#   cv2.imwrite(f"face_{idx}.jpg", face)

# dfs = DeepFace.find(
#   img_path = img1_path, 
#   # db_path = img1_path, 
#   detector_backend = backends[5],
#   align = True,
# )


# objs1 = DeepFace.analyze(
#   img_path = img1_path,
#   actions = ['age', 'gender', 'race', 'emotion'],
#   enforce_detection = False
# )

# objs2 = DeepFace.analyze(
#   img_path = img3_path,
#   actions = ['age', 'gender', 'race', 'emotion'],
#   enforce_detection = False
# )

# print(objs1)
# print("\n"*5)
# print(objs2)