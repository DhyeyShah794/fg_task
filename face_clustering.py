import os
from deepface import DeepFace
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shutil
import cv2
import itertools
import pickle

data = pickle.loads(open("face_encodings.pickle", "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]
image_paths = [os.path.basename(d["imagePath"]) for d in data]

scaler = StandardScaler()
normalized_encodings = scaler.fit_transform(encodings)

clt = DBSCAN(eps=0.36, min_samples=2, metric="cosine", n_jobs=-1)
clt.fit(normalized_encodings)
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("Number of unique faces: {}".format(numUniqueFaces))

output_folder = "clustered_influencer_faces"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

for labelID in labelIDs:
    os.makedirs(os.path.join(output_folder, f"face_{labelID}"))

for i, label in enumerate(clt.labels_):
    image = cv2.imread(data[i]["imagePath"])
    (top, right, bottom, left) = data[i]["loc"]
    face = image[top:bottom, left:right]
    face = cv2.resize(face, (96, 96))
    cv2.imwrite(os.path.join(output_folder, f"face_{label}", image_paths[i]), face)

print("Clustering complete.")
