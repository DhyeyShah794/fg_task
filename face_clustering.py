import os
from deepface import DeepFace
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import itertools

face_images_folder = "faces_copy"

embeddings = []
image_paths = []

for face_image in os.listdir(face_images_folder):
    if face_image.endswith(".jpg"):
        image_path = os.path.join(face_images_folder, face_image)
        try:
            embedding_obj = DeepFace.represent(img_path=image_path, model_name="VGG-Face", enforce_detection = False)
            for obj in embedding_obj:
                embeddings.append(obj["embedding"])
                image_paths.append(image_path)
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

embeddings = np.array(embeddings)

scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(embeddings)

dbscan = DBSCAN(eps=0.65, min_samples=3, metric="cosine")
labels = dbscan.fit_predict(normalized_embeddings)

unique_faces = {}
for idx, label in enumerate(labels):
    if label not in unique_faces:
        unique_faces[label] = []
    unique_faces[label].append(image_paths[idx])

for cluster_id, face_list in unique_faces.items():
    if cluster_id == -1:
        print("Unclustered faces:", len(face_list))
    else:
        print(f"Cluster {cluster_id}: {len(face_list)} faces")

output_clusters_folder = "clustered_influencer_faces"
os.makedirs(output_clusters_folder, exist_ok=True)

for cluster_id, face_list in unique_faces.items():
    cluster_folder = os.path.join(output_clusters_folder, f"cluster_{cluster_id}")
    os.makedirs(cluster_folder, exist_ok=True)
    for face_path in face_list:
        face_name = os.path.basename(face_path)
        output_path = os.path.join(cluster_folder, face_name)
        os.rename(face_path, output_path)

print(f"DBSCAN completed. Results temporarily saved in '{output_clusters_folder}'.")

# Now perform agglomerative clustering using DeepFace.verify. If any one image from two clusters is verified, merge the clusters. Else skip that pair.

# Remove unclustered faces
# unique_faces.pop(-1, None)
# cluster_pairs = list(itertools.combinations(unique_faces.keys(), 2))

# for cluster1, cluster2 in cluster_pairs:
#     cluster1_folder = os.path.join(output_clusters_folder, f"cluster_{cluster1}")
#     cluster2_folder = os.path.join(output_clusters_folder, f"cluster_{cluster2}")
    
#     face1 = os.listdir(cluster1_folder)[0]
#     face2 = os.listdir(cluster2_folder)[0]
#     face1_path = os.path.join(cluster1_folder, face1)
#     face2_path = os.path.join(cluster2_folder, face2)
            
#     result = DeepFace.verify(img1_path=face1_path, img2_path=face2_path, enforce_detection=False)
#     if result["verified"]:
#         for face in os.listdir(cluster2_folder):
#             face_path = os.path.join(cluster2_folder, face)
#             output_path = os.path.join(cluster1_folder, face)
#             os.rename(face_path, output_path)
#         os.rmdir(cluster2_folder)
#         print(f"Merged cluster {cluster2} into cluster {cluster1}.")
#     else:
#         print(f"Clusters {cluster1} and {cluster2} are not similar.")

# print(f"Agglomerative clustering completed. Results saved in '{output_clusters_folder}'.")
