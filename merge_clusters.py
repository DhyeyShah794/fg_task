import os
import cv2
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace
import itertools
import shutil
import random

initial_folder = "clustered_influencer_faces"
if os.path.exists("merged_clusters"):
    shutil.rmtree("merged_clusters")
shutil.copytree(initial_folder, "merged_clusters")
clustered_folder = "merged_clusters"

clusters = sorted(os.listdir(clustered_folder))

if "face_-1" in clusters:
    clusters.remove("face_-1")


def build_cluster_dict(clusters, clustered_folder):
    cluster_dict = {}

    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            cluster1 = os.listdir(os.path.join(clustered_folder, clusters[i]))
            cluster2 = os.listdir(os.path.join(clustered_folder, clusters[j]))
            # Select two random images from each cluster
            c1_img1_path = os.path.join(clustered_folder, clusters[i], random.choice(cluster1))
            c1_img2_path = os.path.join(clustered_folder, clusters[i], random.choice(cluster1))
            c2_img1_path = os.path.join(clustered_folder, clusters[j], random.choice(cluster2))
            c2_img2_path = os.path.join(clustered_folder, clusters[j], random.choice(cluster2))

            pairs = [
                (c1_img1_path, c2_img1_path),
                (c1_img1_path, c2_img2_path),
                (c1_img2_path, c2_img1_path),
                (c1_img2_path, c2_img2_path)
            ]

            # Calculate the average distance between all 4 pairs of images

            results = []
            for pair in pairs:
                result = DeepFace.verify(
                    img1_path = pair[0],
                    img2_path = pair[1],
                    enforce_detection = False,
                    model_name = "Facenet512",
                    detector_backend = "retinaface",
                    align = True
                )
                results.append(result['distance'])
            
            avg_distance = sum(results) / len(results)
            avg_distance = round(avg_distance, 5)
            if avg_distance not in cluster_dict:
                cluster_dict[avg_distance] = [clusters[i], clusters[j]]

    return cluster_dict


def update_dict(cluster_dict, cluster1, cluster2, merged_cluster):
    """Update the cluster dictionary after merging two clusters"""
    for k, v in cluster_dict.items():
        if v[0] == cluster1 or v[0] == cluster2:
            cluster_dict[k][0] = merged_cluster
        if v[1] == cluster1 or v[1] == cluster2:
            cluster_dict[k][1] = merged_cluster
    return cluster_dict


def merge_clusters(cluster_dict, threshold):
    """Merge clusters with the smallest average distance below the threshold"""
    key = min(cluster_dict.keys())

    if key <= threshold:
        value = cluster_dict[key]
        cluster1 = os.listdir(os.path.join(clustered_folder, value[0]))
        cluster2 = os.listdir(os.path.join(clustered_folder, value[1]))
        merged_cluster = value[0] + "_" + value[1]
        os.makedirs(os.path.join(clustered_folder, merged_cluster), exist_ok=True)
        
        print(f"Merging clusters {value[0]} and {value[1]} into {merged_cluster}")

        for img in cluster1:
            shutil.copy(os.path.join(clustered_folder, value[0], img), os.path.join(clustered_folder, merged_cluster, img))
        for img in cluster2:
            shutil.copy(os.path.join(clustered_folder, value[1], img), os.path.join(clustered_folder, merged_cluster, img))
        if os.path.exists(os.path.join(clustered_folder, value[0])):
            shutil.rmtree(os.path.join(clustered_folder, value[0]))
        if os.path.exists(os.path.join(clustered_folder, value[1])):
            shutil.rmtree(os.path.join(clustered_folder, value[1]))

        cluster_dict = update_dict(cluster_dict, value[0], value[1], merged_cluster)
        del cluster_dict[key]


threshold = 0.37
cluster_dict = build_cluster_dict(clusters, clustered_folder)
count = 0

while min(test_dict.keys()) <= threshold and count < 8:
    merge_clusters(test_dict, threshold)
    count += 1

print("Clustering complete.")
