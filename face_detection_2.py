import os
import cv2
from video_facenet.pipelines import process_video

input_folder = "data"
output_folder = "influencer_faces"

os.makedirs(output_folder, exist_ok=True)

for video_file in os.listdir(input_folder):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(input_folder, video_file)
        process_video(video_path, model_path="/lab/20180402-114759/facenet.pb")