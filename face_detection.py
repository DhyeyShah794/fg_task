import cv2
import os

input_folder = "data"
output_folder = "faces"

os.makedirs(output_folder, exist_ok=True)

model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"

face_net = cv2.dnn.readNetFromCaffe(config_file, model_file)

def extract_faces_from_video(video_path, output_dir, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split('.')[0]

    frame_count = 0
    face_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame (reduce computational load)
        if frame_count % frame_skip == 0:
            # Prepare the frame for face detection
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.85:  # Confidence threshold
                    # Get the bounding box of the face
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    (x1, y1, x2, y2) = box.astype("int")

                    # Extract the face region
                    face = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    face_height, face_width = face.shape[:2]

                    # Only save the face if its dimensions are not both less than 80
                    if face_height >= 85 or face_width >= 85:
                        face_path = os.path.join(output_dir, f"{video_name}_face_{face_count}.jpg")
                        cv2.imwrite(face_path, face)
                        face_count += 1

        frame_count += 1

    cap.release()


# Process all videos in the input folder
for video_file in os.listdir(input_folder):
    if video_file.endswith((".mp4")):
        video_path = os.path.join(input_folder, video_file)
        extract_faces_from_video(video_path, output_folder)

print(f"Faces have been extracted and saved in '{output_folder}'.")
