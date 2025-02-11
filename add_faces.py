import cv2
import pickle
import numpy as np
import os

# Ensure data directory exists
data_path = 'data/'
os.makedirs(data_path, exist_ok=True)

# Initialize webcam and face detection
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not access the webcam.")
    exit()

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = []
framesTotal = 51  # Total frames to capture
captureAfterFrame = 2  # Capture every nth frame
frame_counter = 0

# Load existing voter data
names_file = os.path.join(data_path, 'names.pkl')
faces_file = os.path.join(data_path, 'faces_data.pkl')

if os.path.exists(names_file):
    with open(names_file, 'rb') as f:
        existing_names = pickle.load(f)
else:
    existing_names = []

# Get voter ID (Aadhar number)
while True:
    voter_id = input("Enter your Aadhar number: ").strip()
    
    if voter_id in existing_names:
        print("Error: This Aadhar number is already registered.")
        exit()
    
    if voter_id:
        break
    
    print("Aadhar number cannot be empty. Please enter a valid ID.")

print("Face capturing started. Please look at the camera.")

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not capture video.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cropped_face = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(cropped_face, (50, 50))

        if len(faces_data) < framesTotal and frame_counter % captureAfterFrame == 0:
            faces_data.append(resized_face)

        frame_counter += 1
        cv2.putText(frame, f"Captured: {len(faces_data)}/{framesTotal}", (50, 50), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Registration', frame)
    
    # Exit if 'q' is pressed or face data is collected
    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= framesTotal:
        break

# Release resources
video.release()
cv2.destroyAllWindows()

if len(faces_data) < framesTotal:
    print("Insufficient face data collected. Try again.")
    exit()

# Convert to numpy array and reshape
faces_data = np.array(faces_data).reshape((framesTotal, -1))

# Append voter ID to names.pkl
existing_names += [voter_id] * framesTotal
with open(names_file, 'wb') as f:
    pickle.dump(existing_names, f)

# Append face data to faces_data.pkl
if os.path.exists(faces_file):
    with open(faces_file, 'rb') as f:
        existing_faces = pickle.load(f)
    faces_data = np.append(existing_faces, faces_data, axis=0)

with open(faces_file, 'wb') as f:
    pickle.dump(faces_data, f)

print(f"Face data for Aadhar ID {voter_id} successfully stored.")
