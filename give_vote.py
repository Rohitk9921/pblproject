from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

# Function for text-to-speech
def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

# Initialize camera and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists('data/'):
    os.makedirs('data/')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")
COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

# Function to check if voter has already voted
def check_if_exists(value):
    try:
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except FileNotFoundError:
        print("File not found or unable to open the CSV file.")
    return False

# Welcome message
speak("Welcome to the Smart Voting System.")
speak("Please look at the camera for face verification.")
speak("Press 1 for BJP, 2 for Congress, 3 for AAP, 4 for NOTA.")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    output = None  # Initialize output
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Votes.csv")
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    
    imgBackground[370:370 + 480, 225:225 + 640] = frame
    cv2.imshow('frame', imgBackground)
    k = cv2.waitKey(1)
    
    if output is not None:
        voter_exist = check_if_exists(output[0])
        if voter_exist:
            speak("YOU HAVE ALREADY VOTED")
            break

        vote_options = {ord('1'): "BJP", ord('2'): "CONGRESS", ord('3'): "AAP", ord('4'): "NOTA"}
        if k in vote_options:
            party = vote_options[k]
            speak(f"Your vote for {party} has been recorded.")
            time.sleep(5)
            
            with open("Votes.csv", "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow([output[0], party, date, timestamp])
            
            speak("Thank you for participating in the elections.")
            break

video.release()
cv2.destroyAllWindows()