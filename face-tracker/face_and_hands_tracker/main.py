import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Create a black image to draw on
height, width = 600, 800 # adjust this according to your needs
img = np.zeros((height, width, 3))

while True:
    # Read each frame from the webcam
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Clear the image on each frame
    img.fill(0)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # calculate the center point
        center_x = x + w//2
        center_y = y + h//2
        
        # shift to image coordinates
        mapped_x = int(width * (center_x / frame.shape[1]))
        mapped_y = int(height * (center_y / frame.shape[0]))
        
        # Draw a circle at the center point on the other image
        cv2.circle(img, (mapped_x, mapped_y), 5, (0, 255, 0), -1)
    
    # Display the resulting images
    cv2.imshow('Face Tracking', frame)
    cv2.imshow('Face Mapping', img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
