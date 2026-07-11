import cv2
import csv
from datetime import datetime
import time

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Create a line on the left side of the frame
line_x1 = int(cap.get(3) / 2)
line_y = int(cap.get(4))
line_x2 = 280

# Initialize the face count
face_count = 0
previous_count = 0

now = datetime.now()
output_file = now.strftime("%d%m%Y") + ".csv"
timestamp = time.time()
with open(output_file, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    #writer.writerow(["Time","Count"])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 6)
    
    # Draw the line on the frame
    cv2.line(frame, (line_x1, 0), (line_x1, line_y), (255, 0, 0), 1)
    cv2.line(frame, (line_x2, 0), (line_x2, line_y), (255, 0, 0), 1)

    # Draw a rectangle around the faces
    count = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Check if the face is on the left side of the line
        if x + w/2 < line_x1 and x + w/2 > line_x2:
            # Draw a circle at the center of the face
            cv2.circle(frame, (x + w//2, y + h//2), 2, (0, 255, 0), -1)
            count += 1
    if(count >= previous_count):
        face_count += count - previous_count
    previous_count = count
    # Display the resulting frame
    cv2.putText(frame, "Count: {}".format(face_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    # Save the count to the CSV file
    # Check if 15 minutes have passed
    elapsed_time = time.time() - timestamp
    if elapsed_time > 60:  # 15 minutes * 60 seconds/minute
                with open(output_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    #writer.writerow(["Time","Count"])
                    now = datetime.now()
                    writer.writerow([now.strftime("%H:%M"),face_count])
                    face_count = 0
                    timestamp = time.time()


    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
