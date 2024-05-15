import cv2  # Import OpenCV library for computer vision tasks
import numpy as np  # Import NumPy library for numerical operations
import sqlite3  # Import SQLite library for database operations

# Load pre-trained cascade classifier for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open webcam for video capture
cam = cv2.VideoCapture(0)

# Function to insert or update user information in the database
def insert_or_update(Id, Name, age):
    # Connect to SQLite database
    conn = sqlite3.connect("sqlite.db")
    # SQL command to check if user ID exists in the database
    cmd = "SELECT * FROM STUDENTS WHERE ID=?"
    # Execute SQL command with user ID as parameter
    cursor = conn.execute(cmd, (Id,))
    isRecordExist = 0  # Flag to indicate if user record exists in the database
    # Iterate through database records
    for row in cursor:
        isRecordExist = 1  # Set flag to indicate record exists if user ID is found
    # Check if user record exists
    if isRecordExist == 1:
        # Update user name and age in the database
        conn.execute("UPDATE STUDENTS SET NAME=? WHERE ID=?", (Name, Id))
        conn.execute("UPDATE STUDENTS SET AGE=? WHERE ID=?", (age, Id))
    else:
        # Insert new user record into the database
        conn.execute("INSERT INTO STUDENTS (Id,Name,age) values(?,?,?)", (Id, Name, age))
    # Commit changes to the database
    conn.commit()
    # Close database connection
    conn.close()

# Input user information from the user
Id = input('Enter User Id: ')
Name = input('Enter User Name: ')
age = input('Enter User Age: ')

# Call function to insert or update user information in the database
insert_or_update(Id, Name, age)

sampleNum = 0  # Counter to track number of captured face samples
while True:
    # Capture video frame from webcam
    ret, img = cam.read()
    # Convert captured frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    # Iterate through detected faces
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1  # Increment sample counter
        # Save captured face sample as image file
        cv2.imwrite("dataset/user." + str(Id) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
        # Draw rectangle around the detected face on the original image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display the original image with detected face
        cv2.imshow("Face", img)
        cv2.waitKey(100)  # Wait for 100 milliseconds
        cv2.waitKey(1)  # Wait for 1 millisecond
        if sampleNum > 20:  # Break loop if 20 face samples are captured
            break
    if sampleNum > 20:  # Break outer loop if 20 face samples are captured
        break

# Release webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
