import os  # Import os module for file and directory operations
import cv2  # Import OpenCV library for computer vision tasks
import numpy as np  # Import NumPy library for numerical operations

# Create LBPH face recognizer object
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define path to the dataset directory
path = "dataset"

# Function to retrieve images with corresponding IDs from the dataset directory
def get_images_with_id(path):
    # Create a list of image paths by joining the directory path with file names
    images_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []  # List to store face images
    ids = []  # List to store corresponding IDs
    # Iterate through each image path
    for single_image_path in images_paths:
        # Read image using cv2 in grayscale mode
        faceImg = cv2.imread(single_image_path, cv2.IMREAD_GRAYSCALE)
        # Extract ID from the image file name
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        print(id)  # Print the ID (optional)
        faces.append(faceImg)  # Append face image to the faces list
        ids.append(id)  # Append ID to the IDs list
        # Display the face image during training (optional)
        cv2.imshow("Training", faceImg)
        cv2.waitKey(10)  # Wait for 10 milliseconds (optional)
    # Convert IDs and faces lists to NumPy arrays
    return np.array(ids), faces

# Call the function to retrieve IDs and faces from the dataset
ids, faces = get_images_with_id(path)

# Train the recognizer using the retrieved IDs and faces
recognizer.train(faces, ids)

# Save the trained recognizer model to a file
recognizer.save("recognizer/trainingdata.yml")

# Close all OpenCV windows
cv2.destroyAllWindows()
