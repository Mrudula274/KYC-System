import cv2
import numpy as np

# Load the Aadhar card image
image = cv2.imread("hello.jpeg")

# Initialize a counter for ID
face_id = 0

# Define a function to extract and save the face
def extract_and_save_face(image):
    global face_id  # Access the global face_id variable

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any faces were detected
    if len(faces) > 0:
        # Assuming the first face found is the person's face
        x, y, w, h = faces[0]

        # Extract the face
        extracted_face = image[y:y+h, x:x+w]

        # Save the extracted face with an incremented ID
        cv2.imwrite(f"data/user_{face_id}.jpg", extracted_face)

        # Increment the face_id
        face_id += 1

        # Display a success message
        print(f"Face extracted and saved as user_{face_id}.jpg")

        # Display the extracted face
        cv2.imshow("Extracted Face", extracted_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected in the image.")

# Call the function to extract and save the face
extract_and_save_face(image)
