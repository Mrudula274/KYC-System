import cv2
import easyocr
import re
import os
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO

face_filename = ""
def extract_data_and_image(image_path):
    data = []

    # Function to extract and save the face
    def extract_and_save_face(image,extracted_img_name):
        # Convert the image to grayscale for face detection
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
            extracted_face = image[y:y + h, x:x + w]

            # Convert BGR image to RGB
            extracted_face_rgb = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2RGB)

            global face_filename
            # Save the extracted face with an incremented ID
            face_filename = f"data/{extracted_img_name}.jpg"
            cv2.imwrite(face_filename, extracted_face_rgb)

            # Display a success message
            print(f"Face extracted and saved as {extracted_img_name}.jpg")
            print(f"Face extracted and saved as {face_filename}")
        # Load the image using the provi

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        image_path.save(temp_image)
        #imagep = temp_image.name
        print(image_path)


    image = Image.open(image_path)

    # Convert the PIL image to bytes
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # EasyOCR for text extraction
    reader = easyocr.Reader(['mr', 'en'], gpu=False)
    texts = reader.readtext(image_bytes)

    threshold = 0.25
    for t in texts:
        bbox, text, score = t
        data.append(text)

    # Join the list into a single string
    data_text = ' '.join(data)
    data_text= data_text.replace("Government of India", "")




    # Define a regular expression pattern to match the name format on the Aadhaar card
    # This pattern assumes the name is in uppercase and may contain spaces

    # Find the name using the regular expression pattern

    # Define regular expressions to match date of birth, gender, and Aadhaar number
    english_name_pattern = r'[A-Z][a-z]+\s?[A-Z]?[a-z]+\s?[A-Z]?[a-z]*'
    dob_pattern = r'\d{2}/\d{2}/\d{4}'
    gender_pattern = r'FEMALE|MALE'
    aadhaar_pattern = r'\d{4}\s+\d{4}\s+\d{4}'


    # Find all occurrences of names
    name_matches = re.findall(english_name_pattern, data_text)

    # Filter out names containing "Government"
    filtered_names = [name for name in name_matches if "Government" not in name]

    # Extract the first non-government name
    name = filtered_names[0] if filtered_names else None


    # Find all occurrences of each pattern in the data

    dob_matches = re.findall(dob_pattern, data_text)
    gender_matches = re.findall(gender_pattern, data_text)
    aadhaar_matches = re.findall(aadhaar_pattern, data_text)

    # Extract the first occurrence of each data type (dob, gender, aadhaar)
    """
    name = None
    for match in name_matches:
        if "Government of India" not in match:
            name = match
            break
    """
    name = name_matches[0] if name_matches else None
    dob = dob_matches[0] if dob_matches else None
    gender = gender_matches[0] if gender_matches else None
    aadhaar = aadhaar_matches[0] if aadhaar_matches else None

    extracted_img_name = name.split(' ')[0] + "_" + aadhaar.split(' ')[2]


    print("Extracted Name:", name)
    print("Extracted Date of Birth:", dob)
    print("Extracted Gender:", gender)
    print("Extracted Aadhaar Number:", aadhaar)
    print("Extracted image name:", extracted_img_name)

    extract_and_save_face(np.array(image), extracted_img_name)

    # Return the extracted data and image path
    return {
        "name": name,
        "dob": dob,
        "gender": gender,
        "aadhaar_number": aadhaar,
        "image_path": face_filename
    }
"""
image_path = "shruti1.jpeg"
extracted_data = extract_data_and_image(image_path)
print("Extracted Data:", extracted_data)
"""