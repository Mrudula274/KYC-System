"""
from flask_migrate import Migrate
from flask import Flask, render_template, flash, redirect, url_for, session, logging, request
from flask_sqlalchemy import SQLAlchemy
from data_extract import extract_data_and_image
from PIL import Image
from match import match_faces

app = Flask(__name__)
app.secret_key = "abc"
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/Mrudula/Downloads/flask-login-register-form-master/flask-login-register-form-master/database.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:Mysqlmru27%40@localhost:3306/kyc'
db = SQLAlchemy(app)
migrate = Migrate(app, db)


class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)


class Details(db.Model):
    aadhar_number = db.Column(db.String(20), primary_key=True)
    id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(255))
    dob = db.Column(db.String(10))
    gender = db.Column(db.String(10))
    image_path = db.Column(db.String(255))


def is_valid_image(file):
    try:
        Image.open(file)
        return True
    except Exception:
        return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]

        login = user.query.filter_by(username=uname, password=passw).first()
        if login:
            session['user_id'] = login.id
            return redirect(url_for("dashboard"))
        else:
            flash("Please Register before login.", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']

        register = user(username=uname, email=mail, password=passw)
        db.session.add(register)
        db.session.commit()
        session['user_id'] = register.id
        return redirect(url_for("dashboard"))
    return render_template("register.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if 'user_id' in session:
        if request.method == "POST":
            # Check if a file was uploaded
            if 'file' in request.files:
                uploaded_file = request.files['file']
                if uploaded_file.filename != '':
                    if not is_valid_image(uploaded_file):
                        flash("Invalid image format. Please upload a valid image.", "danger")
                    else:
                        extracted_data = extract_data_and_image(uploaded_file)

                    # Create a new Details record and store the extracted data
                    if 'user_id' in session:
                        user_id = session['user_id']
                        new_details = Details(
                            aadhar_number=extracted_data.get('aadhaar_number'),
                            id=user_id,
                            name=extracted_data.get('name'),
                            dob=extracted_data.get('dob'),
                            gender=extracted_data.get('gender'),
                            image_path=extracted_data.get('image_path')
                            # Use the image path from extraction
                        )
                        try:
                            db.session.add(new_details)
                            db.session.commit()
                            # os.remove(uploaded_file_path)

                        except Exception as e:
                            print("Error:", str(e))

                        user_details = Details.query.filter_by(id=session['user_id']).first()
                        image_path_from_db = user_details.image_path
                        print("Image path from DB:", image_path_from_db)
                        match_result = match_faces(image_path_from_db, uploaded_file)
                        print("Match Result:", match_result)
                        flash("Data stored successfully.", "success")
                    else:
                        flash("Please log in to access the dashboard.", "danger")

            else:
                flash("Please upload a file first.", "danger")

        user_id = session['user_id']
        user = user_id
        if user:
            return render_template("dashboard.html", user=user)

    flash("Please log in to access the dashboard.", "danger")
    return redirect(url_for("login"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
"""
import subprocess
from flask_migrate import Migrate
from flask import Flask, render_template, flash, redirect, url_for, session, logging, request,jsonify
from flask_sqlalchemy import SQLAlchemy
from data_extract import extract_data_and_image
from PIL import Image
from match import match_faces
import base64
import cv2
import os
import face_recognition
import numpy as np
import sys


app = Flask(__name__)
app.secret_key = "abc"
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/Mrudula/Downloads/flask-login-register-form-master/flask-login-register-form-master/database.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:Mysqlmru27%40@localhost:3306/kyc'
db = SQLAlchemy(app)
migrate = Migrate(app, db)


image_path_from_db=""
class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)


class Details(db.Model):
    aadhar_number = db.Column(db.String(20), primary_key=True)
    id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(255))
    dob = db.Column(db.String(10))
    gender = db.Column(db.String(10))
    image_path = db.Column(db.String(255))


def is_valid_image(file):
    try:
        Image.open(file)
        return True
    except Exception:
        return False


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/capture')
def done():
    return render_template('cameraCapture.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]

        login = user.query.filter_by(username=uname, password=passw).first()
        if login:
            session['user_id'] = login.id
            return redirect(url_for("dashboard"))
        else:
            flash("Please Register before login.", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']

        register = user(username=uname, email=mail, password=passw)
        db.session.add(register)
        db.session.commit()
        session['user_id'] = register.id
        return redirect(url_for("dashboard"))
    return render_template("register.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if 'user_id' in session:
        if request.method == "POST":
            # Check if a file was uploaded
            if 'file' in request.files:
                uploaded_file = request.files['file']
                if uploaded_file.filename != '':
                    if not is_valid_image(uploaded_file):
                        flash("Invalid image format. Please upload a valid image.", "danger")
                    else:
                        extracted_data = extract_data_and_image(uploaded_file)

                    # Create a new Details record and store the extracted data
                    if 'user_id' in session:
                        user_id = session['user_id']
                        new_details = Details(
                            aadhar_number=extracted_data.get('aadhaar_number'),
                            id=user_id,
                            name=extracted_data.get('name'),
                            dob=extracted_data.get('dob'),
                            gender=extracted_data.get('gender'),
                            image_path=extracted_data.get('image_path')
                            # Use the image path from extraction
                        )
                        try:
                            db.session.add(new_details)
                            db.session.commit()
                            # os.remove(uploaded_file_path)

                        except Exception as e:
                            print("Error:", str(e))

                        global image_path_from_db
                        user_details = Details.query.filter_by(id=session['user_id']).first()
                        image_path_from_db = user_details.image_path
                        print("Image path from DB:", image_path_from_db)
                        # match_result = match_faces(image_path_from_db, uploaded_file)
                        # print("Match Result:", match_result)
                        msg="data stored successfully"

                        # flash("Data stored successfully.", "success")
                        return jsonify({'message': msg})
                    else:

                        flash("Please log in to access the dashboard.", "danger")

            else:
                flash("Please upload a file first.", "danger")

        user_id = session['user_id']
        user = user_id
        if user:
            return render_template("dashboard.html", user=user)

    flash("Please log in to access the dashboard.", "danger")

    return redirect(url_for("login"))


@app.route('/save_photo', methods=['POST'])
def save_photo():
    data_url = request.json.get('photo_data')
    if data_url:
        # Convert the base64 data URL to binary image data
        image_data = base64.b64decode(data_url.split(',')[1])

        # Call the face_capture.py script as a subprocess and pass the image data as an argument

        output_folder = "face_images"
        os.makedirs(output_folder, exist_ok=True)

        msg = detect_and_save_face(image_data, output_folder)
        print(msg)
        return jsonify({'message': msg})

    return jsonify({'error': 'Invalid photo data'})


# Function to save a face image to a folder
def save_face(face, folder_path):
    msg = ""
    count = len(os.listdir(folder_path)) + 1
    face_name = f"user.{count}.jpg"
    cv2.imwrite(os.path.join(folder_path, face_name), face)
    print("Face image saved successfully.")
    image1 = face_recognition.load_image_file(image_path_from_db)
    image2 = face_recognition.load_image_file("face_images/" + face_name)

    # Find face encodings for both images
    face_encoding1 = face_recognition.face_encodings(image1)
    face_encoding2 = face_recognition.face_encodings(image2)

    # Check if at least one face was found in each image
    if len(face_encoding1) > 0 and len(face_encoding2) > 0:
        # Convert face encodings to numpy arrays
        face_encoding1 = np.array(face_encoding1[0])  # Assuming there's only one face in each image
        face_encoding2 = np.array(face_encoding2[0])

        # Calculate the face distance (lower is better)
        face_distance = face_recognition.face_distance([face_encoding1], face_encoding2)

        # Set a matching threshold (you can adjust this threshold)
        matching_threshold = 0.6

        # Check if the face distance is below the threshold
        if face_distance[0] < matching_threshold:
            msg = "Match successful"
            print("Match successful")

        else:
            msg = "Match unsuccessful"
            print("Match unsuccessful")
    else:
        msg = "image not found"
        print("image not found")
    try:
        os.remove(os.path.join(folder_path, face_name))
        print("Image deleted successfully.")
    except OSError as e:
        print(f"Error deleting image: {e}")
    return msg


def detect_and_save_face(image_data, output_folder):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    # Load the Haar Cascade classifier
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Crop and save each detected face
        face = image[y:y + h, x:x + w]
        msg = save_face(face, output_folder)
    return msg


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)