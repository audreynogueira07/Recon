import os
import cv2
import dlib
import numpy as np
import pickle

#Use the function below to send a couple of pictures from a directory if you haven't created the 'pkl' database file before.
def create_face_database():
    # Initialize the dlib face detector
    detector = dlib.get_frontal_face_detector()

    # Load the dlib facial landmarks predictor model
    predictor = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')

    # Load the dlib face recognition model
    facerec = dlib.face_recognition_model_v1('resources/dlib_face_recognition_resnet_model_v1.dat')

    # This is our "database"
    faces_database = {}

    # Iterate over all images in the known faces folder
    for filename in os.listdir('known_faces'):
        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Load the image
            img_known = cv2.imread(f'known_faces/{filename}')

            # Convert the image to grayscale
            gray_known = cv2.cvtColor(img_known, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces_known = detector(gray_known)

            # For each face, extract landmarks and the embedding, and add it to our database
            for i, face_known in enumerate(faces_known):
                # Get the landmarks of the face
                shape_known = predictor(gray_known, face_known)

                # Get the face descriptor (embedding)
                face_descriptor_known = facerec.compute_face_descriptor(img_known, shape_known)

                # Add the descriptor to our database
                # Certify that you're sending only one face per photo
                faces_database[filename] = np.array(face_descriptor_known)
                
                #For more than one, you can use this code -> faces_database[f'{filename}_{i+1}'] = np.array(face_descriptor_known)




    # Save the database to a file using pickle
    with open('faces_database.pkl', 'wb') as f:
        pickle.dump(faces_database, f)
    
    print("Database created successfully.")

def load_face_database():
    # Load the face database from a file using pickle
    with open('faces_database.pkl', 'rb') as f:
        faces_database = pickle.load(f)
    
    print("Database loaded successfully.")
    return faces_database


create_face_database()
load_face_database()
