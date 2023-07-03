import cv2
import dlib
import numpy as np
import pickle

def load_program():
    def face_detector():
        # Prompt the user to enter the path to the image
        image = input("Please enter the path to the image: ")

        # Initialize the dlib face detector
        detector = dlib.get_frontal_face_detector()

        # Load the dlib facial landmarks predictor model
        predictor = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')

        # Load the dlib face recognition model
        facerec = dlib.face_recognition_model_v1('resources/dlib_face_recognition_resnet_model_v1.dat')

        # Load the face database
        with open('faces_database.pkl', 'rb') as f:
            faces_database = pickle.load(f)

        # Load the image
        img = cv2.imread(image)

        # Convert the image to grayscale
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)

        # Iterate over the detected faces
        for face in faces:
            # Extract the coordinates of the bounding box around the face
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            # Draw a rectangle around the face in the image
            cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

            # Find facial landmarks in the face
            landmarks = predictor(image=gray, box=face)

            # Compute the face descriptor (embedding)
            face_descriptor = facerec.compute_face_descriptor(img, landmarks)

            # Search for the closest matching face in the database
            distances = {}
            for name, saved_descriptor in faces_database.items():
                # Calculate the Euclidean distance between the embeddings
                distance = np.linalg.norm(saved_descriptor - np.array(face_descriptor))
                distances[name] = distance

            # Find the smallest distance
            best_match = min(distances, key=distances.get)

            # Load the image of the best match
            img_match = cv2.imread(f'known_faces/{best_match}')

            # Draw the name of the best match on the image
            text = f"{best_match} "
            cv2.putText(img, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Resize the image to 400x400 before displaying
        img = cv2.resize(img, (400, 400))

        # Display the image with the faces drawn
        cv2.imshow(winname="Face Detector", mat=img)

        # Wait for a key press to close the window
        cv2.waitKey(delay=0)

        # Close all opened windows
        cv2.destroyAllWindows()

        # Return the image of the best match
        return img_match


    img_match = face_detector()

    # Resize the best match image to 400x400 before displaying
    img_match = cv2.resize(img_match, (400, 400))

    # Display the image of the best match
    cv2.imshow("Best Match", img_match)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

load_program()
